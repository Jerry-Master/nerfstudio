# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
3D Gaussian Splatting data manager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Type, Union

import torch
from jaxtyping import Float
from pykdtree.kdtree import KDTree
from torch import Tensor
from torch.nn import Parameter
from typing_extensions import Literal

from nerfstudio.cameras.points import Gaussians3D
from nerfstudio.data.datamanagers.base_datamanager import (
    AnnotatedDataParserUnion,
    DataManager,
    DataManagerConfig,
)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
)
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class GaussfactoDataManagerConfig(DataManagerConfig):
    """Configuration for data manager that does not load from a dataset. Instead, it generates random poses."""

    _target: Type = field(default_factory=lambda: GaussfactoDataManager)
    """Target class to initiate."""
    dataparser: AnnotatedDataParserUnion = ColmapDataParserConfig(
        load_3D_points=True, max_2D_matches_per_3D_point=-1, colmap_path=Path("colmap/sparse/0")
    )
    """Specifies the dataparser used to unpack the data."""
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics.
    """
    initial_opacity_value: Float = 0.1
    """Initial opacity value (original paper has 0.1)"""
    intial_scale: Literal["mean_3_nn", "rand", "ones"] = "mean_3_nn"
    """Method to initialize gaussian scales"""


class GaussfactoDataManager(DataManager):  # pylint: disable=abstract-method
    """Rasterization based data manager.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: GaussfactoDataManagerConfig

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        config: GaussfactoDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"

        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time

        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")

        self.points3D_xyz = self.train_dataparser_outputs.metadata["points3D_xyz"]
        self.points3D_rgb = self.train_dataparser_outputs.metadata["points3D_rgb"]
        self.points3D_rgb = self.points3D_rgb.float() / 255  # make rgbs float
        self.points3D_errors = self.train_dataparser_outputs.metadata["points3D_error"]
        self.points3D_num_2d_correspondences = self.train_dataparser_outputs.metadata["points3D_num_points"]

        self.gaussians = self._init_gaussians()
        self.train_dataset = self.create_train_dataset()

        self.train_image_index = 0

        super().__init__()

    def _init_gaussians(self) -> Gaussians3D:
        """Initialize 3D gaussians from SfM points"""
        self.num_points = self.points3D_xyz.shape[0]
        assert self.num_points > 0, "Not enough SfM points to initialize Gaussians"
        CONSOLE.log("Number of Gaussians at initialisation : ", self.num_points)

        opacity = inverse_sigmoid(0.1 * torch.ones((self.num_points, 1), dtype=torch.float, device="cpu"))
        quat = torch.Tensor([1, 0, 0, 0]).unsqueeze(dim=0).repeat(self.num_points, 1).to(torch.float32)
        if self.config.intial_scale == "mean_3_nn":
            xyz_numpy = self.points3D_xyz.cpu().numpy()
            kd_tree = KDTree(xyz_numpy)
            dist, idx = kd_tree.query(xyz_numpy, k=4)
            mean_min_three_dis = dist[:, 1:].mean(axis=1)
            mean_min_three_dis = torch.Tensor(mean_min_three_dis).to(torch.float32)  # * scale_init_value
            scale = torch.ones(self.num_points, 3).to(torch.float32) * mean_min_three_dis.unsqueeze(dim=1)
        elif self.config.intial_scale == "rand":
            scale = torch.rand(size=(self.num_points, 3)).to(torch.float32)
        elif self.config.intial_scale == "ones":
            scale = torch.ones(size=(self.num_points, 3)).to(torch.float32)

        gaussians = Gaussians3D(
            xyzs=self.points3D_xyz,
            rgbs=self.points3D_rgb,
            opacities=opacity,
            quats=quat,
            scales=scale,
        )

        return gaussians

    def create_train_dataset(self) -> InputDataset:
        """Sets up the data loaders for training"""
        return InputDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)

    def next_train(self, step: int) -> Tuple[int, Tensor]:
        """Returns next training image index and image"""

        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        idx = image_batch["image_idx"][self.train_image_index]
        image = image_batch["image"][self.train_image_index]
        self.train_image_index += 1

        return idx, image

    def get_param_groups(
        self,
    ) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        return param_groups


def inverse_sigmoid(x):
    return -torch.log(1 / x - 1)
