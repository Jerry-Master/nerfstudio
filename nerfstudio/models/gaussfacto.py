# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
3D Gaussian Splatting model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.points import Gaussians3D
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import poses as pose_utils

import diff_rast  # make sure to import diff_rast after torch


@dataclass
class GaussfactoConfig(ModelConfig):
    """3D Gaussian Splatting Config"""

    _target: Type = field(default_factory=lambda: Gaussfacto)


class Gaussfacto(Model):
    """3D Gaussian Splatting Model

    Args:
        config: configuration to instantiate model
    """

    config: GaussfactoConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.cameras: Cameras = self.kwargs["cameras"]
        self.gaussians: Gaussians3D = self.kwargs["gaussians"]  # initialized 3D gaussians from SfM

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """

        return {}

    def get_outputs(self, idx: int, image: Tensor):
        """Projects 3D gaussians to 2D image

        Args:
            idx: image index of current training view
            image: rgb image of current training view
        """
        camera: Cameras = self.cameras[idx]  # current camera view to which gaussians are rendered onto
        c2w = camera.camera_to_worlds
        c2w = pose_utils.to4x4(c2w)
        w2c = torch.linalg.inv(c2w)
        fx = float(camera.fx)
        fy = float(camera.fy)
        width = int(camera.width)
        height = int(camera.height)

        proj_matrix = self._get_proj_matrix(w2c=w2c, fx=fx, fy=fy, width=width, height=height)

        glob_scale = 1
        rast_outs = diff_rast.rasterize(
            means3d=self.gaussians.xyzs,
            scales=self.gaussians.scales,
            glob_scale=glob_scale,
            rotations_quat=self.gaussians.quats,
            colors=self.gaussians.rgbs,
            opacity=self.gaussians.opacities,
            view_matrix=w2c,
            proj_matrix=proj_matrix,
            img_height=width,
            img_width=height,
            fx=fx,
            fy=fy,
        )

        rgb = rast_outs[1]

        outputs = {
            "rgb": rgb,
        }
        return outputs

    def _get_proj_matrix(
        self, w2c: Float[Tensor, "3 4"], fx: float, fy: float, width: int, height: int, znear=0.01, zfar=100
    ):
        top = 0.5 * height / fy * znear
        bottom = -top
        right = 0.5 * width / fx * znear
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)

        return P @ w2c

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """

        return {}

    def forward(self, idx: int, image: Tensor):
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        return self.get_outputs(idx, image)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks that run functions at the specified training iterations."""
        callbacks = []

        return callbacks
