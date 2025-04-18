# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import enum
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import resnet18, resnet50, resnext101_32x8d



class GlobalGeMPool2d(nn.Module):
    """Generalized mean pooling.

    Inputs should be non-negative.
    """

    def __init__(
        self,
        pooling_param: float,
    ):
        """
        Args:
            pooling_param: the GeM pooling parameter
        """
        super().__init__()
        self.pooling_param = pooling_param

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.reshape(N, C, H * W)  # Combine spatial dimensions
        mean = x.clamp(min=1e-6).pow(self.pooling_param).mean(dim=2)
        r = 1.0 / self.pooling_param
        return mean.pow(r)


class Implementation(enum.Enum):
    TORCHVISION = enum.auto()


class Backbone(enum.Enum):

    TV_RESNET18 = (resnet18, 512, Implementation.TORCHVISION)
    TV_RESNET50 = (resnet50, 2048, Implementation.TORCHVISION)
    TV_RESNEXT101 = (resnext101_32x8d, 2048, Implementation.TORCHVISION)

    def build(self, dims: int):
        return self.value[0](num_classes=dims, zero_init_residual=True)


class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x)


class Model(nn.Module):
    def __init__(self, backbone: str, dims: int, pool_param: float):
        super().__init__()
        self.backbone_type = Backbone[backbone]
        self.backbone = self.backbone_type.build(dims=dims)
        if pool_param > 1:
            self.backbone.avgpool = GlobalGeMPool2d(pool_param)
            fc = self.backbone.fc
            nn.init.xavier_uniform_(fc.weight)
            nn.init.constant_(fc.bias, 0)
        self.embeddings = L2Norm()

    def forward(self, x):
        x = self.backbone(x)
        return self.embeddings(x)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser = parser.add_argument_group("Model")
        parser.add_argument(
            "--backbone", default="TV_RESNET50", choices=[b.name for b in Backbone]
        )
        parser.add_argument("--dims", default=512, type=int)
        parser.add_argument("--pool_param", default=3, type=float)