# SPDX-License-Identifier: GPL-3.0-or-later
"""
Space-to-Depth (PixelUnshuffle) + per-predictor modulated DCNv2 geometry head.

P3 FPN features [B, C, H/16, W/16] are unshuffled to [B, 4C, H/32, W/32], then each
predictor applies its own DCNv2 with independent offset/mask prediction (symmetry breaking).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import deform_conv2d
except ImportError as exc:  # pragma: no cover
    deform_conv2d = None
    _DEFORM_IMPORT_ERROR = exc
else:
    _DEFORM_IMPORT_ERROR = None

from yolino.utils.logger import Log


class ModulatedDeformConv2d(nn.Module):
    """DCNv2-style modulated deformable conv (offset + mask + learnable conv weight)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        deformable_groups: int = 1,
    ):
        super().__init__()
        if deform_conv2d is None:
            raise ImportError(
                "torchvision.ops.deform_conv2d is required for ModulatedDeformConv2d; "
                "install torchvision with ops support."
            ) from _DEFORM_IMPORT_ERROR

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.deformable_groups = int(deformable_groups)

        k = self.kernel_size
        k2 = k * k
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, k, k))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        offset_channels = 2 * k2 * self.deformable_groups
        mask_channels = k2 * self.deformable_groups
        self.conv_offset_mask = nn.Conv2d(
            in_channels,
            offset_channels + mask_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            bias=True,
        )

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.conv_offset_mask.weight, 0.0)
        nn.init.constant_(self.conv_offset_mask.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_offset_mask(x)
        k2 = self.kernel_size * self.kernel_size
        offset = out[:, : 2 * k2]
        mask = torch.sigmoid(out[:, 2 * k2 :])
        return deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            mask=mask,
        )


class YolinoStdHead(nn.Module):
    """
    P3 -> PixelUnshuffle(2) -> [B, 4*C, H/32, W/32] -> per-predictor DCNv2 (geom) + 1x1 (embed).
    """

    def __init__(
        self,
        in_channels: int,
        num_predictors: int,
        vars_train: int,
        embed_dim: int,
        unshuffle_factor: int = 2,
        dcn_kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_predictors = int(num_predictors)
        self.vars_train = int(vars_train)
        self.embed_dim = int(embed_dim)
        self.unshuffle_factor = int(unshuffle_factor)

        self.pixel_unshuffle = nn.PixelUnshuffle(self.unshuffle_factor)
        std_channels = self.in_channels * (self.unshuffle_factor ** 2)

        pad = dcn_kernel_size // 2
        self.geom_dcns = nn.ModuleList(
            [
                ModulatedDeformConv2d(
                    std_channels,
                    self.vars_train,
                    kernel_size=dcn_kernel_size,
                    padding=pad,
                )
                for _ in range(self.num_predictors)
            ]
        )
        self.embed_convs = nn.ModuleList(
            [
                nn.Conv2d(std_channels, self.embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
                for _ in range(self.num_predictors)
            ]
        )
        Log.info(
            "YolinoStdHead: P3 %dch -> PixelUnshuffle(%d) -> %dch @ stride-32; "
            "%d x DCNv2(geom, %d vars) + %d x 1x1(embed, %d)"
            % (
                self.in_channels,
                self.unshuffle_factor,
                std_channels,
                self.num_predictors,
                self.vars_train,
                self.num_predictors,
                self.embed_dim,
            )
        )

    @property
    def std_channels(self) -> int:
        return self.in_channels * (self.unshuffle_factor ** 2)

    def unshuffle(self, x_p3: torch.Tensor) -> torch.Tensor:
        return self.pixel_unshuffle(x_p3)

    def forward_from_feat(
        self, feat_std: torch.Tensor, geom: bool = True, embed: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        geom_out = None
        embed_out = None
        if geom:
            geom_parts: List[torch.Tensor] = [dcn(feat_std) for dcn in self.geom_dcns]
            geom_out = torch.cat(geom_parts, dim=1)
        if embed:
            embed_parts: List[torch.Tensor] = [conv(feat_std) for conv in self.embed_convs]
            embed_out = torch.cat(embed_parts, dim=1)
        return geom_out, embed_out

    def forward(self, x_p3: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_p3: [B, C, H/16, W/16] FPN P3 map.

        Returns:
            geom: [B, P*vars, H/32, W/32]
            embed: [B, P*embed_dim, H/32, W/32]
            feat_std: [B, std_channels, H/32, W/32] (for E2E grid_sample memory)
        """
        feat_std = self.unshuffle(x_p3)
        geom, embed = self.forward_from_feat(feat_std, geom=True, embed=True)
        return geom, embed, feat_std


class StdFeatRefineDcn(nn.Module):
    """Shared modulated DCNv2 on STD feature map (E2E memory refinement, not per-variable geom)."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        out_ch = int(out_channels) if out_channels is not None else int(channels)
        self.out_channels = out_ch
        self.dcn = ModulatedDeformConv2d(
            int(channels),
            out_ch,
            kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2,
        )
        Log.info(
            "StdFeatRefineDcn: %d -> %d channels (shared DCN before E2E)"
            % (int(channels), out_ch)
        )

    def forward(self, feat_std: torch.Tensor) -> torch.Tensor:
        return self.dcn(feat_std)
