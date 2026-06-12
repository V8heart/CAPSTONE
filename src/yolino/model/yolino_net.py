# Copyright 2023 Karlsruhe Institute of Technology, Institute for Measurement
# and Control Systems
#
# This file is part of YOLinO.
#
# YOLinO is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# YOLinO is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# YOLinO. If not, see <https://www.gnu.org/licenses/>.
#
# ---------------------------------------------------------------------------- #
# ----------------------------- COPYRIGHT ------------------------------------ #
# ---------------------------------------------------------------------------- #
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolino.model.activations import get_activations
from yolino.model.yolino_std_dcn_head import StdFeatRefineDcn, YolinoStdHead
from yolino.model.yolino_detr_head import YolinoDetrBezierHead
from yolino.model.yolino_gnn_head import YolinoGnnSegmentGraphHead
from yolino.model.yolino_center_head import YolinoCenterPolyHead
from yolino.utils.enums import Variables
from yolino.utils.logger import Log

# torchvision ConvNeXt + feature extractor
try:
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    _HAS_NEW_TV_API = True
except ImportError:  # older torchvision
    from torchvision.models import convnext_tiny  # type: ignore
    ConvNeXt_Tiny_Weights = None  # type: ignore
    _HAS_NEW_TV_API = False
from torchvision.models.feature_extraction import create_feature_extractor
try:
    import timm
except ImportError:  # optional dependency, only needed when --backbone=timm
    timm = None


def _make_fpn_norm_layer(norm_type: str, num_channels: int) -> nn.Module:
    """Norm after FPN 3x3 smooth / bottom-up smooth convs (default: groupnorm)."""
    nt = str(norm_type).lower().strip()
    ch = int(num_channels)
    if nt == "groupnorm":
        groups = 32 if ch % 32 == 0 else 16
        return nn.GroupNorm(groups, ch)
    if nt == "batchnorm":
        return nn.BatchNorm2d(ch)
    if nt in ("syncbatchnorm", "sync_bn", "syncbn"):
        return nn.SyncBatchNorm(ch)
    if nt in ("none", "identity"):
        return nn.Identity()
    raise ValueError(
        "fpn_norm must be one of groupnorm|batchnorm|syncbatchnorm|none, got %r" % norm_type
    )


class ChannelAttention(nn.Module):
    """CBAM channel branch: MLP on global avg/max pool (lightweight)."""

    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        hidden = max(1, in_planes // ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_planes, 1, bias=False),
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        return torch.sigmoid(self.mlp(avg) + self.mlp(mx))


class SpatialAttention(nn.Module):
    """CBAM spatial branch: 7x7 conv on channel-aggregated map."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018) — channel then spatial."""

    def __init__(self, in_planes: int, ratio: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class GlobalSelfAttention(nn.Module):
    """
    임베딩 헤드에 Global Context를 부여하기 위한 가벼운 Self-Attention 모듈
    """

    def __init__(self, in_channels):
        super().__init__()
        # 연산량 감소를 위해 채널 수를 1/8로 줄여서 Q, K를 계산합니다.
        self.query = nn.Conv2d(in_channels, max(1, in_channels // 8), 1)
        self.key = nn.Conv2d(in_channels, max(1, in_channels // 8), 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)

        # 학습 초기에 기존 피처맵을 망치지 않도록 0으로 초기화 (매우 중요!)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        N = H * W

        q = self.query(x).view(batch_size, -1, N).permute(0, 2, 1)  # [B, N, C']
        k = self.key(x).view(batch_size, -1, N)                     # [B, C', N]

        energy = torch.bmm(q, k)
        attention = torch.softmax(energy, dim=-1)                   # [B, N, N]

        v = self.value(x).view(batch_size, -1, N)                   # [B, C, N]
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)

        return x + self.gamma * out


class BottomUpFuseToP4(nn.Module):
    """
    Bottom-up fusion into stride-32 (P4) grid while keeping head_level=P4.

    Uses smooth+GN **p2** (same as exported P2) but **m3/m4** are pre-smooth tensors
    after lateral (+ optional top-down add): PANet-style “all smoothed levels” fusion
    is not matched exactly; see smoother-aligned variant as a future option.

        n3 = smooth_bu( down_stride2(p2) + m3 )
        P4_bu = smooth_bu( down_stride2(n3) + m4 )

    Stride-2 uses 3x3 conv. When disabled, forward falls back to P4_td = smooth(m4).
    """

    def __init__(self, out_channels: int, fpn_norm: str = "groupnorm"):
        super().__init__()
        oc = int(out_channels)
        self.down_p2_to_p3 = nn.Conv2d(oc, oc, kernel_size=3, stride=2, padding=1)
        self.smooth_bu3 = nn.Conv2d(oc, oc, kernel_size=3, padding=1)
        self.norm_bu3 = _make_fpn_norm_layer(fpn_norm, oc)
        self.down_p3_to_p4 = nn.Conv2d(oc, oc, kernel_size=3, stride=2, padding=1)
        self.smooth_bu4 = nn.Conv2d(oc, oc, kernel_size=3, padding=1)
        self.norm_bu4 = _make_fpn_norm_layer(fpn_norm, oc)
        for m in (self.down_p2_to_p3, self.smooth_bu3, self.down_p3_to_p4, self.smooth_bu4):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, p2: torch.Tensor, m3: torch.Tensor, m4: torch.Tensor) -> torch.Tensor:
        x = self.down_p2_to_p3(p2) + m3
        x = self.norm_bu3(self.smooth_bu3(x))
        x = self.down_p3_to_p4(x) + m4
        x = self.norm_bu4(self.smooth_bu4(x))
        return x


class Timm_FPN_Backbone(nn.Module):
    """
    Generic timm backbone + FPN wrapper exposing the same output contract as
    ConvNeXt_FPN_Backbone: {"P2","P3","P4"} and optional {"P5"} when force_stride32_head.

    timm variants: resnet50_dilated (OS16, C4 from stage3), resnet50 (OS32, C4 from stage4),
    hrnet_w32.
    """

    _SUPPORTED = ("resnet50_dilated", "resnet50", "hrnet_w32")

    def __init__(self,
                 model_name: str = "resnet50_dilated",
                 pretrained: bool = True,
                 upsample_mode: str = "nearest",
                 out_channels: int = 256,
                 use_fpn: bool = True,
                 use_bottom_up: bool = False,
                 force_stride32_head: bool = False,
                 fpn_norm: str = "groupnorm"):
        super().__init__()
        if timm is None:
            raise ImportError(
                "timm is required for --backbone=timm. Install with `pip install timm`.")

        if model_name not in self._SUPPORTED:
            raise ValueError("Unsupported timm model %r. Supported: %s"
                             % (model_name, list(self._SUPPORTED)))

        self.model_name = model_name
        self.out_channels = int(out_channels)
        self.fpn_norm = str(fpn_norm)
        self.use_fpn = bool(use_fpn)
        self.use_bottom_up = bool(use_bottom_up)
        self.force_stride32_head = bool(force_stride32_head)
        # Standard ResNet50 (OS=32): deepest used level is already stride 32 — no extra stride-2 head block.
        self._p5_alias_p4 = False
        if upsample_mode not in ("nearest", "bilinear"):
            raise ValueError("upsample_mode must be 'nearest' or 'bilinear', got %r" % upsample_mode)
        self.upsample_mode = upsample_mode

        if model_name == "resnet50_dilated":
            self.body = timm.create_model(
                "resnet50",
                pretrained=pretrained,
                features_only=True,
                out_indices=(1, 2, 3),
                output_stride=16,
            )
        elif model_name == "resnet50":
            # No dilated/atrous stage-4: OS=32, use final stage as C4 (stride 32 vs input).
            self.body = timm.create_model(
                "resnet50",
                pretrained=pretrained,
                features_only=True,
                out_indices=(1, 2, 4),
                output_stride=32,
            )
            if self.force_stride32_head:
                self._p5_alias_p4 = True
        else:  # hrnet_w32
            self.body = timm.create_model(
                "hrnet_w32",
                pretrained=pretrained,
                features_only=True,
                out_indices=(1, 2, 3),
            )

        in_channels = list(self.body.feature_info.channels())
        if len(in_channels) != 3:
            raise ValueError("Expected 3 feature maps from timm %s, got %d (%s)"
                             % (model_name, len(in_channels), in_channels))
        self.C2_CH, self.C3_CH, self.C4_CH = int(in_channels[0]), int(in_channels[1]), int(in_channels[2])

        self.lateral_c2 = nn.Conv2d(self.C2_CH, self.out_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(self.C3_CH, self.out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(self.C4_CH, self.out_channels, kernel_size=1)

        self.smooth_p2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.smooth_p4 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

        self.smooth_norm_p2 = _make_fpn_norm_layer(self.fpn_norm, self.out_channels)
        self.smooth_norm_p3 = _make_fpn_norm_layer(self.fpn_norm, self.out_channels)
        self.smooth_norm_p4 = _make_fpn_norm_layer(self.fpn_norm, self.out_channels)

        for m in (self.lateral_c2, self.lateral_c3, self.lateral_c4,
                  self.smooth_p2, self.smooth_p3, self.smooth_p4):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        self.bottom_up = (
            BottomUpFuseToP4(self.out_channels, fpn_norm=self.fpn_norm)
            if self.use_bottom_up else None
        )
        self.head_downsample = None
        if self.force_stride32_head and not self._p5_alias_p4:
            # Simple down-projection: keep high-res backbone path, create a larger-stride head map.
            self.head_downsample = nn.Sequential(
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.GELU(),
            )

    def _upsample_to(self, x: torch.Tensor, size) -> torch.Tensor:
        if self.upsample_mode == "nearest":
            return F.interpolate(x, size=size, mode="nearest")
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor):
        c2, c3, c4 = self.body(x)

        m4 = self.lateral_c4(c4)
        m3 = self.lateral_c3(c3)
        m2 = self.lateral_c2(c2)

        if self.use_fpn:
            m3 = m3 + self._upsample_to(m4, c3.shape[-2:])
            m2 = m2 + self._upsample_to(m3, c2.shape[-2:])

        p3 = self.smooth_norm_p3(self.smooth_p3(m3))
        p2 = self.smooth_norm_p2(self.smooth_p2(m2))
        if self.bottom_up is not None:
            p4 = self.bottom_up(p2, m3, m4)
        else:
            p4 = self.smooth_norm_p4(self.smooth_p4(m4))
        if self._p5_alias_p4:
            p5 = p4
        else:
            p5 = self.head_downsample(p4) if self.head_downsample is not None else None

        if p5 is None:
            Log.debug("TimmBackbone[%s] feats: C2=%s C3=%s C4=%s | P2=%s P3=%s P4=%s"
                      % (self.model_name, tuple(c2.shape), tuple(c3.shape), tuple(c4.shape),
                         tuple(p2.shape), tuple(p3.shape), tuple(p4.shape)))
            return {"P2": p2, "P3": p3, "P4": p4}

        Log.debug("TimmBackbone[%s] feats: C2=%s C3=%s C4=%s | P2=%s P3=%s P4=%s P5=%s"
                  % (self.model_name, tuple(c2.shape), tuple(c3.shape), tuple(c4.shape),
                     tuple(p2.shape), tuple(p3.shape), tuple(p4.shape), tuple(p5.shape)))
        return {"P2": p2, "P3": p3, "P4": p4, "P5": p5}


# --------------------------------------------------------------------------- #
# ConvNeXt-Tiny + FPN backbone
# --------------------------------------------------------------------------- #
class ConvNeXt_FPN_Backbone(nn.Module):
    """
    ConvNeXt-Tiny + FPN backbone tuned for thin-line (e.g. drone power-line) detection.

    ConvNeXt-Tiny stage layout (torchvision):
        features.0  : stem               (stride 4,  96 ch)
        features.1  : stage1 blocks      (stride 4,  96 ch)
        features.2  : downsample         (stride 8,  192 ch)
        features.3  : stage2 blocks  ->  C2 (stride 8,  192 ch)
        features.4  : downsample         (stride 16, 384 ch)
        features.5  : stage3 blocks  ->  C3 (stride 16, 384 ch)
        features.6  : downsample         (stride 32, 768 ch)
        features.7  : stage4 blocks  ->  C4 (stride 32, 768 ch)

    FPN (top-down with element-wise add):
        M4 = 1x1(C4)
        M3 = 1x1(C3) + Upsample(M4)
        M2 = 1x1(C2) + Upsample(M3)
        P2, P3 = 3x3+GN(M2), 3x3+GN(M3)   # anti-aliasing smooth conv

    Optional bottom-up (use_bottom_up): replace P4 with fusion of P2_td + m3 + m4 at
    stride 32 so fine C2-derived features inform the P4 grid without multi-level heads.

    Output dict keys: "P2" (1/8), "P3" (1/16), "P4" (1/32), all `out_channels` channels.
    """

    C2_CH, C3_CH, C4_CH = 192, 384, 768

    # In torchvision's convnext_tiny, the entire stage Sequential is exposed as
    # features.<idx> in the symbolic trace. The last block within each stage is
    # 'features.3.2', 'features.5.8', 'features.7.2' — both work, the first is
    # cleaner because it does not depend on the internal block count.
    _RETURN_NODES_PRIMARY = OrderedDict([
        ("features.3", "C2"),
        ("features.5", "C3"),
        ("features.7", "C4"),
    ])
    _RETURN_NODES_FALLBACK = OrderedDict([
        ("features.3.2", "C2"),
        ("features.5.8", "C3"),
        ("features.7.2", "C4"),
    ])

    def __init__(self,
                 out_channels: int = 256,
                 pretrained: bool = True,
                 upsample_mode: str = "nearest",
                 use_fpn: bool = True,
                 use_bottom_up: bool = False,
                 fpn_norm: str = "groupnorm"):
        super().__init__()
        self.out_channels = int(out_channels)
        self.fpn_norm = str(fpn_norm)
        self.use_fpn = bool(use_fpn)
        self.use_bottom_up = bool(use_bottom_up)
        if upsample_mode not in ("nearest", "bilinear"):
            raise ValueError("upsample_mode must be 'nearest' or 'bilinear', got %r" % upsample_mode)
        self.upsample_mode = upsample_mode

        if _HAS_NEW_TV_API:
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base = convnext_tiny(weights=weights)
        else:
            base = convnext_tiny(pretrained=pretrained)

        try:
            self.body = create_feature_extractor(base, return_nodes=self._RETURN_NODES_PRIMARY)
        except Exception as ex:  # pragma: no cover - defensive for tv version drift
            Log.warning("create_feature_extractor failed with primary nodes (%s). "
                        "Falling back to per-block node names." % str(ex))
            self.body = create_feature_extractor(base, return_nodes=self._RETURN_NODES_FALLBACK)

        # Lateral 1x1 convs: align all C* to the FPN channel.
        self.lateral_c2 = nn.Conv2d(self.C2_CH, self.out_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(self.C3_CH, self.out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(self.C4_CH, self.out_channels, kernel_size=1)

        # 3x3 anti-aliasing convs after top-down add.
        self.smooth_p2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.smooth_p4 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.smooth_norm_p2 = _make_fpn_norm_layer(self.fpn_norm, self.out_channels)
        self.smooth_norm_p3 = _make_fpn_norm_layer(self.fpn_norm, self.out_channels)
        self.smooth_norm_p4 = _make_fpn_norm_layer(self.fpn_norm, self.out_channels)

        for m in (self.lateral_c2, self.lateral_c3, self.lateral_c4,
                  self.smooth_p2, self.smooth_p3, self.smooth_p4):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        self.bottom_up = (
            BottomUpFuseToP4(self.out_channels, fpn_norm=self.fpn_norm)
            if self.use_bottom_up else None
        )
        if self.use_bottom_up and not self.use_fpn:
            Log.warning(
                "use_bottom_up=True with use_fpn=False: m3/m4 are lateral-only (no top-down add). "
                "Bottom-up becomes coarse←fine fusion without FPN semantics — prefer use_fpn=True "
                "for PANet-style comparisons unless intentionally ablating.")

    def _upsample_to(self, x: torch.Tensor, size) -> torch.Tensor:
        if self.upsample_mode == "nearest":
            return F.interpolate(x, size=size, mode="nearest")
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor):
        feats = self.body(x)
        c2, c3, c4 = feats["C2"], feats["C3"], feats["C4"]

        m4 = self.lateral_c4(c4)
        m3 = self.lateral_c3(c3)
        m2 = self.lateral_c2(c2)

        if self.use_fpn:
            m3 = m3 + self._upsample_to(m4, c3.shape[-2:])
            m2 = m2 + self._upsample_to(m3, c2.shape[-2:])

        p3 = self.smooth_norm_p3(self.smooth_p3(m3))
        p2 = self.smooth_norm_p2(self.smooth_p2(m2))

        if self.bottom_up is not None:
            p4 = self.bottom_up(p2, m3, m4)
        else:
            p4 = self.smooth_norm_p4(self.smooth_p4(m4))

        mode = "FPN" if self.use_fpn else "no-FPN"
        if self.bottom_up is not None:
            mode += "+BU"
        Log.debug("ConvNeXt+%s feats: C2=%s C3=%s C4=%s | P2=%s P3=%s P4=%s"
                  % (mode, tuple(c2.shape), tuple(c3.shape), tuple(c4.shape),
                     tuple(p2.shape), tuple(p3.shape), tuple(p4.shape)))
        return {"P2": p2, "P3": p3, "P4": p4}


# --------------------------------------------------------------------------- #
# Darknet-19 + FPN backbone (legacy YOLinO backbone, kept for ablation)
# --------------------------------------------------------------------------- #
class Darknet_FPN_Backbone(nn.Module):
    """
    Darknet-19 (with optional dilation) + FPN backbone, mirroring
    `ConvNeXt_FPN_Backbone` so the rest of `YolinoNet` (heads, optimizer LR
    groups, freeze logic) is shared verbatim across backbones.

    The underlying `Darknet` model exposes three intermediate features when
    constructed with `return_intermediate=True`:
        - C2: module index 10  (256 ch,  stride  8)
        - C3: module index 16  (512 ch,  stride 16)
        - C4: module index 22  (1024 ch, stride 32)

    A standard top-down FPN (1x1 lateral + 3x3 anti-alias smooth + GroupNorm)
    is applied on top, identical to the ConvNeXt variant. The `body` submodule
    is the raw `Darknet`, so its parameters live under `backbone.body.*`,
    which is exactly the prefix the optimizer routes to the `backbone` LR
    group / freeze list.

    Output dict keys: "P2" (1/8), "P3" (1/16), "P4" (1/32).
    """

    C2_CH, C3_CH, C4_CH = 256, 512, 1024

    def __init__(self,
                 cfg_path: str,
                 weights_path: str = None,
                 out_channels: int = 256,
                 pretrained: bool = True,
                 upsample_mode: str = "nearest",
                 use_fpn: bool = True,
                 use_bottom_up: bool = False,
                 fpn_norm: str = "groupnorm"):
        super().__init__()
        import os as _os
        from yolino.model.darknet import Darknet

        if not cfg_path or not _os.path.isfile(cfg_path):
            raise FileNotFoundError(
                "Darknet backbone requires --darknet_cfg pointing to a valid .cfg, "
                "got %r" % cfg_path)

        self.out_channels = int(out_channels)
        self.fpn_norm = str(fpn_norm)
        self.use_fpn = bool(use_fpn)
        self.use_bottom_up = bool(use_bottom_up)
        if upsample_mode not in ("nearest", "bilinear"):
            raise ValueError("upsample_mode must be 'nearest' or 'bilinear', got %r" % upsample_mode)
        self.upsample_mode = upsample_mode

        self.body = Darknet(cfg_path, return_intermediate=True)

        if pretrained and weights_path:
            if _os.path.isfile(weights_path) and _os.path.getsize(weights_path) > 1024:
                try:
                    self.body.load_weights(weights_path)
                    Log.info("Loaded Darknet weights from %s" % weights_path)
                except Exception as ex:  # noqa: BLE001
                    Log.warning(
                        "Could not load Darknet weights from %s (%s). "
                        "Falling back to random init." % (weights_path, ex))
            else:
                Log.warning(
                    "Darknet weights file %s is missing or looks like a placeholder "
                    "(<=1KB). Falling back to random init." % weights_path)
        else:
            Log.info("Darknet backbone using random init (pretrained=%s)" % pretrained)

        self.lateral_c2 = nn.Conv2d(self.C2_CH, self.out_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(self.C3_CH, self.out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(self.C4_CH, self.out_channels, kernel_size=1)

        self.smooth_p2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.smooth_p4 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

        self.smooth_norm_p2 = _make_fpn_norm_layer(self.fpn_norm, self.out_channels)
        self.smooth_norm_p3 = _make_fpn_norm_layer(self.fpn_norm, self.out_channels)
        self.smooth_norm_p4 = _make_fpn_norm_layer(self.fpn_norm, self.out_channels)

        for m in (self.lateral_c2, self.lateral_c3, self.lateral_c4,
                  self.smooth_p2, self.smooth_p3, self.smooth_p4):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        self.bottom_up = (
            BottomUpFuseToP4(self.out_channels, fpn_norm=self.fpn_norm)
            if self.use_bottom_up else None
        )
        if self.use_bottom_up and not self.use_fpn:
            Log.warning(
                "use_bottom_up=True with use_fpn=False: m3/m4 are lateral-only (no top-down add). "
                "Bottom-up becomes coarse←fine fusion without FPN semantics — prefer use_fpn=True "
                "for PANet-style comparisons unless intentionally ablating.")

    def _upsample_to(self, x: torch.Tensor, size) -> torch.Tensor:
        if self.upsample_mode == "nearest":
            return F.interpolate(x, size=size, mode="nearest")
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor):
        c2, c3, c4 = self.body(x)

        m4 = self.lateral_c4(c4)
        m3 = self.lateral_c3(c3)
        m2 = self.lateral_c2(c2)

        if self.use_fpn:
            m3 = m3 + self._upsample_to(m4, c3.shape[-2:])
            m2 = m2 + self._upsample_to(m3, c2.shape[-2:])

        p3 = self.smooth_norm_p3(self.smooth_p3(m3))
        p2 = self.smooth_norm_p2(self.smooth_p2(m2))

        if self.bottom_up is not None:
            p4 = self.bottom_up(p2, m3, m4)
        else:
            p4 = self.smooth_norm_p4(self.smooth_p4(m4))

        mode = "FPN" if self.use_fpn else "no-FPN"
        if self.bottom_up is not None:
            mode += "+BU"
        Log.debug("Darknet+%s feats: C2=%s C3=%s C4=%s | P2=%s P3=%s P4=%s"
                  % (mode, tuple(c2.shape), tuple(c3.shape), tuple(c4.shape),
                     tuple(p2.shape), tuple(p3.shape), tuple(p4.shape)))
        return {"P2": p2, "P3": p3, "P4": p4}


# --------------------------------------------------------------------------- #
# YolinoNet (ConvNeXt-Tiny + FPN)
# --------------------------------------------------------------------------- #
class YolinoNet(nn.Module):
    """
    YOLinO with ConvNeXt-Tiny + FPN backbone.

    The geometry head and the embedding head are both fed from a single FPN level
    (default: P3, stride 16, 256 channels). The model assumes the cell stride
    matches the chosen FPN level — set --scale 16 (or --scale 8 with head_level=P2).
    """

    # Map "FPN level -> stride" for sanity checks against args.scale.
    _LEVEL_TO_STRIDE = {"P2": 8, "P3": 16, "P4": 32, "P5": 32}

    def __init__(self, args, coords):
        super().__init__()

        self.cuda = args.cuda
        self.scale = args.scale
        self.coords = coords
        if len(self.coords.get_position_of_training_vars()) == 0:
            raise ValueError("Network is configured to predict 0 variables! "
                             "Please fix %s, %s" % (self.coords, self.coords.vars_to_train))

        self.num_predictors = args.num_predictors
        self.feature_refine = getattr(args, "feature_refine", "sa_embed_only")
        self.cbam_reduction_ratio = int(getattr(args, "cbam_reduction_ratio", 16))
        self.embed_dim = int(getattr(args, "embed_dim", 8))

        # ----- Backbone + FPN -----
        fpn_out = int(getattr(args, "fpn_out_channels", 256))
        pretrained = bool(getattr(args, "backbone_pretrained", True))
        upsample_mode = str(getattr(args, "fpn_upsample_mode", "nearest"))
        use_fpn = bool(getattr(args, "use_fpn", True))
        use_bottom_up = bool(getattr(args, "use_bottom_up", False))
        fpn_norm = str(getattr(args, "fpn_norm", "groupnorm"))
        backbone_name = str(getattr(args, "backbone", "convnext")).lower()
        self.backbone_name = backbone_name
        if backbone_name == "convnext":
            self.backbone = ConvNeXt_FPN_Backbone(out_channels=fpn_out,
                                                  pretrained=pretrained,
                                                  upsample_mode=upsample_mode,
                                                  use_fpn=use_fpn,
                                                  use_bottom_up=use_bottom_up,
                                                  fpn_norm=fpn_norm)
        elif backbone_name == "darknet":
            cfg_path = getattr(args, "darknet_cfg", None)
            weights_path = getattr(args, "darknet_weights", None)
            self.backbone = Darknet_FPN_Backbone(
                cfg_path=cfg_path,
                weights_path=weights_path,
                out_channels=fpn_out,
                pretrained=pretrained,
                upsample_mode=upsample_mode,
                use_fpn=use_fpn,
                use_bottom_up=use_bottom_up,
                fpn_norm=fpn_norm,
            )
        elif backbone_name == "timm":
            timm_model_name = str(getattr(args, "timm_model_name", "resnet50_dilated")).lower()
            timm_force_stride32_head = bool(getattr(args, "timm_force_stride32_head", False))
            self.backbone = Timm_FPN_Backbone(
                model_name=timm_model_name,
                pretrained=pretrained,
                upsample_mode=upsample_mode,
                out_channels=fpn_out,
                use_fpn=use_fpn,
                use_bottom_up=use_bottom_up,
                force_stride32_head=timm_force_stride32_head,
                fpn_norm=fpn_norm,
            )
        else:
            raise ValueError("Unknown --backbone=%r (expected 'convnext' or 'darknet' or 'timm')"
                             % backbone_name)
        Log.info("YolinoNet backbone=%s, use_fpn=%s, use_bottom_up=%s, fpn_norm=%s, fpn_out=%d, head_level=%s"
                 % (backbone_name, str(use_fpn), str(use_bottom_up), fpn_norm, fpn_out,
                    str(getattr(args, "head_level", "P3"))))

        # FPN level fed to the heads (ignored when --std: P3 + PixelUnshuffle -> stride 32).
        self.use_std = bool(getattr(args, "std", False))
        self.head_level = str(getattr(args, "head_level", "P3"))
        if self.head_level not in self._LEVEL_TO_STRIDE:
            raise ValueError("head_level must be one of %s, got %r"
                             % (list(self._LEVEL_TO_STRIDE.keys()), self.head_level))
        if self.use_std:
            if int(self.scale) != 32:
                raise ValueError(
                    "--std requires --scale 32 (P3 PixelUnshuffle(2) -> stride-32 head map); got scale=%s"
                    % self.scale
                )
            self._head_stride = 32
            self._head_feat_level = "P3+STD"
            Log.info(
                "STD head enabled: feats['P3'] -> PixelUnshuffle(2) -> stride-32 map "
                "(head_level=%s ignored for geometry/embed)."
                % self.head_level
            )
        else:
            self._head_stride = self._LEVEL_TO_STRIDE[self.head_level]
            self._head_feat_level = self.head_level
            if int(self.scale) != self._head_stride:
                Log.warning("Backbone head outputs at stride %d (level %s) but args.scale=%s. "
                            "Set --scale %d so cell_size matches the feature map (and ensure "
                            "img_size %% %d == 0)."
                            % (self._head_stride, self.head_level, self.scale,
                               self._head_stride, self._head_stride))

        in_channels = fpn_out  # FPN channel width before STD unshuffle
        self.vars_train = len(self.coords.get_position_of_training_vars())
        vars_train = self.vars_train

        # ----- Optional trunk refinement before the heads -----
        self.attention = None
        self.cbam = None
        if self.feature_refine in ("sa_embed_only", "sa_shared"):
            self.attention = GlobalSelfAttention(in_channels=in_channels)
        elif self.feature_refine == "cbam_shared":
            self.cbam = CBAM(in_channels, ratio=self.cbam_reduction_ratio)
        elif self.feature_refine == "none":
            pass
        else:
            raise ValueError("Unknown feature_refine=%r (expected none|sa_embed_only|sa_shared|cbam_shared)"
                             % (self.feature_refine,))

        self.std_skip_geom_head = bool(getattr(args, "std_skip_geom_head", False))
        self.std_feat_dcn_before_e2e = bool(getattr(args, "std_feat_dcn_before_e2e", False))
        if self.std_skip_geom_head and not self.use_std:
            raise ValueError("--std_skip_geom_head requires --std true")
        if self.std_feat_dcn_before_e2e and not self.use_std:
            raise ValueError("--std_feat_dcn_before_e2e requires --std true")

        # ----- Heads (classic 1x1 conv vs STD + per-predictor DCNv2) -----
        self.std_head = None
        self.std_feat_dcn = None
        self.yolo = None
        self.embed_head = None
        if self.use_std:
            self.std_head = YolinoStdHead(
                in_channels=in_channels,
                num_predictors=self.num_predictors,
                vars_train=vars_train,
                embed_dim=self.embed_dim,
                unshuffle_factor=2,
            )
            head_feat_channels = self.std_head.std_channels
            if self.std_feat_dcn_before_e2e:
                dcn_out = int(getattr(args, "std_feat_dcn_channels", 0)) or head_feat_channels
                self.std_feat_dcn = StdFeatRefineDcn(
                    head_feat_channels,
                    kernel_size=int(getattr(args, "std_feat_dcn_kernel", 3)),
                    out_channels=dcn_out,
                )
                head_feat_channels = self.std_feat_dcn.out_channels
        else:
            self.yolo = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.num_predictors * vars_train,
                kernel_size=1, stride=1, padding=0, bias=True,
            )
            self.embed_head = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.num_predictors * self.embed_dim,
                kernel_size=1, stride=1, padding=0, bias=True,
            )
            head_feat_channels = in_channels

        # ----- Optional E2E head (yaml: e2e_differentiable_postproc) -----
        # The flag now selects the YOLinO-DETR Hybrid Bézier head; the legacy
        # E2EDifferentiablePostHead remains in-tree but is no longer wired here.
        self.e2e_differentiable_postproc = bool(getattr(args, "e2e_differentiable_postproc", False))
        self.e2e_mode = str(getattr(args, "e2e_mode", "detr")).lower()
        if self.e2e_mode not in (
            "detr", "gnn", "center", "seg_detr", "seg_detr_soft", "std_seg_detr",
            "hough_detr", "learnable_detr",
        ):
            raise ValueError(
                "Unknown --e2e_mode=%r (expected 'detr', 'seg_detr', 'seg_detr_soft', "
                "'std_seg_detr', 'gnn', 'center', 'hough_detr' or 'learnable_detr')"
                % self.e2e_mode
            )
        if self.e2e_mode == "std_seg_detr" and not self.use_std:
            raise ValueError("--e2e_mode=std_seg_detr requires --std true")
        self.e2e_head = None
        self._e2e_geom_activations = None
        self._e2e_conf_idx = -1
        if self.e2e_differentiable_postproc:
            conf_pos = coords.get_position_within_prediction(Variables.CONF)
            if len(conf_pos) != 1:
                raise ValueError(
                    "e2e_differentiable_postproc expects exactly one CONF channel in training layout; got %s"
                    % (list(conf_pos),)
                )
            self._e2e_conf_idx = int(conf_pos[0])
            if self.e2e_mode == "detr":
                self.e2e_head = YolinoDetrBezierHead(
                    line_rep=coords.line_representation.enum,
                    fpn_channels=head_feat_channels,
                    num_queries=int(getattr(args, "e2e_num_queries", 20)),
                    decoder_layers=int(getattr(args, "e2e_decoder_layers", 3)),
                    decoder_heads=int(getattr(args, "e2e_decoder_heads", 8)),
                    decoder_ff=int(getattr(args, "e2e_decoder_ff", 1024)),
                    token_dim=int(getattr(args, "e2e_token_dim", 256)),
                    bezier_degree=int(getattr(args, "e2e_bezier_degree", 3)),
                    bezier_num_samples=int(getattr(args, "e2e_bezier_num_samples", 32)),
                    conf_filter_thresh=float(getattr(args, "e2e_conf_filter_thresh", 0.1)),
                    local_max_filter=bool(getattr(args, "e2e_local_max_filter", False)),
                    local_max_kernel=int(getattr(args, "e2e_local_max_kernel", 5)),
                    anchor_ctrl_offsets=bool(getattr(args, "e2e_bezier_anchor_ctrl_offsets", True)),
                )
                Log.info(
                    "YolinoDetrBezierHead enabled (K=%d, decoder_layers=%d, token_dim=%d, conf_idx=%d, "
                    "local_max=%s/k=%d, anchor_ctrl_offsets=%s)"
                    % (
                        self.e2e_head.num_queries,
                        int(getattr(args, "e2e_decoder_layers", 3)),
                        self.e2e_head.token_dim,
                        self._e2e_conf_idx,
                        str(self.e2e_head.local_max_filter),
                        int(self.e2e_head.local_max_kernel),
                        str(self.e2e_head.anchor_ctrl_offsets),
                    )
                )
            elif self.e2e_mode == "seg_detr":
                from yolino.model.yolino_seg_detr_head import YolinoSegDetrBezierHead

                self.e2e_head = YolinoSegDetrBezierHead(
                    line_rep=coords.line_representation.enum,
                    fpn_channels=head_feat_channels,
                    num_queries=int(getattr(args, "e2e_num_queries", 20)),
                    decoder_layers=int(getattr(args, "e2e_decoder_layers", 3)),
                    decoder_heads=int(getattr(args, "e2e_decoder_heads", 8)),
                    decoder_ff=int(getattr(args, "e2e_decoder_ff", 1024)),
                    token_dim=int(getattr(args, "e2e_token_dim", 256)),
                    bezier_degree=int(getattr(args, "e2e_bezier_degree", 3)),
                    bezier_num_samples=int(getattr(args, "e2e_bezier_num_samples", 32)),
                    conf_filter_thresh=float(getattr(args, "e2e_conf_filter_thresh", 0.1)),
                    dropout=float(getattr(args, "gnn_dropout", 0.1)),
                )
                Log.info(
                    "YolinoSegDetrBezierHead enabled (K=%d, learned queries, segment-token memory only; "
                    "no center heatmap / no Top-K anchor init, conf_idx=%d)"
                    % (self.e2e_head.num_queries, self._e2e_conf_idx)
                )
            elif self.e2e_mode == "seg_detr_soft":
                from yolino.model.yolino_seg_detr_head import YolinoSegDetrSoftInitBezierHead

                soft_mode = str(getattr(args, "e2e_soft_query_init", "residual_topk")).lower()
                self.e2e_head = YolinoSegDetrSoftInitBezierHead(
                    line_rep=coords.line_representation.enum,
                    fpn_channels=head_feat_channels,
                    num_queries=int(getattr(args, "e2e_num_queries", 20)),
                    decoder_layers=int(getattr(args, "e2e_decoder_layers", 3)),
                    decoder_heads=int(getattr(args, "e2e_decoder_heads", 8)),
                    decoder_ff=int(getattr(args, "e2e_decoder_ff", 1024)),
                    token_dim=int(getattr(args, "e2e_token_dim", 256)),
                    bezier_degree=int(getattr(args, "e2e_bezier_degree", 3)),
                    bezier_num_samples=int(getattr(args, "e2e_bezier_num_samples", 32)),
                    conf_filter_thresh=float(getattr(args, "e2e_conf_filter_thresh", 0.1)),
                    dropout=float(getattr(args, "gnn_dropout", 0.1)),
                    soft_init_mode=soft_mode,
                    local_max_filter=bool(getattr(args, "e2e_local_max_filter", False)),
                    local_max_kernel=int(getattr(args, "e2e_local_max_kernel", 5)),
                    softmax_pool_temperature=float(
                        getattr(args, "e2e_softmax_pool_temperature", 1.0)
                    ),
                )
                Log.info(
                    "YolinoSegDetrSoftInitBezierHead enabled (K=%d, soft_init=%s, segment memory; "
                    "no center heatmap, conf_idx=%d, local_max=%s)"
                    % (
                        self.e2e_head.num_queries,
                        self.e2e_head.soft_init_mode,
                        self._e2e_conf_idx,
                        str(self.e2e_head.local_max_filter),
                    )
                )
            elif self.e2e_mode == "std_seg_detr":
                from yolino.model.yolino_seg_detr_head import YolinoGridSegDetrBezierHead

                self.e2e_head = YolinoGridSegDetrBezierHead(
                    line_rep=coords.line_representation.enum,
                    fpn_channels=head_feat_channels,
                    num_predictors=self.num_predictors,
                    num_queries=int(getattr(args, "e2e_num_queries", 20)),
                    decoder_layers=int(getattr(args, "e2e_decoder_layers", 3)),
                    decoder_heads=int(getattr(args, "e2e_decoder_heads", 8)),
                    decoder_ff=int(getattr(args, "e2e_decoder_ff", 1024)),
                    token_dim=int(getattr(args, "e2e_token_dim", 256)),
                    bezier_degree=int(getattr(args, "e2e_bezier_degree", 3)),
                    bezier_num_samples=int(getattr(args, "e2e_bezier_num_samples", 32)),
                    polyline_num_points=int(getattr(args, "e2e_polyline_num_points", 0)),
                    conf_filter_thresh=float(getattr(args, "e2e_conf_filter_thresh", 0.1)),
                    dropout=float(getattr(args, "gnn_dropout", 0.1)),
                )
                Log.info(
                    "YolinoGridSegDetrBezierHead enabled (STD feat%s, skip_geom=%s, K=%d, "
                    "grid tokens x P=%d, output=%s)"
                    % (
                        "+FeatDCN" if self.std_feat_dcn is not None else "",
                        str(self.std_skip_geom_head),
                        self.e2e_head.num_queries,
                        self.num_predictors,
                        self.e2e_head.output_mode,
                    )
                )
            elif self.e2e_mode == "gnn":
                _gnn_token_dim = getattr(args, "gnn_token_dim", None)
                _gnn_token_dim = int(
                    _gnn_token_dim if _gnn_token_dim is not None
                    else getattr(args, "e2e_token_dim", 256)
                )
                self.e2e_head = YolinoGnnSegmentGraphHead(
                    line_rep=coords.line_representation.enum,
                    fpn_channels=head_feat_channels,
                    token_dim=_gnn_token_dim,
                    max_nodes=int(getattr(args, "gnn_max_nodes", 256)),
                    node_conf_thresh=float(getattr(args, "gnn_node_conf_thresh", 0.1)),
                    knn_k=int(getattr(args, "gnn_knn_k", 16)),
                    knn_min_dir_dot=float(getattr(args, "gnn_knn_min_dir_dot", 0.0)),
                    edge_radius_px=float(getattr(args, "gnn_edge_radius_px", 0.0)),
                    adjacency_mode=str(getattr(args, "gnn_adjacency_mode", "global")),
                    max_lateral_px=float(getattr(args, "gnn_max_lateral_px", 48.0)),
                    max_lateral_sym=bool(getattr(args, "gnn_max_lateral_sym", False)),
                    lateral_on_overlap_only=bool(getattr(args, "gnn_lateral_on_overlap_only", False)),
                    lateral_overlap_window_px=float(getattr(args, "gnn_lateral_overlap_window_px", 24.0)),
                    max_along_px=float(getattr(args, "gnn_max_along_px", 0.0)),
                    max_end_gap_px=float(getattr(args, "gnn_max_end_gap_px", 0.0)),
                    min_dir_dot=float(getattr(args, "gnn_min_dir_dot", 0.0)),
                    directional_min_sep_px=float(getattr(args, "gnn_directional_min_sep_px", 8.0)),
                    directional_k=int(getattr(args, "gnn_directional_k", 2)),
                    directional_include_all=bool(getattr(args, "gnn_directional_include_all", False)),
                    context_k=int(getattr(args, "gnn_context_k", 4)),
                    context_lat_min_px=float(getattr(args, "gnn_context_lat_min_px", 12.0)),
                    context_lat_max_px=float(getattr(args, "gnn_context_lat_max_px", 40.0)),
                    context_max_along_px=float(getattr(args, "gnn_context_max_along_px", 200.0)),
                    context_min_dir_dot=float(getattr(args, "gnn_context_min_dir_dot", 0.85)),
                    gat_layers=int(getattr(args, "gnn_gat_layers", 3)),
                    heads=int(getattr(args, "gnn_heads", 4)),
                    dropout=float(getattr(args, "gnn_dropout", 0.1)),
                    soft_nms_enabled=bool(getattr(args, "gnn_soft_nms", False)),
                    soft_nms_mid_sigma_px=float(getattr(args, "gnn_soft_nms_mid_sigma_px", 16.0)),
                    soft_nms_min_dir_dot=float(getattr(args, "gnn_soft_nms_min_dir_dot", 0.96)),
                    soft_nms_decay_method=str(getattr(args, "gnn_soft_nms_decay", "linear")),
                    soft_nms_score_floor=float(getattr(args, "gnn_soft_nms_score_floor", 0.001)),
                    soft_nms_prefilter_conf=float(getattr(args, "gnn_soft_nms_prefilter_conf", 0.05)),
                    soft_nms_max_segments=int(getattr(args, "gnn_soft_nms_max_segments", 1024)),
                    segment_merge_enabled=bool(getattr(args, "gnn_segment_merge", False)),
                    segment_merge_lat_px=float(getattr(args, "gnn_segment_merge_lat_px", 6.0)),
                    segment_merge_dir_dot_min=float(getattr(args, "gnn_segment_merge_dir_dot_min", 0.98)),
                    segment_merge_end_gap_px=float(getattr(args, "gnn_segment_merge_end_gap_px", 8.0)),
                    segment_merge_iters=int(getattr(args, "gnn_segment_merge_iters", 3)),
                    segment_merge_prefilter_conf=float(getattr(args, "gnn_segment_merge_prefilter_conf", 0.05)),
                    use_hard_geom_gate=bool(getattr(args, "gnn_use_hard_geom_gate", True)),
                    soft_geom_gate_enabled=bool(getattr(args, "gnn_soft_geom_gate", False)),
                    soft_geom_sigma_lat_px=float(getattr(args, "gnn_soft_geom_sigma_lat_px", 32.0)),
                    soft_geom_dir_floor=float(getattr(args, "gnn_soft_geom_dir_floor", 0.5)),
                    soft_geom_prior_eps=float(getattr(args, "gnn_soft_geom_prior_eps", 1e-6)),
                    node_use_visual_feat=bool(getattr(args, "gnn_node_use_visual_feat", True)),
                    edge_feat_signed=bool(getattr(args, "gnn_edge_feat_signed", False)),
                    gat_geom_bias=bool(getattr(args, "gnn_gat_geom_bias", False)),
                    gat_geom_bias_w_along=float(getattr(args, "gnn_gat_geom_bias_w_along", 1.0)),
                    gat_geom_bias_w_lat=float(getattr(args, "gnn_gat_geom_bias_w_lat", 0.5)),
                    gat_geom_bias_tau_along=float(getattr(args, "gnn_gat_geom_bias_tau_along", 40.0)),
                    gat_geom_bias_tau_lat=float(getattr(args, "gnn_gat_geom_bias_tau_lat", 20.0)),
                )
                _adj = str(getattr(args, "gnn_adjacency_mode", "global")).lower()
                if _adj == "global":
                    _k_eff = int(self.e2e_head.max_nodes)
                elif _adj == "directional2":
                    _k_eff = int(getattr(args, "gnn_directional_k", 2))
                elif _adj == "directional2_ctx":
                    if bool(getattr(args, "gnn_directional_include_all", False)):
                        _k_eff = -1  # variable (batch max on-line + context_k)
                    else:
                        _k_eff = int(getattr(args, "gnn_directional_k", 2)) + int(
                            getattr(args, "gnn_context_k", 4)
                        )
                elif _adj == "directional2_global":
                    _k_eff = int(getattr(args, "gnn_knn_k", 16))
                else:
                    _k_eff = int(getattr(args, "gnn_knn_k", 16))
                Log.info(
                    "YolinoGnnSegmentGraphHead enabled (adjacency=%s, N_max=%d, K=%d, layers=%d, heads=%d, "
                    "token_dim=%d, conf_thresh=%.3f, radius_px=%.1f, lat_px=%.1f sym=%s along_px=%.1f "
                    "end_px=%.1f min_dir_dot=%.2f, soft_nms=%s, hard_geom=%s soft_geom=%s, conf_idx=%d)"
                    % (
                        str(getattr(args, "gnn_adjacency_mode", "global")),
                        self.e2e_head.max_nodes,
                        _k_eff,
                        len(self.e2e_head.gat_layers),
                        int(getattr(args, "gnn_heads", 4)),
                        self.e2e_head.token_dim,
                        self.e2e_head.node_conf_thresh,
                        self.e2e_head.edge_radius_px,
                        float(getattr(args, "gnn_max_lateral_px", 48.0)),
                        bool(getattr(args, "gnn_max_lateral_sym", False)),
                        float(getattr(args, "gnn_max_along_px", 0.0)),
                        float(getattr(args, "gnn_max_end_gap_px", 0.0)),
                        float(getattr(args, "gnn_min_dir_dot", 0.0)),
                        bool(getattr(args, "gnn_soft_nms", False)),
                        bool(getattr(args, "gnn_use_hard_geom_gate", True)),
                        bool(getattr(args, "gnn_soft_geom_gate", False)),
                        self._e2e_conf_idx,
                    )
                )
            elif self.e2e_mode == "center":
                self.e2e_head = YolinoCenterPolyHead(
                    line_rep=coords.line_representation.enum,
                    fpn_channels=head_feat_channels,
                    token_dim=int(getattr(args, "e2e_token_dim", 256)),
                    num_queries=int(getattr(args, "center_num_queries", 20)),
                    num_points=int(getattr(args, "center_num_points", 10)),
                    decoder_layers=int(getattr(args, "center_decoder_layers", 4)),
                    decoder_heads=int(getattr(args, "center_decoder_heads", 8)),
                    decoder_ff=int(getattr(args, "center_decoder_ff", 1024)),
                    dropout=float(getattr(args, "center_dropout", 0.1)),
                    local_radius_px=float(getattr(args, "center_local_radius_px", 64.0)),
                    mask_mode=str(getattr(args, "center_mask_mode", "per_center")),
                    init_spread_px=float(getattr(args, "center_init_spread_px", 32.0)),
                    nms_kernel=int(getattr(args, "center_nms_kernel", 3)),
                    peak_thresh=float(getattr(args, "center_peak_thresh", 0.05)),
                    peak_nms_dist_px=float(getattr(args, "center_peak_nms_dist_px", 12.0)),
                    delta_bound=str(getattr(args, "center_delta_bound", "tanh")),
                    delta_max_px=float(getattr(args, "center_delta_max_px", 64.0)),
                )
                Log.info(
                    "YolinoCenterPolyHead enabled (K=%d, N=%d, layers=%d, heads=%d, "
                    "token_dim=%d, local_radius_px=%.1f, init_spread_px=%.1f, "
                    "mask_mode=%s, delta_bound=%s, delta_max_px=%.1f, peak_nms_px=%.1f, conf_idx=%d)"
                    % (
                        self.e2e_head.K,
                        self.e2e_head.N,
                        self.e2e_head.decoder_layers,
                        self.e2e_head.decoder_heads,
                        self.e2e_head.token_dim,
                        self.e2e_head.local_radius_px,
                        self.e2e_head.init_spread_px,
                        self.e2e_head.mask_mode,
                        self.e2e_head.delta_bound,
                        self.e2e_head.delta_max_px,
                        float(getattr(self.e2e_head, "peak_nms_dist_px", 0.0)),
                        self._e2e_conf_idx,
                    )
                )
            elif self.e2e_mode == "learnable_detr":
                from yolino.model.yolino_learnable_detr_head import YolinoLearnableDetrPolyHead

                self.e2e_head = YolinoLearnableDetrPolyHead(
                    fpn_channels=head_feat_channels,
                    token_dim=int(getattr(args, "e2e_token_dim", 256)),
                    num_queries=int(getattr(args, "e2e_num_queries", 20)),
                    decoder_layers=int(getattr(args, "e2e_decoder_layers", 6)),
                    decoder_heads=int(getattr(args, "e2e_decoder_heads", 8)),
                    decoder_ff=int(getattr(args, "e2e_decoder_ff", 1024)),
                    dropout=float(getattr(args, "gnn_dropout", 0.1)),
                    max_step_norm=float(getattr(args, "e2e_hough_max_step_norm", 0.1)),
                    mem_encoder_layers=int(getattr(args, "e2e_hough_mem_encoder_layers", 1)),
                    dn_mode=str(getattr(args, "e2e_hough_dn_mode", "simple")),
                    dn_groups=int(getattr(args, "e2e_hough_dn_groups", 3)),
                    dn_sigma_xy=float(getattr(args, "e2e_hough_dn_sigma_xy", 0.05)),
                    dn_length_scale=float(getattr(args, "e2e_hough_dn_length_scale", 0.2)),
                    dn_rot_deg=float(getattr(args, "e2e_hough_dn_rot_deg", 10.0)),
                    dn_off_epoch=int(getattr(args, "e2e_dn_off_epoch", -1)),
                    gt_vertical_angle_deg=float(getattr(args, "e2e_gt_vertical_angle_deg", 80.0)),
                )
                Log.info(
                    "YolinoLearnableDetrPolyHead enabled (K=%d, layers=%d, heads=%d, token_dim=%d, "
                    "max_step_norm=%.3f, mem_enc=%d, dn=%s, dn_groups=%d, dn_off_epoch=%d)"
                    % (
                        self.e2e_head.K,
                        self.e2e_head.decoder_layers_n,
                        self.e2e_head.decoder_heads,
                        self.e2e_head.token_dim,
                        self.e2e_head.max_step_norm,
                        self.e2e_head.mem_encoder_layers,
                        self.e2e_head.dn_mode,
                        self.e2e_head.dn_groups,
                        self.e2e_head.dn_off_epoch,
                    )
                )
            elif self.e2e_mode == "hough_detr":
                from yolino.model.yolino_hough_detr_head import YolinoHoughDetrPolyHead

                self.e2e_head = YolinoHoughDetrPolyHead(
                    line_rep=coords.line_representation.enum,
                    fpn_channels=head_feat_channels,
                    token_dim=int(getattr(args, "e2e_token_dim", 256)),
                    num_queries=int(getattr(args, "e2e_num_queries", 20)),
                    decoder_layers=int(getattr(args, "e2e_decoder_layers", 4)),
                    decoder_heads=int(getattr(args, "e2e_decoder_heads", 8)),
                    decoder_ff=int(getattr(args, "e2e_decoder_ff", 1024)),
                    dropout=float(getattr(args, "gnn_dropout", 0.1)),
                    conf_thresh=float(getattr(args, "e2e_hough_seg_conf_thresh", 0.3)),
                    max_segments=int(getattr(args, "e2e_hough_max_segments", 512)),
                    dbscan_eps=float(getattr(args, "e2e_hough_dbscan_eps", 0.05)),
                    dbscan_min_samples=int(getattr(args, "e2e_hough_dbscan_min_samples", 2)),
                    rho_weight=float(getattr(args, "e2e_hough_rho_weight", 1.0)),
                    theta_weight=float(getattr(args, "e2e_hough_theta_weight", 1.0)),
                    L_init_default=float(getattr(args, "e2e_hough_L_init_default", 0.3)),
                    pt_radius_norm=float(getattr(args, "e2e_hough_pt_radius_norm", 0.08)),
                    max_step_norm=float(getattr(args, "e2e_hough_max_step_norm", 0.05)),
                    mem_encoder_layers=int(getattr(args, "e2e_hough_mem_encoder_layers", 1)),
                    dn_mode=str(getattr(args, "e2e_hough_dn_mode", "simple")),
                    dn_groups=int(getattr(args, "e2e_hough_dn_groups", 3)),
                    dn_sigma_xy=float(getattr(args, "e2e_hough_dn_sigma_xy", 0.05)),
                    dn_length_scale=float(getattr(args, "e2e_hough_dn_length_scale", 0.2)),
                    dn_rot_deg=float(getattr(args, "e2e_hough_dn_rot_deg", 10.0)),
                    gt_vertical_angle_deg=float(getattr(args, "e2e_gt_vertical_angle_deg", 80.0)),
                )
                Log.info(
                    "YolinoHoughDetrPolyHead enabled (K=%d, layers=%d, heads=%d, token_dim=%d, "
                    "pt_radius_norm=%.3f, max_step_norm=%.3f, mem_enc=%d, dn=%s, dn_groups=%d, "
                    "conf_idx=%d)"
                    % (
                        self.e2e_head.K,
                        self.e2e_head.decoder_layers,
                        self.e2e_head.decoder_heads,
                        self.e2e_head.token_dim,
                        self.e2e_head.pt_radius_norm,
                        self.e2e_head.max_step_norm,
                        self.e2e_head.mem_encoder_layers,
                        self.e2e_head.dn_mode,
                        self.e2e_head.dn_groups,
                        self._e2e_conf_idx,
                    )
                )
            self._e2e_geom_activations = get_activations(args.activations, coords, args.linerep)

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): with shape [batch, 3, H, W], dtype=float32, values in [0,1]

        Returns:
            (geom_pred, embed_pred) or (geom_pred, embed_pred, e2e_out):
                geom_pred  with shape [batch, cells, preds, vars_train]
                embed_pred with shape [batch, cells, preds, embed_dim]
                e2e_out (only if ``e2e_differentiable_postproc``): dict from :class:`YolinoDetrBezierHead`
                with K Bézier polyline predictions and objectness logits. Computed inside ``forward`` so
                DDP sees all ``e2e_head`` parameters in the same autograd pass as the backbone.
                where cells = (H/stride) * (W/stride) for the chosen FPN head level.
        """
        in_h = int(x.shape[2])
        in_w = int(x.shape[3])
        feats = self.backbone(x)
        if self.use_std:
            if "P3" not in feats:
                raise ValueError(
                    "--std requires backbone feats['P3']; got %s" % sorted(list(feats.keys()))
                )
            x = feats["P3"]
        else:
            if self.head_level not in feats:
                raise ValueError("Requested head_level=%s but backbone returned %s. "
                                 "For timm with stride-16 top level, enable --timm_force_stride32_head=True "
                                 "and use --head_level=P5 for a stride-32 head map."
                                 % (self.head_level, sorted(list(feats.keys()))))
            x = feats[self.head_level]

        mode = self.feature_refine
        if mode == "none":
            x_geom, x_embed = x, x
        elif mode == "sa_embed_only":
            x_geom, x_embed = x, self.attention(x)
        elif mode == "sa_shared":
            x_ref = self.attention(x)
            x_geom, x_embed = x_ref, x_ref
        elif mode == "cbam_shared":
            x_ref = self.cbam(x)
            x_geom, x_embed = x_ref, x_ref
        else:
            raise ValueError("Unknown feature_refine=%r" % (mode,))

        if self.use_std:
            feat_std = self.std_head.unshuffle(x_geom)
            feat_e2e = self.std_feat_dcn(feat_std) if self.std_feat_dcn is not None else feat_std
            if self.std_skip_geom_head:
                hf, wf = feat_std.shape[-2:]
                geom_raw = torch.zeros(
                    feat_std.shape[0],
                    self.num_predictors * self.vars_train,
                    hf,
                    wf,
                    device=feat_std.device,
                    dtype=feat_std.dtype,
                )
                geom_pred = self.reshape_prediction(geom_raw)
                _, embed_raw = self.std_head.forward_from_feat(feat_std, geom=False, embed=True)
                if embed_raw is None:
                    embed_raw = torch.zeros(
                        feat_std.shape[0],
                        self.num_predictors * self.embed_dim,
                        hf,
                        wf,
                        device=feat_std.device,
                        dtype=feat_std.dtype,
                    )
                embed_pred = self.reshape_embedding(embed_raw)
            else:
                geom_raw, embed_raw = self.std_head.forward_from_feat(feat_std, geom=True, embed=True)
                geom_pred = self.reshape_prediction(geom_raw)
                embed_pred = self.reshape_embedding(embed_raw)
            x_geom = feat_e2e
        else:
            geom_pred = self.reshape_prediction(self.yolo(x_geom))
            embed_pred = self.reshape_embedding(self.embed_head(x_embed))

        Log.debug("YolinoNet forward shapes geom=%s (expected [B, cells, P, vars_train]) "
                  "embed=%s (expected [B, cells, P, %d], P=%d, head_feat=%s)"
                  % (tuple(geom_pred.shape), tuple(embed_pred.shape),
                     self.embed_dim, self.num_predictors, self._head_feat_level))
        if self.e2e_head is not None:
            stride_f = float(self.scale)
            # Heads that consume GT during forward (hough_detr DN) read it from
            # this transient stash set by ForwardRunner; we clear it immediately
            # to avoid cross-batch leakage.
            pending_gt = getattr(self, "_pending_e2e_gt_pack", None)
            if hasattr(self, "_pending_e2e_gt_pack"):
                self._pending_e2e_gt_pack = None
            pending_epoch = getattr(self, "_pending_e2e_epoch", None)
            if hasattr(self, "_pending_e2e_epoch"):
                self._pending_e2e_epoch = None
            if self.e2e_mode == "std_seg_detr":
                e2e_out = self.e2e_head(
                    x_geom,
                    stride_f,
                    in_h,
                    in_w,
                    self._e2e_conf_idx,
                )
            elif self.e2e_mode == "hough_detr":
                geom_act_e2e, _ = self._e2e_geom_activations((geom_pred, embed_pred))
                e2e_out = self.e2e_head(
                    geom_act_e2e.detach(),
                    x_geom,
                    stride_f,
                    in_h,
                    in_w,
                    self._e2e_conf_idx,
                    e2e_gt_pack=pending_gt,
                )
            elif self.e2e_mode == "learnable_detr":
                # Learnable-query head: no geom_act dependency (queries are
                # ``nn.Embedding`` params; DN GT comes from ``pending_gt``).
                e2e_out = self.e2e_head(
                    x_geom,
                    stride_f,
                    in_h,
                    in_w,
                    e2e_gt_pack=pending_gt,
                    current_epoch=pending_epoch,
                )
            else:
                geom_act_e2e, _ = self._e2e_geom_activations((geom_pred, embed_pred))
                e2e_out = self.e2e_head(
                    geom_act_e2e,
                    x_geom,
                    stride_f,
                    in_h,
                    in_w,
                    self._e2e_conf_idx,
                )
            return geom_pred, embed_pred, e2e_out
        return geom_pred, embed_pred

    def run_e2e_post(self, geom_act: torch.Tensor, pack: dict):
        """Run E2E head on **activated** geometry (same layout as training loss)."""
        if self.e2e_head is None:
            return None
        if self.e2e_mode == "std_seg_detr":
            return self.e2e_head(
                pack["head_feat"],
                pack["stride"],
                pack["img_h"],
                pack["img_w"],
                self._e2e_conf_idx,
            )
        if self.e2e_mode == "hough_detr":
            return self.e2e_head(
                geom_act.detach(),
                pack["head_feat"],
                pack["stride"],
                pack["img_h"],
                pack["img_w"],
                self._e2e_conf_idx,
                e2e_gt_pack=None,
            )
        if self.e2e_mode == "learnable_detr":
            return self.e2e_head(
                pack["head_feat"],
                pack["stride"],
                pack["img_h"],
                pack["img_w"],
                e2e_gt_pack=None,
                current_epoch=None,
            )
        return self.e2e_head(
            geom_act,
            pack["head_feat"],
            pack["stride"],
            pack["img_h"],
            pack["img_w"],
            self._e2e_conf_idx,
        )

    def reshape_prediction(self, pred):
        """

        Args:
            pred (torch.Tensor): with shape [batch, preds*vars, rows, cols]

        Returns:
            torch.Tensor: with shape [batch, cells, preds, vars]
        """
        batch_size = pred.shape[0]
        pred = pred.permute(0, 2, 3, 1)
        pred = pred.reshape(batch_size, -1, self.num_predictors, self.coords.num_vars_to_train())
        return pred

    def reshape_embedding(self, pred):
        """

        Args:
            pred (torch.Tensor): with shape [batch, preds*embed_dim, rows, cols]

        Returns:
            torch.Tensor: with shape [batch, cells, preds, embed_dim]
        """
        batch_size = pred.shape[0]
        pred = pred.permute(0, 2, 3, 1)  # [batch, rows, cols, preds*embed_dim]
        pred = pred.reshape(batch_size, -1, self.num_predictors, self.embed_dim)
        return pred

    def receptive_field(self, input_size):
        """
        input_size: (channels, H, W)
        """
        from torch_receptive_field import receptive_field
        return receptive_field(self, input_size=input_size)


def get_test_input(shape, batch_size):
    return torch.rand(batch_size, shape[2], shape[0], shape[1])


def get_test_label(cells, batch_size):
    return torch.rand(batch_size, cells, 1, 1)
