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
        P{2,3,4} = 3x3(M{2,3,4})       # anti-aliasing smooth conv

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
                 use_fpn: bool = True):
        super().__init__()
        self.out_channels = int(out_channels)
        self.use_fpn = bool(use_fpn)
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
        # FPN-specific normalization.
        # Use GroupNorm (batch-size agnostic, avoids cuDNN BatchNorm kernel path instability).
        gn_groups = 32 if self.out_channels % 32 == 0 else 16
        self.smooth_norm_p2 = nn.GroupNorm(gn_groups, self.out_channels)
        self.smooth_norm_p3 = nn.GroupNorm(gn_groups, self.out_channels)
        self.smooth_norm_p4 = nn.GroupNorm(gn_groups, self.out_channels)

        for m in (self.lateral_c2, self.lateral_c3, self.lateral_c4,
                  self.smooth_p2, self.smooth_p3, self.smooth_p4):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

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

        p4 = self.smooth_norm_p4(self.smooth_p4(m4))
        p3 = self.smooth_norm_p3(self.smooth_p3(m3))
        p2 = self.smooth_norm_p2(self.smooth_p2(m2))

        mode = "FPN" if self.use_fpn else "no-FPN"
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
                 use_fpn: bool = True):
        super().__init__()
        import os as _os
        from yolino.model.darknet import Darknet

        if not cfg_path or not _os.path.isfile(cfg_path):
            raise FileNotFoundError(
                "Darknet backbone requires --darknet_cfg pointing to a valid .cfg, "
                "got %r" % cfg_path)

        self.out_channels = int(out_channels)
        self.use_fpn = bool(use_fpn)
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

        gn_groups = 32 if self.out_channels % 32 == 0 else 16
        self.smooth_norm_p2 = nn.GroupNorm(gn_groups, self.out_channels)
        self.smooth_norm_p3 = nn.GroupNorm(gn_groups, self.out_channels)
        self.smooth_norm_p4 = nn.GroupNorm(gn_groups, self.out_channels)

        for m in (self.lateral_c2, self.lateral_c3, self.lateral_c4,
                  self.smooth_p2, self.smooth_p3, self.smooth_p4):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

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

        p4 = self.smooth_norm_p4(self.smooth_p4(m4))
        p3 = self.smooth_norm_p3(self.smooth_p3(m3))
        p2 = self.smooth_norm_p2(self.smooth_p2(m2))

        mode = "FPN" if self.use_fpn else "no-FPN"
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
    _LEVEL_TO_STRIDE = {"P2": 8, "P3": 16, "P4": 32}

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
        backbone_name = str(getattr(args, "backbone", "convnext")).lower()
        self.backbone_name = backbone_name
        if backbone_name == "convnext":
            self.backbone = ConvNeXt_FPN_Backbone(out_channels=fpn_out,
                                                  pretrained=pretrained,
                                                  upsample_mode=upsample_mode,
                                                  use_fpn=use_fpn)
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
            )
        else:
            raise ValueError("Unknown --backbone=%r (expected 'convnext' or 'darknet')"
                             % backbone_name)
        Log.info("YolinoNet backbone=%s, use_fpn=%s, fpn_out=%d, head_level=%s"
                 % (backbone_name, str(use_fpn), fpn_out, str(getattr(args, "head_level", "P3"))))

        # FPN level fed to the heads.
        self.head_level = str(getattr(args, "head_level", "P3"))
        if self.head_level not in self._LEVEL_TO_STRIDE:
            raise ValueError("head_level must be one of %s, got %r"
                             % (list(self._LEVEL_TO_STRIDE.keys()), self.head_level))
        head_stride = self._LEVEL_TO_STRIDE[self.head_level]
        if int(self.scale) != head_stride:
            Log.warning("Backbone head outputs at stride %d (level %s) but args.scale=%s. "
                        "Set --scale %d so cell_size matches the feature map (and ensure "
                        "img_size %% %d == 0)."
                        % (head_stride, self.head_level, self.scale, head_stride, head_stride))

        in_channels = fpn_out  # heads consume the FPN channel directly

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

        # ----- Heads -----
        self.yolo = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_predictors * len(self.coords.get_position_of_training_vars()),
            kernel_size=1, stride=1, padding=0, bias=True,
        )
        self.embed_head = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_predictors * self.embed_dim,
            kernel_size=1, stride=1, padding=0, bias=True,
        )

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): with shape [batch, 3, H, W], dtype=float32, values in [0,1]

        Returns:
            (geom_pred, embed_pred):
                geom_pred  with shape [batch, cells, preds, vars_train]
                embed_pred with shape [batch, cells, preds, embed_dim]
                where cells = (H/stride) * (W/stride) for the chosen FPN head level.
        """
        feats = self.backbone(x)
        x = feats[self.head_level]  # default: P3 (1/16, fpn_out channels)

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

        # 1. Geometry
        geom_pred = self.yolo(x_geom)
        geom_pred = self.reshape_prediction(geom_pred)   # [B, cells, P, vars_train]
        # 2. Embedding
        embed_pred = self.embed_head(x_embed)
        embed_pred = self.reshape_embedding(embed_pred)  # [B, cells, P, embed_dim]

        Log.debug("YolinoNet forward shapes geom=%s (expected [B, cells, P, vars_train]) "
                  "embed=%s (expected [B, cells, P, %d], P=%d, head_level=%s)"
                  % (tuple(geom_pred.shape), tuple(embed_pred.shape),
                     self.embed_dim, self.num_predictors, self.head_level))
        return geom_pred, embed_pred

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
