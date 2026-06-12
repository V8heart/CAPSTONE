# Copyright 2023 Karlsruhe Institute of Technology, Institute for Measurement
# and Control Systems
#
# Differentiable E2E polyline head skeleton (soft local pooling, optional
# feature-aware attention, Bézier tensor ops). Original scipy/BFS postproc
# remains separate in yolino.postprocessing.line_fit.
#
# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolino.utils.enums import LINE, Variables
from yolino.utils.logger import Log


def bernstein_matrix(degree: int, num_samples: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Matrix T of shape [num_samples, degree + 1] with rows B(t_i) for uniform t in [0, 1].
    curve_points = T @ P  (P: [degree+1, 2] control points) → [num_samples, 2].
    """
    n = int(degree)
    if n < 1:
        raise ValueError("Bézier degree must be >= 1 (at least two control points).")
    t = torch.linspace(0.0, 1.0, int(num_samples), device=device, dtype=dtype)
    row_list = []
    for i in range(n + 1):
        c = math.comb(n, i)
        row = c * ((1.0 - t) ** (n - i)) * (t**i)
        row_list.append(row)
    return torch.stack(row_list, dim=1)


def bezier_sample_points(control_points: torch.Tensor, bernstein_t: torch.Tensor) -> torch.Tensor:
    """
    Args:
        control_points: [B, N, K, 2] (K = degree+1)
        bernstein_t: [T, K] precomputed Bernstein basis rows (same K)
    Returns:
        [B, N, T, 2]
    """
    # [B, N, T, 2] = einsum('tk,bnkd->bntd', bernstein_t, control_points)
    return torch.einsum("tk,bnkd->bntd", bernstein_t, control_points)


class SoftArgmaxSegmentTokens(nn.Module):
    """
    Phase 1: replace hard threshold / local winner with temperature-scaled softmax
    over a (2k+1)^2 neighborhood, then weighted sum of neighbor activations.
    Operates on the head grid (same H x W as geometry logits after reshape).
    """

    def __init__(self, window_size: int = 3, temperature: float = 0.5):
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd (e.g. 3 or 5).")
        self.window_size = int(window_size)
        self.temperature = float(temperature)

    def forward(self, geom_act: torch.Tensor, conf_index: int) -> torch.Tensor:
        """
        Args:
            geom_act: [B, H, W, P, V] activated training variables (sigmoid/linear per config).
            conf_index: channel index of confidence within V.
        Returns:
            soft_geom: [B, H, W, P, V] same layout, conf unchanged (copied through softmax weights).
        """
        b, h, w, p, v = geom_act.shape
        k = self.window_size
        pad = k // 2
        # [B*P, V, H, W]
        x = geom_act.permute(0, 3, 4, 1, 2).reshape(b * p, v, h, w)
        conf = x[:, conf_index : conf_index + 1]  # [B*P,1,H,W]
        conf_unf = F.unfold(conf, kernel_size=k, padding=pad)  # [B*P, k*k, H*W]
        conf_unf = conf_unf / max(self.temperature, 1e-6)
        wts = F.softmax(conf_unf, dim=1)  # softmax over neighborhood

        soft_ch = []
        for c in range(v):
            ch = x[:, c : c + 1]
            ch_unf = F.unfold(ch, kernel_size=k, padding=pad)
            mixed = (wts * ch_unf).sum(dim=1, keepdim=True)  # [B*P,1,H*W]
            mixed = mixed.view(b * p, 1, h, w)
            soft_ch.append(mixed)
        out = torch.cat(soft_ch, dim=1)  # [B*P, V, H, W]
        out = out.view(b, p, v, h, w).permute(0, 3, 4, 1, 2).contiguous()
        return out


def mid_dir_geom_to_midpoints_pixels(
    geom_act: torch.Tensor,
    stride: float,
    img_h: int,
    img_w: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert MID_DIR activated geometry to midpoint and endpoints in **pixel** coords.

    Channel semantics match the grid / ``get_image_lines`` UV_SPLIT convention used by
    ``plot.draw_cell`` (see comment there: row-like, col-like in the training tensor).
    For MID_DIR from ``to_md_cell_space``, the first four slots are::

        ch0 = midpoint fraction along the **row** axis within the cell (vertical / V),
        ch1 = midpoint fraction along the **col** axis within the cell (horizontal / H),
        ch2 = segment delta along **row** (dV, same units as ch0),
        ch3 = segment delta along **col** (dH, same units as ch1).

    Pixel positions follow ``Grid.get_image_lines``: ``V = row * stride + ...``,
    ``H = col * stride + ...``.  For ``F.grid_sample`` / OpenCV we return **image (x, y)**
    with ``x = H`` (column / width) and ``y = V`` (row / height), i.e. the same ordering
    as ``pixels_to_grid_sample_grid`` expects.

    geom_act: [B, H, W, P, 4+] — uses first four channels as above.
    Returns:
        mid_px: [B, H, W, P, 2]  (x_px, y_px) = (H, V)
        end_a:  [B, H, W, P, 2]
        end_b:  [B, H, W, P, 2]
    """
    b, hh, ww, p, _ = geom_act.shape
    device, dtype = geom_act.device, geom_act.dtype
    col = torch.arange(ww, device=device, dtype=dtype).view(1, 1, ww, 1, 1).expand(b, hh, ww, p, 1)
    row = torch.arange(hh, device=device, dtype=dtype).view(1, hh, 1, 1, 1).expand(b, hh, ww, p, 1)
    s = torch.tensor(float(stride), device=device, dtype=dtype)
    mid_v = geom_act[..., 0:1]
    mid_h = geom_act[..., 1:2]
    d_v = geom_act[..., 2:3]
    d_h = geom_act[..., 3:4]
    # V / H in pixels (row-major image indices).
    v_mid = (row + mid_v) * s
    h_mid = (col + mid_h) * s
    half_v = d_v * (0.5 * s)
    half_h = d_h * (0.5 * s)
    v_a = v_mid - half_v
    h_a = h_mid - half_h
    v_b = v_mid + half_v
    h_b = h_mid + half_h
    # (x, y) for grid_sample / cv2.line without UV_SPLIT swap = (H, V).
    mid_px = torch.cat([h_mid, v_mid], dim=-1)
    end_a = torch.cat([h_a, v_a], dim=-1)
    end_b = torch.cat([h_b, v_b], dim=-1)
    return mid_px, end_a, end_b


def pixels_to_grid_sample_grid(xy_px: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
    """
    xy_px: [B, N, 2] with (x=width dim, y=height dim) in pixel coords.
    Returns grid for F.grid_sample: [B, N, 1, 2] with (x_norm, y_norm) in [-1, 1].
    """
    x = xy_px[..., 0:1]
    y = xy_px[..., 1:2]
    gx = 2.0 * x / max(float(img_w - 1), 1.0) - 1.0
    gy = 2.0 * y / max(float(img_h - 1), 1.0) - 1.0
    g = torch.cat([gx, gy], dim=-1).view(xy_px.shape[0], xy_px.shape[1], 1, 2)
    return g


class TokenVisualSampler(nn.Module):
    """Sample FPN features at token midpoints using differentiable grid_sample."""

    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_dim)

    def forward(self, feat_map: torch.Tensor, mid_px: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
        """
        feat_map: [B, C, Hf, Wf]
        mid_px:   [B, N, 2]  (flattened spatial * predictors)
        """
        b, c, hf, wf = feat_map.shape
        g = pixels_to_grid_sample_grid(mid_px, img_h, img_w)
        # align_corners=True matches linear index mapping used above
        samp = F.grid_sample(feat_map, g, mode="bilinear", padding_mode="border", align_corners=True)
        vec = samp.view(b, c, -1).transpose(1, 2)  # [B, N, C]
        return self.proj(vec)


class GeometricPositionalEncoding(nn.Module):
    """Small MLP on (mid_px norm, dir unit) → pos_dim."""

    def __init__(self, pos_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pos_dim, pos_dim),
        )

    def forward(self, mid_px: torch.Tensor, dir_vec: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
        mx = 2.0 * mid_px[..., 0:1] / max(float(img_w - 1), 1.0) - 1.0
        my = 2.0 * mid_px[..., 1:2] / max(float(img_h - 1), 1.0) - 1.0
        d = dir_vec
        d = d / (d.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        inp = torch.cat([mx, my, d], dim=-1)
        return self.mlp(inp)


class FeatureAwareAffinityEncoder(nn.Module):
    """
    Phase 2 (extended): TransformerEncoder on fused tokens; affinity from attention map.
    Optional lightweight cross-attn: pool image features as memory (single vector broadcast).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        use_cross_image_context: bool = False,
    ):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.use_cross = bool(use_cross_image_context)
        if self.use_cross:
            self.cross_q = nn.Linear(d_model, d_model)
            self.cross_k = nn.Linear(d_model, d_model)
            self.cross_v = nn.Linear(d_model, d_model)
            self.cross_scale = d_model**0.5
        self.affinity_proj = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor, image_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        tokens: [B, N, D]
        image_context: [B, D] global pooled feature (optional)
        Returns:
            tokens_out: [B, N, D]
            affinity:   [B, N, N] row-softmax (stochastic adjacency prior)
        """
        h = self.encoder(tokens)
        if self.use_cross and image_context is not None:
            q = self.cross_q(h)
            k = self.cross_k(image_context).unsqueeze(1)
            v = self.cross_v(image_context).unsqueeze(1)
            att = torch.softmax((q * k).sum(-1, keepdim=True) / self.cross_scale, dim=1)
            h = h + att * v

        q2 = h
        k2 = self.affinity_proj(h)
        affinity_logits = torch.matmul(q2, k2.transpose(-2, -1)) / (h.shape[-1] ** 0.5)
        affinity = F.softmax(affinity_logits, dim=-1)
        return h, affinity


class E2EDifferentiablePostHead(nn.Module):
    """
    Optional head: soft local conf pooling → segment tokens → (optional) visual sampling
    + transformer affinity → per-token Bézier control points + sampled curve points.

    Enabled from yaml via YolinoNet; loss wiring is separate (see e2e_polyline_criterion).
    """

    def __init__(
        self,
        line_rep: LINE,
        fpn_channels: int,
        window_size: int = 3,
        softargmax_temperature: float = 0.5,
        token_dim: int = 128,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_ff: int = 256,
        bezier_degree: int = 3,
        bezier_num_samples: int = 32,
        feature_aware: bool = True,
        cross_image_context: bool = False,
    ):
        super().__init__()
        if line_rep != LINE.MID_DIR:
            Log.warning(
                "E2EDifferentiablePostHead is only wired for MID_DIR geometry in this skeleton; "
                "got %s — MID_DIR-style channels (u,v,dx,dy) are assumed." % line_rep
            )
        self.line_rep = line_rep
        self.window_size = int(window_size)
        self.soft_pool = SoftArgmaxSegmentTokens(window_size=window_size, temperature=softargmax_temperature)
        self.feature_aware = bool(feature_aware)
        self.posenc = GeometricPositionalEncoding(pos_dim=token_dim)
        if self.feature_aware:
            self.visual = TokenVisualSampler(fpn_channels, token_dim)
        else:
            self.geom_only = nn.Linear(4, token_dim)
        fuse_in = token_dim + token_dim
        self.fuse = nn.Linear(fuse_in, token_dim)
        self.tr = FeatureAwareAffinityEncoder(
            d_model=token_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_ff,
            dropout=0.1,
            use_cross_image_context=cross_image_context,
        )
        self.cross_image_context = bool(cross_image_context)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        degree = int(bezier_degree)
        self.bezier_degree = degree
        self.num_ctrl = degree + 1
        self.bezier_num_samples = int(bezier_num_samples)

        self.ctrl_head = nn.Linear(token_dim, self.num_ctrl * 2)

    def forward(
        self,
        geom_act: torch.Tensor,
        feat_map: torch.Tensor,
        stride: float,
        img_h: int,
        img_w: int,
        conf_channel: int,
    ) -> Dict[str, torch.Tensor]:
        """
        geom_act: [B, cells, P, V] — **activated** (post-sigmoid) predictions.
        feat_map: FPN map [B, C, H, W] aligned with the head grid (same H,W as cells).
        """
        b, ncell, p, v = geom_act.shape
        hw = feat_map.shape[-2] * feat_map.shape[-1]
        if ncell != hw:
            raise ValueError(
                "E2E head expects geom_act cells == H*W of feat_map. Got cells=%d, feat H*W=%d."
                % (ncell, hw)
            )
        h, w = feat_map.shape[-2], feat_map.shape[-1]
        g = geom_act.view(b, h, w, p, v)
        g_soft = self.soft_pool(g, conf_index=conf_channel)
        mid_px, end_a, end_b = mid_dir_geom_to_midpoints_pixels(g_soft, stride=stride, img_h=img_h, img_w=img_w)
        dir_vec = end_b - end_a

        mid_flat = mid_px.reshape(b, h * w * p, 2)
        pos = self.posenc(mid_px.reshape(b, h * w * p, 2), dir_vec.reshape(b, h * w * p, 2), img_h, img_w)

        if self.feature_aware:
            vis = self.visual(feat_map, mid_flat, img_h, img_w)
        else:
            vis = self.geom_only(g_soft[..., :4].reshape(b, h * w * p, 4))

        tok = self.fuse(torch.cat([pos, vis], dim=-1))

        img_ctx = None
        if self.cross_image_context:
            img_ctx = self.global_pool(feat_map).flatten(1)
        tok2, affinity = self.tr(tok, image_context=img_ctx)

        ctrl = self.ctrl_head(tok2).view(b, h * w * p, self.num_ctrl, 2)
        # Anchor control points around segment midpoint in pixel space (differentiable).
        mid_exp = mid_flat.unsqueeze(2).expand_as(ctrl)
        ctrl_px = mid_exp + ctrl * float(stride)

        bern = bernstein_matrix(self.bezier_degree, self.bezier_num_samples, geom_act.device, geom_act.dtype)
        curve = bezier_sample_points(ctrl_px, bern)

        return {
            "soft_geom": g_soft,
            "token_mid_px": mid_flat,
            "token_dir": F.normalize(dir_vec.reshape(b, h * w * p, 2), dim=-1, eps=1e-6),
            "tokens": tok2,
            "affinity": affinity,
            "bezier_ctrl_px": ctrl_px,
            "bezier_curve_px": curve,
        }
