# SPDX-License-Identifier: GPL-3.0-or-later
"""
YOLinO-DETR Hybrid Bézier polyline head.

Replaces the per-token Bézier prediction of :mod:`yolino.model.e2e_polyline_modules`
with a DETR-style set-prediction over K polyline queries:

1. Phase 1 (Segment-Token fusion): each grid (cell, predictor) becomes a 256-D
   "segment token" = Linear(concat(visual_feat_at_midpoint, sine_pe_4d(x,y,dx,dy))).
2. Phase 2 (Top-K anchor init): the top-K confidence tokens become initial queries
   and their midpoints become reference points (anchors).
3. Phase 3 (Transformer decoder): K queries cross-attend over the full set of
   valid segment tokens (masked by ``conf > conf_filter_thresh``).
4. Phase 4 (Direct Bézier head): a 3-layer MLP predicts ``num_ctrl=degree+1``
   control-point offsets per query (in sigmoid-logit space, DETR-style), and a
   separate linear layer predicts objectness logits.

The 32-pt Bernstein expansion of the control points is appended to the output
dict for downstream loss / visualization.

The legacy :class:`yolino.model.e2e_polyline_modules.E2EDifferentiablePostHead`
remains in-tree but is no longer wired by :class:`YolinoNet`.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolino.model.e2e_polyline_modules import (
    bernstein_matrix,
    bezier_sample_points,
    mid_dir_geom_to_midpoints_pixels,
    pixels_to_grid_sample_grid,
)
from yolino.utils.enums import LINE
from yolino.utils.logger import Log


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Numerically stable inverse sigmoid (logit) for x in (0, 1)."""
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


class YolinoDetrBezierHead(nn.Module):
    """DETR-style polyline head producing K Bézier curves per image."""

    def __init__(
        self,
        line_rep: LINE,
        fpn_channels: int,
        num_queries: int = 20,
        decoder_layers: int = 3,
        decoder_heads: int = 8,
        decoder_ff: int = 1024,
        token_dim: int = 256,
        bezier_degree: int = 3,
        bezier_num_samples: int = 32,
        conf_filter_thresh: float = 0.1,
        local_max_filter: bool = False,
        local_max_kernel: int = 5,
        anchor_ctrl_offsets: bool = True,
        dropout: float = 0.1,
        sine_freq_bands: int = 16,
    ):
        super().__init__()
        if line_rep != LINE.MID_DIR:
            Log.warning(
                "YolinoDetrBezierHead is wired for MID_DIR geometry; got %s. "
                "Channels (u,v,dx,dy) are assumed in the first four geom slots."
                % line_rep
            )
        self.line_rep = line_rep
        self.num_queries = int(num_queries)
        self.token_dim = int(token_dim)
        self.bezier_degree = int(bezier_degree)
        self.num_ctrl = self.bezier_degree + 1
        self.bezier_num_samples = int(bezier_num_samples)
        self.conf_filter_thresh = float(conf_filter_thresh)
        self.local_max_filter = bool(local_max_filter)
        k_lm = int(local_max_kernel)
        if k_lm < 1:
            k_lm = 1
        if k_lm % 2 == 0:
            Log.warning(
                "YolinoDetrBezierHead local_max_kernel must be odd; got %d, using %d."
                % (k_lm, k_lm + 1)
            )
            k_lm += 1
        self.local_max_kernel = k_lm
        self.anchor_ctrl_offsets = bool(anchor_ctrl_offsets)
        self.sine_freq_bands = int(sine_freq_bands)

        # Visual feature projection from FPN channels to token_dim.
        self.visual_proj = nn.Linear(int(fpn_channels), self.token_dim)

        # Sine positional encoding over 4 channels (mid_x_norm, mid_y_norm, dx, dy).
        pe_in = 4 * 2 * self.sine_freq_bands
        self.pe_proj = nn.Linear(pe_in, self.token_dim)

        # Fusion of visual + positional features into a single token.
        self.fuse = nn.Linear(2 * self.token_dim, self.token_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.token_dim,
            nhead=int(decoder_heads),
            dim_feedforward=int(decoder_ff),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(decoder_layers))

        # 3-layer MLP: per-query control logits (either offsets from ref in logit space, or direct [0,1] logits).
        self.ctrl_head = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.num_ctrl * 2),
        )

        # Objectness head: per-query "is this a real wire?" logit.
        self.objectness_head = nn.Linear(self.token_dim, 1)

        # Pre-multiplied sinusoid frequencies (2^k * π, k in [0, L)).
        bands = (2.0 ** torch.arange(self.sine_freq_bands).float()) * math.pi
        self.register_buffer("_freq_bands", bands, persistent=False)

    def _sine_pe_4d(self, x4: torch.Tensor) -> torch.Tensor:
        """
        x4: [..., 4] features in roughly [-1, 1] (mid_x_norm, mid_y_norm, dx, dy).
        Returns: [..., 4 * 2 * L]
        """
        # [..., 4, 1] * [L] -> [..., 4, L]
        f = x4.unsqueeze(-1) * self._freq_bands
        sin = torch.sin(f)
        cos = torch.cos(f)
        pe = torch.cat([sin, cos], dim=-1)  # [..., 4, 2L]
        return pe.flatten(-2)  # [..., 4*2L]

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
        Args:
            geom_act: [B, cells, P, V] activated geometry predictions (sigmoid/linear per config).
            feat_map: [B, C, Hf, Wf] FPN feature map aligned with the head grid (cells == Hf*Wf).
            stride: pixel stride of the head grid.
            img_h / img_w: original input image height/width in pixels.
            conf_channel: channel index of confidence within ``V``.
        """
        b, ncell, p, v = geom_act.shape
        hf, wf = feat_map.shape[-2:]
        if ncell != hf * wf:
            raise ValueError(
                "YolinoDetrBezierHead expects cells == Hf*Wf; got cells=%d, feat H*W=%d"
                % (ncell, hf * wf)
            )

        g = geom_act.view(b, hf, wf, p, v)
        mid_px, end_a, end_b = mid_dir_geom_to_midpoints_pixels(
            g, stride=stride, img_h=img_h, img_w=img_w
        )
        dir_vec = end_b - end_a

        n_tokens = hf * wf * p
        mid_flat = mid_px.reshape(b, n_tokens, 2)
        dir_flat = dir_vec.reshape(b, n_tokens, 2)

        # Visual feature: F.grid_sample at each midpoint of the FPN map.
        grid = pixels_to_grid_sample_grid(mid_flat, img_h, img_w)
        sampled = F.grid_sample(
            feat_map, grid, mode="bilinear", padding_mode="border", align_corners=True
        )  # [B, C, N, 1]
        sampled = sampled.view(b, feat_map.shape[1], n_tokens).transpose(1, 2)  # [B, N, C]
        visual_feat = self.visual_proj(sampled)

        # Sine PE 4D over (mid_x_norm, mid_y_norm, dx_unit, dy_unit).
        scale_w = max(float(img_w - 1), 1.0)
        scale_h = max(float(img_h - 1), 1.0)
        mx = 2.0 * mid_flat[..., 0:1] / scale_w - 1.0
        my = 2.0 * mid_flat[..., 1:2] / scale_h - 1.0
        dnorm = dir_flat / dir_flat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        pe_in = torch.cat([mx, my, dnorm], dim=-1)  # [B, N, 4]
        pe_feat = self.pe_proj(self._sine_pe_4d(pe_in))

        tokens = self.fuse(torch.cat([visual_feat, pe_feat], dim=-1))  # [B, N, D]

        # Confidence filter.
        conf_flat = g[..., conf_channel].reshape(b, n_tokens)
        valid_mask = conf_flat > self.conf_filter_thresh  # [B, N]
        empty_rows = valid_mask.sum(dim=1) == 0
        if empty_rows.any():
            # Fallback: keep all tokens to avoid NaN softmax when nothing passes the threshold.
            valid_mask = valid_mask.clone()
            valid_mask[empty_rows] = True
        memory_key_padding_mask = ~valid_mask  # True = ignore.

        # Top-K anchor init by confidence (sorted descending).
        # Optional: CenterNet-style local-maxima NMS on the 2D conf grid before topk.
        # This only affects query selection; the memory (Keys/Values) for the decoder
        # is unchanged so non-peak tokens still contribute to cross-attention.
        if self.local_max_filter:
            # [B, H, W, P] -> [B, P, H, W] for per-predictor 2D max-pool.
            conf_2d = g[..., conf_channel].permute(0, 3, 1, 2).contiguous()
            klm = self.local_max_kernel
            local_max = F.max_pool2d(conf_2d, kernel_size=klm, stride=1, padding=klm // 2)
            peak_mask_2d = (conf_2d == local_max)  # [B, P, H, W]
            peak_mask_flat = peak_mask_2d.permute(0, 2, 3, 1).reshape(b, n_tokens)
            # Suppress non-peaks to a value below any valid conf so topk skips them.
            # We use a negative sentinel because conf could be 0 after sigmoid in edge cases.
            neg_inf = torch.finfo(conf_flat.dtype).min
            conf_for_topk = torch.where(peak_mask_flat, conf_flat, torch.full_like(conf_flat, neg_inf))
        else:
            conf_for_topk = conf_flat

        k = min(self.num_queries, n_tokens)
        _, topk_idx = torch.topk(conf_for_topk, k=k, dim=1)
        # Always log the un-filtered confidence of the selected indices for monitoring.
        topk_conf = torch.gather(conf_flat, 1, topk_idx)

        # Gather K queries and their reference (anchor) midpoints.
        idx_d = topk_idx.unsqueeze(-1).expand(-1, -1, self.token_dim)
        initial_queries = torch.gather(tokens, 1, idx_d)  # [B, K, D]
        idx_xy = topk_idx.unsqueeze(-1).expand(-1, -1, 2)
        ref_points_px = torch.gather(mid_flat, 1, idx_xy)  # [B, K, 2]
        ref_dir_norm = torch.gather(dnorm, 1, idx_xy)  # [B, K, 2]

        # Query positional embedding: sine PE of (ref_x_norm, ref_y_norm, dir).
        rx = 2.0 * ref_points_px[..., 0:1] / scale_w - 1.0
        ry = 2.0 * ref_points_px[..., 1:2] / scale_h - 1.0
        q_pe_in = torch.cat([rx, ry, ref_dir_norm], dim=-1)
        q_pe = self.pe_proj(self._sine_pe_4d(q_pe_in))
        queries = initial_queries + q_pe  # [B, K, D]

        decoded = self.decoder(
            tgt=queries,
            memory=tokens,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # [B, K, D]

        raw_ctrl_logits = self.ctrl_head(decoded).view(b, k, self.num_ctrl, 2)
        if self.anchor_ctrl_offsets:
            ref_norm_01 = torch.stack(
                [ref_points_px[..., 0] / scale_w, ref_points_px[..., 1] / scale_h], dim=-1
            ).clamp(min=1e-5, max=1.0 - 1e-5)
            ref_logit = _inverse_sigmoid(ref_norm_01)  # [B, K, 2]
            ctrl_logit = ref_logit.unsqueeze(2) + raw_ctrl_logits  # [B, K, num_ctrl, 2]
            ctrl_norm_01 = ctrl_logit.sigmoid()
        else:
            ctrl_norm_01 = raw_ctrl_logits.sigmoid()
        scale_xy = torch.tensor([scale_w, scale_h], device=ctrl_norm_01.device, dtype=ctrl_norm_01.dtype)
        ctrl_px = ctrl_norm_01 * scale_xy  # [B, K, num_ctrl, 2]

        # Objectness logits per query.
        obj_logits = self.objectness_head(decoded).squeeze(-1)  # [B, K]

        # Bernstein expansion (4 ctrl -> T sample points).
        bern = bernstein_matrix(
            self.bezier_degree, self.bezier_num_samples, ctrl_px.device, ctrl_px.dtype
        )
        curve_px = bezier_sample_points(ctrl_px, bern)  # [B, K, T, 2]

        return {
            "bezier_ctrl_px": ctrl_px,
            "bezier_curve_px": curve_px,
            "objectness_logits": obj_logits,
            "ref_points_px": ref_points_px,
            "topk_conf": topk_conf,
            "valid_token_mask": valid_mask,
        }
