# SPDX-License-Identifier: GPL-3.0-or-later
"""YOLinO Learnable-Query DETR polyline head (``--e2e_mode=learnable_detr``, exp53).

Spec (one-liner): DN queries teach the decoder *how* to shape a 5-pt polyline from
P4 features; learnable queries then learn *where* the wires are using the same
decoder.

Pipeline
========

* **Memory**: ``feat_map [B, C, Hf, Wf]`` (P4 from frozen ImageNet backbone +
  trainable FPN), flattened to ``[B, Hf*Wf, C]``, projected to ``D`` via
  ``visual_proj``, fused with sine 2-D positional encoding, optionally pushed
  through ``mem_encoder_layers`` self-attention layers.
* **Matching queries (K)**: ``query_content [K, D]`` and ``query_ref_pts [K, 2]``
  are ``nn.Embedding`` parameters. Initial 5-pt reference is **5 copies of the
  sigmoid-decoded 2-D ref**; the decoder spreads them via Δref per layer.
* **DN queries (N_inst × G, train-time only, gated by ``dn_off_epoch``)**: GT
  5-pt + noise via :func:`yolino.model.yolino_hough_dn.build_simple_dn` or
  :func:`build_lcdn`. DN content is a separate ``Linear(4, D)`` projection of
  ``(cx, cy, dx, dy)`` parameters from the noisy 5-pt.
* **Decoder**: ``decoder_layers`` of self-attn (block-diagonal mask: matching ↔
  DN isolated) → cross-attn (full P4 memory, query gets sine PE on the current
  ``ref_pts[:, :, 2, :]`` mid-point) → FFN → ``Δref [Q, 5, 2]`` predicted from
  the content vector. Each layer's ``ref_pts_for_loss = (ref.detach() + Δ).clamp``
  is stored for deep supervision; the next layer treats it as a detached input.
* **Output**: ``polylines_px``, ``ref_pts_layers``, ``objectness_logits``,
  plus DN bookkeeping (``dn_polylines_px``, ``dn_targets_px``, ``dn_valid``).
  Output schema is **the same** as :class:`YolinoHoughDetrPolyHead` so the
  existing :func:`compute_detr_e2e_loss` and trainer logging paths just work.

The ``dn_off_epoch`` gate disables DN slot generation when ``current_epoch >=
dn_off_epoch`` (training only); after the cut-off the decoder runs the matching
block alone, which matches the inference path.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolino.model.yolino_hough_dn import (
    build_dn_self_attn_mask,
    build_lcdn,
    build_simple_dn,
)
from yolino.utils.logger import Log


def _sine_pe(x: torch.Tensor, freq_bands: torch.Tensor) -> torch.Tensor:
    """Sine PE for the last dim. ``x [..., D] -> [..., D*2*L]``."""
    f = x.unsqueeze(-1) * freq_bands
    return torch.cat([torch.sin(f), torch.cos(f)], dim=-1).flatten(-2)


# ---------------------------------------------------------------------- #
# Decoder layer                                                          #
# ---------------------------------------------------------------------- #
class _LearnableDetrDecoderLayer(nn.Module):
    """Self-attn (block-diag mask) → cross-attn over P4 (full) → FFN → Δref MLP.

    The cross-attn query receives a sine PE built from ``ref_pts[:, :, 2, :]``
    (the 5-pt mid-point) so iterative refinement can re-attend to the moving
    reference. The Δref MLP outputs ``5*2 = 10`` values per query that are
    ``tanh``-bounded × ``max_step_norm`` (normalized image coords).
    """

    def __init__(self, dim: int, heads: int, ff: int, dropout: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(
                "e2e_token_dim (%d) must be divisible by heads (%d)" % (dim, heads)
            )
        self.dim = dim
        self.heads = heads

        self.norm_q1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm_q2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.drop2 = nn.Dropout(dropout)

        self.norm_q3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff, dim),
        )
        self.drop3 = nn.Dropout(dropout)

        self.delta_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 5 * 2),
        )

    def forward(
        self,
        content: torch.Tensor,            # [B, Q, D]
        ref_pe: torch.Tensor,             # [B, Q, D]
        memory: torch.Tensor,             # [B, N_mem, D]
        self_attn_mask: Optional[torch.Tensor],  # [Q, Q] bool, True=blocked
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = content
        xn1 = self.norm_q1(x)
        sa, _ = self.self_attn(
            xn1, xn1, xn1,
            attn_mask=self_attn_mask,
            need_weights=False,
        )
        x = x + self.drop1(sa)

        xn2 = self.norm_q2(x) + ref_pe
        ca, _ = self.cross_attn(
            xn2, memory, memory,
            need_weights=False,
        )
        x = x + self.drop2(ca)

        xn3 = self.norm_q3(x)
        x = x + self.drop3(self.ffn(xn3))

        delta_raw = self.delta_mlp(x)             # [B, Q, 10]
        return x, delta_raw


# ---------------------------------------------------------------------- #
# YolinoLearnableDetrPolyHead                                            #
# ---------------------------------------------------------------------- #
class YolinoLearnableDetrPolyHead(nn.Module):
    """exp53 learnable-query DETR 5-pt head.

    Args mirror :class:`YolinoHoughDetrPolyHead` where applicable so the existing
    args/config plumbing is reused. New knobs:

    * ``dn_off_epoch``  — int ≥ 0; when ``current_epoch >= dn_off_epoch`` (training)
                          DN slot generation is skipped. ``-1`` keeps DN on forever.
    * ``max_step_norm`` — normalized cap on per-layer Δ per keypoint (default 0.10,
                          spec value).
    """

    def __init__(
        self,
        fpn_channels: int,
        token_dim: int = 256,
        num_queries: int = 20,
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        decoder_ff: int = 1024,
        dropout: float = 0.1,
        sine_freq_bands: int = 16,
        max_step_norm: float = 0.1,
        mem_encoder_layers: int = 1,
        dn_mode: str = "simple",
        dn_groups: int = 3,
        dn_sigma_xy: float = 0.05,
        dn_length_scale: float = 0.2,
        dn_rot_deg: float = 10.0,
        dn_off_epoch: int = -1,
        gt_vertical_angle_deg: float = 80.0,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.K = int(num_queries)
        self.N = 5
        self.decoder_layers_n = int(decoder_layers)
        self.decoder_heads = int(decoder_heads)
        self.dropout = float(dropout)
        self.sine_freq_bands = int(sine_freq_bands)
        self.max_step_norm = float(max_step_norm)
        self.mem_encoder_layers = int(mem_encoder_layers)

        self.dn_mode = str(dn_mode).lower()
        if self.dn_mode not in ("none", "simple", "lcdn"):
            raise ValueError(
                "dn_mode must be one of 'none', 'simple', 'lcdn'; got %r" % self.dn_mode
            )
        self.dn_groups = int(max(0, dn_groups))
        self.dn_sigma_xy = float(dn_sigma_xy)
        self.dn_length_scale = float(dn_length_scale)
        self.dn_rot_deg = float(dn_rot_deg)
        self.dn_off_epoch = int(dn_off_epoch)
        self.gt_vertical_angle_deg = float(gt_vertical_angle_deg)

        bands = (2.0 ** torch.arange(self.sine_freq_bands).float()) * math.pi
        self.register_buffer("_freq_bands", bands, persistent=False)
        self._pe_dim_2 = 2 * 2 * self.sine_freq_bands

        # ---- Memory pipeline. ----
        self.visual_proj = nn.Linear(int(fpn_channels), self.token_dim)
        self.pe_proj_mem = nn.Linear(self._pe_dim_2, self.token_dim)
        self.fuse_mem = nn.Linear(2 * self.token_dim, self.token_dim)
        if self.mem_encoder_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.token_dim,
                nhead=self.decoder_heads,
                dim_feedforward=int(decoder_ff),
                dropout=self.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.mem_encoder = nn.TransformerEncoder(enc_layer, num_layers=self.mem_encoder_layers)
        else:
            self.mem_encoder = None

        # ---- Learnable queries (matching path). ----
        self.query_content = nn.Embedding(self.K, self.token_dim)
        nn.init.normal_(self.query_content.weight, std=0.02)
        self.query_ref_pts = nn.Embedding(self.K, 2)
        # Uniform in logit space so sigmoid stays well inside (0, 1) at init.
        nn.init.uniform_(self.query_ref_pts.weight, a=-1.5, b=1.5)

        # ---- DN content projection from (cx, cy, dx, dy). ----
        self.dn_proj = nn.Sequential(
            nn.Linear(4, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )

        # ---- Query ref PE (2-D normalized mid-point → D). ----
        self.point_pe_proj = nn.Linear(self._pe_dim_2, self.token_dim)

        # ---- Decoder stack. ----
        self.layers = nn.ModuleList(
            [
                _LearnableDetrDecoderLayer(self.token_dim, self.decoder_heads, int(decoder_ff), self.dropout)
                for _ in range(self.decoder_layers_n)
            ]
        )
        self.norm_out = nn.LayerNorm(self.token_dim)

        # ---- Output head. ----
        self.objectness_head = nn.Linear(self.token_dim, 1)

    # ------------------------------------------------------------------ #
    # Memory builder                                                      #
    # ------------------------------------------------------------------ #
    def _build_memory(
        self,
        feat_map: torch.Tensor,
        stride: float,
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        b, _, hf, wf = feat_map.shape
        device, dtype = feat_map.device, feat_map.dtype
        visual = feat_map.flatten(2).transpose(1, 2)                # [B, HW, C]
        visual = self.visual_proj(visual)                            # [B, HW, D]

        col = torch.arange(wf, device=device, dtype=dtype).view(1, 1, wf).expand(b, hf, wf)
        row = torch.arange(hf, device=device, dtype=dtype).view(1, hf, 1).expand(b, hf, wf)
        x_px = (col + 0.5) * float(stride)
        y_px = (row + 0.5) * float(stride)
        mem_xy = torch.stack([x_px, y_px], dim=-1).reshape(b, hf * wf, 2)

        sw = max(float(img_w - 1), 1.0)
        sh = max(float(img_h - 1), 1.0)
        mx = 2.0 * mem_xy[..., 0:1] / sw - 1.0
        my = 2.0 * mem_xy[..., 1:2] / sh - 1.0
        pe_in = torch.cat([mx, my], dim=-1)
        pe_feat = self.pe_proj_mem(_sine_pe(pe_in, self._freq_bands))   # [B, HW, D]

        mem = self.fuse_mem(torch.cat([visual, pe_feat], dim=-1))
        if self.mem_encoder is not None:
            mem = self.mem_encoder(mem)
        return mem

    # ------------------------------------------------------------------ #
    # Matching / DN initializers                                          #
    # ------------------------------------------------------------------ #
    def _matching_init(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        content = self.query_content.weight.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        ref_2d = self.query_ref_pts.weight.sigmoid()                  # [K, 2] in (0, 1)
        ref_2d = ref_2d.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        ref_5pt = ref_2d.unsqueeze(2).expand(-1, -1, self.N, -1).contiguous()
        return content.to(dtype), ref_5pt.to(dtype)

    def _dn_init(
        self,
        e2e_gt_pack: Optional[dict],
        img_h: int,
        img_w: int,
        device: torch.device,
        dtype: torch.dtype,
        b: int,
        current_epoch: Optional[int],
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        dn_disabled = (
            self.dn_mode == "none"
            or self.dn_groups <= 0
            or e2e_gt_pack is None
            or self.training is False
            or (
                self.dn_off_epoch >= 0
                and current_epoch is not None
                and int(current_epoch) >= int(self.dn_off_epoch)
            )
        )
        if dn_disabled:
            return (
                torch.zeros((b, 0, self.token_dim), device=device, dtype=dtype),
                torch.zeros((b, 0, self.N, 2), device=device, dtype=dtype),
                torch.zeros((b, 0, self.N, 2), device=device, dtype=dtype),
                torch.zeros((b, 0), dtype=torch.bool, device=device),
                0,
            )

        from yolino.model.e2e_polyline_order import canonicalize_polyline_xy
        from yolino.model.e2e_train_bridge import resample_polyline_xy

        padded = e2e_gt_pack["padded"].to(device=device, dtype=dtype, non_blocking=True)
        inst_m = e2e_gt_pack["inst_mask"].to(device=device, non_blocking=True)
        pt_m = e2e_gt_pack["pt_mask"].to(device=device, non_blocking=True)

        bi_size, ni, _, _ = padded.shape
        gt_5pt_px = padded.new_zeros((bi_size, ni, self.N, 2))
        for bi in range(bi_size):
            for ki in range(ni):
                if not bool(inst_m[bi, ki].item()):
                    continue
                if int(pt_m[bi, ki].sum().item()) < 2:
                    continue
                gt_5pt_px[bi, ki] = resample_polyline_xy(padded[bi, ki], pt_m[bi, ki], self.N)
        sw = max(float(img_w - 1), 1.0)
        sh = max(float(img_h - 1), 1.0)
        scale = padded.new_tensor([sw, sh]).view(1, 1, 1, 2)
        gt_5pt_norm = (gt_5pt_px / scale).clamp(0.0, 1.0)
        vdeg = float(self.gt_vertical_angle_deg)
        for bi in range(bi_size):
            for ki in range(ni):
                if not bool(inst_m[bi, ki].item()):
                    continue
                gt_5pt_norm[bi, ki] = canonicalize_polyline_xy(
                    gt_5pt_norm[bi, ki], vertical_angle_deg=vdeg
                )

        if self.dn_mode == "simple":
            dn = build_simple_dn(
                gt_5pt_norm, inst_m,
                num_groups=self.dn_groups,
                sigma_xy=self.dn_sigma_xy,
                generator=generator,
            )
        else:
            dn = build_lcdn(
                gt_5pt_norm, inst_m,
                num_groups=self.dn_groups,
                sigma_xy=self.dn_sigma_xy,
                scale_range=self.dn_length_scale,
                rot_deg=self.dn_rot_deg,
                generator=generator,
            )
        dn_refs = dn["refs"]                                         # [B, N*G, 5, 2]
        dn_targets = dn["targets"]
        dn_valid = dn["valid"]

        # DN content from the *noisy* 5-pt (p3 mid + endpoint chord vector).
        cx = dn_refs[..., 2, 0]
        cy = dn_refs[..., 2, 1]
        dx = dn_refs[..., 4, 0] - dn_refs[..., 0, 0]
        dy = dn_refs[..., 4, 1] - dn_refs[..., 0, 1]
        feat = torch.stack([cx, cy, dx, dy], dim=-1)
        content_dn = self.dn_proj(feat.detach())
        return content_dn, dn_refs, dn_targets, dn_valid, int(dn_refs.shape[1])

    # ------------------------------------------------------------------ #
    # Query positional encoding from current ref_pts                      #
    # ------------------------------------------------------------------ #
    def _query_pe(self, ref_pts: torch.Tensor) -> torch.Tensor:
        """``ref_pts: [B, Q, 5, 2]`` in [0, 1]. Use p3 mid-point → sine PE → ``[B, Q, D]``."""
        ref_mid = ref_pts[..., 2, :]                                 # [B, Q, 2]
        rx = 2.0 * ref_mid[..., 0:1] - 1.0
        ry = 2.0 * ref_mid[..., 1:2] - 1.0
        xy_n = torch.cat([rx, ry], dim=-1)
        return self.point_pe_proj(_sine_pe(xy_n, self._freq_bands))

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        feat_map: torch.Tensor,
        stride: float,
        img_h: int,
        img_w: int,
        e2e_gt_pack: Optional[dict] = None,
        current_epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        b = int(feat_map.shape[0])
        device = feat_map.device
        dtype = feat_map.dtype

        mem = self._build_memory(feat_map, stride, img_h, img_w)

        content_match, ref_match_norm = self._matching_init(b, device, dtype)
        content_dn, ref_dn_norm, dn_targets_norm, dn_valid, n_dn = self._dn_init(
            e2e_gt_pack, img_h, img_w, device, dtype, b, current_epoch
        )

        K = self.K
        content = torch.cat([content_match, content_dn], dim=1)              # [B, K + N_dn, D]
        ref_pts = torch.cat([ref_match_norm, ref_dn_norm], dim=1)            # [B, K + N_dn, 5, 2]
        n_total = K + n_dn

        if n_dn > 0:
            self_attn_mask = build_dn_self_attn_mask(
                n_matching=K, dn_group_sizes=(n_dn,), device=device
            )
        else:
            self_attn_mask = None

        sw = max(float(img_w - 1), 1.0)
        sh = max(float(img_h - 1), 1.0)
        scale_xy = torch.tensor([sw, sh], device=device, dtype=ref_pts.dtype).view(1, 1, 1, 2)

        ref_pts_layers_norm: List[torch.Tensor] = []
        ref_initial_norm = ref_pts.clone().detach()

        for li, layer in enumerate(self.layers):
            ref_pe = self._query_pe(ref_pts)
            content, delta_raw = layer(content, ref_pe, mem, self_attn_mask)
            delta_xy = delta_raw.reshape(b, n_total, self.N, 2)
            delta_xy = torch.tanh(delta_xy) * float(self.max_step_norm)

            ref_for_loss = (ref_pts.detach() + delta_xy).clamp(0.0, 1.0)
            ref_pts_layers_norm.append(ref_for_loss)
            ref_pts = ref_for_loss.detach()

        content = self.norm_out(content)
        final_norm = ref_pts_layers_norm[-1]

        match_final_norm = final_norm[:, :K]
        dn_final_norm = final_norm[:, K:]
        layers_match_norm = torch.stack([t[:, :K] for t in ref_pts_layers_norm], dim=0)
        layers_dn_norm = torch.stack([t[:, K:] for t in ref_pts_layers_norm], dim=0)

        match_px = match_final_norm * scale_xy
        layers_match_px = layers_match_norm * scale_xy.unsqueeze(0)
        dn_px = dn_final_norm * scale_xy
        layers_dn_px = layers_dn_norm * scale_xy.unsqueeze(0)
        dn_targets_px = dn_targets_norm * scale_xy
        ref_init_match_px = ref_initial_norm[:, :K] * scale_xy

        obj_logits = self.objectness_head(content[:, :K]).squeeze(-1)

        return {
            # DETR-compat keys.
            "polylines_px": match_px,
            "bezier_curve_px": match_px,
            "objectness_logits": obj_logits,
            "ref_points_px_init": ref_init_match_px,
            "ref_pts_layers": layers_match_px,
            # DN branch outputs.
            "dn_polylines_px": dn_px,
            "dn_polylines_layers": layers_dn_px,
            "dn_targets_px": dn_targets_px,
            "dn_valid": dn_valid,
            # Bookkeeping (DDP autograd + viz hooks).
            "node_feat": content,
        }
