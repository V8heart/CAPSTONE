# SPDX-License-Identifier: GPL-3.0-or-later
"""YOLinO-DETR Hybrid 5-pt head with Hough anchors and Localized Steering Deformable
Attention (LSDA) — ``--e2e_mode=hough_detr`` for ``exp51``.

Architecture (see exp51 plan §2):

1. **Path A — Hough Anchor (no-grad).**
   ``segments_to_hough_anchors`` clusters local YOLinO segments in normalized
   ``(rho, theta)`` Hough space (DBSCAN) and produces a *master anchor*
   ``(cx, cy, theta, L_init)`` per cluster. ``anchor_to_5pt_ref`` lays a
   deterministic 5-pt soft reference along the anchor direction; an
   ``MLP(4 -> D)`` (the **content query Linear** reused by DN, see plan §5.1)
   embeds the anchor parameters as the matching query token.

2. **Path B — Memory (P4 feature tokens).**
   Flatten ``P4 [B, C, Hf, Wf]`` to ``[B, Hf*Wf, D]`` (linear visual proj +
   sine PE in normalized image coords + optional ``nn.TransformerEncoder``).
   Each memory token has its pixel-space ``(x, y)`` recorded so the LSDA
   union-of-circles mask can be built in pixel distance.

3. **Decoder (L layers, ``_LSDALineDecoderLayer``).**
   Per-layer pipeline:
       Self-attn over [B, K_total, D] queries with optional **block-diagonal DN
       mask** (matching block isolated from each DN group).
     → expand per-keypoint queries to ``[B, K_total*5, D]`` (content token + 2-D
       sine PE on the current ``ref_pts``).
     → **Union-of-5-circles cross-attn**: point query ``(k, j)`` attends memory
       token ``m`` iff ``||p_m - ref_pts[k, j]|| <= pt_radius_px``; a zero-row
       fallback opens the nearest token to avoid NaN softmax (mirrors
       ``yolino_center_head._LocalizedDecoderLayer``).
     → FFN.
     → **Δref MLP** produces ``delta_xy [B, K_total, 5, 2]``, bounded by
       ``tanh × max_step_norm``. Reference update follows the corrected detach
       policy from plan §6:

           ref_pts_for_loss = (ref_pts.detach() + delta_xy).clamp(0, 1)
           # loss flows through delta_xy of THIS layer
           ref_pts_next    = ref_pts_for_loss.detach()
           # next layer treats the new ref as a constant (no history grad)

     → ``content_next = content + q5_out[:, :, 2, :]`` (p3 center token only;
       mean-reduce dilutes positional info — see plan §2 Step 7).

4. **Output head.**
   * ``polylines_px = ref_pts_last * (img_w-1, img_h-1)`` (final-ref mode).
   * ``objectness_logits = Linear(D, 1)(content_last)`` per query.
   * DN slot outputs are stripped from the matching outputs and forwarded
     separately in ``dn_polylines_px`` / ``dn_targets`` for the criterion's
     direct L1 path.

All Path A outputs and intermediate ``ref_pts_next`` carry no gradient. The
trainer also wraps ``geom_act.detach()`` upstream (caller passes the detached
geometry); the head additionally re-detaches inside Path A as a safety net.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolino.model.e2e_polyline_modules import pixels_to_grid_sample_grid
from yolino.model.yolino_hough_cluster import (
    anchor_to_5pt_ref,
    segments_to_hough_anchors,
)
from yolino.model.yolino_hough_dn import (
    build_dn_self_attn_mask,
    build_lcdn,
    build_simple_dn,
    gt_5pt_anchor_params,
)
from yolino.utils.enums import LINE
from yolino.utils.logger import Log


def _sine_pe(x: torch.Tensor, freq_bands: torch.Tensor) -> torch.Tensor:
    """Sine PE for the last dim. ``x [..., D] -> [..., D*2*L]``."""
    f = x.unsqueeze(-1) * freq_bands
    return torch.cat([torch.sin(f), torch.cos(f)], dim=-1).flatten(-2)


# ---------------------------------------------------------------------- #
# LSDA decoder layer                                                     #
# ---------------------------------------------------------------------- #
class _LSDALineDecoderLayer(nn.Module):
    """One decoder layer with **per-instance** self-attn (over ``K_total``) and
    **per-keypoint** cross-attn (over ``K_total * 5``) using a union-of-5-circles
    LSDA mask.

    The self-attn input is the per-query content token ``[B, K_total, D]``; the
    per-keypoint cross-attn input is ``content + PE(ref_pts)`` expanded over 5
    keypoints. The Δref MLP outputs per-keypoint ``Δxy`` over the *cross-attn*
    output (so each keypoint refines independently).

    Notes:
        * ``self_attn_mask`` semantics follow :class:`torch.nn.MultiheadAttention`:
          ``True`` means *masked* (entries blocked). The mask is broadcast
          across all batches.
        * ``cross_mask`` is built externally so it can reuse a precomputed
          memory ``(x, y)`` grid. Shape ``[B*heads, K_total*5, N_all]`` float
          with ``-inf`` blocked entries; or ``None`` for full cross-attn.
        * ``zero_row_mask [B, K_total*5]`` zeros the cross-attn message for
          queries that ended up with no valid keys (post-fallback).
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
            nn.Linear(dim, 2),
        )

    def forward(
        self,
        content: torch.Tensor,          # [B, K, D]
        per_pt_pe: torch.Tensor,        # [B, K, 5, D] PE on current ref_pts
        mem: torch.Tensor,              # [B, N_all, D]
        cross_mask: Optional[torch.Tensor],   # [B*heads, K*5, N_all] float (or None)
        zero_row_mask: Optional[torch.Tensor],  # [B, K*5] bool
        self_attn_mask: Optional[torch.Tensor],  # [K, K] bool (True=blocked)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(content_out [B, K, D], delta_xy [B, K, 5, 2])``."""
        # ----- per-instance self-attn over K queries -----
        x = content
        xn = self.norm_q1(x)
        sa, _ = self.self_attn(
            xn, xn, xn,
            attn_mask=self_attn_mask,
            need_weights=False,
        )
        x = x + self.drop1(sa)

        # ----- per-keypoint cross-attn over K*5 queries -----
        b, k, d = x.shape
        # query tokens: content broadcast + 2-D ref PE.
        q5 = x.unsqueeze(2).expand(b, k, 5, d) + per_pt_pe        # [B, K, 5, D]
        q5_flat = q5.reshape(b, k * 5, d)

        xn2 = self.norm_q2(q5_flat)
        ca, _ = self.cross_attn(xn2, mem, mem, attn_mask=cross_mask, need_weights=False)
        if zero_row_mask is not None:
            ca = ca.masked_fill(zero_row_mask.unsqueeze(-1), 0.0)
        q5_after_ca = q5_flat + self.drop2(ca)

        # ----- FFN -----
        xn3 = self.norm_q3(q5_after_ca)
        q5_out = q5_after_ca + self.drop3(self.ffn(xn3))           # [B, K*5, D]

        # ----- Δref MLP (per-keypoint) -----
        delta_raw = self.delta_mlp(q5_out)                          # [B, K*5, 2]
        delta_xy = delta_raw.reshape(b, k, 5, 2)

        # ----- content update: p3 center token only (plan §2 Step 7) -----
        q5_out_k5 = q5_out.reshape(b, k, 5, d)
        content_out = x + q5_out_k5[:, :, 2, :]                     # [B, K, D]

        return content_out, delta_xy


# ---------------------------------------------------------------------- #
# YolinoHoughDetrPolyHead                                                #
# ---------------------------------------------------------------------- #
class YolinoHoughDetrPolyHead(nn.Module):
    """5-point polyline head with Hough anchor + LSDA decoder + DN.

    Forward returns the standard E2E DETR dict (``polylines_px``,
    ``objectness_logits``, alias ``bezier_curve_px`` for trainer compat) plus
    Hough/DN diagnostics for TB overlays and the DN loss path:

    * ``polylines_px``         ``[B, K, 5, 2]`` final 5-pt prediction (pixels).
    * ``bezier_curve_px``      alias for ``polylines_px`` (trainer DDP stub
                               indexes this key).
    * ``objectness_logits``    ``[B, K]``.
    * ``ref_points_px_init``   ``[B, K, 5, 2]`` initial Hough 5-pt ref (pixels).
    * ``ref_pts_layers``       ``[L, B, K, 5, 2]`` per-layer refined ref (pixels)
                               for deep-supervision aux loss (each retains grad
                               through its own layer's ``delta_xy``).
    * ``hough_anchors``        dict with ``cx``, ``cy``, ``theta``, ``L_init``,
                               ``valid`` (each ``[B, K]``, all detached, no grad).
    * ``dn_polylines_px``      ``[B, N_dn_total, 5, 2]`` predicted DN refs in
                               pixels (final layer). Used by the criterion's
                               direct L1.
    * ``dn_targets_px``        ``[B, N_dn_total, 5, 2]`` clean GT 5-pt in pixels.
    * ``dn_valid``             ``[B, N_dn_total]`` bool.
    * ``dn_polylines_layers``  ``[L, B, N_dn_total, 5, 2]`` per-layer DN refs
                               (for optional DN aux supervision).

    Notes:
        * When ``e2e_hough_dn_mode == "none"`` or no GT pack is supplied, the
          DN-related keys carry length-0 tensors but remain present so the
          downstream loss/trainer code does not need to branch on absence.
    """

    def __init__(
        self,
        line_rep: LINE,
        fpn_channels: int,
        token_dim: int = 256,
        num_queries: int = 20,
        decoder_layers: int = 4,
        decoder_heads: int = 8,
        decoder_ff: int = 1024,
        dropout: float = 0.1,
        sine_freq_bands: int = 16,
        # Hough cluster knobs.
        conf_thresh: float = 0.3,
        max_segments: int = 512,
        dbscan_eps: float = 0.05,
        dbscan_min_samples: int = 2,
        rho_weight: float = 1.0,
        theta_weight: float = 1.0,
        L_init_default: float = 0.3,
        # Decoder knobs.
        pt_radius_norm: float = 0.08,
        max_step_norm: float = 0.05,
        mem_encoder_layers: int = 1,
        # DN knobs.
        dn_mode: str = "simple",
        dn_groups: int = 3,
        dn_sigma_xy: float = 0.05,
        dn_length_scale: float = 0.2,
        dn_rot_deg: float = 10.0,
        gt_vertical_angle_deg: float = 80.0,
    ):
        super().__init__()
        if line_rep != LINE.MID_DIR:
            Log.warning(
                "YolinoHoughDetrPolyHead is wired for MID_DIR geometry; got %s. "
                "Path A assumes the standard (mid_v, mid_h, d_v, d_h) layout." % line_rep
            )
        self.line_rep = line_rep
        self.token_dim = int(token_dim)
        self.K = int(num_queries)
        self.N = 5  # 5-pt soft reference is the head's signature
        self.decoder_layers = int(decoder_layers)
        self.decoder_heads = int(decoder_heads)
        self.dropout = float(dropout)
        self.sine_freq_bands = int(sine_freq_bands)

        # Hough cluster knobs.
        self.conf_thresh = float(conf_thresh)
        self.max_segments = int(max_segments)
        self.dbscan_eps = float(dbscan_eps)
        self.dbscan_min_samples = int(dbscan_min_samples)
        self.rho_weight = float(rho_weight)
        self.theta_weight = float(theta_weight)
        self.L_init_default = float(L_init_default)

        # Decoder knobs.
        self.pt_radius_norm = float(pt_radius_norm)
        self.max_step_norm = float(max_step_norm)
        self.mem_encoder_layers = int(max(0, mem_encoder_layers))

        # DN knobs.
        self.dn_mode = str(dn_mode).lower()
        if self.dn_mode not in ("none", "simple", "lcdn"):
            raise ValueError(
                "e2e_hough_dn_mode must be one of 'none', 'simple', 'lcdn'; got %r"
                % self.dn_mode
            )
        self.dn_groups = int(max(0, dn_groups))
        self.dn_sigma_xy = float(dn_sigma_xy)
        self.dn_length_scale = float(dn_length_scale)
        self.dn_rot_deg = float(dn_rot_deg)
        self.gt_vertical_angle_deg = float(gt_vertical_angle_deg)

        # ---- Sine PE buffer (shared across all sine PE usages). ----
        bands = (2.0 ** torch.arange(self.sine_freq_bands).float()) * math.pi
        self.register_buffer("_freq_bands", bands, persistent=False)
        self._pe_dim_2 = 2 * 2 * self.sine_freq_bands

        # ---- Memory pipeline (Path B). ----
        self.visual_proj = nn.Linear(int(fpn_channels), self.token_dim)
        self.pe_proj_mem = nn.Linear(self._pe_dim_2, self.token_dim)
        self.fuse_mem = nn.Linear(2 * self.token_dim, self.token_dim)
        if self.mem_encoder_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.token_dim,
                nhead=self.decoder_heads,
                dim_feedforward=decoder_ff,
                dropout=self.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.mem_encoder = nn.TransformerEncoder(enc_layer, num_layers=self.mem_encoder_layers)
        else:
            self.mem_encoder = None

        # ---- Content query Linear (4 anchor params -> D). Reused for DN. ----
        # Anchor params input order: (cx, cy, cos_theta, sin_theta).
        self.anchor_to_content = nn.Sequential(
            nn.Linear(4, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )

        # ---- Point-PE projection (2-D normalized ref -> D). ----
        self.point_pe_proj = nn.Linear(self._pe_dim_2, self.token_dim)
        # Slot embedding per-keypoint (5 slots, anti-collapse).
        self.slot_embed = nn.Embedding(self.N, self.token_dim)

        # ---- Decoder stack. ----
        self.layers = nn.ModuleList(
            [
                _LSDALineDecoderLayer(self.token_dim, self.decoder_heads, int(decoder_ff), self.dropout)
                for _ in range(self.decoder_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(self.token_dim)

        # ---- Output head. ----
        self.objectness_head = nn.Linear(self.token_dim, 1)

    # ------------------------------------------------------------------ #
    # Memory                                                              #
    # ------------------------------------------------------------------ #
    def _build_memory(
        self,
        feat_map: torch.Tensor,
        stride: float,
        img_h: int,
        img_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flatten ``P4`` into memory tokens with sine PE on normalized grid ``(x, y)``.

        Returns:
            mem ``[B, Hf*Wf, D]``, mem_xy ``[B, Hf*Wf, 2]`` (pixel ``(x, y)``).
        """
        b, c, hf, wf = feat_map.shape
        device = feat_map.device
        dtype = feat_map.dtype
        # Visual proj on a [B, Hf*Wf, C] sequence.
        visual = feat_map.flatten(2).transpose(1, 2)              # [B, Hf*Wf, C]
        visual = self.visual_proj(visual)                          # [B, Hf*Wf, D]

        # Pixel-space (x, y) at each feature cell center.
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
        pe_feat = self.pe_proj_mem(_sine_pe(pe_in, self._freq_bands))   # [B, Hf*Wf, D]

        mem = self.fuse_mem(torch.cat([visual, pe_feat], dim=-1))   # [B, Hf*Wf, D]
        if self.mem_encoder is not None:
            mem = self.mem_encoder(mem)
        return mem, mem_xy

    # ------------------------------------------------------------------ #
    # Per-keypoint PE                                                     #
    # ------------------------------------------------------------------ #
    def _per_keypoint_pe(self, ref_norm: torch.Tensor) -> torch.Tensor:
        """Build content-free per-keypoint PE.

        Args:
            ref_norm: ``[B, K, 5, 2]`` normalized ``(x, y)`` reference.

        Returns:
            ``[B, K, 5, D]`` = point_pe_proj(sine_pe(2*xy - 1)) + slot_embed.
        """
        rx = 2.0 * ref_norm[..., 0:1] - 1.0
        ry = 2.0 * ref_norm[..., 1:2] - 1.0
        xy_n = torch.cat([rx, ry], dim=-1)
        point_pe = self.point_pe_proj(_sine_pe(xy_n, self._freq_bands))   # [B, K, 5, D]
        b, k, _, d = point_pe.shape
        slot = self.slot_embed.weight.view(1, 1, self.N, d).expand(b, k, self.N, d)
        return point_pe + slot

    # ------------------------------------------------------------------ #
    # Cross-attn mask: union-of-5-circles                                 #
    # ------------------------------------------------------------------ #
    def _build_lsda_mask(
        self,
        ref_pts_px: torch.Tensor,   # [B, K, 5, 2] pixels
        mem_xy: torch.Tensor,       # [B, N_all, 2] pixels
        radius_px: float,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Union-of-5-circles LSDA mask.

        Point query ``(k, j)`` attends memory token ``m`` iff its pixel distance
        to ``ref_pts_px[k, j]`` is ``<= radius_px``. Queries left with zero
        valid keys are patched with the nearest token (avoid NaN softmax) and
        flagged in ``zero_row`` so the caller can zero the message.
        """
        b, k, _, _ = ref_pts_px.shape
        n_all = mem_xy.shape[1]
        if radius_px <= 0.0:
            return None, torch.zeros((b, k * 5), dtype=torch.bool, device=ref_pts_px.device)
        # Distance per keypoint: [B, K*5, N_all].
        ref_flat = ref_pts_px.reshape(b, k * 5, 2).float()
        d = torch.cdist(ref_flat, mem_xy.float(), p=2)         # [B, K*5, N_all]
        within = d <= float(radius_px)                          # [B, K*5, N_all]

        any_valid = within.any(dim=-1)                          # [B, K*5]
        if (~any_valid).any():
            nearest_idx = d.argmin(dim=-1, keepdim=True)        # [B, K*5, 1]
            scatter_helper = torch.zeros_like(within)
            scatter_helper.scatter_(-1, nearest_idx, True)
            within = torch.where(any_valid.unsqueeze(-1), within, scatter_helper)

        float_mask = torch.zeros_like(within, dtype=torch.float32)
        float_mask = float_mask.masked_fill(~within, float("-inf"))

        cross_mask = float_mask.unsqueeze(1).expand(-1, self.decoder_heads, -1, -1).reshape(
            b * self.decoder_heads, k * 5, n_all
        )
        zero_row = (~any_valid)
        return cross_mask, zero_row

    # ------------------------------------------------------------------ #
    # Hough anchor → matching content + ref_pts                            #
    # ------------------------------------------------------------------ #
    def _matching_init(
        self,
        anchors: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build matching ``content [B, K, D]`` and ``ref_pts_norm [B, K, 5, 2]`` from
        the (no-grad) Hough anchors.

        Both outputs are **freshly built** (anchor params themselves are
        detached, but the Linear that embeds them is trainable).
        """
        cx = anchors["cx"]            # [B, K]
        cy = anchors["cy"]
        theta = anchors["theta"]
        L_init = anchors["L_init"]
        valid = anchors["valid"]       # [B, K]
        # Reseed invalid slots to image center / horizontal so the input to
        # ``anchor_to_5pt_ref`` stays in-range; their queries are not Hungarian-
        # supervised because matching uses validity mask through objectness.
        cx = torch.where(valid, cx, torch.full_like(cx, 0.5))
        cy = torch.where(valid, cy, torch.full_like(cy, 0.5))
        L_init = torch.where(valid, L_init, torch.full_like(L_init, self.L_init_default))
        theta = torch.where(valid, theta, torch.zeros_like(theta))

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        # Anchor param embedding (Linear is trainable).
        anchor_feat = torch.stack([cx, cy, cos_t, sin_t], dim=-1)   # [B, K, 4]
        content = self.anchor_to_content(anchor_feat.detach())       # [B, K, D]

        ref_pts_norm = anchor_to_5pt_ref(cx, cy, theta, L_init)      # [B, K, 5, 2] detached
        return content, ref_pts_norm

    # ------------------------------------------------------------------ #
    # DN init                                                              #
    # ------------------------------------------------------------------ #
    def _dn_init(
        self,
        e2e_gt_pack: Optional[dict],
        img_h: int,
        img_w: int,
        device: torch.device,
        dtype: torch.dtype,
        b: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Build DN matching block.

        Returns:
            content_dn   : ``[B, N_dn, D]``  (zero-sized when DN disabled)
            ref_pts_dn   : ``[B, N_dn, 5, 2]`` normalized
            dn_targets   : ``[B, N_dn, 5, 2]`` normalized (clean GT)
            dn_valid     : ``[B, N_dn]`` bool
            n_dn_per_batch : int  number of DN slots per batch (same across batch)

        When ``e2e_gt_pack`` is None or ``dn_mode == "none"`` or ``dn_groups == 0``
        returns length-0 tensors.
        """
        if (
            self.dn_mode == "none"
            or self.dn_groups <= 0
            or e2e_gt_pack is None
            or self.training is False
        ):
            return (
                torch.zeros((b, 0, self.token_dim), device=device, dtype=dtype),
                torch.zeros((b, 0, 5, 2), device=device, dtype=dtype),
                torch.zeros((b, 0, 5, 2), device=device, dtype=dtype),
                torch.zeros((b, 0), dtype=torch.bool, device=device),
                0,
            )

        padded = e2e_gt_pack["padded"].to(device=device, dtype=dtype, non_blocking=True)
        inst_m = e2e_gt_pack["inst_mask"].to(device=device, non_blocking=True)
        pt_m = e2e_gt_pack["pt_mask"].to(device=device, non_blocking=True)

        # Resample each GT polyline to 5 points via arc-length (lazy import to
        # keep top-level import simple).
        from yolino.model.e2e_train_bridge import resample_polyline_xy
        bi_size, ni, _, _ = padded.shape
        gt_5pt_px = padded.new_zeros((bi_size, ni, 5, 2))
        for bi in range(bi_size):
            for ki in range(ni):
                if not bool(inst_m[bi, ki].item()):
                    continue
                if int(pt_m[bi, ki].sum().item()) < 2:
                    continue
                gt_5pt_px[bi, ki] = resample_polyline_xy(padded[bi, ki], pt_m[bi, ki], 5)
        # Normalize to [0, 1].
        sw = max(float(img_w - 1), 1.0)
        sh = max(float(img_h - 1), 1.0)
        scale = padded.new_tensor([sw, sh]).view(1, 1, 1, 2)
        gt_5pt_norm = (gt_5pt_px / scale).clamp(0.0, 1.0)
        # Flip-only canonicalize on the 5-pt chain (path order already set in padded).
        from yolino.model.e2e_polyline_order import canonicalize_polyline_xy

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
        dn_refs = dn["refs"]             # [B, N*G, 5, 2]
        dn_targets = dn["targets"]
        dn_valid = dn["valid"]
        # DN content = anchor_to_content on jittered anchor params.
        cx_g, cy_g, theta_g, L_g = gt_5pt_anchor_params(dn_refs)
        anchor_feat_dn = torch.stack([cx_g, cy_g, torch.cos(theta_g), torch.sin(theta_g)], dim=-1)
        content_dn = self.anchor_to_content(anchor_feat_dn.detach())
        return content_dn, dn_refs, dn_targets, dn_valid, int(dn_refs.shape[1])

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        geom_act: torch.Tensor,
        feat_map: torch.Tensor,
        stride: float,
        img_h: int,
        img_w: int,
        conf_channel: int,
        e2e_gt_pack: Optional[dict] = None,
    ) -> Dict[str, torch.Tensor]:
        b = int(geom_act.shape[0])
        device = geom_act.device
        dtype = feat_map.dtype
        hf, wf = feat_map.shape[-2:]
        K = self.K

        # ---- Path B: Memory. ----
        mem, mem_xy = self._build_memory(feat_map, stride, img_h, img_w)

        # ---- Path A: Hough anchors (no-grad). ----
        anchors = segments_to_hough_anchors(
            geom_act.detach(),
            stride=stride, img_h=img_h, img_w=img_w,
            conf_channel=conf_channel, hf=int(hf), wf=int(wf),
            num_anchors=K,
            conf_thresh=self.conf_thresh,
            max_segments=self.max_segments,
            dbscan_eps=self.dbscan_eps,
            dbscan_min_samples=self.dbscan_min_samples,
            rho_weight=self.rho_weight,
            theta_weight=self.theta_weight,
            L_init_default=self.L_init_default,
        )

        # ---- Matching block (K queries). ----
        content_match, ref_pts_match_norm = self._matching_init(anchors)  # [B, K, D], [B, K, 5, 2]

        # ---- DN block (N_dn queries, train-time only). ----
        content_dn, ref_pts_dn_norm, dn_targets_norm, dn_valid, n_dn = self._dn_init(
            e2e_gt_pack, img_h, img_w, device, dtype, b
        )

        # Concat along K axis to share a single decoder pass.
        content = torch.cat([content_match, content_dn], dim=1)               # [B, K + N_dn, D]
        ref_pts_norm = torch.cat([ref_pts_match_norm, ref_pts_dn_norm], dim=1)  # [B, K + N_dn, 5, 2]
        n_total = K + n_dn

        # ---- Block-diagonal self-attn mask. ----
        if n_dn > 0:
            # DN groups are interleaved per-instance in our build (N inst × G
            # groups per instance, with all groups for one inst consecutive).
            # For block-diagonal isolation we treat *all* DN slots as **one**
            # block separated from matching (simpler and equally effective for
            # avoiding info leakage between matching ↔ DN ↔ DN-other-inst).
            self_attn_mask = build_dn_self_attn_mask(
                n_matching=K, dn_group_sizes=(n_dn,), device=device
            )
        else:
            self_attn_mask = None

        # ---- Decoder loop. ----
        sw = max(float(img_w - 1), 1.0)
        sh = max(float(img_h - 1), 1.0)
        scale_xy = torch.tensor([sw, sh], device=device, dtype=ref_pts_norm.dtype).view(1, 1, 1, 2)
        radius_px = max(self.pt_radius_norm * float(img_w - 1), 1.0)

        ref_pts_layers_norm: List[torch.Tensor] = []
        ref_pts_initial_norm = ref_pts_norm.clone().detach()
        ref_pts = ref_pts_norm                       # detached on entry to layer 0

        for li, layer in enumerate(self.layers):
            # Per-layer PE on the *current* ref points.
            per_pt_pe = self._per_keypoint_pe(ref_pts)
            # LSDA cross-attn mask in pixel space (recomputed because ref moves).
            ref_pts_px = ref_pts * scale_xy
            cross_mask, zero_row = self._build_lsda_mask(
                ref_pts_px, mem_xy, radius_px=radius_px
            )

            content, delta_xy_raw = layer(
                content=content,
                per_pt_pe=per_pt_pe,
                mem=mem,
                cross_mask=cross_mask,
                zero_row_mask=zero_row,
                self_attn_mask=self_attn_mask,
            )
            # Δ bounded by tanh * max_step_norm.
            delta_xy = torch.tanh(delta_xy_raw) * float(self.max_step_norm)

            # Detach-policy correction (plan §2 Step 6 + §6):
            #   - loss target: (ref.detach() + delta).clamp(0,1) -- delta keeps grad
            #   - next layer:  loss_target.detach() -- no history grad through ref
            ref_pts_for_loss = (ref_pts.detach() + delta_xy).clamp(0.0, 1.0)
            ref_pts_layers_norm.append(ref_pts_for_loss)
            ref_pts = ref_pts_for_loss.detach()

        content = self.norm_out(content)
        polylines_norm_final = ref_pts_layers_norm[-1]                          # [B, K_total, 5, 2]

        # ---- Split back into matching / DN. ----
        polylines_match_norm = polylines_norm_final[:, :K]
        polylines_dn_norm = polylines_norm_final[:, K:]
        layers_match_norm = torch.stack([t[:, :K] for t in ref_pts_layers_norm], dim=0)
        layers_dn_norm = torch.stack([t[:, K:] for t in ref_pts_layers_norm], dim=0)

        # ---- Output head: pixels + objectness. ----
        polylines_match_px = polylines_match_norm * scale_xy
        layers_match_px = layers_match_norm * scale_xy.unsqueeze(0)
        polylines_dn_px = polylines_dn_norm * scale_xy
        layers_dn_px = layers_dn_norm * scale_xy.unsqueeze(0)
        dn_targets_px = dn_targets_norm * scale_xy
        ref_init_match_px = ref_pts_initial_norm[:, :K] * scale_xy

        obj_logits = self.objectness_head(content[:, :K]).squeeze(-1)          # [B, K]

        return {
            # DETR-compat keys (for trainer DDP stub + criterion fallback).
            "polylines_px": polylines_match_px,
            "bezier_curve_px": polylines_match_px,
            "objectness_logits": obj_logits,
            # Hough head extras.
            "ref_points_px_init": ref_init_match_px,
            "ref_pts_layers": layers_match_px,
            "hough_anchors": {
                "cx": anchors["cx"],
                "cy": anchors["cy"],
                "theta": anchors["theta"],
                "L_init": anchors["L_init"],
                "valid": anchors["valid"],
            },
            "hough_segments": {
                "mid_xy": anchors["mid_xy_seg"],
                "end_a": anchors["end_a_seg"],
                "end_b": anchors["end_b_seg"],
                "conf": anchors["conf_seg"],
            },
            # DN branch outputs (for direct L1 in the criterion).
            "dn_polylines_px": polylines_dn_px,
            "dn_polylines_layers": layers_dn_px,
            "dn_targets_px": dn_targets_px,
            "dn_valid": dn_valid,
            # Node feat keep-alive for downstream DDP / aux probes.
            "node_feat": content,
        }
