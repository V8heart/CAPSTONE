# SPDX-License-Identifier: GPL-3.0-or-later
"""
YOLinO Center-DETR polyline head (``--e2e_mode=center``).

Architecture (MapTR-flavoured, dependency-free):

1. *Center heatmap* — a small conv head on the FPN feature map produces a
   1-channel center logit map; peaks (top-K after max-pool NMS) become K
   polyline-instance queries.
2. *Point-query expansion* — each peak spawns N learnable "point queries" with
   asymmetric initial offsets (``nn.Embedding(N, 2)``) so the N points are
   spatially distinct from step 0 (anti-collapse).
3. *Localized cross-attention* — memory = YOLinO segment tokens (visual +
   positional + direction fuse, identical to the GNN head's
   ``_build_node_tokens``). Per-center radius mask restricts each query to
   tokens near its peak (per_center mode; see plan §3.5 for trade-off).
4. *Iterative refinement* — each decoder layer outputs a raw Δoffset from a small
   MLP; it is **bounded** (default: ``tanh(raw) * center_delta_max_px``) before
   adding to ``ref_xy`` to avoid early-training blow-ups / out-of-image jumps.
   Alternative: ``(2*sigmoid(raw)-1) * max_px``. The last layer's ``ref_xy`` is
   the final polyline; all layers are kept for deep supervision.

This module is fully differentiable and DDP-safe (a ``* 0`` aux term keeps
every parameter in the autograd graph even when no GT is available).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolino.model.e2e_polyline_modules import (
    mid_dir_geom_to_midpoints_pixels,
    pixels_to_grid_sample_grid,
)
from yolino.utils.enums import LINE
from yolino.utils.logger import Log


def _sine_pe_4d(x4: torch.Tensor, freq_bands: torch.Tensor) -> torch.Tensor:
    """Sine PE for 4-D inputs (``[..., 4] -> [..., 4*2*L]``). Mirrors GNN head."""
    f = x4.unsqueeze(-1) * freq_bands
    return torch.cat([torch.sin(f), torch.cos(f)], dim=-1).flatten(-2)


def _sine_pe_2d(x2: torch.Tensor, freq_bands: torch.Tensor) -> torch.Tensor:
    """Sine PE for 2-D inputs (``[..., 2] -> [..., 2*2*L]``)."""
    f = x2.unsqueeze(-1) * freq_bands
    return torch.cat([torch.sin(f), torch.cos(f)], dim=-1).flatten(-2)


class _LocalizedDecoderLayer(nn.Module):
    """One decoder layer: self-attn over K*N point queries + localized cross-attn.

    The cross-attention mask is **per-center** (see plan §3.5): we compute pixel
    distance between each peak and every memory token once per forward, mask
    out tokens further than ``radius_px`` (or memory-invalid), then expand to
    the K*N point-query axis with ``.expand`` (no extra allocation).
    """

    def __init__(self, dim: int, heads: int, ff: int, dropout: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("center_token_dim (%d) must be divisible by heads (%d)" % (dim, heads))
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

        # Per-layer Δref MLP: small head producing (Δx, Δy) per point query.
        self.delta_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
        )

    def forward(
        self,
        q: torch.Tensor,          # [B, K*N, D]
        mem: torch.Tensor,        # [B, N_all, D]
        cross_mask: Optional[torch.Tensor],  # [B*heads, K*N, N_all] float (-inf or 0); or None
        zero_row_mask: Optional[torch.Tensor],  # [B, K*N] bool: True if query has 0 keys (force zero msg)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (q_out, delta_xy [B, K*N, 2])."""
        x = q
        # Self-attn over all K*N point queries (intra + inter-instance).
        xn = self.norm_q1(x)
        sa, _ = self.self_attn(xn, xn, xn, need_weights=False)
        x = x + self.drop1(sa)

        # Localized cross-attn.
        xn = self.norm_q2(x)
        ca, _ = self.cross_attn(xn, mem, mem, attn_mask=cross_mask, need_weights=False)
        if zero_row_mask is not None:
            # For queries with zero valid keys, MultiheadAttention's softmax of all -inf
            # produces NaN; we already preempt that by leaving one key open (see caller).
            # Belt-and-braces: zero out the message for those rows entirely.
            ca = ca.masked_fill(zero_row_mask.unsqueeze(-1), 0.0)
        x = x + self.drop2(ca)

        # FFN.
        xn = self.norm_q3(x)
        x = x + self.drop3(self.ffn(xn))

        # Per-point Δxy in pixels.
        delta_xy = self.delta_mlp(x)
        return x, delta_xy


class YolinoCenterPolyHead(nn.Module):
    """Center heatmap + N-point polyline decoder over YOLinO segment tokens."""

    def __init__(
        self,
        line_rep: LINE,
        fpn_channels: int,
        token_dim: int = 256,
        num_queries: int = 20,
        num_points: int = 10,
        decoder_layers: int = 4,
        decoder_heads: int = 8,
        decoder_ff: int = 1024,
        dropout: float = 0.1,
        local_radius_px: float = 64.0,
        mask_mode: str = "per_center",
        init_spread_px: float = 32.0,
        nms_kernel: int = 3,
        peak_thresh: float = 0.05,
        peak_nms_dist_px: float = 12.0,
        sine_freq_bands: int = 16,
        delta_bound: str = "tanh",
        delta_max_px: float = 64.0,
    ):
        super().__init__()
        if line_rep != LINE.MID_DIR:
            Log.warning(
                "YolinoCenterPolyHead is wired for MID_DIR geometry; got %s. "
                "Channels (mid_v, mid_h, d_v, d_h) are assumed in the first four slots." % line_rep
            )
        self.line_rep = line_rep
        self.token_dim = int(token_dim)
        self.K = int(num_queries)
        self.N = int(num_points)
        self.decoder_layers = int(decoder_layers)
        self.decoder_heads = int(decoder_heads)
        self.dropout = float(dropout)
        self.local_radius_px = float(local_radius_px)
        self.mask_mode = str(mask_mode)
        if self.mask_mode not in ("per_center", "per_point"):
            raise ValueError("center_mask_mode must be 'per_center' or 'per_point', got %r" % self.mask_mode)
        if self.mask_mode == "per_point":
            Log.warning(
                "center_mask_mode=per_point requested but not implemented yet; falling back to per_center."
            )
            self.mask_mode = "per_center"
        self.init_spread_px = float(init_spread_px)
        self.nms_kernel = max(1, int(nms_kernel) | 1)  # force odd
        self.peak_thresh = float(peak_thresh)
        self.peak_nms_dist_px = float(peak_nms_dist_px)
        self.sine_freq_bands = int(sine_freq_bands)
        self.delta_bound = str(delta_bound).lower()
        if self.delta_bound not in ("tanh", "sigmoid_signed", "none"):
            raise ValueError("delta_bound must be 'tanh', 'sigmoid_signed', or 'none', got %r" % delta_bound)
        self.delta_max_px = float(delta_max_px)
        if self.delta_bound != "none" and self.delta_max_px <= 0.0:
            raise ValueError("delta_max_px must be > 0 when delta_bound is not 'none'")

        # --- Center heatmap head ---
        self.center_head = nn.Sequential(
            nn.Conv2d(int(fpn_channels), int(fpn_channels), kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32 if int(fpn_channels) % 32 == 0 else 16, int(fpn_channels)),
            nn.GELU(),
            nn.Conv2d(int(fpn_channels), 1, kernel_size=1, bias=True),
        )
        # CornerNet-style bias init: focal loss starts hot otherwise.
        nn.init.constant_(self.center_head[-1].bias, -2.19)

        # --- Memory token pipeline (mirrors GNN head._build_node_tokens) ---
        self.visual_proj = nn.Linear(int(fpn_channels), self.token_dim)
        pe_in_4 = 4 * 2 * self.sine_freq_bands
        self.pe_proj_mem = nn.Linear(pe_in_4, self.token_dim)
        self.fuse_mem = nn.Linear(2 * self.token_dim, self.token_dim)

        # --- Query/point-query pipeline ---
        pe_in_2 = 2 * 2 * self.sine_freq_bands
        self.peak_pe_proj = nn.Linear(pe_in_2, self.token_dim)         # K peak xy -> D
        self.point_pe_proj = nn.Linear(pe_in_2, self.token_dim)         # per-point ref xy -> D
        self.slot_embed = nn.Embedding(self.N, self.token_dim)          # N slot embedding
        # Initial slot offsets: linspace(-1,1)*init_spread on x, 0 on y (asymmetry break).
        self.slot_init_offset = nn.Embedding(self.N, 2)
        with torch.no_grad():
            if self.N == 1:
                init_lin = torch.zeros(1)
            else:
                init_lin = torch.linspace(-1.0, 1.0, self.N)
            init = torch.zeros(self.N, 2)
            init[:, 0] = init_lin * self.init_spread_px
            self.slot_init_offset.weight.copy_(init)

        # --- Decoder stack ---
        self.layers = nn.ModuleList(
            [_LocalizedDecoderLayer(self.token_dim, self.decoder_heads, int(decoder_ff), self.dropout)
             for _ in range(self.decoder_layers)]
        )
        self.norm_out = nn.LayerNorm(self.token_dim)

        bands = (2.0 ** torch.arange(self.sine_freq_bands).float()) * math.pi
        self.register_buffer("_freq_bands", bands, persistent=False)

    # ------------------------------------------------------------------ #
    # Memory builder                                                      #
    # ------------------------------------------------------------------ #
    def _build_memory(
        self,
        geom_act: torch.Tensor,
        feat_map: torch.Tensor,
        stride: float,
        img_h: int,
        img_w: int,
        conf_channel: int,
    ):
        b, ncell, p, v = geom_act.shape
        hf, wf = feat_map.shape[-2:]
        if ncell != hf * wf:
            raise ValueError(
                "YolinoCenterPolyHead expects cells == Hf*Wf; got cells=%d, feat H*W=%d"
                % (ncell, hf * wf)
            )

        g = geom_act.view(b, hf, wf, p, v)
        # mid_dir_geom_to_midpoints_pixels returns (x_px, y_px) = (H, V) order
        # (see plan + comment in e2e_polyline_modules.py).
        mid_px, end_a, end_b = mid_dir_geom_to_midpoints_pixels(
            g, stride=stride, img_h=img_h, img_w=img_w
        )
        dir_vec = end_b - end_a

        n_tokens = hf * wf * p
        mid_flat = mid_px.reshape(b, n_tokens, 2)
        dir_flat = dir_vec.reshape(b, n_tokens, 2)
        conf_flat = g[..., conf_channel].reshape(b, n_tokens)

        grid = pixels_to_grid_sample_grid(mid_flat, img_h, img_w)
        sampled = F.grid_sample(
            feat_map, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        sampled = sampled.view(b, feat_map.shape[1], n_tokens).transpose(1, 2)
        visual_feat = self.visual_proj(sampled)

        scale_w = max(float(img_w - 1), 1.0)
        scale_h = max(float(img_h - 1), 1.0)
        mx = 2.0 * mid_flat[..., 0:1] / scale_w - 1.0
        my = 2.0 * mid_flat[..., 1:2] / scale_h - 1.0
        dnorm = dir_flat / dir_flat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        pe_in = torch.cat([mx, my, dnorm], dim=-1)
        pe_feat = self.pe_proj_mem(_sine_pe_4d(pe_in, self._freq_bands))

        mem = self.fuse_mem(torch.cat([visual_feat, pe_feat], dim=-1))  # [B, N_all, D]
        return mem, mid_flat, conf_flat, scale_w, scale_h

    # ------------------------------------------------------------------ #
    # Peak extraction                                                     #
    # ------------------------------------------------------------------ #
    def _extract_peaks(
        self,
        center_logits: torch.Tensor,  # [B, 1, Hf, Wf]
        stride: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pick up to ``K`` peaks: heatmap local-max mask, score sort, then optional **pixel-space NMS**.

        When ``peak_nms_dist_px > 0``, we greedily scan cells in descending masked score and keep a peak only
        if its pixel center is at least that far from all already-kept peaks (duplicate suppression + MapTR-style
        ``peak ≈ instance`` approximation). Pads with zeros / invalid for unused slots.
        """
        b, _, hf, wf = center_logits.shape
        stride_f = float(stride)
        device = center_logits.device
        dtype = center_logits.dtype
        prob = torch.sigmoid(center_logits)  # [B,1,Hf,Wf]
        pad = self.nms_kernel // 2
        local_max = F.max_pool2d(prob, kernel_size=self.nms_kernel, stride=1, padding=pad)
        peak_mask = (prob == local_max).float()

        flat_idx = torch.arange(hf * wf, device=device, dtype=torch.long)
        row = (flat_idx // wf).to(dtype)
        col = (flat_idx % wf).to(dtype)
        x_px_all = (col + 0.5) * stride_f
        y_px_all = (row + 0.5) * stride_f
        flat_xy = torch.stack([x_px_all, y_px_all], dim=-1)  # [HW, 2]

        all_xy = []
        all_sc = []
        all_ok = []
        for bi in range(b):
            sc = (prob[bi, 0] * peak_mask[bi, 0]).flatten()
            idx_desc = torch.argsort(sc, descending=True)
            picked_xy: list[torch.Tensor] = []
            picked_sc: list[float] = []
            nms_d = float(self.peak_nms_dist_px)
            for t in range(int(idx_desc.numel())):
                idx = int(idx_desc[t].item())
                scc = float(sc[idx].item())
                if scc <= self.peak_thresh:
                    break
                xy = flat_xy[idx]
                if nms_d <= 0.0:
                    ok = True
                else:
                    ok = True
                    for py in picked_xy:
                        if torch.norm(xy - py, p=2).item() < nms_d:
                            ok = False
                            break
                if ok:
                    picked_xy.append(xy.clone())
                    picked_sc.append(scc)
                    if len(picked_xy) >= self.K:
                        break
            while len(picked_xy) < self.K:
                picked_xy.append(torch.zeros(2, device=device, dtype=dtype))
                picked_sc.append(0.0)
            all_xy.append(torch.stack(picked_xy[: self.K], dim=0))
            all_sc.append(
                torch.tensor(picked_sc[: self.K], device=device, dtype=dtype)
            )
            all_ok.append(
                torch.tensor([s > self.peak_thresh for s in picked_sc[: self.K]], device=device, dtype=torch.bool)
            )
        peaks_xy = torch.stack(all_xy, dim=0)
        peak_score = torch.stack(all_sc, dim=0)
        peak_valid = torch.stack(all_ok, dim=0)
        return peaks_xy, peak_score, peak_valid

    # ------------------------------------------------------------------ #
    # Cross-attn mask                                                     #
    # ------------------------------------------------------------------ #
    def _build_cross_mask(
        self,
        peaks_xy: torch.Tensor,    # [B, K, 2]
        mem_xy: torch.Tensor,      # [B, N_all, 2]
        mem_valid: torch.Tensor,   # [B, N_all] bool
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Per-center radius mask.

        Returns:
          cross_mask: ``[B*heads, K*N, N_all]`` float (-inf where masked, 0 elsewhere),
                      or None when masking is disabled.
          zero_row:   ``[B, K*N]`` bool, True for queries that ended up with zero
                      eligible keys (we keep one fallback key open to avoid NaN
                      softmax; the message is zeroed downstream).
        """
        b, k, _ = peaks_xy.shape
        n_all = mem_xy.shape[1]
        if self.local_radius_px <= 0.0:
            return None, torch.zeros((b, k * self.N), dtype=torch.bool, device=peaks_xy.device)

        # cdist per-center: [B, K, N_all]
        d = torch.cdist(peaks_xy.float(), mem_xy.float(), p=2)
        within = d <= float(self.local_radius_px)
        within = within & mem_valid.unsqueeze(1)  # [B, K, N_all]

        # Avoid zero-row softmax NaN: for any (b, k) with no valid keys, force the
        # nearest single token to be eligible. The downstream layer will zero the
        # message for these rows via zero_row_mask.
        any_valid = within.any(dim=-1)  # [B, K]
        if (~any_valid).any():
            nearest_idx = d.argmin(dim=-1, keepdim=True)  # [B, K, 1]
            scatter_helper = torch.zeros_like(within)
            scatter_helper.scatter_(-1, nearest_idx, True)
            within = torch.where(any_valid.unsqueeze(-1), within, scatter_helper)

        # Expand to [B, K, N, N_all] without allocating: we reshape later.
        mask_kxn = within.unsqueeze(2).expand(-1, -1, self.N, -1).reshape(b, k * self.N, n_all)

        # Float mask for MultiheadAttention: 0 for keep, -inf for mask.
        float_mask = torch.zeros_like(mask_kxn, dtype=torch.float32)
        float_mask = float_mask.masked_fill(~mask_kxn, float("-inf"))

        # Replicate per-head.
        cross_mask = float_mask.unsqueeze(1).expand(-1, self.decoder_heads, -1, -1).reshape(
            b * self.decoder_heads, k * self.N, n_all
        )

        # zero-row mask is per-batch-per-query (B,K*N) for the queries we patched.
        zero_row = (~any_valid).unsqueeze(-1).expand(-1, -1, self.N).reshape(b, k * self.N)
        return cross_mask, zero_row

    def _bound_delta_xy(self, raw: torch.Tensor) -> torch.Tensor:
        """Map raw MLP outputs to bounded pixel Δ (per x and y)."""
        if self.delta_bound == "none":
            return raw
        m = self.delta_max_px
        if self.delta_bound == "tanh":
            return torch.tanh(raw) * m
        # sigmoid_signed: (0,1) → (-1,1), then scale; same max magnitude as tanh path.
        return (torch.sigmoid(raw) * 2.0 - 1.0) * m

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        geom_act: torch.Tensor,
        feat_map: torch.Tensor,
        stride: float,
        img_h: int,
        img_w: int,
        conf_channel: int,
    ) -> Dict[str, torch.Tensor]:
        b = geom_act.shape[0]
        K, N, D = self.K, self.N, self.token_dim
        device = geom_act.device

        # --- 1. memory (segment tokens) ---
        mem, mem_xy, mem_conf, scale_w, scale_h = self._build_memory(
            geom_act, feat_map, stride, img_h, img_w, conf_channel
        )
        # mem_valid: tokens with conf > 0 (raw, before topk); using a soft threshold
        # avoids "all -inf" rows during early training when most conf is low.
        mem_valid = mem_conf > 1e-4

        # --- 2. center heatmap + peaks ---
        center_logits = self.center_head(feat_map)  # [B,1,Hf,Wf]
        peaks_xy, peak_score, peak_valid = self._extract_peaks(center_logits, stride=stride)

        # --- 3. point queries ---
        sw = max(scale_w, 1.0)
        sh = max(scale_h, 1.0)

        # initial ref_xy = peak + slot_init_offset (broadcast)
        slot_offsets = self.slot_init_offset.weight.unsqueeze(0).unsqueeze(0)  # [1,1,N,2]
        ref_xy = peaks_xy.unsqueeze(2) + slot_offsets  # [B,K,N,2]

        def _make_q_tokens(ref_xy_in: torch.Tensor) -> torch.Tensor:
            rx = 2.0 * ref_xy_in[..., 0:1] / sw - 1.0
            ry = 2.0 * ref_xy_in[..., 1:2] / sh - 1.0
            xy_norm = torch.cat([rx, ry], dim=-1)  # [B,K,N,2]
            point_pe = self.point_pe_proj(_sine_pe_2d(xy_norm, self._freq_bands))  # [B,K,N,D]
            slot = self.slot_embed.weight.unsqueeze(0).unsqueeze(0).expand(b, K, -1, -1)  # [B,K,N,D]
            # also inject peak PE on every point query (instance-level cue)
            px = 2.0 * peaks_xy[..., 0:1] / sw - 1.0
            py = 2.0 * peaks_xy[..., 1:2] / sh - 1.0
            peak_xy_norm = torch.cat([px, py], dim=-1)  # [B,K,2]
            peak_pe = self.peak_pe_proj(_sine_pe_2d(peak_xy_norm, self._freq_bands))  # [B,K,D]
            peak_pe = peak_pe.unsqueeze(2).expand(-1, -1, N, -1)  # [B,K,N,D]
            return point_pe + slot + peak_pe

        q = _make_q_tokens(ref_xy).reshape(b, K * N, D)

        # --- 4. localized cross-attn mask (per-center, recomputed only once
        # since peaks don't change across layers in this implementation) ---
        cross_mask, zero_row = self._build_cross_mask(peaks_xy, mem_xy, mem_valid)

        # --- 5. decoder stack with iterative Δref refinement ---
        polylines_aux = []
        for layer in self.layers:
            q, delta_xy_flat = layer(q, mem, cross_mask, zero_row)
            delta_xy_flat = self._bound_delta_xy(delta_xy_flat)
            delta_xy = delta_xy_flat.reshape(b, K, N, 2)
            ref_xy = ref_xy + delta_xy
            polylines_aux.append(ref_xy)
            # Refresh point PE with the updated ref_xy (the mask stays peak-anchored).
            q = _make_q_tokens(ref_xy).reshape(b, K * N, D) + q

        q = self.norm_out(q)
        polylines_px = ref_xy  # [B, K, N, 2]
        # [L, B, K, N, 2] aux output for deep supervision.
        polylines_aux_t = torch.stack(polylines_aux, dim=0)

        return {
            "center_logits": center_logits,        # [B,1,Hf,Wf]
            "center_peaks_xy": peaks_xy,           # [B,K,2]
            "peak_score": peak_score,              # [B,K]
            "peak_valid": peak_valid,              # [B,K] bool
            "polylines_px": polylines_px,          # [B,K,N,2]
            "polylines_aux": polylines_aux_t,      # [L,B,K,N,2]
            "node_feat": q,                        # [B,K*N,D]; aliased for DDP-aux symmetry
            "mem_valid": mem_valid,                # [B,N_all] bool
            "mem_xy": mem_xy,                      # [B,N_all,2]; for viz
            "mem_conf": mem_conf,                  # [B,N_all]; for viz
        }
