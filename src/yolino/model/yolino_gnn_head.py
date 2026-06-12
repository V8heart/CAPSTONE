# SPDX-License-Identifier: GPL-3.0-or-later
"""
YOLinO-GNN segment graph head.

Graph over **segment** nodes (top-conf ``gnn_max_nodes`` midpoints):

1. *Node initialisation* — ``geom_act`` → mid/end/dir + FPN ``grid_sample`` → fused tokens.
2. *Geometric edge prior* — ``gnn_adjacency_mode``:
   - ``global``: **all N×N** directed candidates (minus self), then hard collinearity masks.
   - ``directional2``: per node, **two** candidates along segment direction (nearest forward /
     nearest backward on the same line; ``K=2``).
   - ``directional2_ctx``: ``cat(on-line, context)`` gated **K≈20** for GAT + edge MLP + ``cross_ignore``
     loss; **on-line K≈6** only for CC / viz / degree / RW (``line_neighbors`` in output).
   - ``directional2_global``: **GAT on global N×N** (hard-gated); edge MLP + ``cross_ignore`` on
     **Euclidean top-``gnn_knn_k``** among pairs with ``|dir_i·dir_j| >= gnn_knn_min_dir_dot``
     (no lateral/along/end gate); CC / viz / degree / RW on same slots.
   - ``knn``: legacy Euclidean top-``gnn_knn_k`` neighbours only.
3. *Message passing* — stacked GAT layers (softmax over the K neighbour axis; for global,
   ``K = N``).
4. *Edge head* — ``[h_i, h_j, edge_geom]`` → logit per candidate edge. Geometry uses
   ``|dir_dot|``, ``|s|``, ``d_⊥``, plus **min endpoint distance** over the four
   endpoint pairings ``(A_i,B_i)×(A_j,B_j)`` (normalized), so the head sees true
   stitch geometry, not midpoints alone.

The trainer dispatches on ``"edge_logits"``.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from yolino.model.e2e_polyline_modules import (
    mid_dir_geom_to_midpoints_pixels,
    pixels_to_grid_sample_grid,
)
from yolino.model.segment_geom_soft_gate import compute_edge_prior
from yolino.model.segment_soft_nms import segment_soft_nms_batch
from yolino.postprocessing.segment_merge import merge_segments
from yolino.utils.enums import LINE
from yolino.utils.logger import Log


_BIG_DIST = 1.0e9


def _sine_pe_4d(x4: torch.Tensor, freq_bands: torch.Tensor) -> torch.Tensor:
    """Sine PE for 4-D inputs (typically (mid_x_norm, mid_y_norm, dx_unit, dy_unit))."""
    f = x4.unsqueeze(-1) * freq_bands  # [..., 4, L]
    return torch.cat([torch.sin(f), torch.cos(f)], dim=-1).flatten(-2)


class _GATLayer(nn.Module):
    """Multi-head GAT over a fixed [B, N, K] neighbour layout (global: K=N)."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("token_dim (%d) must be divisible by gnn_heads (%d)" % (dim, heads))
        self.h = int(heads)
        self.d = dim // self.h
        self.norm1 = nn.LayerNorm(dim)
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        neighbors: torch.Tensor,
        neigh_valid: torch.Tensor,
        geom_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, d = x.shape
        k = neighbors.shape[-1]
        h, dh = self.h, self.d
        hd = h * dh

        xn = self.norm1(x)
        q = self.proj_q(xn).view(b, n, h, dh)
        k_full = self.proj_k(xn).view(b, n, hd)
        v_full = self.proj_v(xn).view(b, n, hd)

        idx_e = neighbors.reshape(b, n * k, 1).expand(b, n * k, hd)
        k_neigh = torch.gather(k_full, 1, idx_e).view(b, n, k, h, dh)
        v_neigh = torch.gather(v_full, 1, idx_e).view(b, n, k, h, dh)

        att = (q.unsqueeze(2) * k_neigh).sum(dim=-1) / math.sqrt(dh)  # [B, N, K, H]
        if geom_bias is not None:
            att = att + geom_bias.unsqueeze(-1)
        att = att.masked_fill(~neigh_valid.unsqueeze(-1), float("-inf"))

        all_masked = ~neigh_valid.any(dim=-1, keepdim=True)  # [B, N, 1]
        att_safe = torch.where(
            all_masked.unsqueeze(-1).expand_as(att),
            torch.zeros_like(att),
            att,
        )
        w = F.softmax(att_safe, dim=2)
        w = w.masked_fill(all_masked.unsqueeze(-1).expand_as(w), 0.0)
        w = self.attn_drop(w)

        msg = (w.unsqueeze(-1) * v_neigh).sum(dim=2).reshape(b, n, hd)
        x = x + self.resid_drop(self.proj_out(msg))
        x = x + self.ffn(self.norm2(x))
        return x


class YolinoGnnSegmentGraphHead(nn.Module):
    """Segment-graph E2E head producing per-edge connectivity logits."""

    def __init__(
        self,
        line_rep: LINE,
        fpn_channels: int,
        token_dim: int = 256,
        max_nodes: int = 256,
        node_conf_thresh: float = 0.1,
        knn_k: int = 16,
        knn_min_dir_dot: float = 0.0,
        edge_radius_px: float = 0.0,
        adjacency_mode: str = "global",
        max_lateral_px: float = 48.0,
        max_lateral_sym: bool = False,
        lateral_on_overlap_only: bool = False,
        lateral_overlap_window_px: float = 24.0,
        max_along_px: float = 0.0,
        max_end_gap_px: float = 0.0,
        min_dir_dot: float = 0.0,
        directional_min_sep_px: float = 8.0,
        directional_k: int = 2,
        directional_include_all: bool = False,
        context_k: int = 4,
        context_lat_min_px: float = 12.0,
        context_lat_max_px: float = 40.0,
        context_max_along_px: float = 200.0,
        context_min_dir_dot: float = 0.85,
        gat_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        sine_freq_bands: int = 16,
        soft_nms_enabled: bool = False,
        soft_nms_mid_sigma_px: float = 16.0,
        soft_nms_min_dir_dot: float = 0.96,
        soft_nms_decay_method: str = "linear",
        soft_nms_score_floor: float = 0.001,
        soft_nms_prefilter_conf: float = 0.05,
        soft_nms_max_segments: int = 1024,
        segment_merge_enabled: bool = False,
        segment_merge_lat_px: float = 6.0,
        segment_merge_dir_dot_min: float = 0.98,
        segment_merge_end_gap_px: float = 8.0,
        segment_merge_iters: int = 3,
        segment_merge_prefilter_conf: float = 0.05,
        use_hard_geom_gate: bool = True,
        soft_geom_gate_enabled: bool = False,
        soft_geom_sigma_lat_px: float = 32.0,
        soft_geom_dir_floor: float = 0.5,
        soft_geom_prior_eps: float = 1e-6,
        node_use_visual_feat: bool = True,
        edge_feat_signed: bool = False,
        gat_geom_bias: bool = False,
        gat_geom_bias_w_along: float = 1.0,
        gat_geom_bias_w_lat: float = 0.5,
        gat_geom_bias_tau_along: float = 40.0,
        gat_geom_bias_tau_lat: float = 20.0,
    ):
        super().__init__()
        if line_rep != LINE.MID_DIR:
            Log.warning(
                "YolinoGnnSegmentGraphHead is wired for MID_DIR geometry; got %s. "
                "Channels (u,v,dx,dy) are assumed in the first four geom slots." % line_rep
            )
        self.line_rep = line_rep
        self.token_dim = int(token_dim)
        self.max_nodes = int(max_nodes)
        self.node_conf_thresh = float(node_conf_thresh)
        self.knn_k = int(knn_k)
        self.knn_min_dir_dot = float(knn_min_dir_dot)
        self.edge_radius_px = float(edge_radius_px)
        self.adjacency_mode = str(adjacency_mode).lower().strip()
        if self.adjacency_mode not in ("global", "knn", "directional2", "directional2_ctx", "directional2_global"):
            raise ValueError(
                "adjacency_mode must be 'global', 'knn', 'directional2', 'directional2_ctx', "
                "or 'directional2_global', got %r" % adjacency_mode
            )
        self.directional_min_sep_px = float(directional_min_sep_px)
        self.directional_k = max(1, int(directional_k))
        self.directional_include_all = bool(directional_include_all)
        self.context_k = max(0, int(context_k))
        self.context_lat_min_px = float(context_lat_min_px)
        self.context_lat_max_px = float(context_lat_max_px)
        self.context_max_along_px = float(context_max_along_px)
        self.context_min_dir_dot = float(context_min_dir_dot)
        self.max_lateral_px = float(max_lateral_px)
        self.max_lateral_sym = bool(max_lateral_sym)
        self.lateral_on_overlap_only = bool(lateral_on_overlap_only)
        self.lateral_overlap_window_px = float(lateral_overlap_window_px)
        self.max_along_px = float(max_along_px)
        self.max_end_gap_px = float(max_end_gap_px)
        self.min_dir_dot = float(min_dir_dot)
        if self.adjacency_mode == "global" and self.max_lateral_px <= 0.0:
            Log.warning(
                "GNN adjacency_mode=global with gnn_max_lateral_px<=0 keeps all off-diagonal N×N edges; "
                "set a positive lateral threshold (recommended) to drop parallel-offset wires."
            )
        self.sine_freq_bands = int(sine_freq_bands)
        self.soft_nms_enabled = bool(soft_nms_enabled)
        self.soft_nms_mid_sigma_px = float(soft_nms_mid_sigma_px)
        self.soft_nms_min_dir_dot = float(soft_nms_min_dir_dot)
        self.soft_nms_decay_method = str(soft_nms_decay_method).lower().strip()
        self.soft_nms_score_floor = float(soft_nms_score_floor)
        self.soft_nms_prefilter_conf = float(soft_nms_prefilter_conf)
        self.soft_nms_max_segments = int(soft_nms_max_segments)
        self.segment_merge_enabled = bool(segment_merge_enabled)
        self.segment_merge_lat_px = float(segment_merge_lat_px)
        self.segment_merge_dir_dot_min = float(segment_merge_dir_dot_min)
        self.segment_merge_end_gap_px = float(segment_merge_end_gap_px)
        self.segment_merge_iters = int(segment_merge_iters)
        self.segment_merge_prefilter_conf = float(segment_merge_prefilter_conf)
        self.use_hard_geom_gate = bool(use_hard_geom_gate)
        self.soft_geom_gate_enabled = bool(soft_geom_gate_enabled)
        self.soft_geom_sigma_lat_px = float(soft_geom_sigma_lat_px)
        self.soft_geom_dir_floor = float(soft_geom_dir_floor)
        self.soft_geom_prior_eps = float(soft_geom_prior_eps)
        self.node_use_visual_feat = bool(node_use_visual_feat)
        self.edge_feat_signed = bool(edge_feat_signed)
        self.gat_geom_bias_enabled = bool(gat_geom_bias)
        self.gat_geom_bias_w_along = float(gat_geom_bias_w_along)
        self.gat_geom_bias_w_lat = float(gat_geom_bias_w_lat)
        self.gat_geom_bias_tau_along = float(gat_geom_bias_tau_along)
        self.gat_geom_bias_tau_lat = float(gat_geom_bias_tau_lat)

        # Node feature pipeline.
        self.visual_proj = nn.Linear(int(fpn_channels), self.token_dim)
        pe_in = 4 * 2 * self.sine_freq_bands
        self.pe_proj = nn.Linear(pe_in, self.token_dim)
        self.fuse = nn.Linear(2 * self.token_dim, self.token_dim)

        bands = (2.0 ** torch.arange(self.sine_freq_bands).float()) * math.pi
        self.register_buffer("_freq_bands", bands, persistent=False)

        self.gat_layers = nn.ModuleList(
            [_GATLayer(self.token_dim, heads=int(heads), dropout=float(dropout))
             for _ in range(int(gat_layers))]
        )

        # legacy 9-D: [dist, dx, dy, |dir_dot|, |along|, lateral, end_gap, conf_i, conf_j]
        # signed 11-D: [dist, dx, dy, dir_dot, along, lateral, end_gap, len_i, len_j, conf_i, conf_j]
        self._edge_feat_dim = 11 if self.edge_feat_signed else 9
        self.edge_head = nn.Sequential(
            nn.Linear(2 * self.token_dim + self._edge_feat_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.token_dim, self.token_dim // 2),
            nn.GELU(),
            nn.Linear(self.token_dim // 2, 1),
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _build_node_tokens(
        self,
        geom_act: torch.Tensor,
        feat_map: torch.Tensor,
        stride: float,
        img_h: int,
        img_w: int,
        conf_channel: int,
    ):
        """Compute full segment-token tensor + raw geometry / conf for all (cell, predictor) slots."""
        b, ncell, p, v = geom_act.shape
        hf, wf = feat_map.shape[-2:]
        if ncell != hf * wf:
            raise ValueError(
                "YolinoGnnSegmentGraphHead expects cells == Hf*Wf; got cells=%d, feat H*W=%d"
                % (ncell, hf * wf)
            )

        g = geom_act.view(b, hf, wf, p, v)
        mid_px, end_a, end_b = mid_dir_geom_to_midpoints_pixels(g, stride=stride, img_h=img_h, img_w=img_w)
        dir_vec = end_b - end_a

        n_tokens = hf * wf * p
        mid_flat = mid_px.reshape(b, n_tokens, 2)
        dir_flat = dir_vec.reshape(b, n_tokens, 2)
        conf_flat = g[..., conf_channel].reshape(b, n_tokens)

        grid = pixels_to_grid_sample_grid(mid_flat, img_h, img_w)
        if self.node_use_visual_feat:
            sampled = F.grid_sample(
                feat_map, grid, mode="bilinear", padding_mode="border", align_corners=True
            )
            sampled = sampled.view(b, feat_map.shape[1], n_tokens).transpose(1, 2)
            visual_feat = self.visual_proj(sampled)
        else:
            visual_feat = torch.zeros(
                (b, n_tokens, self.token_dim), device=geom_act.device, dtype=geom_act.dtype
            )

        scale_w = max(float(img_w - 1), 1.0)
        scale_h = max(float(img_h - 1), 1.0)
        mx = 2.0 * mid_flat[..., 0:1] / scale_w - 1.0
        my = 2.0 * mid_flat[..., 1:2] / scale_h - 1.0
        dnorm = dir_flat / dir_flat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        pe_in = torch.cat([mx, my, dnorm], dim=-1)
        pe_feat = self.pe_proj(_sine_pe_4d(pe_in, self._freq_bands))

        tokens = self.fuse(torch.cat([visual_feat, pe_feat], dim=-1))
        end_a_flat = end_a.reshape(b, n_tokens, 2)
        end_b_flat = end_b.reshape(b, n_tokens, 2)
        return tokens, mid_flat, dir_flat, dnorm, conf_flat, end_a_flat, end_b_flat, scale_w, scale_h

    def _select_nodes(
        self,
        tokens: torch.Tensor,
        mid_flat: torch.Tensor,
        dir_flat: torch.Tensor,
        dnorm: torch.Tensor,
        conf_flat: torch.Tensor,
        end_a_flat: torch.Tensor,
        end_b_flat: torch.Tensor,
    ):
        """Threshold-by-confidence first, then top-K among survivors."""
        b, n_all, _ = tokens.shape
        n_keep = min(self.max_nodes, n_all)
        # Enforce "confidence cut first, then top-K": below-threshold slots can
        # still appear only as invalid padding when survivors < n_keep.
        neg_inf = torch.full_like(conf_flat, -1.0e9)
        conf_rank = torch.where(conf_flat > self.node_conf_thresh, conf_flat, neg_inf)
        top_rank_conf, top_idx = torch.topk(conf_rank, k=n_keep, dim=1)
        top_conf = torch.gather(conf_flat, 1, top_idx)
        idx_d = top_idx.unsqueeze(-1).expand(-1, -1, self.token_dim)
        node_feat = torch.gather(tokens, 1, idx_d)
        idx_xy = top_idx.unsqueeze(-1).expand(-1, -1, 2)
        node_mid = torch.gather(mid_flat, 1, idx_xy)
        node_dir = torch.gather(dir_flat, 1, idx_xy)
        node_dnorm = torch.gather(dnorm, 1, idx_xy)
        node_end_a = torch.gather(end_a_flat, 1, idx_xy)
        node_end_b = torch.gather(end_b_flat, 1, idx_xy)
        node_valid = top_rank_conf > (-1.0e8)
        return node_feat, node_mid, node_dir, node_dnorm, node_end_a, node_end_b, top_conf, node_valid, top_idx

    def _build_knn(
        self,
        node_mid: torch.Tensor,
        node_valid: torch.Tensor,
        node_dnorm: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Euclidean kNN. Returns ``neighbors`` ``[B,N,K]``, ``neigh_valid``, ``neigh_dist``.

        When ``node_dnorm`` is given and ``knn_min_dir_dot > 0``, only pairs with
        ``|dir_i·dir_j| >= knn_min_dir_dot`` are eligible (distance gate still optional via
        ``edge_radius_px``).
        """
        b, n, _ = node_mid.shape
        dist = torch.cdist(node_mid.float(), node_mid.float(), p=2)

        dist = dist.masked_fill(
            torch.eye(n, device=dist.device, dtype=torch.bool).unsqueeze(0).expand(b, -1, -1),
            _BIG_DIST,
        )
        dist = dist.masked_fill(~node_valid.unsqueeze(1), _BIG_DIST)
        dist = dist.masked_fill(~node_valid.unsqueeze(2), _BIG_DIST)
        if self.edge_radius_px > 0.0:
            dist = dist.masked_fill(dist > self.edge_radius_px, _BIG_DIST)
        if node_dnorm is not None and self.knn_min_dir_dot > 0.0:
            dn_i = node_dnorm.unsqueeze(2).expand(b, n, n, 2)
            dn_j = node_dnorm.unsqueeze(1).expand(b, n, n, 2)
            abs_dir_dot = (dn_i * dn_j).sum(dim=-1).abs()
            dist = dist.masked_fill(abs_dir_dot < self.knn_min_dir_dot, _BIG_DIST)

        k_eff = max(1, min(self.knn_k, n - 1))
        neigh_dist, neighbors = torch.topk(-dist, k=k_eff, dim=-1)
        neigh_dist = -neigh_dist
        finite = neigh_dist < (_BIG_DIST * 0.5)
        nv_neigh = torch.gather(node_valid.unsqueeze(1).expand(-1, n, -1), 2, neighbors)
        neigh_valid = finite & nv_neigh & node_valid.unsqueeze(-1)
        neighbors = neighbors.clamp_min(0)
        return neighbors, neigh_valid, neigh_dist

    def _pairwise_collinearity_mask(
        self,
        node_mid: torch.Tensor,
        node_dnorm: torch.Tensor,
        node_ea: torch.Tensor,
        node_eb: torch.Tensor,
        node_valid: torch.Tensor,
    ) -> torch.Tensor:
        """``[B,N,N]`` bool mask for directed i→j: same-line hard gates (no learnable params).

        Gates (when thresholds > 0): lateral j|i, optional lateral i|j, |dir_i·dir_j|,
        along-track |s|, min endpoint gap, optional midpoint radius.
        """
        b, n, _ = node_mid.shape
        device = node_mid.device

        mid_i = node_mid.unsqueeze(2).expand(b, n, n, 2)
        mid_j = node_mid.unsqueeze(1).expand(b, n, n, 2)
        delta_ij = mid_j - mid_i
        delta_ji = mid_i - mid_j

        dn_i = node_dnorm.unsqueeze(2).expand(b, n, n, 2)
        dn_j = node_dnorm.unsqueeze(1).expand(b, n, n, 2)

        d_perp_j_on_i = (delta_ij[..., 0] * dn_i[..., 1] - delta_ij[..., 1] * dn_i[..., 0]).abs()
        d_perp_i_on_j = (delta_ji[..., 0] * dn_j[..., 1] - delta_ji[..., 1] * dn_j[..., 0]).abs()
        abs_dir_dot = (dn_i * dn_j).sum(dim=-1).abs()
        s_abs = (delta_ij * dn_i).sum(dim=-1).abs()
        mid_dist = torch.norm(delta_ij.float(), dim=-1)

        ea_i = node_ea.unsqueeze(2).expand(b, n, n, 2)
        eb_i = node_eb.unsqueeze(2).expand(b, n, n, 2)
        ea_j = node_ea.unsqueeze(1).expand(b, n, n, 2)
        eb_j = node_eb.unsqueeze(1).expand(b, n, n, 2)
        d_end = torch.minimum(
            torch.minimum(torch.norm(ea_i - ea_j, dim=-1), torch.norm(ea_i - eb_j, dim=-1)),
            torch.minimum(torch.norm(eb_i - ea_j, dim=-1), torch.norm(eb_i - eb_j, dim=-1)),
        )

        eye = torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0).expand(b, -1, -1)
        ok = node_valid.unsqueeze(2) & node_valid.unsqueeze(1) & ~eye

        if self.max_lateral_px > 0.0:
            lat_j_on_i_ok = d_perp_j_on_i <= self.max_lateral_px
            if self.lateral_on_overlap_only and self.lateral_overlap_window_px > 0.0:
                overlap_zone = s_abs <= self.lateral_overlap_window_px
                lat_j_on_i_ok = torch.where(overlap_zone, lat_j_on_i_ok, torch.ones_like(lat_j_on_i_ok))
            ok = ok & lat_j_on_i_ok
        if self.max_lateral_sym and self.max_lateral_px > 0.0:
            lat_i_on_j_ok = d_perp_i_on_j <= self.max_lateral_px
            if self.lateral_on_overlap_only and self.lateral_overlap_window_px > 0.0:
                overlap_zone = s_abs <= self.lateral_overlap_window_px
                lat_i_on_j_ok = torch.where(overlap_zone, lat_i_on_j_ok, torch.ones_like(lat_i_on_j_ok))
            ok = ok & lat_i_on_j_ok
        if self.min_dir_dot > 0.0:
            ok = ok & (abs_dir_dot >= self.min_dir_dot)
        if self.max_along_px > 0.0:
            ok = ok & (s_abs <= self.max_along_px)
        if self.max_end_gap_px > 0.0:
            ok = ok & (d_end <= self.max_end_gap_px)
        if self.edge_radius_px > 0.0:
            ok = ok & (mid_dist <= self.edge_radius_px)
        return ok

    def _build_global_adjacency(
        self,
        node_mid: torch.Tensor,
        node_valid: torch.Tensor,
        node_dnorm: torch.Tensor,
        node_ea: torch.Tensor,
        node_eb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full N×N directed graph minus self; hard collinearity / stitch masks."""
        b, n, _ = node_mid.shape
        device, dtype = node_mid.device, node_mid.dtype

        mid_src = node_mid.unsqueeze(2).expand(b, n, n, 2)
        mid_neigh = node_mid.unsqueeze(1).expand(b, n, n, 2)
        delta = mid_neigh - mid_src
        neigh_dist = torch.norm(delta.float(), dim=-1).to(dtype)

        if self.use_hard_geom_gate:
            neigh_valid = self._pairwise_collinearity_mask(
                node_mid, node_dnorm, node_ea, node_eb, node_valid,
            )
        else:
            b, n, _ = node_mid.shape
            device = node_mid.device
            eye = torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0).expand(b, -1, -1)
            neigh_valid = node_valid.unsqueeze(2) & node_valid.unsqueeze(1) & ~eye
        neigh_dist = neigh_dist.masked_fill(~neigh_valid, _BIG_DIST)
        neighbors = torch.arange(n, device=device, dtype=torch.long).view(1, 1, n).expand(b, n, n)
        return neighbors, neigh_valid, neigh_dist

    def _build_directional2_adjacency(
        self,
        node_mid: torch.Tensor,
        node_valid: torch.Tensor,
        node_dnorm: torch.Tensor,
        node_ea: torch.Tensor,
        node_eb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-node directional candidates along ``dn_i`` (forward + backward), up to ``directional_k``.

        Reuses :meth:`_pairwise_collinearity_mask` (exp43-style gates). Signed along-track
        ``s_ij = (mid_j - mid_i) · dn_i``:
          - forward slots: smallest positive ``s`` among gated pairs
          - backward slots: largest negative ``s`` (closest behind) among gated pairs
        """
        b, n, _ = node_mid.shape
        device, dtype = node_mid.device, node_mid.dtype
        eps = float(self.directional_min_sep_px)
        k_total = int(self.directional_k)
        k_fwd = (k_total + 1) // 2
        k_bwd = k_total - k_fwd
        k_fwd_eff = max(1, min(k_fwd, max(n - 1, 1)))
        k_bwd_eff = max(1, min(k_bwd, max(n - 1, 1)))

        pair_ok = self._pairwise_collinearity_mask(
            node_mid, node_dnorm, node_ea, node_eb, node_valid,
        )

        mid_i = node_mid.unsqueeze(2).expand(b, n, n, 2)
        mid_j = node_mid.unsqueeze(1).expand(b, n, n, 2)
        dn_i = node_dnorm.unsqueeze(2).expand(b, n, n, 2)
        delta_ij = mid_j - mid_i
        s_signed = (delta_ij * dn_i).sum(dim=-1)

        pos_ok = pair_ok & (s_signed > eps)
        neg_ok = pair_ok & (s_signed < -eps)

        big = torch.tensor(_BIG_DIST, device=device, dtype=dtype)
        fwd_cost = torch.where(pos_ok, s_signed, big)
        bwd_cost = torch.where(neg_ok, -s_signed, big)

        _, fwd_j = torch.topk(fwd_cost, k=k_fwd_eff, dim=-1, largest=False)
        _, bwd_j = torch.topk(bwd_cost, k=k_bwd_eff, dim=-1, largest=False)

        fwd_valid = torch.gather(pos_ok, 2, fwd_j)
        bwd_valid = torch.gather(neg_ok, 2, bwd_j)

        if k_fwd_eff < k_fwd:
            pad_f = k_fwd - k_fwd_eff
            fwd_j = F.pad(fwd_j, (0, pad_f), value=0)
            fwd_valid = F.pad(fwd_valid, (0, pad_f), value=False)
        if k_bwd_eff < k_bwd:
            pad_b = k_bwd - k_bwd_eff
            bwd_j = F.pad(bwd_j, (0, pad_b), value=0)
            bwd_valid = F.pad(bwd_valid, (0, pad_b), value=False)

        neighbors = torch.cat([fwd_j, bwd_j], dim=-1)
        neigh_valid = torch.cat([fwd_valid, bwd_valid], dim=-1)

        mid_stack = node_mid.unsqueeze(1).expand(b, n, n, 2)
        k_slots = neighbors.shape[-1]
        idx_xy = neighbors.unsqueeze(-1).expand(b, n, k_slots, 2)
        mid_neigh = torch.gather(mid_stack, 2, idx_xy)
        delta = mid_neigh - node_mid.unsqueeze(2)
        neigh_dist = torch.norm(delta.float(), dim=-1).to(dtype)
        neigh_dist = neigh_dist.masked_fill(~neigh_valid, _BIG_DIST)
        return neighbors, neigh_valid, neigh_dist

    def _build_directional_all_adjacency(
        self,
        node_mid: torch.Tensor,
        node_valid: torch.Tensor,
        node_dnorm: torch.Tensor,
        node_ea: torch.Tensor,
        node_eb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """All hard-gate on-line candidates per node (variable K, padded to batch max).

        Includes every ``j`` with ``pair_ok[i,j]`` and ``|s_ij| > directional_min_sep_px``,
        ordered by increasing ``|s|`` (nearest along-track first). No top-K cap.
        """
        b, n, _ = node_mid.shape
        device, dtype = node_mid.device, node_mid.dtype
        eps = float(self.directional_min_sep_px)

        pair_ok = self._pairwise_collinearity_mask(
            node_mid, node_dnorm, node_ea, node_eb, node_valid,
        )
        mid_i = node_mid.unsqueeze(2).expand(b, n, n, 2)
        mid_j = node_mid.unsqueeze(1).expand(b, n, n, 2)
        dn_i = node_dnorm.unsqueeze(2).expand(b, n, n, 2)
        delta_ij = mid_j - mid_i
        s_signed = (delta_ij * dn_i).sum(dim=-1)
        s_abs = s_signed.abs()
        cand = pair_ok & (s_abs > eps)

        big = torch.tensor(_BIG_DIST, device=device, dtype=dtype)
        sort_key = torch.where(cand, s_abs, big)
        order = torch.argsort(sort_key, dim=-1)
        k = max(1, int(cand.sum(dim=-1).max().item()))
        k = min(k, n)
        neighbors = order[:, :, :k]
        neigh_valid = torch.gather(cand, 2, neighbors)

        mid_stack = node_mid.unsqueeze(1).expand(b, n, n, 2)
        idx_xy = neighbors.unsqueeze(-1).expand(b, n, k, 2)
        mid_neigh = torch.gather(mid_stack, 2, idx_xy)
        neigh_dist = torch.norm((mid_neigh - node_mid.unsqueeze(2)).float(), dim=-1).to(dtype)
        neigh_dist = neigh_dist.masked_fill(~neigh_valid, _BIG_DIST)
        return neighbors, neigh_valid, neigh_dist

    def _build_line_adjacency(
        self,
        node_mid: torch.Tensor,
        node_valid: torch.Tensor,
        node_dnorm: torch.Tensor,
        node_ea: torch.Tensor,
        node_eb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.directional_include_all:
            return self._build_directional_all_adjacency(
                node_mid, node_valid, node_dnorm, node_ea, node_eb,
            )
        return self._build_directional2_adjacency(
            node_mid, node_valid, node_dnorm, node_ea, node_eb,
        )

    def _build_directional2_ctx_adjacency(
        self,
        node_mid: torch.Tensor,
        node_valid: torch.Tensor,
        node_dnorm: torch.Tensor,
        node_ea: torch.Tensor,
        node_eb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """On-line directional + parallel context; ``gat = cat(line, ctx)`` for GAT/edge/loss.

        Returns:
            line_neighbors, line_neigh_valid, line_neigh_dist  (connect / viz / degree / RW),
            gat_neighbors, gat_neigh_valid, gat_neigh_dist  (GAT + edge logits + BCE)
        """
        line_neighbors, line_neigh_valid, line_neigh_dist = self._build_line_adjacency(
            node_mid, node_valid, node_dnorm, node_ea, node_eb,
        )
        b, n, k_line = line_neighbors.shape
        device, dtype = node_mid.device, node_mid.dtype

        if self.context_k <= 0:
            return (
                line_neighbors, line_neigh_valid, line_neigh_dist,
                line_neighbors, line_neigh_valid, line_neigh_dist,
            )

        mid_i = node_mid.unsqueeze(2).expand(b, n, n, 2)
        mid_j = node_mid.unsqueeze(1).expand(b, n, n, 2)
        delta_ij = mid_j - mid_i
        dn_i = node_dnorm.unsqueeze(2).expand(b, n, n, 2)
        dn_j = node_dnorm.unsqueeze(1).expand(b, n, n, 2)

        d_perp = (delta_ij[..., 0] * dn_i[..., 1] - delta_ij[..., 1] * dn_i[..., 0]).abs()
        abs_dir_dot = (dn_i * dn_j).sum(dim=-1).abs()
        s_abs = (delta_ij * dn_i).sum(dim=-1).abs()
        mid_dist = torch.norm(delta_ij.float(), dim=-1).to(dtype)

        eye = torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0).expand(b, -1, -1)
        parallel_ok = node_valid.unsqueeze(2) & node_valid.unsqueeze(1) & ~eye
        if self.context_min_dir_dot > 0.0:
            parallel_ok = parallel_ok & (abs_dir_dot >= self.context_min_dir_dot)
        parallel_ok = parallel_ok & (d_perp >= self.context_lat_min_px) & (d_perp <= self.context_lat_max_px)
        if self.context_max_along_px > 0.0:
            parallel_ok = parallel_ok & (s_abs <= self.context_max_along_px)

        line_adj = torch.zeros(b, n, n, dtype=torch.bool, device=device)
        idx_line = line_neighbors.clamp(0, n - 1)
        line_adj.scatter_(2, idx_line, line_neigh_valid)
        parallel_ok = parallel_ok & ~line_adj

        big = torch.tensor(_BIG_DIST, device=device, dtype=dtype)
        score = torch.where(parallel_ok, d_perp, big)
        k_ctx = max(1, min(int(self.context_k), max(n - 1, 1)))
        _, ctx_j = torch.topk(score, k=k_ctx, dim=-1, largest=False)
        ctx_valid = torch.gather(parallel_ok, 2, ctx_j)

        idx_xy = ctx_j.unsqueeze(-1).expand(b, n, k_ctx, 2)
        mid_stack = node_mid.unsqueeze(1).expand(b, n, n, 2)
        ctx_mid = torch.gather(mid_stack, 2, idx_xy)
        ctx_dist = torch.norm((ctx_mid - node_mid.unsqueeze(2)).float(), dim=-1).to(dtype)
        ctx_dist = ctx_dist.masked_fill(~ctx_valid, _BIG_DIST)

        gat_neighbors = torch.cat([line_neighbors, ctx_j], dim=-1)
        gat_neigh_valid = torch.cat([line_neigh_valid, ctx_valid], dim=-1)
        gat_neigh_dist = torch.cat([line_neigh_dist, ctx_dist], dim=-1)
        return (
            line_neighbors, line_neigh_valid, line_neigh_dist,
            gat_neighbors, gat_neigh_valid, gat_neigh_dist,
        )

    def _build_directional2_global_adjacency(
        self,
        node_mid: torch.Tensor,
        node_valid: torch.Tensor,
        node_dnorm: torch.Tensor,
        node_ea: torch.Tensor,
        node_eb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Euclidean top-``knn_k`` (optional dir filter) for edges; global N×N (hard-gated) for GAT."""
        conn_neighbors, conn_neigh_valid, conn_neigh_dist = self._build_knn(
            node_mid, node_valid, node_dnorm=node_dnorm,
        )
        gat_neighbors, gat_neigh_valid, gat_neigh_dist = self._build_global_adjacency(
            node_mid, node_valid, node_dnorm, node_ea, node_eb,
        )
        return (
            conn_neighbors, conn_neigh_valid, conn_neigh_dist,
            gat_neighbors, gat_neigh_valid, gat_neigh_dist,
        )

    def _apply_geom_mask_to_knn(
        self,
        neighbors: torch.Tensor,
        neigh_valid: torch.Tensor,
        neigh_dist: torch.Tensor,
        node_mid: torch.Tensor,
        node_dnorm: torch.Tensor,
        node_ea: torch.Tensor,
        node_eb: torch.Tensor,
        node_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the same hard geometry gates as global mode to kNN neighbour slots."""
        if not self.use_hard_geom_gate:
            return neighbors, neigh_valid, neigh_dist
        if (
            self.max_lateral_px <= 0.0
            and not self.max_lateral_sym
            and self.min_dir_dot <= 0.0
            and self.max_along_px <= 0.0
            and self.max_end_gap_px <= 0.0
            and self.edge_radius_px <= 0.0
        ):
            return neighbors, neigh_valid, neigh_dist
        pair_ok = self._pairwise_collinearity_mask(
            node_mid, node_dnorm, node_ea, node_eb, node_valid,
        )
        b, n, k = neighbors.shape
        idx = neighbors.clamp(0, n - 1)
        slot_ok = torch.gather(pair_ok, 2, idx)
        bad = ~slot_ok & neigh_valid
        neigh_valid = neigh_valid & slot_ok
        neigh_dist = neigh_dist.masked_fill(bad, _BIG_DIST)
        return neighbors, neigh_valid, neigh_dist

    def _gat_geom_attention_bias(
        self,
        node_mid: torch.Tensor,
        node_dnorm: torch.Tensor,
        neighbors: torch.Tensor,
        neigh_valid: torch.Tensor,
    ) -> torch.Tensor | None:
        """Optional additive bias on GAT logits: favour on-line neighbours, penalise lateral offset."""
        if not self.gat_geom_bias_enabled:
            return None
        b, n, k = neighbors.shape
        idx_xy = neighbors.unsqueeze(-1).expand(b, n, k, 2)
        mid_stack = node_mid.unsqueeze(1).expand(b, n, n, 2)
        mid_neigh = torch.gather(mid_stack, 2, idx_xy)
        delta = mid_neigh - node_mid.unsqueeze(2)
        dn_src = node_dnorm.unsqueeze(2).expand(b, n, k, 2)
        s_signed = (delta * dn_src).sum(dim=-1)
        d_perp = (
            delta[..., 0] * dn_src[..., 1] - delta[..., 1] * dn_src[..., 0]
        ).abs()
        tau_a = max(self.gat_geom_bias_tau_along, 1e-6)
        tau_l = max(self.gat_geom_bias_tau_lat, 1e-6)
        along_term = self.gat_geom_bias_w_along * torch.exp(-s_signed.abs() / tau_a)
        lat_term = self.gat_geom_bias_w_lat * (d_perp / tau_l)
        bias = along_term - lat_term
        return bias.masked_fill(~neigh_valid, 0.0)

    def _edge_geom_features(
        self,
        node_mid: torch.Tensor,
        node_dnorm: torch.Tensor,
        node_conf: torch.Tensor,
        node_ea: torch.Tensor,
        node_eb: torch.Tensor,
        neighbors: torch.Tensor,
        neigh_dist: torch.Tensor,
        scale_w: float,
        scale_h: float,
    ) -> torch.Tensor:
        """Per-edge features ``[B, N, K, 9|11]`` (signed mode keeps along/dir_dot sign + segment lengths)."""
        b, n, _ = node_mid.shape
        k = neighbors.shape[-1]

        idx_xy = neighbors.unsqueeze(-1).expand(b, n, k, 2)
        mid_stack = node_mid.unsqueeze(1).expand(b, n, n, 2)
        dn_stack = node_dnorm.unsqueeze(1).expand(b, n, n, 2)
        ea_stack = node_ea.unsqueeze(1).expand(b, n, n, 2)
        eb_stack = node_eb.unsqueeze(1).expand(b, n, n, 2)
        mid_neigh = torch.gather(mid_stack, 2, idx_xy)
        dir_neigh = torch.gather(dn_stack, 2, idx_xy)
        ea_neigh = torch.gather(ea_stack, 2, idx_xy)
        eb_neigh = torch.gather(eb_stack, 2, idx_xy)
        idx_c = neighbors
        conf_neigh = torch.gather(node_conf.unsqueeze(1).expand(b, n, n), 2, idx_c)

        diag_norm = max(math.hypot(scale_w, scale_h), 1.0)
        dist_norm = (neigh_dist / diag_norm).unsqueeze(-1)
        delta = mid_neigh - node_mid.unsqueeze(2)
        dx_norm = (delta[..., 0:1] / max(scale_w, 1.0))
        dy_norm = (delta[..., 1:2] / max(scale_h, 1.0))
        dir_dot = (node_dnorm.unsqueeze(2) * dir_neigh).sum(dim=-1, keepdim=True)

        dn_src = node_dnorm.unsqueeze(2).expand(b, n, k, 2)
        s_signed = (delta * dn_src).sum(dim=-1, keepdim=True)
        d_perp = (
            delta[..., 0:1] * dn_src[..., 1:2] - delta[..., 1:2] * dn_src[..., 0:1]
        ).abs()
        s_norm = s_signed.abs() / diag_norm
        d_perp_norm = d_perp / diag_norm

        # Min distance over endpoint pairs (A_i,B_i) x (A_j,B_j).
        ea_i = node_ea.unsqueeze(2).expand(b, n, k, 2)
        eb_i = node_eb.unsqueeze(2).expand(b, n, k, 2)
        d_aa = torch.norm(ea_i - ea_neigh, dim=-1, keepdim=True)
        d_ab = torch.norm(ea_i - eb_neigh, dim=-1, keepdim=True)
        d_ba = torch.norm(eb_i - ea_neigh, dim=-1, keepdim=True)
        d_bb = torch.norm(eb_i - eb_neigh, dim=-1, keepdim=True)
        d_end = torch.minimum(torch.minimum(d_aa, d_ab), torch.minimum(d_ba, d_bb))
        d_end_norm = d_end / diag_norm

        conf_src = node_conf.unsqueeze(-1).expand(-1, -1, k).unsqueeze(-1)
        conf_dst = conf_neigh.unsqueeze(-1)

        if self.edge_feat_signed:
            along_norm = s_signed / diag_norm
            len_i = torch.norm(node_ea - node_eb, dim=-1).unsqueeze(-1).expand(b, n, k) / diag_norm
            len_j = torch.norm(ea_neigh - eb_neigh, dim=-1, keepdim=True) / diag_norm
            feats = torch.cat(
                [
                    dist_norm, dx_norm, dy_norm, dir_dot, along_norm, d_perp_norm,
                    d_end_norm, len_i.unsqueeze(-1), len_j, conf_src, conf_dst,
                ],
                dim=-1,
            )
        else:
            abs_dir_dot = dir_dot.abs()
            feats = torch.cat(
                [
                    dist_norm, dx_norm, dy_norm, abs_dir_dot, s_norm, d_perp_norm,
                    d_end_norm, conf_src, conf_dst,
                ],
                dim=-1,
            )
        return feats

    def _build_adjacency(
        self,
        node_mid: torch.Tensor,
        node_valid: torch.Tensor,
        node_dnorm: torch.Tensor,
        node_ea: torch.Tensor,
        node_eb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.adjacency_mode == "global":
            return self._build_global_adjacency(
                node_mid, node_valid, node_dnorm, node_ea, node_eb,
            )
        if self.adjacency_mode == "directional2":
            return self._build_line_adjacency(
                node_mid, node_valid, node_dnorm, node_ea, node_eb,
            )
        if self.adjacency_mode == "directional2_ctx":
            line_n, line_v, line_d, _, _, _ = self._build_directional2_ctx_adjacency(
                node_mid, node_valid, node_dnorm, node_ea, node_eb,
            )
            return line_n, line_v, line_d
        if self.adjacency_mode == "directional2_global":
            line_n, line_v, line_d, _, _, _ = self._build_directional2_global_adjacency(
                node_mid, node_valid, node_dnorm, node_ea, node_eb,
            )
            return line_n, line_v, line_d
        neighbors, neigh_valid, neigh_dist = self._build_knn(node_mid, node_valid)
        return self._apply_geom_mask_to_knn(
            neighbors, neigh_valid, neigh_dist,
            node_mid, node_dnorm, node_ea, node_eb, node_valid,
        )

    # ------------------------------------------------------------------ #
    # Public                                                              #
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
        """
        Returns dict with:
          - ``neighbors``    ``[B, N, K]`` — ``K=N`` (global), ``K=2`` (directional2), or ``K=knn_k``.
          - ``edge_logits``  ``[B, N, K]`` — directed i→j; invalid slots are ``-10``.
        """
        tokens, mid_all, dir_all, dn_all, conf_all, ea_all, eb_all, sw, sh = self._build_node_tokens(
            geom_act=geom_act, feat_map=feat_map, stride=stride,
            img_h=img_h, img_w=img_w, conf_channel=conf_channel,
        )

        segment_merge_stats = None
        if self.segment_merge_enabled:
            (tokens, mid_all, dir_all, dn_all, conf_all,
             ea_all, eb_all, segment_merge_stats) = self._apply_segment_merge(
                feat_map=feat_map,
                tokens=tokens,
                mid_all=mid_all,
                dir_all=dir_all,
                dn_all=dn_all,
                conf_all=conf_all,
                ea_all=ea_all,
                eb_all=eb_all,
                img_h=img_h,
                img_w=img_w,
                scale_w=sw,
                scale_h=sh,
            )

        soft_nms_stats = None
        soft_nms_debug = None
        if self.soft_nms_enabled:
            conf_before = conf_all
            conf_all, soft_nms_stats = segment_soft_nms_batch(
                conf_all,
                mid_all,
                dn_all,
                mid_sigma_px=self.soft_nms_mid_sigma_px,
                min_dir_dot=self.soft_nms_min_dir_dot,
                decay_method=self.soft_nms_decay_method,
                score_floor=self.soft_nms_score_floor,
                prefilter_conf=self.soft_nms_prefilter_conf,
                max_segments=self.soft_nms_max_segments,
            )
            if not self.training:
                soft_nms_debug = {
                    "mid_px": mid_all.detach(),
                    "end_a_px": ea_all.detach(),
                    "end_b_px": eb_all.detach(),
                    "conf_before": conf_before.detach(),
                    "conf_after": conf_all.detach(),
                    "viz_thresh": float(self.node_conf_thresh),
                }

        node_feat, node_mid, node_dir, node_dn, node_ea, node_eb, node_conf, node_valid, node_src_flat_idx = self._select_nodes(
            tokens, mid_all, dir_all, dn_all, conf_all, ea_all, eb_all,
        )

        line_neighbors = line_neigh_valid = line_neigh_dist = None
        if self.adjacency_mode in ("directional2_ctx", "directional2_global"):
            if self.adjacency_mode == "directional2_ctx":
                (
                    line_neighbors, line_neigh_valid, line_neigh_dist,
                    gat_neighbors, gat_neigh_valid, gat_neigh_dist,
                ) = self._build_directional2_ctx_adjacency(
                    node_mid, node_valid, node_dn, node_ea, node_eb,
                )
                neighbors = gat_neighbors
                neigh_valid = gat_neigh_valid
                neigh_dist = gat_neigh_dist
            else:
                (
                    line_neighbors, line_neigh_valid, line_neigh_dist,
                    gat_neighbors, gat_neigh_valid, gat_neigh_dist,
                ) = self._build_directional2_global_adjacency(
                    node_mid, node_valid, node_dn, node_ea, node_eb,
                )
                neighbors = line_neighbors
                neigh_valid = line_neigh_valid
                neigh_dist = line_neigh_dist
        else:
            neighbors, neigh_valid, neigh_dist = self._build_adjacency(
                node_mid, node_valid, node_dn, node_ea, node_eb,
            )
            gat_neighbors, gat_neigh_valid = neighbors, neigh_valid

        edge_feat = self._edge_geom_features(
            node_mid=node_mid, node_dnorm=node_dn, node_conf=node_conf,
            node_ea=node_ea, node_eb=node_eb,
            neighbors=neighbors, neigh_dist=neigh_dist, scale_w=sw, scale_h=sh,
        )

        h = node_feat
        gat_geom_bias = self._gat_geom_attention_bias(
            node_mid, node_dn, gat_neighbors, gat_neigh_valid,
        )
        for layer in self.gat_layers:
            h = layer(h, gat_neighbors, gat_neigh_valid, geom_bias=gat_geom_bias)

        b, n, d = h.shape
        k = neighbors.shape[-1]
        idx_e = neighbors.reshape(b, n * k, 1).expand(b, n * k, d)
        h_neigh = torch.gather(h, 1, idx_e).view(b, n, k, d)
        h_src = h.unsqueeze(2).expand(-1, -1, k, -1)
        pair = torch.cat([h_src, h_neigh, edge_feat], dim=-1)
        edge_logits = self.edge_head(pair).squeeze(-1)

        soft_geom_stats = None
        if self.soft_geom_gate_enabled:
            edge_prior, soft_geom_stats = compute_edge_prior(
                node_mid,
                node_dn,
                node_valid,
                sigma_lat_px=self.soft_geom_sigma_lat_px,
                dir_floor=self.soft_geom_dir_floor,
                prior_eps=self.soft_geom_prior_eps,
            )
            prior_k = torch.gather(edge_prior, 2, neighbors.clamp(0, n - 1))
            edge_logits = edge_logits + torch.log(prior_k.clamp(min=self.soft_geom_prior_eps))

        edge_logits = torch.where(
            neigh_valid, edge_logits, torch.full_like(edge_logits, -10.0)
        )

        gnn_adj_stats = None
        if self.adjacency_mode in ("directional2_ctx", "directional2_global") and line_neighbors is not None:
            with torch.no_grad():
                k_line = int(line_neighbors.shape[-1])
                k_gat = int(gat_neighbors.shape[-1])
                nv = node_valid.unsqueeze(-1)
                line_cnt = (line_neigh_valid & nv).float().sum(dim=-1)
                if self.adjacency_mode == "directional2_global":
                    gat_cnt = (gat_neigh_valid & nv).float().sum(dim=-1)
                    frac_gat_used = (
                        gat_neigh_valid.float().sum() / max(gat_neigh_valid.numel(), 1.0)
                    )
                    ctx_cnt = torch.zeros_like(line_cnt)
                    frac_ctx_used = 0.0
                elif k_gat > k_line:
                    ctx_valid = gat_neigh_valid[:, :, k_line:]
                    ctx_cnt = (ctx_valid & nv).float().sum(dim=-1)
                    frac_ctx_used = (
                        ctx_valid.float().sum() / max(ctx_valid.numel(), 1.0)
                    )
                    gat_cnt = None
                    frac_gat_used = 0.0
                else:
                    ctx_cnt = torch.zeros_like(line_cnt)
                    frac_ctx_used = 0.0
                    gat_cnt = None
                    frac_gat_used = 0.0
                node_denom = node_valid.float().sum().clamp(min=1.0)
                gnn_adj_stats = {
                    "k_line": float(k_line),
                    "k_gat": float(k_gat),
                    "k_ctx": float(max(k_gat - k_line, 0)),
                    "mean_line_neighbors": float((line_cnt * node_valid.float()).sum() / node_denom),
                    "mean_ctx_neighbors": float((ctx_cnt * node_valid.float()).sum() / node_denom),
                    "frac_nodes_with_ctx": float(
                        ((ctx_cnt > 0) & node_valid).float().sum() / node_denom
                    ),
                    "frac_ctx_slots_used": float(frac_ctx_used),
                }
                if gat_cnt is not None:
                    gnn_adj_stats["mean_gat_neighbors"] = float(
                        (gat_cnt * node_valid.float()).sum() / node_denom
                    )
                    gnn_adj_stats["frac_gat_slots_used"] = float(frac_gat_used)

        out = {
            "node_feat": h,
            "node_mid_px": node_mid,
            "node_end_a_px": node_ea,
            "node_end_b_px": node_eb,
            "node_valid": node_valid,
            "node_conf": node_conf,
            "node_src_flat_idx": node_src_flat_idx,
            "neighbors": neighbors,
            "neigh_valid": neigh_valid,
            "neigh_dist": neigh_dist,
            "gat_neighbors": gat_neighbors,
            "gat_neigh_valid": gat_neigh_valid,
            "edge_logits": edge_logits,
            "edge_feat": edge_feat,
        }
        if line_neighbors is not None:
            out["line_neighbors"] = line_neighbors
            out["line_neigh_valid"] = line_neigh_valid
        if soft_nms_stats is not None:
            out["soft_nms_stats"] = soft_nms_stats
        if soft_nms_debug is not None:
            out["soft_nms_debug"] = soft_nms_debug
        if soft_geom_stats is not None:
            out["soft_geom_stats"] = soft_geom_stats
        if segment_merge_stats is not None:
            out["segment_merge_stats"] = segment_merge_stats
        if gnn_adj_stats is not None:
            out["gnn_adj_stats"] = gnn_adj_stats
        return out

    def _apply_segment_merge(
        self,
        *,
        feat_map: torch.Tensor,
        tokens: torch.Tensor,
        mid_all: torch.Tensor,
        dir_all: torch.Tensor,
        dn_all: torch.Tensor,
        conf_all: torch.Tensor,
        ea_all: torch.Tensor,
        eb_all: torch.Tensor,
        img_h: int,
        img_w: int,
        scale_w: float,
        scale_h: float,
    ):
        """Run :func:`merge_segments` per image and rebuild tokens at merged midpoints.

        The merge is non-differentiable (numpy union-find + extent projection); we
        therefore detach the inputs, replace the corresponding slot rows in-place,
        and **re-sample visual features at the new mid-px positions** via
        ``grid_sample``. Surplus slots are zeroed and flagged invalid via conf=0.
        """
        b, n_all = conf_all.shape
        device = conf_all.device
        new_conf = torch.zeros_like(conf_all)
        new_mid = torch.zeros_like(mid_all)
        new_dir = torch.zeros_like(dir_all)
        new_dn = torch.zeros_like(dn_all)
        new_ea = torch.zeros_like(ea_all)
        new_eb = torch.zeros_like(eb_all)
        stats_n_before = conf_all.new_zeros((b,))
        stats_n_after = conf_all.new_zeros((b,))

        cf_np_all = conf_all.detach().cpu().numpy()
        ea_np_all = ea_all.detach().cpu().numpy()
        eb_np_all = eb_all.detach().cpu().numpy()

        for bi in range(b):
            cf = cf_np_all[bi].astype(np.float32)
            ea_np = ea_np_all[bi].astype(np.float32)
            eb_np = eb_np_all[bi].astype(np.float32)
            mask = cf > self.segment_merge_prefilter_conf
            stats_n_before[bi] = float(mask.sum())
            if not bool(mask.any()):
                continue
            idx = np.where(mask)[0]
            ea_m, eb_m, cf_m = merge_segments(
                ea_np[idx], eb_np[idx], cf[idx],
                lat_px=self.segment_merge_lat_px,
                dir_dot_min=self.segment_merge_dir_dot_min,
                end_gap_px=self.segment_merge_end_gap_px,
                iters=self.segment_merge_iters,
            )
            order = np.argsort(-cf_m)
            keep = min(int(order.size), n_all)
            order = order[:keep]
            cf_m = cf_m[order]; ea_m = ea_m[order]; eb_m = eb_m[order]
            stats_n_after[bi] = float(keep)
            if keep == 0:
                continue

            mid_m = 0.5 * (ea_m + eb_m)
            vec_m = eb_m - ea_m
            L_m = np.linalg.norm(vec_m, axis=-1)
            safe = np.maximum(L_m, 1e-8)
            dn_m = vec_m / safe[:, None]

            cf_t = torch.from_numpy(cf_m).to(device=device, dtype=conf_all.dtype)
            ea_t = torch.from_numpy(ea_m).to(device=device, dtype=ea_all.dtype)
            eb_t = torch.from_numpy(eb_m).to(device=device, dtype=eb_all.dtype)
            mid_t = torch.from_numpy(mid_m).to(device=device, dtype=mid_all.dtype)
            dir_t = torch.from_numpy(vec_m).to(device=device, dtype=dir_all.dtype)
            dn_t = torch.from_numpy(dn_m).to(device=device, dtype=dn_all.dtype)

            sl = slice(0, keep)
            new_conf[bi, sl] = cf_t
            new_ea[bi, sl] = ea_t
            new_eb[bi, sl] = eb_t
            new_mid[bi, sl] = mid_t
            new_dir[bi, sl] = dir_t
            new_dn[bi, sl] = dn_t

        # Re-sample FPN features at the merged midpoints; non-filled slots
        # land at (0, 0) which the conf=0 mask later excludes via _select_nodes.
        grid = pixels_to_grid_sample_grid(new_mid, img_h, img_w)
        sampled = F.grid_sample(
            feat_map, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        sampled = sampled.view(b, feat_map.shape[1], n_all).transpose(1, 2)
        visual_feat = self.visual_proj(sampled)

        mx = 2.0 * new_mid[..., 0:1] / max(float(scale_w), 1.0) - 1.0
        my = 2.0 * new_mid[..., 1:2] / max(float(scale_h), 1.0) - 1.0
        pe_in = torch.cat([mx, my, new_dn], dim=-1)
        pe_feat = self.pe_proj(_sine_pe_4d(pe_in, self._freq_bands))
        new_tokens = self.fuse(torch.cat([visual_feat, pe_feat], dim=-1))

        stats = {
            "n_before_merge": stats_n_before,
            "n_after_merge": stats_n_after,
        }
        return new_tokens, new_mid, new_dir, new_dn, new_conf, new_ea, new_eb, stats
