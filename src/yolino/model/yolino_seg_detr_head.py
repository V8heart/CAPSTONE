# SPDX-License-Identifier: GPL-3.0-or-later
"""
Segment-token DETR Bézier heads.

* **Memory** = all YOLinO segment tokens: FPN ``grid_sample`` at segment mid + sine PE(mid, dir).
* **Queries**:
  - :class:`YolinoSegDetrBezierHead` — fixed K **learned** embeddings (exp44).
  - :class:`YolinoSegDetrSoftInitBezierHead` — segment-informed soft init (exp45):
    ``residual_topk`` (Top-K token + learnable residual + PE) or
    ``softmax_pool`` (learned slot attends over segments).

Reuses :func:`yolino.model.yolino_detr_criterion.compute_detr_e2e_loss` (Hungarian set loss).
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

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


def _conf_for_topk(
    conf_flat: torch.Tensor,
    g: torch.Tensor,
    conf_channel: int,
    local_max_filter: bool,
    local_max_kernel: int,
    n_tokens: int,
) -> torch.Tensor:
    """Confidence map for Top-K query selection (optional 2D local-max NMS)."""
    if not local_max_filter:
        return conf_flat
    b = conf_flat.shape[0]
    hf, wf, p = g.shape[1], g.shape[2], g.shape[3]
    conf_2d = g[..., conf_channel].permute(0, 3, 1, 2).contiguous()
    klm = local_max_kernel
    local_max = F.max_pool2d(conf_2d, kernel_size=klm, stride=1, padding=klm // 2)
    peak_mask_flat = (conf_2d == local_max).permute(0, 2, 3, 1).reshape(b, n_tokens)
    neg_inf = torch.finfo(conf_flat.dtype).min
    return torch.where(peak_mask_flat, conf_flat, torch.full_like(conf_flat, neg_inf))


class _SegDetrBezierHeadBase(nn.Module):
    """Shared segment-token memory + Bézier decoder trunk."""

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
        polyline_num_points: int = 0,
        conf_filter_thresh: float = 0.1,
        dropout: float = 0.1,
        sine_freq_bands: int = 16,
    ):
        super().__init__()
        if line_rep != LINE.MID_DIR:
            Log.warning(
                "_SegDetrBezierHeadBase expects MID_DIR geometry; got %s." % line_rep
            )
        self.line_rep = line_rep
        self.num_queries = int(num_queries)
        self.token_dim = int(token_dim)
        self.bezier_degree = int(bezier_degree)
        self.num_ctrl = self.bezier_degree + 1
        self.bezier_num_samples = int(bezier_num_samples)
        self.polyline_num_points = int(polyline_num_points)
        self.output_mode = "polyline" if self.polyline_num_points > 0 else "bezier"
        self.conf_filter_thresh = float(conf_filter_thresh)
        self.sine_freq_bands = int(sine_freq_bands)

        self.visual_proj = nn.Linear(int(fpn_channels), self.token_dim)
        pe_in = 4 * 2 * self.sine_freq_bands
        self.pe_proj = nn.Linear(pe_in, self.token_dim)
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

        out_pts = self.polyline_num_points if self.output_mode == "polyline" else self.num_ctrl
        self.ctrl_head = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, out_pts * 2),
        )
        self.objectness_head = nn.Linear(self.token_dim, 1)

        bands = (2.0 ** torch.arange(self.sine_freq_bands).float()) * math.pi
        self.register_buffer("_freq_bands", bands, persistent=False)

    def _sine_pe_4d(self, x4: torch.Tensor) -> torch.Tensor:
        f = x4.unsqueeze(-1) * self._freq_bands
        return torch.cat([torch.sin(f), torch.cos(f)], dim=-1).flatten(-2)

    def _encode_segment_memory(
        self,
        geom_act: torch.Tensor,
        feat_map: torch.Tensor,
        stride: float,
        img_h: int,
        img_w: int,
        conf_channel: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        """Returns tokens, mid_flat, dnorm, conf_flat, g, scale_w, scale_h."""
        b, ncell, p, _ = geom_act.shape
        hf, wf = feat_map.shape[-2:]
        if ncell != hf * wf:
            raise ValueError(
                "_SegDetrBezierHeadBase expects cells == Hf*Wf; got cells=%d, feat H*W=%d"
                % (ncell, hf * wf)
            )

        g = geom_act.view(b, hf, wf, p, geom_act.shape[-1])
        mid_px, end_a, end_b = mid_dir_geom_to_midpoints_pixels(
            g, stride=stride, img_h=img_h, img_w=img_w
        )
        dir_flat = (end_b - end_a).reshape(b, hf * wf * p, 2)
        n_tokens = hf * wf * p
        mid_flat = mid_px.reshape(b, n_tokens, 2)

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
        pe_feat = self.pe_proj(self._sine_pe_4d(pe_in))
        tokens = self.fuse(torch.cat([visual_feat, pe_feat], dim=-1))

        conf_flat = g[..., conf_channel].reshape(b, n_tokens)
        valid_mask = conf_flat > self.conf_filter_thresh
        empty_rows = valid_mask.sum(dim=1) == 0
        if empty_rows.any():
            valid_mask = valid_mask.clone()
            valid_mask[empty_rows] = True

        return tokens, mid_flat, dnorm, conf_flat, g, scale_w, scale_h

    def _decode_bezier(
        self,
        queries: torch.Tensor,
        tokens: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        scale_w: float,
        scale_h: float,
    ) -> Dict[str, torch.Tensor]:
        b, k, _ = queries.shape
        decoded = self.decoder(
            tgt=queries,
            memory=tokens,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        ctrl_norm_01 = self.ctrl_head(decoded).view(b, k, self.num_ctrl, 2).sigmoid()
        scale_xy = torch.tensor(
            [scale_w, scale_h], device=ctrl_norm_01.device, dtype=ctrl_norm_01.dtype
        )
        ctrl_px = ctrl_norm_01 * scale_xy
        obj_logits = self.objectness_head(decoded).squeeze(-1)
        bern = bernstein_matrix(
            self.bezier_degree, self.bezier_num_samples, ctrl_px.device, ctrl_px.dtype
        )
        curve_px = bezier_sample_points(ctrl_px, bern)
        return {
            "bezier_ctrl_px": ctrl_px,
            "bezier_curve_px": curve_px,
            "objectness_logits": obj_logits,
        }

    def _decode_polyline_points(
        self,
        queries: torch.Tensor,
        tokens: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        scale_w: float,
        scale_h: float,
    ) -> Dict[str, torch.Tensor]:
        """Direct N-point polyline regression (no Bernstein), pixels in [0,W]x[0,H]."""
        b, k, _ = queries.shape
        n = self.polyline_num_points
        decoded = self.decoder(
            tgt=queries,
            memory=tokens,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        pts_norm_01 = self.ctrl_head(decoded).view(b, k, n, 2).sigmoid()
        scale_xy = torch.tensor(
            [scale_w, scale_h], device=pts_norm_01.device, dtype=pts_norm_01.dtype
        )
        polylines_px = pts_norm_01 * scale_xy
        obj_logits = self.objectness_head(decoded).squeeze(-1)
        return {
            "polylines_px": polylines_px,
            "bezier_curve_px": polylines_px,
            "objectness_logits": obj_logits,
            "polyline_num_points": n,
        }


class YolinoSegDetrBezierHead(_SegDetrBezierHeadBase):
    """DETR Bézier head with learned queries and segment-token memory only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_embed = nn.Embedding(self.num_queries, self.token_dim)
        nn.init.normal_(self.query_embed.weight, std=0.02)

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
        tokens, mid_flat, dnorm, conf_flat, g, scale_w, scale_h = self._encode_segment_memory(
            geom_act, feat_map, stride, img_h, img_w, conf_channel
        )
        valid_mask = conf_flat > self.conf_filter_thresh
        empty_rows = valid_mask.sum(dim=1) == 0
        if empty_rows.any():
            valid_mask = valid_mask.clone()
            valid_mask[empty_rows] = True
        memory_key_padding_mask = ~valid_mask

        k = self.num_queries
        queries = self.query_embed.weight.unsqueeze(0).expand(b, k, -1)
        out = self._decode_bezier(queries, tokens, memory_key_padding_mask, scale_w, scale_h)
        n_tokens = tokens.shape[1]
        out["valid_token_mask"] = valid_mask
        out["segment_token_count"] = torch.full(
            (b,), float(n_tokens), device=out["bezier_ctrl_px"].device, dtype=out["bezier_ctrl_px"].dtype,
        )
        return out


class YolinoSegDetrSoftInitBezierHead(_SegDetrBezierHeadBase):
    """
    Segment-informed soft query init + same segment memory as seg_detr.

    * ``residual_topk``: gather Top-K segment tokens + learnable residual + sine PE(ref mid, dir).
    * ``softmax_pool``: each query slot softmax-pools segment tokens, then adds slot embedding.
    """

    def __init__(
        self,
        *args,
        soft_init_mode: str = "residual_topk",
        local_max_filter: bool = False,
        local_max_kernel: int = 5,
        softmax_pool_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        mode = str(soft_init_mode).lower()
        if mode not in ("residual_topk", "softmax_pool"):
            raise ValueError(
                "YolinoSegDetrSoftInitBezierHead soft_init_mode must be 'residual_topk' or "
                "'softmax_pool'; got %r" % soft_init_mode
            )
        self.soft_init_mode = mode
        self.local_max_filter = bool(local_max_filter)
        k_lm = int(local_max_kernel)
        if k_lm < 1:
            k_lm = 1
        if k_lm % 2 == 0:
            k_lm += 1
        self.local_max_kernel = k_lm
        self.softmax_pool_temperature = max(float(softmax_pool_temperature), 1e-6)

        if self.soft_init_mode == "residual_topk":
            self.query_residual = nn.Embedding(self.num_queries, self.token_dim)
            nn.init.normal_(self.query_residual.weight, std=0.02)
        else:
            self.query_slot = nn.Embedding(self.num_queries, self.token_dim)
            nn.init.normal_(self.query_slot.weight, std=0.02)

    def _query_pe_from_refs(
        self, ref_points_px: torch.Tensor, ref_dir_norm: torch.Tensor, scale_w: float, scale_h: float,
    ) -> torch.Tensor:
        rx = 2.0 * ref_points_px[..., 0:1] / scale_w - 1.0
        ry = 2.0 * ref_points_px[..., 1:2] / scale_h - 1.0
        q_pe_in = torch.cat([rx, ry, ref_dir_norm], dim=-1)
        return self.pe_proj(self._sine_pe_4d(q_pe_in))

    def _init_queries_residual_topk(
        self,
        tokens: torch.Tensor,
        mid_flat: torch.Tensor,
        dnorm: torch.Tensor,
        conf_flat: torch.Tensor,
        g: torch.Tensor,
        conf_channel: int,
        scale_w: float,
        scale_h: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n_tokens, _ = tokens.shape
        k = min(self.num_queries, n_tokens)
        conf_for_topk = _conf_for_topk(
            conf_flat, g, conf_channel, self.local_max_filter, self.local_max_kernel, n_tokens,
        )
        _, topk_idx = torch.topk(conf_for_topk, k=k, dim=1)
        topk_conf = torch.gather(conf_flat, 1, topk_idx)

        idx_d = topk_idx.unsqueeze(-1).expand(-1, -1, self.token_dim)
        initial_queries = torch.gather(tokens, 1, idx_d)
        idx_xy = topk_idx.unsqueeze(-1).expand(-1, -1, 2)
        ref_points_px = torch.gather(mid_flat, 1, idx_xy)
        ref_dir_norm = torch.gather(dnorm, 1, idx_xy)

        residual = self.query_residual.weight[:k].unsqueeze(0).expand(b, -1, -1)
        q_pe = self._query_pe_from_refs(ref_points_px, ref_dir_norm, scale_w, scale_h)
        queries = initial_queries + residual + q_pe
        return queries, ref_points_px, topk_conf

    def _init_queries_softmax_pool(
        self,
        tokens: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, n_tokens, d = tokens.shape
        k = self.num_queries
        slots = self.query_slot.weight.unsqueeze(0).expand(b, k, -1)
        scores = torch.bmm(slots, tokens.transpose(1, 2)) / math.sqrt(float(d))
        scores = scores / self.softmax_pool_temperature
        scores = scores.masked_fill(memory_key_padding_mask.unsqueeze(1), float("-inf"))
        alpha = F.softmax(scores, dim=-1)
        pooled = torch.bmm(alpha, tokens)
        return pooled + slots

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
        tokens, mid_flat, dnorm, conf_flat, g, scale_w, scale_h = self._encode_segment_memory(
            geom_act, feat_map, stride, img_h, img_w, conf_channel
        )
        valid_mask = conf_flat > self.conf_filter_thresh
        empty_rows = valid_mask.sum(dim=1) == 0
        if empty_rows.any():
            valid_mask = valid_mask.clone()
            valid_mask[empty_rows] = True
        memory_key_padding_mask = ~valid_mask
        n_tokens = tokens.shape[1]

        ref_points_px = None
        topk_conf = None
        if self.soft_init_mode == "residual_topk":
            queries, ref_points_px, topk_conf = self._init_queries_residual_topk(
                tokens, mid_flat, dnorm, conf_flat, g, conf_channel, scale_w, scale_h,
            )
        else:
            queries = self._init_queries_softmax_pool(tokens, memory_key_padding_mask)

        out = self._decode_bezier(queries, tokens, memory_key_padding_mask, scale_w, scale_h)
        out["valid_token_mask"] = valid_mask
        out["segment_token_count"] = torch.full(
            (b,), float(n_tokens), device=out["bezier_ctrl_px"].device, dtype=out["bezier_ctrl_px"].dtype,
        )
        out["soft_init_mode"] = self.soft_init_mode
        if ref_points_px is not None:
            out["ref_points_px"] = ref_points_px
        if topk_conf is not None:
            out["topk_conf"] = topk_conf
        return out


class YolinoGridSegDetrBezierHead(_SegDetrBezierHeadBase):
    """
    seg_detr memory without YOLinO geom head: tokens at fixed cell centers × P predictor dirs.

    Expects feat_map already STD-refined (optionally through StdFeatRefineDcn). No geom_act.
    """

    def __init__(
        self,
        *args,
        num_predictors: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_predictors = int(num_predictors)
        self.query_embed = nn.Embedding(self.num_queries, self.token_dim)
        nn.init.normal_(self.query_embed.weight, std=0.02)
        self.pred_dir_embed = nn.Embedding(self.num_predictors, 2)
        nn.init.normal_(self.pred_dir_embed.weight, std=0.02)
        self.score_convs = nn.ModuleList(
            [
                nn.Conv2d(self.visual_proj.in_features, 1, kernel_size=1, bias=True)
                for _ in range(self.num_predictors)
            ]
        )

    def forward(
        self,
        feat_map: torch.Tensor,
        stride: float,
        img_h: int,
        img_w: int,
        conf_channel: int = -1,
        geom_act: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        del conf_channel, geom_act
        b, _, hf, wf = feat_map.shape
        p = self.num_predictors
        n_tokens = hf * wf * p
        device, dtype = feat_map.device, feat_map.dtype

        ys = (torch.arange(hf, device=device, dtype=dtype) + 0.5) * float(stride)
        xs = (torch.arange(wf, device=device, dtype=dtype) + 0.5) * float(stride)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        mid_flat = torch.stack(
            [xx.reshape(-1), yy.reshape(-1)], dim=-1
        ).unsqueeze(0).expand(b, -1, -1)
        mid_flat = mid_flat.unsqueeze(2).expand(b, hf * wf, p, 2).reshape(b, n_tokens, 2)

        dirs = F.normalize(self.pred_dir_embed.weight, dim=-1)
        dir_flat = dirs.view(1, 1, p, 2).expand(b, hf * wf, p, 2).reshape(b, n_tokens, 2)

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
        pe_feat = self.pe_proj(self._sine_pe_4d(torch.cat([mx, my, dir_flat], dim=-1)))
        tokens = self.fuse(torch.cat([visual_feat, pe_feat], dim=-1))

        conf_parts = [conv(feat_map) for conv in self.score_convs]
        conf_map = torch.cat(conf_parts, dim=1)
        conf_flat = conf_map.reshape(b, p, hf * wf).permute(0, 2, 1).reshape(b, n_tokens)
        conf_flat = torch.sigmoid(conf_flat)

        valid_mask = conf_flat > self.conf_filter_thresh
        empty_rows = valid_mask.sum(dim=1) == 0
        if empty_rows.any():
            valid_mask = valid_mask.clone()
            valid_mask[empty_rows] = True
        memory_key_padding_mask = ~valid_mask

        k = self.num_queries
        queries = self.query_embed.weight.unsqueeze(0).expand(b, k, -1)
        if self.output_mode == "polyline":
            out = self._decode_polyline_points(
                queries, tokens, memory_key_padding_mask, scale_w, scale_h,
            )
        else:
            out = self._decode_bezier(queries, tokens, memory_key_padding_mask, scale_w, scale_h)
        out["valid_token_mask"] = valid_mask
        ref_dev = out.get("polylines_px", out["bezier_curve_px"])
        out["segment_token_count"] = torch.full(
            (b,), float(n_tokens), device=ref_dev.device, dtype=ref_dev.dtype,
        )
        out["grid_token_mode"] = True
        return out
