# SPDX-License-Identifier: GPL-3.0-or-later
"""
Soft-NMS for YOLinO line segments (pre-GNN deduplication).

Duplicates are suppressed by **confidence decay**, not hard removal. Overlap uses
**lateral distance only** (perpendicular offset of ``mid_j`` to the source segment
line); along-track separation does not enter the suppress score, so collinear
fragments spaced along the same wire are not decayed.

Typical use: ``YolinoGnnSegmentGraphHead`` applies this on the full
(cell × predictor) segment pool before top-k node selection.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch


def _lateral_dist_px(
    mid_i: torch.Tensor,
    dn_i: torch.Tensor,
    mid_j: torch.Tensor,
) -> torch.Tensor:
    """Perpendicular distance from each ``mid_j`` to the line through ``mid_i`` with direction ``dn_i``."""
    delta = mid_j - mid_i
    return (delta[..., 0] * dn_i[..., 1] - delta[..., 1] * dn_i[..., 0]).abs()


def _pairwise_overlap(
    mid: torch.Tensor,
    dnorm: torch.Tensor,
    i: int,
    j_ids: torch.Tensor,
    mid_sigma_px: float,
    min_dir_dot: float,
) -> torch.Tensor:
    """Overlap weights in [0, 1] for segment i vs indices j_ids (same length).

    ``overlap = exp(-d_lateral^2 / (2 * sigma_lat^2))``; ``d_along`` is not used.
    ``min_dir_dot`` is kept for API compatibility but not applied (lateral-only suppress).
    """
    del min_dir_dot  # lateral-only; direction is not part of the decay score
    if j_ids.numel() == 0:
        return mid.new_zeros((0,))
    mid_i = mid[i : i + 1]
    dn_i = dnorm[i : i + 1]
    d_lateral = _lateral_dist_px(mid_i, dn_i, mid[j_ids])
    sigma_lat = max(float(mid_sigma_px), 1e-6)
    return torch.exp(-(d_lateral ** 2) / (2.0 * sigma_lat ** 2)).clamp(0.0, 1.0)


def segment_soft_nms_1d(
    conf: torch.Tensor,
    mid: torch.Tensor,
    dnorm: torch.Tensor,
    *,
    mid_sigma_px: float = 16.0,
    min_dir_dot: float = 0.96,
    decay_method: str = "linear",
    score_floor: float = 0.001,
    prefilter_conf: float = 0.05,
    max_segments: int = 1024,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Soft-NMS on one image's segments.

    Args:
        conf: ``[N]`` scores (e.g. sigmoid confidence).
        mid: ``[N, 2]`` pixel midpoints.
        dnorm: ``[N, 2]`` unit direction vectors.
    Returns:
        decayed conf ``[N]`` and scalar stats.
    """
    n = int(conf.numel())
    stats = {
        "n_total": float(n),
        "n_prefilter": 0.0,
        "n_processed": 0.0,
        "n_above_floor_after": 0.0,
        "mean_decay_ratio": 1.0,
    }
    if n == 0:
        return conf, stats

    out = conf.clone()
    keep = out >= float(prefilter_conf)
    if not bool(keep.any()):
        return out, stats

    idx = torch.where(keep)[0]
    stats["n_prefilter"] = float(idx.numel())
    if idx.numel() > int(max_segments):
        sub_conf = out[idx]
        topk = torch.topk(sub_conf, k=int(max_segments), dim=0).indices
        idx = idx[topk]

    sub_mid = mid[idx]
    sub_dn = dnorm[idx]
    sub_scores = out[idx].clone()
    order = torch.argsort(sub_scores, descending=True)
    stats["n_processed"] = float(order.numel())

    decay_ratios = []
    method = str(decay_method).lower().strip()
    floor = float(score_floor)

    for ord_pos in range(order.numel()):
        si = int(order[ord_pos])
        if float(sub_scores[si]) < floor:
            continue
        active = sub_scores >= floor
        active[si] = False
        j_local = torch.where(active)[0]
        if j_local.numel() == 0:
            continue
        overlap = _pairwise_overlap(
            sub_mid, sub_dn, si, j_local, mid_sigma_px, min_dir_dot,
        )
        if overlap.numel() == 0:
            continue
        if method == "gaussian":
            decay = torch.exp(-(overlap ** 2))
        else:
            decay = (1.0 - overlap).clamp(0.0, 1.0)
        before = sub_scores[j_local].clone()
        sub_scores[j_local] = sub_scores[j_local] * decay
        valid = before > 1e-8
        if bool(valid.any()):
            decay_ratios.append((sub_scores[j_local][valid] / before[valid]).mean().item())

    out[idx] = sub_scores
    stats["n_above_floor_after"] = float((out >= floor).sum().item())
    if decay_ratios:
        stats["mean_decay_ratio"] = float(sum(decay_ratios) / len(decay_ratios))
    return out, stats


def segment_soft_nms_batch(
    conf: torch.Tensor,
    mid: torch.Tensor,
    dnorm: torch.Tensor,
    *,
    mid_sigma_px: float = 16.0,
    min_dir_dot: float = 0.96,
    decay_method: str = "linear",
    score_floor: float = 0.001,
    prefilter_conf: float = 0.05,
    max_segments: int = 1024,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Batched soft-NMS.

    Args:
        conf: ``[B, N]``
        mid: ``[B, N, 2]``
        dnorm: ``[B, N, 2]``
    Returns:
        decayed conf ``[B, N]`` and dict of per-batch scalar tensors for logging.
    """
    b, _ = conf.shape
    out = conf.clone()
    n_pref = conf.new_zeros((b,))
    n_proc = conf.new_zeros((b,))
    n_after = conf.new_zeros((b,))
    mean_decay = conf.new_ones((b,))

    for bi in range(b):
        out[bi], st = segment_soft_nms_1d(
            out[bi], mid[bi], dnorm[bi],
            mid_sigma_px=mid_sigma_px,
            min_dir_dot=min_dir_dot,
            decay_method=decay_method,
            score_floor=score_floor,
            prefilter_conf=prefilter_conf,
            max_segments=max_segments,
        )
        n_pref[bi] = st["n_prefilter"]
        n_proc[bi] = st["n_processed"]
        n_after[bi] = st["n_above_floor_after"]
        mean_decay[bi] = st["mean_decay_ratio"]

    return out, {
        "n_prefilter": n_pref,
        "n_processed": n_proc,
        "n_above_floor_after": n_after,
        "mean_decay_ratio": mean_decay,
        "n_total": conf.new_full((b,), float(conf.shape[1])),
    }
