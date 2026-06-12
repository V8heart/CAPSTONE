# SPDX-License-Identifier: GPL-3.0-or-later
"""Loss for YolinoCenterPolyHead (center heatmap + N-point polyline regression).

Components (weighted sum):
  1. *Gaussian focal loss* on the center heatmap. Targets = max over instances of
     a 2-D Gaussian centered at each GT polyline's arc-length midpoint with σ
     proportional to the polyline length (clamped). CornerNet penalty.
  2. *Bidirectional min L1* on N-point polylines. Peaks are matched to GT centers
     with ``center_poly_match_mode``: ``greedy_1to1`` (default) sorts valid (peak,GT)
     pairs by distance and greedily assigns so **each GT gets at most one peak and
     each peak at most one GT**; ``nearest`` keeps the old per-peak independent
     nearest-GT rule (duplicates allowed). Only matched pairs contribute poly loss.
  3. *Aux* deep-supervision: same L1 on intermediate decoder layers, weighted.

DDP-safety: even when no GT is assignable, every learnable parameter is touched
via a tiny ``* 0`` stub in the returned ``total`` so DDP's allreduce is happy.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _gaussian_kernel_size(sigma: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.round(3.0 * sigma), min=2.0)


def _build_center_heatmap_target(
    centers_xy: torch.Tensor,    # [NI_total, 2] pixel (x, y)
    sigmas: torch.Tensor,        # [NI_total] pixel σ
    H: int, W: int, stride: float, device, dtype
) -> torch.Tensor:
    """Render the per-instance Gaussian peaks onto a [1, Hf, Wf] target (max-merged).

    A peak at GT pixel (cx, cy) hits feature index ``(round(cx/s), round(cy/s))``;
    σ on the feature grid is ``sigma_px / stride``. Background ≈ 0.
    """
    Hf = max(1, int(H // max(1, stride)))
    Wf = max(1, int(W // max(1, stride)))
    target = torch.zeros((1, Hf, Wf), device=device, dtype=dtype)
    if centers_xy.numel() == 0:
        return target
    centers_grid = centers_xy / float(stride)
    sigma_grid = (sigmas / float(stride)).clamp(min=0.7)

    # For each GT, draw Gaussian in a local window of radius ceil(3σ) around its center.
    # The Gaussian is centered at the **integer-rounded** cell (cx_int, cy_int) so
    # that the target hits exactly 1.0 at that cell — required for CornerNet/CenterNet
    # focal loss to identify the positive pixel via ``target == 1.0``.
    for i in range(centers_xy.shape[0]):
        cx = float(centers_grid[i, 0].item())
        cy = float(centers_grid[i, 1].item())
        cx_int = int(round(cx))
        cy_int = int(round(cy))
        if cx_int < 0 or cx_int >= Wf or cy_int < 0 or cy_int >= Hf:
            continue
        s = float(sigma_grid[i].item())
        rad = int(_gaussian_kernel_size(sigmas.new_tensor(s)).item())
        x0 = max(cx_int - rad, 0)
        x1 = min(cx_int + rad + 1, Wf)
        y0 = max(cy_int - rad, 0)
        y1 = min(cy_int + rad + 1, Hf)
        if x1 <= x0 or y1 <= y0:
            continue
        ys = torch.arange(y0, y1, device=device, dtype=dtype).unsqueeze(1)  # [hh,1]
        xs = torch.arange(x0, x1, device=device, dtype=dtype).unsqueeze(0)  # [1,ww]
        g = torch.exp(-((xs - cx_int) ** 2 + (ys - cy_int) ** 2) / (2.0 * s * s + 1e-9))
        target[0, y0:y1, x0:x1] = torch.maximum(target[0, y0:y1, x0:x1], g)
    return target


def _gaussian_focal_loss(
    pred_logits: torch.Tensor,    # [B,1,Hf,Wf]
    target: torch.Tensor,         # [B,1,Hf,Wf] in [0,1]
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """CornerNet/CenterNet penalty-reduced focal loss for keypoint heatmaps.

    Loss = -1/Npos * Σ [ pos: (1-p)^α log p ; neg: (1-y)^β p^α log(1-p) ].
    """
    p = pred_logits.sigmoid().clamp(min=1e-6, max=1.0 - 1e-6)
    pos_mask = target.eq(1.0).float()
    neg_mask = 1.0 - pos_mask
    pos_loss = -((1.0 - p) ** alpha) * torch.log(p) * pos_mask
    neg_loss = -((1.0 - target) ** beta) * (p ** alpha) * torch.log(1.0 - p) * neg_mask
    npos = pos_mask.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / npos


def _greedy_peak_gt_pairs(dist: torch.Tensor, radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """1:1 bipartite matching: each peak ↔ at most one GT, each GT ↔ at most one peak.

    Greedy by ascending distance among pairs with ``dist[k,j] <= radius`` (set-matching
    approximation; optimal assignment would need Hungarian on a padded cost matrix).

    Args:
        dist: ``[K, Ni]`` on any device.
    Returns:
        ``k_idx`` ``[M]``, ``j_idx`` ``[M]`` with ``M <= min(K, Ni)``.
    """
    device = dist.device
    K, Ni = dist.shape
    pairs: list[tuple[float, int, int]] = []
    valid = dist <= radius
    for k in range(K):
        for j in range(Ni):
            if bool(valid[k, j].item()):
                pairs.append((float(dist[k, j].item()), k, j))
    pairs.sort(key=lambda x: x[0])
    used_k: set[int] = set()
    used_j: set[int] = set()
    ks: list[int] = []
    js: list[int] = []
    for _, k, j in pairs:
        if k in used_k or j in used_j:
            continue
        used_k.add(k)
        used_j.add(j)
        ks.append(k)
        js.append(j)
    if not ks:
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
    return (
        torch.tensor(ks, device=device, dtype=torch.long),
        torch.tensor(js, device=device, dtype=torch.long),
    )


def _resample_polyline(pts: torch.Tensor, n: int) -> torch.Tensor:
    """Resample an (P,2) polyline to N evenly-spaced (arc-length) points.

    Falls back to repeated first point if all points coincide.
    """
    p = pts.shape[0]
    if p == 0:
        return pts.new_zeros((n, 2))
    if p == 1:
        return pts.expand(n, 2).clone()
    seg = (pts[1:] - pts[:-1]).norm(dim=-1)
    total = float(seg.sum().item())
    if total <= 1e-9:
        return pts[:1].expand(n, 2).clone()
    cum = torch.cat([pts.new_zeros((1,)), torch.cumsum(seg, dim=0)], dim=0)  # [P]
    s_targets = torch.linspace(0.0, total, steps=n, device=pts.device, dtype=pts.dtype)
    # For each target s, find segment k s.t. cum[k] <= s <= cum[k+1].
    idx = torch.searchsorted(cum, s_targets, right=False).clamp(min=1, max=p - 1)
    left = idx - 1
    s_left = cum[left]
    s_right = cum[idx]
    denom = (s_right - s_left).clamp(min=1e-9)
    t = ((s_targets - s_left) / denom).unsqueeze(-1)
    out = (1.0 - t) * pts[left] + t * pts[idx]
    return out


# --------------------------------------------------------------------------- #
# Main entry                                                                  #
# --------------------------------------------------------------------------- #
def compute_center_e2e_loss(
    pred: Dict[str, torch.Tensor],
    gt: Dict[str, torch.Tensor],
    args,
    img_h: int,
    img_w: int,
    stride: float,
    model_params=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the combined center+polyline loss.

    Args:
        pred: outputs of :class:`YolinoCenterPolyHead.forward`.
        gt:   batched dict with keys ``padded`` ``[B,NI,MP,2]``, ``inst_mask`` ``[B,NI]``,
              ``pt_mask`` ``[B,NI,MP]``, ``center_xy`` ``[B,NI,2]``, ``poly_length`` ``[B,NI]``.
        args: argparse Namespace with ``center_*`` knobs.
        img_h/img_w/stride: feature-map geometry.
        model_params: iterable of nn.Parameters to attach the DDP-safe zero stub.

    Returns:
        total_loss (scalar tensor) + scalar component dict for logging.
    """
    device = pred["center_logits"].device
    dtype = pred["center_logits"].dtype

    B, _, Hf, Wf = pred["center_logits"].shape
    K = int(getattr(args, "center_num_queries", 20))
    N = int(getattr(args, "center_num_points", 10))
    bidir = bool(getattr(args, "center_bidirectional_l1", True))

    focal_w = float(getattr(args, "center_focal_weight", 1.0))
    poly_w = float(getattr(args, "center_poly_weight", 5.0))
    aux_w = float(getattr(args, "center_aux_weight", 0.5))
    endpt_w = float(getattr(args, "center_endpoint_weight", 0.0))
    match_r = float(getattr(args, "center_match_radius_px", 24.0))
    focal_alpha = float(getattr(args, "center_focal_alpha", 2.0))
    focal_beta = float(getattr(args, "center_focal_beta", 4.0))

    # --- Build per-image target heatmap ---
    target_heat = pred["center_logits"].new_zeros((B, 1, Hf, Wf))
    centers_xy = gt.get("center_xy")  # [B, NI, 2]
    inst_mask = gt.get("inst_mask")  # [B, NI]
    poly_length = gt.get("poly_length")  # [B, NI]
    padded = gt.get("padded")  # [B, NI, MP, 2]
    pt_mask = gt.get("pt_mask")  # [B, NI, MP]

    sigma_floor = 2.0
    for b_idx in range(B):
        ok = inst_mask[b_idx].bool() if inst_mask is not None else None
        if ok is None or ok.sum() == 0:
            continue
        ctr = centers_xy[b_idx][ok]
        leng = poly_length[b_idx][ok].clamp(min=4.0)
        # σ proportional to sqrt(length) keeps small lines from blowing up the FP zone.
        sigmas = torch.clamp(0.15 * torch.sqrt(leng), min=sigma_floor, max=float(stride) * 6.0)
        target_heat[b_idx] = _build_center_heatmap_target(
            ctr, sigmas, img_h, img_w, stride, device, dtype
        )

    focal = _gaussian_focal_loss(pred["center_logits"], target_heat, alpha=focal_alpha, beta=focal_beta)

    # --- N-point polyline supervision ---
    pred_poly_last = pred["polylines_px"]  # [B,K,N,2]
    pred_poly_aux = pred.get("polylines_aux")  # [L,B,K,N,2] (incl. last)
    peaks_xy = pred["center_peaks_xy"]  # [B,K,2]

    poly_loss_last = pred_poly_last.new_zeros(())
    poly_loss_aux = pred_poly_last.new_zeros(())
    endpoint_loss = pred_poly_last.new_zeros(())
    num_matched = 0.0
    num_gt_total = 0.0
    match_mode = str(getattr(args, "center_poly_match_mode", "greedy_1to1") or "greedy_1to1").lower()

    for b_idx in range(B):
        if inst_mask is None:
            continue
        ok = inst_mask[b_idx].bool()
        if ok.sum() == 0:
            continue
        ni = int(ok.sum().item())
        num_gt_total += float(ni)
        ctr_b = centers_xy[b_idx][ok]            # [Ni,2]
        pad_b = padded[b_idx][ok]                # [Ni,MP,2]
        pm_b = pt_mask[b_idx][ok]                # [Ni,MP] bool

        # Resample each GT to N points (forward), then build reverse.
        gt_n_list = []
        for j in range(ctr_b.shape[0]):
            valid = pm_b[j].bool()
            pts = pad_b[j][valid]
            if pts.shape[0] < 2:
                gt_n_list.append(pts.new_zeros((N, 2)))
                continue
            gt_n_list.append(_resample_polyline(pts.float(), N))
        if len(gt_n_list) == 0:
            continue
        gt_fwd = torch.stack(gt_n_list, dim=0).to(device=device, dtype=dtype)  # [Ni,N,2]
        gt_rev = torch.flip(gt_fwd, dims=[1])

        d = torch.cdist(peaks_xy[b_idx].float(), ctr_b.float(), p=2)  # [K, Ni]

        if match_mode == "nearest":
            nearest_d, nearest_gt = d.min(dim=-1)
            valid_q = nearest_d <= match_r
            if valid_q.sum() == 0:
                continue
            idx_gt = nearest_gt[valid_q]
            pred_last_v = pred_poly_last[b_idx][valid_q]
            mk = torch.where(valid_q)[0]
        else:
            mk, mj = _greedy_peak_gt_pairs(d, match_r)
            if mk.numel() == 0:
                continue
            idx_gt = mj
            pred_last_v = pred_poly_last[b_idx][mk]

        tgt_fwd = gt_fwd[idx_gt]
        tgt_rev = gt_rev[idx_gt]
        l1_fwd = (pred_last_v - tgt_fwd).abs().mean(dim=(-1, -2))
        l1_rev = (pred_last_v - tgt_rev).abs().mean(dim=(-1, -2))
        if bidir:
            l1 = torch.minimum(l1_fwd, l1_rev)
            orient_fwd = l1_fwd <= l1_rev
        else:
            l1 = l1_fwd
            orient_fwd = torch.ones_like(l1_fwd, dtype=torch.bool)
        poly_loss_last = poly_loss_last + l1.sum()
        num_matched += float(mk.numel() if match_mode != "nearest" else valid_q.sum().item())

        if endpt_w > 0.0:
            tgt_end = torch.where(
                orient_fwd.unsqueeze(-1).unsqueeze(-1),
                tgt_fwd, tgt_rev
            )
            ep = pred_last_v[:, [0, -1], :] - tgt_end[:, [0, -1], :]
            endpoint_loss = endpoint_loss + ep.abs().mean(dim=(-1, -2)).sum()

        if pred_poly_aux is not None and pred_poly_aux.shape[0] > 1:
            sel = mk if match_mode != "nearest" else torch.where(valid_q)[0]
            for li in range(pred_poly_aux.shape[0] - 1):
                pred_aux_v = pred_poly_aux[li, b_idx][sel]
                l1f = (pred_aux_v - tgt_fwd).abs().mean(dim=(-1, -2))
                l1r = (pred_aux_v - tgt_rev).abs().mean(dim=(-1, -2))
                la = torch.minimum(l1f, l1r) if bidir else l1f
                poly_loss_aux = poly_loss_aux + la.sum()

    denom = max(num_matched, 1.0)
    poly_loss_last = poly_loss_last / denom
    endpoint_loss = endpoint_loss / denom
    if pred_poly_aux is not None and pred_poly_aux.shape[0] > 1:
        poly_loss_aux = poly_loss_aux / (denom * max(1, pred_poly_aux.shape[0] - 1))

    total = focal_w * focal + poly_w * poly_loss_last + aux_w * poly_loss_aux + endpt_w * endpoint_loss

    # DDP zero-stub touch of all model params (so all-reduce sees them).
    if model_params is not None:
        zero = pred["center_logits"].new_zeros(())
        for p in model_params:
            if p is not None and p.requires_grad:
                zero = zero + p.sum() * 0.0
        total = total + zero

    match_ratio = float(num_matched) / max(float(num_gt_total), 1.0)
    return total, {
        "center/focal": float(focal.detach().item()),
        "center/poly_last": float(poly_loss_last.detach().item()),
        "center/poly_aux": float(poly_loss_aux.detach().item()) if torch.is_tensor(poly_loss_aux) else float(poly_loss_aux),
        "center/endpoint": float(endpoint_loss.detach().item()),
        "center/num_matched": float(num_matched),
        "center/num_gt": float(num_gt_total),
        "center/match_ratio": match_ratio,
        "center/num_pos_cells": float(target_heat.eq(1.0).sum().item()),
    }
