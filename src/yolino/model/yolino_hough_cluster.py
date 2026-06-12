# SPDX-License-Identifier: GPL-3.0-or-later
"""Hough-space DBSCAN clustering on YOLinO local segments (``exp51`` Path A).

This module is **strictly non-differentiable** — every output is detached and
produced under :func:`torch.no_grad`. Gradients flow into the decoder only via
the small content-query ``Linear(4, 256)`` and the per-keypoint refinement
inside :class:`yolino_hough_detr_head.YolinoHoughDetrPolyHead`.

Pipeline:

1. Convert YOLinO geometry ``[B, H, W, P, V]`` to per-segment
   ``(midpoint_xy, theta, conf)`` in pixel space using
   :func:`yolino.model.e2e_polyline_modules.mid_dir_geom_to_midpoints_pixels`.
2. Threshold on ``conf > seg_conf_thresh`` and cap to ``max_segments`` per image
   (Top-K by conf).
3. Run DBSCAN on the **normalized Hough features** ``(rho / image_diag,
   theta * theta_weight / pi)`` with isotropic ``eps`` to obtain wire clusters.
4. For each valid cluster (``size >= dbscan_min_samples`` after DBSCAN), build
   a *master anchor*:

   * ``cx, cy`` = highest-confidence segment's midpoint.
   * ``theta_prior`` = ``atan2(sum(conf * sin theta), sum(conf * cos theta))``.
     ``theta`` and ``theta + pi`` are treated as the same line direction (we
     fold to ``[-pi/2, pi/2)`` before the weighted average).
   * ``L_init`` = ``max(span_norm, L_init_default * 0.3)`` where ``span_norm``
     is the cluster's along-``theta`` projected span divided by the image
     diagonal. The minimum guarantees that the 5-pt reference does not
     collapse to a single point when DBSCAN finds a single-segment cluster.

The output ``cx, cy`` are in **normalized image coordinates** ``[0, 1]``, the
same coordinate system the 5-pt reference operates in.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

from yolino.model.e2e_polyline_modules import mid_dir_geom_to_midpoints_pixels

try:
    from sklearn.cluster import DBSCAN  # type: ignore

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    DBSCAN = None  # type: ignore
    _SKLEARN_AVAILABLE = False


def _fold_theta_to_half_circle(theta: torch.Tensor) -> torch.Tensor:
    """Fold ``theta`` into ``[-pi/2, pi/2)`` (line direction, sign-invariant)."""
    t = theta
    t = torch.where(t >= math.pi / 2, t - math.pi, t)
    t = torch.where(t < -math.pi / 2, t + math.pi, t)
    return t


@torch.no_grad()
def _segments_from_geom(
    geom_act: torch.Tensor,
    stride: float,
    img_h: int,
    img_w: int,
    conf_channel: int,
    hf: int,
    wf: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Activate geometry → per-segment midpoint, theta, conf, endpoints (in pixels).

    Args:
        geom_act: ``[B, ncell, P, V]`` activated geometry (``ncell == hf*wf``).
        stride, img_h, img_w: head stride / image size.
        conf_channel: V-axis index of the confidence channel.
        hf, wf: feat-map grid size.

    Returns:
        mid_xy:  ``[B, N_all, 2]`` pixels.
        theta:   ``[B, N_all]`` segment direction in radians (folded to half-circle).
        conf:    ``[B, N_all]`` raw confidence.
        end_a:   ``[B, N_all, 2]`` pixels (one endpoint).
        end_b:   ``[B, N_all, 2]`` pixels (other endpoint).
    """
    b, ncell, p, v = geom_act.shape
    if ncell != hf * wf:
        raise ValueError(
            "yolino_hough_cluster expects ncell == Hf*Wf; got ncell=%d, Hf*Wf=%d"
            % (ncell, hf * wf)
        )
    g = geom_act.view(b, hf, wf, p, v)
    mid_px, end_a, end_b = mid_dir_geom_to_midpoints_pixels(
        g, stride=stride, img_h=img_h, img_w=img_w
    )
    dir_vec = end_b - end_a  # [B, hf, wf, p, 2]
    # atan2(dy, dx) → segment direction in image space.
    theta = torch.atan2(dir_vec[..., 1], dir_vec[..., 0])
    theta = _fold_theta_to_half_circle(theta)
    conf = g[..., conf_channel]

    n_all = hf * wf * p
    mid_xy_flat = mid_px.reshape(b, n_all, 2)
    theta_flat = theta.reshape(b, n_all)
    conf_flat = conf.reshape(b, n_all)
    end_a_flat = end_a.reshape(b, n_all, 2)
    end_b_flat = end_b.reshape(b, n_all, 2)
    return mid_xy_flat, theta_flat, conf_flat, end_a_flat, end_b_flat


@torch.no_grad()
def segments_to_hough_anchors(
    geom_act: torch.Tensor,
    stride: float,
    img_h: int,
    img_w: int,
    conf_channel: int,
    hf: int,
    wf: int,
    num_anchors: int,
    conf_thresh: float = 0.3,
    max_segments: int = 512,
    dbscan_eps: float = 0.05,
    dbscan_min_samples: int = 2,
    rho_weight: float = 1.0,
    theta_weight: float = 1.0,
    L_init_default: float = 0.3,
) -> dict:
    """Cluster segments in normalized Hough space and produce master anchors.

    All outputs are **detached** (``.detach()``) and live on the same device /
    dtype as ``geom_act``.

    Args:
        geom_act: ``[B, ncell, P, V]`` activated geometry (caller should pass
                  ``geom_act.detach()`` — we additionally call ``.detach()``).
        stride, img_h, img_w, conf_channel, hf, wf: see :func:`_segments_from_geom`.
        num_anchors: ``K`` slots to pad / truncate to.
        conf_thresh: only segments with conf above this are considered.
        max_segments: per-image segment cap before DBSCAN (Top-K by conf).
        dbscan_eps, dbscan_min_samples: DBSCAN parameters in the normalized
            Hough space.
        rho_weight, theta_weight: feature weights inside the Hough vector. The
            default of ``1.0`` treats both axes equally after normalization.
        L_init_default: fallback / lower-bound on the normalized cluster span.

    Returns:
        Dict with keys (all detached, no grad):

        * ``cx``, ``cy``    : ``[B, K]`` normalized in ``[0, 1]``.
        * ``theta``         : ``[B, K]`` radians (folded to half-circle).
        * ``L_init``        : ``[B, K]`` normalized in ``[0, 1]``.
        * ``valid``         : ``[B, K]`` bool — True for filled slots.
        * ``mid_xy_seg``    : ``[B, N_all, 2]`` pixels (debug / TB overlay).
        * ``theta_seg``     : ``[B, N_all]``  radians (debug).
        * ``conf_seg``      : ``[B, N_all]``  raw conf (debug).
        * ``end_a_seg``     : ``[B, N_all, 2]`` pixels.
        * ``end_b_seg``     : ``[B, N_all, 2]`` pixels.

    Notes:
        * **Non-differentiable**: DBSCAN runs in NumPy. ``cx, cy, theta,
          L_init, valid`` are all detached returns.
        * When sklearn is unavailable the function falls back to a Top-K
          confidence anchor (each segment becomes its own "cluster"). This
          keeps the head functional on systems without scikit-learn.
    """
    geom_act = geom_act.detach()
    b = int(geom_act.shape[0])
    device = geom_act.device
    dtype = geom_act.dtype

    mid_xy_seg, theta_seg, conf_seg, end_a_seg, end_b_seg = _segments_from_geom(
        geom_act, stride, img_h, img_w, conf_channel, hf, wf
    )

    cx_out = torch.zeros((b, num_anchors), device=device, dtype=dtype)
    cy_out = torch.zeros_like(cx_out)
    theta_out = torch.zeros_like(cx_out)
    L_init_out = torch.full_like(cx_out, float(L_init_default))
    valid_out = torch.zeros((b, num_anchors), device=device, dtype=torch.bool)

    img_w_f = max(float(img_w - 1), 1.0)
    img_h_f = max(float(img_h - 1), 1.0)
    img_diag = float(math.hypot(img_w_f, img_h_f))
    L_init_floor = float(L_init_default) * 0.3

    for bi in range(b):
        conf_b = conf_seg[bi]
        mid_b = mid_xy_seg[bi]
        theta_b = theta_seg[bi]

        # 1. Threshold + Top-K to limit DBSCAN cost.
        keep_mask = conf_b > float(conf_thresh)
        n_keep = int(keep_mask.sum().item())
        if n_keep == 0:
            continue
        kept_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
        if int(kept_idx.numel()) > int(max_segments):
            top_conf, top_local = torch.topk(conf_b[kept_idx], k=int(max_segments))
            kept_idx = kept_idx[top_local]

        seg_xy = mid_b[kept_idx]            # [Nk, 2] pixels
        seg_theta = theta_b[kept_idx]       # [Nk] rad
        seg_conf = conf_b[kept_idx]         # [Nk]

        n_kept = int(kept_idx.numel())
        if n_kept == 0:
            continue

        # 2. Normalized Hough features.
        # ``seg_theta`` is the segment **direction**. The Hough parametrisation
        # uses the line's **normal**: ``theta_n = seg_theta + pi/2`` (mod pi). All
        # collinear segments share the same ``(rho, theta_n)`` regardless of
        # along-line position.
        theta_n = _fold_theta_to_half_circle(seg_theta + math.pi / 2.0)
        rho_px = seg_xy[:, 0] * torch.cos(theta_n) + seg_xy[:, 1] * torch.sin(theta_n)
        rho_norm = rho_px / max(img_diag, 1.0)
        theta_norm = theta_n / (math.pi / 2.0)  # half-circle → [-1, 1]

        feats_np = (
            torch.stack(
                [rho_norm * float(rho_weight), theta_norm * float(theta_weight)],
                dim=-1,
            )
            .float()
            .cpu()
            .numpy()
        )

        # 3. DBSCAN (or fallback).
        if _SKLEARN_AVAILABLE and DBSCAN is not None:
            labels = DBSCAN(
                eps=float(dbscan_eps),
                min_samples=int(dbscan_min_samples),
            ).fit_predict(feats_np)
        else:
            # Fallback: every kept segment is its own cluster (Top-K conf, no clustering).
            labels = np.arange(n_kept, dtype=np.int64)

        unique_labels = sorted(int(lbl) for lbl in set(labels.tolist()) if int(lbl) >= 0)
        if not unique_labels:
            continue

        # 4. Rank clusters by total confidence so the most reliable wires
        #    end up as anchors 0..K-1 (helps Hungarian initially).
        cluster_confs: list[tuple[float, int]] = []
        for lbl in unique_labels:
            mask_lbl = labels == lbl
            cluster_confs.append((float(seg_conf[mask_lbl].sum().item()), lbl))
        cluster_confs.sort(key=lambda x: x[0], reverse=True)
        ranked_labels = [lbl for _, lbl in cluster_confs][:num_anchors]

        for slot, lbl in enumerate(ranked_labels):
            mask_lbl = torch.as_tensor(labels == lbl, dtype=torch.bool, device=device)
            xy_c = seg_xy[mask_lbl]
            theta_c = seg_theta[mask_lbl]
            conf_c = seg_conf[mask_lbl]

            # cx, cy from the highest-conf segment (anchor seed).
            j_best = int(torch.argmax(conf_c).item())
            cx_px = float(xy_c[j_best, 0].item())
            cy_px = float(xy_c[j_best, 1].item())

            # theta_prior: confidence-weighted circular mean on the half-circle.
            sin_avg = float((conf_c * torch.sin(2.0 * theta_c)).sum().item())
            cos_avg = float((conf_c * torch.cos(2.0 * theta_c)).sum().item())
            if abs(sin_avg) < 1e-9 and abs(cos_avg) < 1e-9:
                theta_prior = float(theta_c[j_best].item())
            else:
                theta_prior = 0.5 * math.atan2(sin_avg, cos_avg)

            # L_init: along-theta_prior projected span / image diag, lower-bounded.
            if int(xy_c.shape[0]) >= 1:
                # Use endpoints for a tighter span than midpoints alone.
                end_a_c = end_a_seg[bi][kept_idx][mask_lbl]
                end_b_c = end_b_seg[bi][kept_idx][mask_lbl]
                all_pts = torch.cat([xy_c, end_a_c, end_b_c], dim=0)
                rel = all_pts - all_pts.new_tensor([cx_px, cy_px])
                proj = rel[:, 0] * math.cos(theta_prior) + rel[:, 1] * math.sin(theta_prior)
                span_px = float((proj.max() - proj.min()).item())
            else:
                span_px = 0.0
            span_norm = span_px / max(img_diag, 1.0)
            L_init = max(span_norm, L_init_floor)
            L_init = min(L_init, 1.0)

            cx_out[bi, slot] = cx_px / img_w_f
            cy_out[bi, slot] = cy_px / img_h_f
            theta_out[bi, slot] = float(theta_prior)
            L_init_out[bi, slot] = float(L_init)
            valid_out[bi, slot] = True

    return {
        "cx": cx_out.detach().clamp(0.0, 1.0),
        "cy": cy_out.detach().clamp(0.0, 1.0),
        "theta": theta_out.detach(),
        "L_init": L_init_out.detach().clamp(min=0.0, max=1.0),
        "valid": valid_out.detach(),
        "mid_xy_seg": mid_xy_seg.detach(),
        "theta_seg": theta_seg.detach(),
        "conf_seg": conf_seg.detach(),
        "end_a_seg": end_a_seg.detach(),
        "end_b_seg": end_b_seg.detach(),
    }


def anchor_to_5pt_ref(
    cx: torch.Tensor,
    cy: torch.Tensor,
    theta: torch.Tensor,
    L_init: torch.Tensor,
) -> torch.Tensor:
    """Convert master anchors to a 5-point soft reference in normalized image coords.

    Args:
        cx, cy, theta, L_init: each ``[B, K]``. ``cx, cy, L_init`` are normalized
            into ``[0, 1]``; ``theta`` is in radians.

    Returns:
        ``[B, K, 5, 2]`` normalized ``(x, y)`` reference points, clamped to
        ``[0, 1]``, **detached** (no grad — these are anchor seeds, the decoder
        learns deltas around them).

    Formula:
        P_i = anchor + i * (L_init / 4) * (cos theta, sin theta)  for i in [-2, -1, 0, 1, 2]
        → P_3 (index 2) is the anchor center; P_1 and P_5 are the endpoints.
    """
    if cx.shape != cy.shape or cx.shape != theta.shape or cx.shape != L_init.shape:
        raise ValueError(
            "cx, cy, theta, L_init must share shape; got cx=%s, cy=%s, theta=%s, L_init=%s"
            % (tuple(cx.shape), tuple(cy.shape), tuple(theta.shape), tuple(L_init.shape))
        )
    if cx.dim() != 2:
        raise ValueError("Expected [B, K] inputs, got %s" % (tuple(cx.shape),))
    delta = L_init / 4.0                                            # [B, K]
    ts = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=cx.device, dtype=cx.dtype)
    offs = ts.view(1, 1, 5) * delta.unsqueeze(-1)                   # [B, K, 5]
    dx = torch.cos(theta).unsqueeze(-1) * offs                      # [B, K, 5]
    dy = torch.sin(theta).unsqueeze(-1) * offs
    x = (cx.unsqueeze(-1) + dx).clamp(0.0, 1.0)
    y = (cy.unsqueeze(-1) + dy).clamp(0.0, 1.0)
    return torch.stack([x, y], dim=-1).detach()                     # [B, K, 5, 2]
