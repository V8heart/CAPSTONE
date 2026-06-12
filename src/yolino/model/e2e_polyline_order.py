# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic polyline ordering for the YOLinO-DETR Hybrid 5-pt head (``exp51``).

The TTPLA dataset stores polylines as variable-length vertex lists. After
data-augmentation (rotation / crop), the original left-right ordering applied
at load time (``__get_labels__``) can be reversed for near-vertical lines. The
Hungarian matching in :func:`compute_detr_e2e_loss` does **not** account for
direction, so an unstable order biases the L1 endpoint cost and the 5-pt
decoder cannot stabilise its query order.

``canonicalize_polyline_xy`` enforces a deterministic **traversal direction**
on a polyline's vertices (in pixel ``(x, y)``) by **reversing** the sequence
when endpoints point the wrong way. It does **not** reorder interior vertices
(e.g. by sorting on ``x``), which would destroy arc-length structure on dense
GT polylines and collapse DN / 5-pt resampling to short zig-zags.

Convention:

* **mostly-horizontal** (``|dx_span| >= |dy_span|``): ``P1`` is the path start
  with smaller ``x`` than the path end (left → right).
* **mostly-vertical** (``|dy_span| > |dx_span|`` and narrow ``x`` range):
  ``P1`` has smaller ``y`` than the path end (top → bottom).

The vertical exception uses ``vertical_angle_deg`` (default 80°): the polyline
is treated as vertical when ``|dx_span|/total_span <= cos(vertical_angle_deg)``.

Downstream ``resample_polyline_xy(..., num_out=5)`` must run on path-ordered
vertices (before or after a flip-only canonicalize); arc-length is preserved.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


def _vertical_threshold(vertical_angle_deg: float) -> float:
    """Threshold on ``|dx_span|/total_span`` below which we treat the polyline as vertical.

    With ``vertical_angle_deg = 80`` we require the span direction to be within 10°
    of the vertical axis, i.e. ``|dx_span|/total_span <= cos(80°) ≈ 0.174``.
    """
    return math.cos(math.radians(float(vertical_angle_deg)))


def _is_vertical_polyline(
    pts_xy: torch.Tensor,
    vertical_angle_deg: float,
) -> bool:
    x = pts_xy[:, 0]
    y = pts_xy[:, 1]
    dx_span = (x.max() - x.min()).abs()
    dy_span = (y.max() - y.min()).abs()
    total = (dx_span * dx_span + dy_span * dy_span).clamp(min=1e-12).sqrt()
    cos_thr = _vertical_threshold(vertical_angle_deg)
    return bool((dy_span >= dx_span).item()) and bool((dx_span <= cos_thr * total).item())


def canonicalize_polyline_xy(
    pts_xy: torch.Tensor,
    vertical_angle_deg: float = 80.0,
) -> torch.Tensor:
    """Flip traversal direction if endpoints violate left→right / top→bottom order.

    Args:
        pts_xy: ``[T, 2]`` polyline vertices in pixel ``(x, y)`` along the wire path
                (not sorted by coordinate). ``T >= 2`` required for a flip.
        vertical_angle_deg: see module docstring.

    Returns:
        ``[T, 2]`` same vertices, possibly reversed along the path.
    """
    if pts_xy.dim() != 2 or int(pts_xy.shape[-1]) != 2:
        raise ValueError("pts_xy must have shape [T, 2]; got %s" % (tuple(pts_xy.shape),))
    t = int(pts_xy.shape[0])
    if t < 2:
        return pts_xy

    if _is_vertical_polyline(pts_xy, vertical_angle_deg):
        # Top → bottom: path start should have smaller y than path end.
        if float(pts_xy[0, 1].item()) > float(pts_xy[-1, 1].item()):
            return pts_xy.flip(0)
    else:
        # Left → right: path start should have smaller x than path end.
        if float(pts_xy[0, 0].item()) > float(pts_xy[-1, 0].item()):
            return pts_xy.flip(0)
    return pts_xy


def canonicalize_polyline_yx(
    pts_yx: torch.Tensor,
    vertical_angle_deg: float = 80.0,
) -> torch.Tensor:
    """Same as :func:`canonicalize_polyline_xy` but the input/output are in ``(y, x)`` order.

    Convenience wrapper for callers that work in the ``(row, col)`` convention
    used by ``TTPLADataset.__get_labels__``.
    """
    if pts_yx.dim() != 2 or int(pts_yx.shape[-1]) != 2:
        raise ValueError("pts_yx must have shape [T, 2]; got %s" % (tuple(pts_yx.shape),))
    xy = torch.stack([pts_yx[:, 1], pts_yx[:, 0]], dim=-1)
    xy_can = canonicalize_polyline_xy(xy, vertical_angle_deg=vertical_angle_deg)
    return torch.stack([xy_can[:, 1], xy_can[:, 0]], dim=-1)


def canonicalize_pack_inplace(
    padded: torch.Tensor,
    pt_mask: torch.Tensor,
    inst_mask: torch.Tensor,
    vertical_angle_deg: float = 80.0,
) -> None:
    """In-place canonicalization of a ``_build_e2e_gt_pack`` tensor block.

    Args:
        padded:    ``[Ni, Mp, 2]`` (x, y) in pixels (single image).
        pt_mask:   ``[Ni, Mp]`` bool — True for valid vertex slots.
        inst_mask: ``[Ni]`` bool — True for valid instance slots.
        vertical_angle_deg: forwarded to :func:`canonicalize_polyline_xy`.

    Padding slots (``pt_mask = False``) are left untouched (stay zero).
    """
    if padded.dim() != 3 or int(padded.shape[-1]) != 2:
        raise ValueError("padded must be [Ni, Mp, 2]; got %s" % (tuple(padded.shape),))
    ni = int(padded.shape[0])
    for k in range(ni):
        if not bool(inst_mask[k].item()):
            continue
        npt = int(pt_mask[k].sum().item())
        if npt < 2:
            continue
        reordered = canonicalize_polyline_xy(padded[k, :npt], vertical_angle_deg=vertical_angle_deg)
        padded[k, :npt] = reordered
