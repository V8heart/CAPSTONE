# SPDX-License-Identifier: GPL-3.0-or-later
"""Denoising Training (DN / LCDN) helpers for ``YolinoHoughDetrPolyHead`` (exp51).

A DN branch feeds *noisy GT* into the decoder alongside the Hough matching
queries. Each DN slot has a fixed i↔i correspondence with a GT polyline, so
its loss does **not** require Hungarian matching — direct L1 against the
clean GT 5-pt teaches the decoder to denoise.

This module provides:

* :func:`build_simple_dn`  — Gaussian Δxy on the 5-pt vertices (Phase 3).
* :func:`build_lcdn`       — anchor-parameter jitter (length / rotation /
                              center) then re-reconstruction via
                              :func:`anchor_to_5pt_ref` (Phase 4).
* :func:`gt_5pt_anchor_params` — reduce GT 5-pt to ``(cx, cy, theta, L)`` so DN
                              content queries can be embedded through the same
                              ``Linear(4, D)`` used by Path A.
* :func:`build_dn_self_attn_mask` — block-diagonal self-attn mask that
                              isolates the matching block from the DN groups
                              (and DN groups from each other).

All functions are deterministic given a ``generator`` argument; the default
is a fresh CPU ``torch.Generator`` seeded from PyTorch's global RNG.
"""
from __future__ import annotations

from typing import Optional, Tuple

import math
import torch

from yolino.model.yolino_hough_cluster import anchor_to_5pt_ref


def gt_5pt_anchor_params(
    gt_5pts_norm: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce a 5-pt polyline to its anchor parameters ``(cx, cy, theta, L)``.

    Args:
        gt_5pts_norm: ``[..., 5, 2]`` 5-pt polyline in normalized ``[0, 1]``
            ``(x, y)``. The vertices are assumed canonical (left→right / top→
            bottom) — see :mod:`yolino.model.e2e_polyline_order`.

    Returns:
        cx, cy: ``[...]`` arithmetic mean of the 5 vertices.
        theta : ``[...]`` direction from ``P1`` (endpoint) to ``P5``
                (other endpoint), in radians.
        L     : ``[...]`` chord length ``||P5 - P1||`` (normalized).
    """
    if int(gt_5pts_norm.shape[-2]) != 5 or int(gt_5pts_norm.shape[-1]) != 2:
        raise ValueError(
            "gt_5pts_norm must have shape [..., 5, 2]; got %s" % (tuple(gt_5pts_norm.shape),)
        )
    cx = gt_5pts_norm[..., 0].mean(dim=-1)
    cy = gt_5pts_norm[..., 1].mean(dim=-1)
    p1 = gt_5pts_norm[..., 0, :]
    p5 = gt_5pts_norm[..., 4, :]
    dx = p5[..., 0] - p1[..., 0]
    dy = p5[..., 1] - p1[..., 1]
    theta = torch.atan2(dy, dx)
    L = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-6)
    return cx, cy, theta, L


def _repeat_groups(
    x: torch.Tensor,
    num_groups: int,
    dim: int,
) -> torch.Tensor:
    """``repeat_interleave``-like helper that broadcasts a ``[B, N, ...]`` tensor
    to ``[B, N * G, ...]`` by repeating each instance ``G`` times (consecutively).
    """
    return x.repeat_interleave(int(num_groups), dim=dim)


def build_simple_dn(
    gt_5pts_norm: torch.Tensor,
    inst_mask: torch.Tensor,
    num_groups: int = 3,
    sigma_xy: float = 0.05,
    generator: Optional[torch.Generator] = None,
) -> dict:
    """Simple Gaussian xy noise on the canonical GT 5-pt polylines.

    Args:
        gt_5pts_norm: ``[B, N, 5, 2]`` GT 5-pt in normalized ``[0, 1]`` ``(x, y)``.
        inst_mask:    ``[B, N]`` bool — True for valid GT instance slots.
        num_groups:   ``G`` — number of DN copies per GT polyline (DN slots).
        sigma_xy:     stddev of additive Gaussian noise (normalized units).
        generator:    optional torch.Generator for reproducible noise.

    Returns:
        Dict with keys:
          ``refs``    : ``[B, N*G, 5, 2]`` noisy 5-pt refs (normalized, clamped to ``[0, 1]``).
          ``targets`` : ``[B, N*G, 5, 2]`` clean GT 5-pt (same i↔i correspondence as refs).
          ``valid``   : ``[B, N*G]``      bool — repeats ``inst_mask`` per group.
          ``n_slots`` : ``int``           ``N * G``.
    """
    if gt_5pts_norm.dim() != 4 or gt_5pts_norm.shape[-2:] != (5, 2):
        raise ValueError(
            "gt_5pts_norm must be [B, N, 5, 2]; got %s" % (tuple(gt_5pts_norm.shape),)
        )
    b, n_inst, _, _ = gt_5pts_norm.shape
    g = int(num_groups)
    if g <= 0:
        return {
            "refs":    gt_5pts_norm.new_zeros((b, 0, 5, 2)),
            "targets": gt_5pts_norm.new_zeros((b, 0, 5, 2)),
            "valid":   torch.zeros((b, 0), dtype=torch.bool, device=gt_5pts_norm.device),
            "n_slots": 0,
        }
    refs = _repeat_groups(gt_5pts_norm, g, dim=1)        # [B, N*G, 5, 2]
    targets = refs.clone()
    valid = _repeat_groups(inst_mask, g, dim=1)          # [B, N*G]

    noise = torch.empty_like(refs)
    if generator is not None:
        noise.normal_(mean=0.0, std=float(sigma_xy), generator=generator)
    else:
        noise.normal_(mean=0.0, std=float(sigma_xy))
    # Mask noise on invalid slots so refs stay at zero there.
    valid_xy = valid.unsqueeze(-1).unsqueeze(-1).to(noise.dtype)
    refs = (refs + noise * valid_xy).clamp(0.0, 1.0)
    # Padding rows stay zero (they were zero in gt_5pts_norm).
    refs = refs * valid_xy
    targets = targets * valid_xy
    return {
        "refs": refs.detach(),
        "targets": targets.detach(),
        "valid": valid.detach(),
        "n_slots": int(b * 0 + n_inst * g),  # per-image slot count
    }


def build_lcdn(
    gt_5pts_norm: torch.Tensor,
    inst_mask: torch.Tensor,
    num_groups: int = 3,
    sigma_xy: float = 0.02,
    scale_range: float = 0.2,
    rot_deg: float = 10.0,
    L_init_floor: float = 0.05,
    generator: Optional[torch.Generator] = None,
) -> dict:
    """Length-scale + rotation jitter DN (DT-LSD style).

    1. Reduce GT to anchor params ``(cx, cy, theta, L)``.
    2. Jitter:
       * ``cx, cy`` += ``N(0, sigma_xy)``
       * ``L``       *= ``1 + U(-scale_range, scale_range)``
       * ``theta``   += ``U(-rot_deg, rot_deg) * pi / 180``
    3. Reconstruct 5-pt via :func:`anchor_to_5pt_ref` for stable scale/rotation
       perturbation (vs. per-vertex Gaussian).

    Targets are the **original (clean) GT 5-pt** so the head learns to undo the
    LCDN jitter.

    Args:
        gt_5pts_norm: ``[B, N, 5, 2]`` GT 5-pt in normalized ``[0, 1]``.
        inst_mask:    ``[B, N]`` bool.
        num_groups, sigma_xy, scale_range, rot_deg: jitter ranges.
        L_init_floor: clamp on jittered L (prevents L≈0).
        generator: optional Generator for reproducible jitter.

    Returns:
        Same schema as :func:`build_simple_dn`.
    """
    if gt_5pts_norm.dim() != 4 or gt_5pts_norm.shape[-2:] != (5, 2):
        raise ValueError(
            "gt_5pts_norm must be [B, N, 5, 2]; got %s" % (tuple(gt_5pts_norm.shape),)
        )
    b, n_inst, _, _ = gt_5pts_norm.shape
    g = int(num_groups)
    if g <= 0:
        return {
            "refs":    gt_5pts_norm.new_zeros((b, 0, 5, 2)),
            "targets": gt_5pts_norm.new_zeros((b, 0, 5, 2)),
            "valid":   torch.zeros((b, 0), dtype=torch.bool, device=gt_5pts_norm.device),
            "n_slots": 0,
        }

    cx, cy, theta, L = gt_5pts_norm_to_anchor_grouped(gt_5pts_norm, g)
    # Match the layout for the LCDN reconstruction.
    valid = _repeat_groups(inst_mask, g, dim=1)  # [B, N*G]

    def _gen(shape, lo: float, hi: float) -> torch.Tensor:
        u = torch.empty(shape, device=gt_5pts_norm.device, dtype=gt_5pts_norm.dtype)
        if generator is not None:
            u.uniform_(lo, hi, generator=generator)
        else:
            u.uniform_(lo, hi)
        return u

    def _gen_normal(shape, std: float) -> torch.Tensor:
        e = torch.empty(shape, device=gt_5pts_norm.device, dtype=gt_5pts_norm.dtype)
        if generator is not None:
            e.normal_(mean=0.0, std=float(std), generator=generator)
        else:
            e.normal_(mean=0.0, std=float(std))
        return e

    dxy_x = _gen_normal(cx.shape, sigma_xy)
    dxy_y = _gen_normal(cy.shape, sigma_xy)
    scale = 1.0 + _gen(L.shape, -float(scale_range), float(scale_range))
    rot = _gen(theta.shape, -float(rot_deg), float(rot_deg)) * (math.pi / 180.0)

    cx_j = (cx + dxy_x).clamp(0.0, 1.0)
    cy_j = (cy + dxy_y).clamp(0.0, 1.0)
    L_j = (L * scale).clamp(min=float(L_init_floor), max=1.0)
    theta_j = theta + rot

    refs = anchor_to_5pt_ref(cx_j, cy_j, theta_j, L_j)  # [B, N*G, 5, 2]
    targets = _repeat_groups(gt_5pts_norm, g, dim=1)
    valid_xy = valid.unsqueeze(-1).unsqueeze(-1).to(refs.dtype)
    refs = refs * valid_xy
    targets = targets * valid_xy
    return {
        "refs": refs.detach(),
        "targets": targets.detach(),
        "valid": valid.detach(),
        "n_slots": int(n_inst * g),
    }


def gt_5pts_norm_to_anchor_grouped(
    gt_5pts_norm: torch.Tensor,
    num_groups: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repeat-interleave the GT 5-pt anchor params per group.

    Args:
        gt_5pts_norm: ``[B, N, 5, 2]``.
        num_groups: ``G``.

    Returns:
        cx, cy, theta, L: each ``[B, N*G]``.
    """
    cx_n, cy_n, theta_n, L_n = gt_5pt_anchor_params(gt_5pts_norm)  # all [B, N]
    cx_g = _repeat_groups(cx_n, num_groups, dim=1)
    cy_g = _repeat_groups(cy_n, num_groups, dim=1)
    theta_g = _repeat_groups(theta_n, num_groups, dim=1)
    L_g = _repeat_groups(L_n, num_groups, dim=1)
    return cx_g, cy_g, theta_g, L_g


def build_dn_self_attn_mask(
    n_matching: int,
    dn_group_sizes: Tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """Block-diagonal mask for self-attention over ``[matching | DN_1 | DN_2 | ... | DN_G]``.

    Returns:
        ``[K_total, K_total]`` bool mask where True means **masked** (blocked).
        ``False`` on the block-diagonal; ``True`` everywhere else.
        ``K_total = n_matching + sum(dn_group_sizes)``.

    The convention matches :class:`torch.nn.MultiheadAttention` ``attn_mask``
    semantics (``True``/``-inf`` blocks the attention).
    """
    sizes = [int(n_matching)] + [int(s) for s in dn_group_sizes]
    k_total = sum(sizes)
    mask = torch.ones((k_total, k_total), dtype=torch.bool, device=device)
    cursor = 0
    for s in sizes:
        if s <= 0:
            continue
        mask[cursor : cursor + s, cursor : cursor + s] = False
        cursor += s
    return mask
