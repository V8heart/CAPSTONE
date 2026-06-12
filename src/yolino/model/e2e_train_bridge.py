# SPDX-License-Identifier: GPL-3.0-or-later
"""E2E training loss: Top-M token selection, Hungarian matching, Chamfer (optional normalized coords)."""
from __future__ import annotations

import math
from typing import Dict

import torch




def bernstein_basis_rows_at_u(degree: int, u: torch.Tensor) -> torch.Tensor:
    """Bernstein basis B_i^degree(u) for each row in u. Returns [N, degree+1]."""
    n = int(degree)
    t = u.reshape(-1).clamp(0.0, 1.0)
    cols = []
    for i in range(n + 1):
        c = math.comb(n, i)
        cols.append(c * ((1.0 - t) ** (n - i)) * (t**i))
    return torch.stack(cols, dim=1)


def bezier_least_squares_ctrl_xy(
    pts: torch.Tensor,
    pt_valid: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """
    Fit degree-`degree` Bézier control points (K=degree+1) to valid vertices in x,y by uniform u and least squares.
    Returns [K, 2] on the same device/dtype as ``pts``.
    """
    k = int(degree) + 1
    v = pts[pt_valid]
    n = int(v.shape[0])
    if n < 2:
        return pts.new_zeros((k, 2))
    u = torch.linspace(0.0, 1.0, n, device=pts.device, dtype=pts.dtype)
    b = bernstein_basis_rows_at_u(int(degree), u)
    if n >= k:
        sol = torch.linalg.lstsq(b, v, rcond=1e-5).solution
        return sol
    ridge = 1e-4 * torch.eye(k, device=pts.device, dtype=pts.dtype)
    bt = b.T
    ctrl = torch.linalg.solve(bt @ b + ridge, bt @ v)
    return ctrl


def resample_polyline_xy(
    pts: torch.Tensor,
    pt_valid: torch.Tensor,
    num_out: int,
) -> torch.Tensor:
    """
    pts: [T, 2] pixel x,y; pt_valid: [T] bool — valid vertices in order.
    Returns [num_out, 2] along cumulative arc length (differentiable).
    """
    if int(pt_valid.sum()) < 2:
        return pts.new_zeros((num_out, 2))
    v = pts[pt_valid]
    seg = torch.diff(v, dim=0)
    dist = torch.norm(seg, p=2, dim=1)
    cum = torch.cat([v.new_zeros((1,)), torch.cumsum(dist, dim=0)])
    total = cum[-1].clamp(min=1e-6)
    targets = torch.linspace(0.0, 1.0, int(num_out), device=pts.device, dtype=pts.dtype) * total
    out = []
    for ti in range(int(num_out)):
        t = targets[ti]
        idx = torch.searchsorted(cum, t, right=True) - 1
        idx = idx.clamp(0, len(v) - 2)
        t0, t1 = cum[idx], cum[idx + 1]
        alpha = ((t - t0) / (t1 - t0).clamp(min=1e-6)).clamp(0.0, 1.0)
        out.append((1.0 - alpha) * v[idx] + alpha * v[idx + 1])
    return torch.stack(out, dim=0)
