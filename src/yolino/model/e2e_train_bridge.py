# SPDX-License-Identifier: GPL-3.0-or-later
"""E2E training loss: Top-M token selection, Hungarian matching, Chamfer (optional normalized coords)."""
from __future__ import annotations

import math
from typing import Dict

import torch

from yolino.model.e2e_polyline_modules import bernstein_matrix
from yolino.utils.enums import Variables

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


def _effective_e2e_lambda(epoch: int, args) -> float:
    w = float(getattr(args, "e2e_loss_weight", 0.0) or 0.0)
    if w <= 0:
        return 0.0
    warm = int(getattr(args, "e2e_warmup_epochs", 0) or 0)
    start = int(getattr(args, "e2e_warmup_start_epoch", 0) or 0)
    if warm <= 0:
        return w if epoch >= start else 0.0
    if epoch < start:
        return 0.0
    t = min(1.0, float(epoch - start + 1) / float(warm))
    return w * t


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


def compute_e2e_loss(
    e2e_out: Dict[str, torch.Tensor],
    geom_act: torch.Tensor,
    coords,
    e2e_gt_pack: Dict[str, torch.Tensor],
    img_h: int,
    img_w: int,
    epoch: int,
    args,
) -> torch.Tensor:
    """
    Top-M conf tokens vs GT instances; scipy Hungarian; mean Chamfer on matched pairs.
    """
    if linear_sum_assignment is None:
        raise ImportError("scipy is required for E2E Hungarian.")
    lam = _effective_e2e_lambda(epoch, args)
    if lam <= 0:
        return e2e_out["bezier_curve_px"].sum() * 0.0

    conf_idx = int(coords.get_position_within_prediction(Variables.CONF)[0])
    top_m = int(getattr(args, "e2e_hungarian_top_m", 128))
    t_res = int(getattr(args, "e2e_gt_resample_t", 32))
    gt_target = str(getattr(args, "e2e_gt_chamfer_target", "arc_length") or "arc_length").lower()
    bezier_deg = int(getattr(args, "e2e_bezier_degree", 3))
    normalize = bool(getattr(args, "e2e_normalize_coords", True))
    if normalize:
        scale = torch.tensor(
            [max(float(img_w - 1), 1.0), max(float(img_h - 1), 1.0)],
            device=geom_act.device,
            dtype=geom_act.dtype,
        )
    else:
        scale = torch.ones(2, device=geom_act.device, dtype=geom_act.dtype)

    curves = e2e_out["bezier_curve_px"]
    b, _, _, _ = curves.shape
    loss_acc = curves.new_zeros(())
    n_pairs = 0

    padded = e2e_gt_pack["padded"]
    inst_m = e2e_gt_pack["inst_mask"]
    pt_m = e2e_gt_pack["pt_mask"]

    for bi in range(b):
        conf_flat = geom_act[bi].reshape(-1, geom_act.shape[-1])[:, conf_idx]
        m_take = min(top_m, int(conf_flat.numel()))
        if m_take < 1:
            continue
        _, topi = torch.topk(conf_flat, m_take)
        pred_c = curves[bi, topi]

        inst_b = inst_m[bi]
        n_gt = int(inst_b.sum().item())
        if n_gt == 0:
            continue
        gt_idx = torch.where(inst_b)[0][:n_gt]
        gt_curves = []
        bern_uniform = bernstein_matrix(bezier_deg, t_res, padded.device, padded.dtype)
        for j in range(n_gt):
            gj = int(gt_idx[j])
            if gt_target == "bezier_ls":
                ctrl = bezier_least_squares_ctrl_xy(padded[bi, gj], pt_m[bi, gj], bezier_deg)
                gt_curves.append(torch.einsum("tk,kd->td", bern_uniform, ctrl))
            else:
                gt_curves.append(resample_polyline_xy(padded[bi, gj], pt_m[bi, gj], t_res))
        gt_stack = torch.stack(gt_curves, dim=0)

        mp, n = pred_c.shape[0], gt_stack.shape[0]
        scl = scale.view(1, 1, 1, 2)
        pred_e = (pred_c.unsqueeze(1) / scl).expand(mp, n, t_res, 2).reshape(mp * n, t_res, 2)
        gt_e = (gt_stack.unsqueeze(0) / scl).expand(mp, n, t_res, 2).reshape(mp * n, t_res, 2)
        d = torch.cdist(pred_e, gt_e, p=1.0)
        part_a = d.min(dim=2)[0].mean(dim=1)
        part_b = d.min(dim=1)[0].mean(dim=1)
        cost_t = (part_a + part_b).reshape(mp, n)
        ri, ci = linear_sum_assignment(cost_t.detach().float().cpu().numpy())
        if len(ri) == 0:
            continue
        r = torch.as_tensor(ri, device=cost_t.device, dtype=torch.long)
        c = torch.as_tensor(ci, device=cost_t.device, dtype=torch.long)
        loss_acc = loss_acc + cost_t[r, c].sum()
        n_pairs += int(r.numel())

    if n_pairs == 0:
        return curves.sum() * 0.0
    return lam * (loss_acc / float(n_pairs))
