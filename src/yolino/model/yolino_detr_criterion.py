# SPDX-License-Identifier: GPL-3.0-or-later
"""
Set-prediction criterion for :class:`yolino.model.yolino_detr_head.YolinoDetrBezierHead`.

Matches K predicted Bézier curves against M GT polylines per image via the
Hungarian algorithm (scipy), then computes:

    L_total = L_l1  (32-pt point-to-point L1, normalized to [0,1] image)
            + L_endpoint  (extra L1 on the two endpoints, * endpoint_weight)
            + obj_weight * L_obj  (BCE: 1 for matched queries, 0 otherwise)
          [+ chamfer_aux_weight * L_chamfer  if enabled]

The total loss is multiplied by the standard E2E warmup factor
:func:`yolino.model.e2e_train_bridge._effective_e2e_lambda` so the existing
``e2e_loss_weight`` / ``e2e_warmup_*`` knobs continue to work.

GT polylines come from ``e2e_gt_pack`` (TTPLA collate), shape:
    padded     : [B, Ni, Mp, 2]   (x,y in pixels)
    inst_mask  : [B, Ni]          (bool, True = valid instance)
    pt_mask    : [B, Ni, Mp]      (bool, True = valid vertex)
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from yolino.model.e2e_train_bridge import _effective_e2e_lambda, resample_polyline_xy

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


def _straightness_collinearity_loss(pts: torch.Tensor) -> torch.Tensor:
    """Mean perpendicular distance of interior points to the endpoint chord (pixels, normalized).

    pts: [Nm, N, 2] with N >= 3. Encourages near-straight polylines (직진성).
    """
    n = int(pts.shape[1])
    if n < 3:
        return pts.new_zeros(())
    p0 = pts[:, 0]
    p1 = pts[:, -1]
    chord = p1 - p0
    chord_len = chord.norm(dim=-1).clamp(min=1e-6)
    inner = pts[:, 1:-1]
    rel = inner - p0.unsqueeze(1)
    cross = (rel[..., 0] * chord[..., 1].unsqueeze(1) - rel[..., 1] * chord[..., 0].unsqueeze(1)).abs()
    perp = cross / chord_len.unsqueeze(1)
    return perp.mean()


def _chamfer_l1(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Bidirectional Chamfer-L1 over point sets.

    pred / gt: [..., T, 2]
    Returns scalar of shape ``pred.shape[:-2]``.
    """
    d = torch.cdist(pred, gt, p=1.0)  # [..., T, T]
    a = d.min(dim=-1).values.mean(dim=-1)
    b = d.min(dim=-2).values.mean(dim=-1)
    return a + b


def _gather_gt_curves(
    padded: torch.Tensor,
    inst_mask: torch.Tensor,
    pt_mask: torch.Tensor,
    t_res: int,
) -> torch.Tensor:
    """
    Returns: [M_valid, t_res, 2] arc-length resampled GT polylines for one image.
    """
    idx = torch.where(inst_mask)[0]
    if idx.numel() == 0:
        return padded.new_zeros((0, t_res, 2))
    curves = []
    for j in range(int(idx.numel())):
        gj = int(idx[j])
        if int(pt_mask[gj].sum().item()) < 2:
            continue
        curves.append(resample_polyline_xy(padded[gj], pt_mask[gj], t_res))
    if len(curves) == 0:
        return padded.new_zeros((0, t_res, 2))
    return torch.stack(curves, dim=0)


def compute_detr_e2e_loss(
    e2e_out: Dict[str, torch.Tensor],
    e2e_gt_pack: Dict[str, torch.Tensor],
    img_h: int,
    img_w: int,
    epoch: int,
    args,
) -> Dict[str, torch.Tensor]:
    """
    Hungarian-matched set loss for the YOLinO-DETR Bézier head.

    Returns a dict::

        {
          "total":     scalar tensor (already scaled by lambda; what trainer adds to sum_loss),
          "l1":        detached scalar — mean matched 32-pt L1 (normalized image coords),
          "endpoint":  detached scalar — mean matched endpoint L1,
          "obj":       detached scalar — BCE on objectness logits,
          "chamfer":   detached scalar — mean matched Chamfer-L1 (0 if disabled),
          "n_matched": float — number of Hungarian-matched pairs over the batch,
          "n_gt":      float — total GT polylines seen this batch,
          "lam":       float — warmup-scaled e2e_loss_weight at this epoch,
        }

    Falls back to a zero-loss anchored on output graphs when no usable GT exists
    so that the autograd graph stays connected for DDP.
    """
    if linear_sum_assignment is None:
        raise ImportError("scipy is required for YOLinO-DETR Hungarian matching.")

    pred_curve = e2e_out.get("polylines_px", e2e_out["bezier_curve_px"])  # [B, K, T, 2]
    obj_logits = e2e_out["objectness_logits"]  # [B, K]
    b, k, t_pred, _ = pred_curve.shape
    device = pred_curve.device
    dtype = pred_curve.dtype

    # Anchor for DDP / no-GT batches: zero-graphed tensor that touches every output we care about.
    zero_anchor = pred_curve.sum() * 0.0 + obj_logits.sum() * 0.0
    zero_scalar = torch.zeros((), device=device, dtype=dtype)

    lam = _effective_e2e_lambda(epoch, args)

    if lam <= 0:
        return {
            "total": zero_anchor,
            "l1": zero_scalar,
            "endpoint": zero_scalar,
            "obj": zero_scalar,
            "chamfer": zero_scalar,
            "straightness": zero_scalar,
            "dn": zero_scalar,
            "aux_layer": zero_scalar,
            "n_matched": 0.0,
            "n_gt": 0.0,
            "lam": float(lam),
        }

    t_res = int(getattr(args, "e2e_gt_resample_t", 32))
    if t_res != t_pred:
        # Pred is sampled at T = e2e_bezier_num_samples (default 32). We resample GT to the same T.
        t_res = t_pred
    endpoint_w = float(getattr(args, "e2e_endpoint_weight", 2.0))
    obj_w = float(getattr(args, "e2e_objectness_weight", 1.0))
    chamfer_w = float(getattr(args, "e2e_chamfer_aux_weight", 0.0))
    straight_w = float(getattr(args, "e2e_straightness_weight", 0.0))

    scale_xy = torch.tensor(
        [max(float(img_w - 1), 1.0), max(float(img_h - 1), 1.0)],
        device=device,
        dtype=dtype,
    ).view(1, 1, 1, 2)

    padded = e2e_gt_pack["padded"].to(device=device, dtype=dtype, non_blocking=True)
    inst_m = e2e_gt_pack["inst_mask"].to(device=device, non_blocking=True)
    pt_m = e2e_gt_pack["pt_mask"].to(device=device, non_blocking=True)

    loss_l1 = pred_curve.new_zeros(())
    loss_end = pred_curve.new_zeros(())
    loss_chamfer = pred_curve.new_zeros(())
    loss_straight = pred_curve.new_zeros(())
    n_matched = 0
    n_gt_total = 0

    # Objectness target accumulator per batch.
    obj_target = torch.zeros_like(obj_logits)
    has_any_gt = False

    for bi in range(b):
        gt_curves = _gather_gt_curves(padded[bi], inst_m[bi], pt_m[bi], t_res)
        n_gt = int(gt_curves.shape[0])
        if n_gt == 0:
            continue
        has_any_gt = True
        n_gt_total += n_gt

        pred_b = pred_curve[bi]  # [K, T, 2]
        pred_n = pred_b / scale_xy.squeeze(0)  # [K, T, 2] in [0,1]
        gt_n = gt_curves / scale_xy.squeeze(0)  # [M, T, 2]

        # Pointwise L1 matching cost (mean over all T points and xy).
        diff = (pred_n.unsqueeze(1) - gt_n.unsqueeze(0)).abs()  # [K, M, T, 2]
        cost_l1 = diff.mean(dim=(-1, -2))  # [K, M]

        # Endpoint extra L1 (sum of |start-start| + |end-end|, mean over x,y).
        pred_ep = torch.stack([pred_n[:, 0], pred_n[:, -1]], dim=1)  # [K, 2, 2]
        gt_ep = torch.stack([gt_n[:, 0], gt_n[:, -1]], dim=1)  # [M, 2, 2]
        cost_ep = (pred_ep.unsqueeze(1) - gt_ep.unsqueeze(0)).abs().mean(dim=(-1, -2))  # [K, M]

        # Objectness contribution to cost: prefer matching queries with high conf.
        cost_obj = (1.0 - obj_logits[bi].sigmoid()).unsqueeze(1).expand(k, n_gt)  # [K, M]

        cost_matrix = cost_l1 + endpoint_w * cost_ep + obj_w * cost_obj

        ri, ci = linear_sum_assignment(cost_matrix.detach().float().cpu().numpy())
        if len(ri) == 0:
            continue
        r = torch.as_tensor(ri, device=device, dtype=torch.long)
        c = torch.as_tensor(ci, device=device, dtype=torch.long)

        matched_pred = pred_n[r]  # [Nm, T, 2]
        matched_gt = gt_n[c]  # [Nm, T, 2]

        loss_l1 = loss_l1 + (matched_pred - matched_gt).abs().mean(dim=(-1, -2)).sum()
        # Endpoint extra L1
        pred_ep_m = torch.stack([matched_pred[:, 0], matched_pred[:, -1]], dim=1)
        gt_ep_m = torch.stack([matched_gt[:, 0], matched_gt[:, -1]], dim=1)
        loss_end = loss_end + (pred_ep_m - gt_ep_m).abs().mean(dim=(-1, -2)).sum()

        if chamfer_w > 0.0:
            loss_chamfer = loss_chamfer + _chamfer_l1(matched_pred, matched_gt).sum()

        if straight_w > 0.0 and matched_pred.shape[1] >= 3:
            loss_straight = loss_straight + _straightness_collinearity_loss(matched_pred).sum()

        obj_target[bi, r] = 1.0
        n_matched += int(r.numel())

    obj_loss = F.binary_cross_entropy_with_logits(obj_logits, obj_target, reduction="mean")

    if not has_any_gt or n_matched == 0:
        total = lam * (obj_w * obj_loss) + zero_anchor
        return {
            "total": total,
            "l1": zero_scalar,
            "endpoint": zero_scalar,
            "obj": obj_loss.detach(),
            "chamfer": zero_scalar,
            "straightness": zero_scalar,
            "dn": zero_scalar,
            "aux_layer": zero_scalar,
            "n_matched": float(n_matched),
            "n_gt": float(n_gt_total),
            "lam": float(lam),
        }

    avg_l1 = loss_l1 / float(n_matched)
    avg_end = loss_end / float(n_matched)
    avg_chamfer = loss_chamfer / float(n_matched) if chamfer_w > 0.0 else zero_scalar
    avg_straight = loss_straight / float(n_matched) if straight_w > 0.0 else zero_scalar

    total = avg_l1 + endpoint_w * avg_end + obj_w * obj_loss
    if chamfer_w > 0.0:
        total = total + chamfer_w * avg_chamfer
    if straight_w > 0.0:
        total = total + straight_w * avg_straight

    # -------- Hough-DETR head extensions (DN + per-layer aux) -------- #
    # When the head emits ``dn_polylines_px`` / ``dn_targets_px`` (exp51), add a
    # direct L1 (no Hungarian) per DN slot. Slots with ``dn_valid == False`` are
    # excluded. The DN weight is independent of the main lambda so DN trains
    # from epoch 0 even when the main e2e_loss_weight is in warmup.
    dn_total_scalar = zero_scalar
    if "dn_polylines_px" in e2e_out and "dn_targets_px" in e2e_out:
        dn_pred = e2e_out["dn_polylines_px"]
        dn_tgt = e2e_out["dn_targets_px"]
        dn_valid = e2e_out.get("dn_valid", None)
        if dn_pred.numel() > 0 and dn_tgt.numel() > 0:
            dn_w = float(getattr(args, "e2e_hough_dn_loss_weight", 0.0) or 0.0)
            if dn_w > 0.0:
                # Normalize to [0,1] for scale-stability with the matching L1.
                dn_pred_n = dn_pred / scale_xy.squeeze(0).unsqueeze(0)
                dn_tgt_n = dn_tgt / scale_xy.squeeze(0).unsqueeze(0)
                if dn_valid is not None:
                    valid_xy = dn_valid.unsqueeze(-1).unsqueeze(-1).to(dn_pred.dtype)
                    dn_pred_n = dn_pred_n * valid_xy
                    dn_tgt_n = dn_tgt_n * valid_xy
                    n_dn_valid = float(dn_valid.sum().item()) + 1e-6
                else:
                    n_dn_valid = float(dn_pred.shape[0] * dn_pred.shape[1]) + 1e-6
                dn_l1 = (dn_pred_n - dn_tgt_n).abs().sum() / max(n_dn_valid, 1.0)
                total = total + dn_w * dn_l1
                dn_total_scalar = dn_l1.detach()

    # Per-intermediate-layer aux Hungarian loss (DINO-style independent matching).
    aux_total_scalar = zero_scalar
    aux_w = float(getattr(args, "e2e_hough_aux_layer_weight", 0.0) or 0.0)
    if aux_w > 0.0 and "ref_pts_layers" in e2e_out and isinstance(
        e2e_out["ref_pts_layers"], torch.Tensor
    ):
        layers_px = e2e_out["ref_pts_layers"]  # [L, B, K, 5, 2] (last layer == polylines_px)
        if layers_px.dim() == 5 and int(layers_px.shape[0]) >= 2:
            n_aux_layers = int(layers_px.shape[0]) - 1
            for li in range(n_aux_layers):
                pred_li = layers_px[li]
                pred_li_n = pred_li / scale_xy.squeeze(0)
                aux_l1_layer = pred_li.new_zeros(())
                aux_n_match = 0
                for bi in range(b):
                    gt_curves_bi = _gather_gt_curves(padded[bi], inst_m[bi], pt_m[bi], t_res)
                    if int(gt_curves_bi.shape[0]) == 0:
                        continue
                    gt_n = gt_curves_bi / scale_xy.squeeze(0)
                    diff_bi = (pred_li_n[bi].unsqueeze(1) - gt_n.unsqueeze(0)).abs()
                    cost_li = diff_bi.mean(dim=(-1, -2))                 # [K, M]
                    ri_a, ci_a = linear_sum_assignment(cost_li.detach().float().cpu().numpy())
                    if len(ri_a) == 0:
                        continue
                    ra = torch.as_tensor(ri_a, device=device, dtype=torch.long)
                    ca = torch.as_tensor(ci_a, device=device, dtype=torch.long)
                    aux_l1_layer = aux_l1_layer + (
                        pred_li_n[bi][ra] - gt_n[ca]
                    ).abs().mean(dim=(-1, -2)).sum()
                    aux_n_match += int(ra.numel())
                if aux_n_match > 0:
                    total = total + aux_w * aux_l1_layer / float(aux_n_match)
                    aux_total_scalar = (aux_total_scalar + aux_l1_layer.detach()
                                        / float(aux_n_match))

    return {
        "total": lam * total,
        "l1": avg_l1.detach(),
        "endpoint": avg_end.detach(),
        "obj": obj_loss.detach(),
        "chamfer": avg_chamfer.detach() if isinstance(avg_chamfer, torch.Tensor) else zero_scalar,
        "straightness": avg_straight.detach() if isinstance(avg_straight, torch.Tensor) else zero_scalar,
        "dn": dn_total_scalar,
        "aux_layer": aux_total_scalar,
        "n_matched": float(n_matched),
        "n_gt": float(n_gt_total),
        "lam": float(lam),
    }
