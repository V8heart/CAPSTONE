# SPDX-License-Identifier: GPL-3.0-or-later
"""
Connectivity + topology losses for :class:`yolino.model.yolino_gnn_head.YolinoGnnSegmentGraphHead`.

Optionally:
  * strict next-hop positives (GT vertex index adjacency),
  * soft degree regularization on predicted edge probabilities,
  * soft-chain Chamfer assembly loss (:mod:`yolino.model.gnn_polyline_assembly`).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from yolino.model.gnn_polyline_assembly import (
    compute_soft_chain_chamfer_loss,
    compute_soft_rw_chamfer_loss,
)
from yolino.model.gnn_topology_loss import build_transition_matrix, random_walk_topology_loss
from yolino.model.gnn_topology_metrics import instance_iou_mean


def _effective_chamfer_weight(epoch: int, args) -> float:
    w = float(getattr(args, "gnn_chamfer_weight", 0.0) or 0.0)
    if w <= 0.0:
        return 0.0
    warm = int(getattr(args, "gnn_chamfer_warmup_epochs", 0) or 0)
    start = int(getattr(args, "gnn_chamfer_warmup_start_epoch", 0) or 0)
    if warm <= 0:
        return w if epoch >= start else 0.0
    if epoch < start:
        return 0.0
    t = min(1.0, float(epoch - start + 1) / float(warm))
    return w * t


def _effective_gnn_lambda(epoch: int, args) -> float:
    w = float(getattr(args, "gnn_loss_weight", 0.0) or 0.0)
    if w <= 0:
        return 0.0
    warm = int(getattr(args, "gnn_warmup_epochs", 0) or 0)
    start = int(getattr(args, "gnn_warmup_start_epoch", 0) or 0)
    if warm <= 0:
        return w if epoch >= start else 0.0
    if epoch < start:
        return 0.0
    t = min(1.0, float(epoch - start + 1) / float(warm))
    return w * t


def _parse_eval_edge_threshs(args) -> Tuple[float, ...]:
    raw = str(getattr(args, "gnn_eval_edge_threshs", "0.35,0.5") or "0.35,0.5")
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            continue
    return tuple(out) if out else (0.35, 0.5)


def _wire_endpoint_adjacent_mask(
    node_ea: torch.Tensor,
    node_eb: torch.Tensor,
    neighbors: torch.Tensor,
    neigh_valid: torch.Tensor,
    max_end_gap_px: float,
) -> torch.Tensor:
    """True where candidate edge endpoints are within ``max_end_gap_px`` (wire stitch)."""
    b, n, k = neighbors.shape
    ea_i = node_ea.unsqueeze(2).expand(b, n, k, 2)
    eb_i = node_eb.unsqueeze(2).expand(b, n, k, 2)
    idx = neighbors.unsqueeze(-1).expand(b, n, k, 2)
    ea_j = torch.gather(node_ea.unsqueeze(1).expand(b, n, n, 2), 2, idx)
    eb_j = torch.gather(node_eb.unsqueeze(1).expand(b, n, n, 2), 2, idx)
    d = torch.minimum(
        torch.minimum(torch.norm(ea_i - ea_j, dim=-1), torch.norm(ea_i - eb_j, dim=-1)),
        torch.minimum(torch.norm(eb_i - ea_j, dim=-1), torch.norm(eb_i - eb_j, dim=-1)),
    )
    return neigh_valid & (d <= float(max_end_gap_px))


def _edge_prf_at_threshold(
    edge_logits: torch.Tensor,
    pos_mask: torch.Tensor,
    eval_mask: torch.Tensor,
    thresh: float,
) -> Tuple[float, float, float]:
    with torch.no_grad():
        pred = torch.sigmoid(edge_logits) >= float(thresh)
        gt = pos_mask
        m = eval_mask
        tp = (pred & gt & m).sum().float()
        fp = (pred & (~gt) & m).sum().float()
        fn = ((~pred) & gt & m).sum().float()
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1 = 2.0 * prec * rec / (prec + rec + 1e-6)
    return float(prec.item()), float(rec.item()), float(f1.item())


def _assign_nodes_to_instances_and_vertices(
    node_mid: torch.Tensor,
    node_valid: torch.Tensor,
    padded: torch.Tensor,
    inst_mask: torch.Tensor,
    pt_mask: torch.Tensor,
    radius_px: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign each node to nearest GT polyline vertex; return (instance_id, vertex_index).

    ``vertex_index`` is the index along the **valid** vertex subsequence ``pts[valid]`` (0 .. T-1).
    Background nodes get ``(-1, -1)``.
    """
    b, n, _ = node_mid.shape
    ni, mp = padded.shape[1], padded.shape[2]
    device = node_mid.device

    seg_inst = torch.full((b, n), -1, dtype=torch.long, device=device)
    seg_vidx = torch.full((b, n), -1, dtype=torch.long, device=device)
    seg_min_dist = torch.full((b, n), float("inf"), dtype=torch.float32, device=device)

    for bi in range(b):
        if int(inst_mask[bi].sum().item()) == 0:
            continue
        nm = node_mid[bi].float()
        for ii in range(ni):
            if not bool(inst_mask[bi, ii].item()):
                continue
            pts = padded[bi, ii]
            valid = pt_mask[bi, ii]
            n_valid = int(valid.sum().item())
            if n_valid < 1:
                continue
            pts_v = pts[valid].float()
            d = torch.cdist(nm, pts_v, p=2)
            min_d, vidx = d.min(dim=1)
            better = min_d < seg_min_dist[bi]
            seg_min_dist[bi] = torch.where(better, min_d, seg_min_dist[bi])
            seg_inst[bi] = torch.where(better, torch.full_like(seg_inst[bi], ii), seg_inst[bi])
            seg_vidx[bi] = torch.where(better, vidx.long(), seg_vidx[bi])

    far = seg_min_dist > float(radius_px)
    seg_inst = seg_inst.masked_fill(far, -1)
    seg_vidx = seg_vidx.masked_fill(far, -1)
    seg_inst = seg_inst.masked_fill(~node_valid, -1)
    seg_vidx = seg_vidx.masked_fill(~node_valid, -1)
    return seg_inst, seg_vidx


def _vertex_adjacent(
    vi: torch.Tensor,
    vj: torch.Tensor,
    T_inst: torch.Tensor,
    allow_closed: bool,
) -> torch.Tensor:
    """``[B,N,K]`` bool: GT vertex indices adjacent along polyline (same batch/instance implied upstream)."""
    ok = (vj == vi + 1) | (vj == vi - 1)
    if not allow_closed:
        return ok
    tm1 = (T_inst - 1).clamp(min=1)
    wrap = ((vi == 0) & (vj == tm1)) | ((vi == tm1) & (vj == 0))
    return ok | wrap


def _supervision_endpoint_mask(
    seg_inst: torch.Tensor,
    seg_inst_neigh: torch.Tensor,
    neigh_valid: torch.Tensor,
    args,
) -> torch.Tensor:
    """Edges eligible for GNN BCE / degree / RW supervision."""
    src_fg = seg_inst.unsqueeze(-1) >= 0
    dst_fg = seg_inst_neigh >= 0
    if bool(getattr(args, "gnn_supervise_matched_nodes_only", True)):
        fg = src_fg & dst_fg
    else:
        fg = src_fg | dst_fg
    return neigh_valid & fg


def _degree_soft_penalty(
    edge_logits: torch.Tensor,
    neighbors: torch.Tensor,
    neigh_valid: torch.Tensor,
    row_cap: float,
    col_cap: float,
    edge_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean ReLU excess of soft row- and column-sums of ``sigmoid(logit)`` (masked edges only)."""
    b, n, k = edge_logits.shape
    p = torch.sigmoid(edge_logits) * neigh_valid.float()
    if edge_mask is not None:
        p = p * edge_mask.float()
    adj = torch.zeros(b, n, n, device=p.device, dtype=p.dtype)
    adj.scatter_add_(2, neighbors.clamp(0, n - 1), p)
    row_sum = adj.sum(dim=2)
    col_sum = adj.sum(dim=1)
    l_row = F.relu(row_sum - float(row_cap)).pow(2).mean()
    l_col = F.relu(col_sum - float(col_cap)).pow(2).mean()
    return 0.5 * (l_row + l_col)


def _gnn_edge_cross_ignore_loss(
    edge_logits: torch.Tensor,
    pos_mask: torch.Tensor,
    cross_mask: torch.Tensor,
    args,
    remote_pos_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pos = chain-adjacent; ignore = same-instance non-adjacent; cross = strong negative only.

    When ``remote_pos_mask`` is set (``--gnn_same_inst_ignore_as_pos``), former ignore edges
    are trained as positives with ``gnn_pos_weight_remote``.
    """
    parts = []
    pos_w = float(getattr(args, "gnn_pos_weight", 1.0) or 1.0)
    remote_w = float(getattr(args, "gnn_pos_weight_remote", 1.0) or 1.0)
    cross_w = float(getattr(args, "gnn_cross_instance_weight", 10.0) or 10.0)

    if bool(pos_mask.any()):
        lp = F.binary_cross_entropy_with_logits(
            edge_logits[pos_mask],
            torch.ones_like(edge_logits[pos_mask]),
            reduction="none",
        )
        if abs(pos_w - 1.0) > 1e-6:
            lp = lp * pos_w
        parts.append(lp)

    if remote_pos_mask is not None and bool(remote_pos_mask.any()):
        lr = F.binary_cross_entropy_with_logits(
            edge_logits[remote_pos_mask],
            torch.ones_like(edge_logits[remote_pos_mask]),
            reduction="none",
        )
        if abs(remote_w - 1.0) > 1e-6:
            lr = lr * remote_w
        parts.append(lr)

    if bool(cross_mask.any()):
        lc = F.binary_cross_entropy_with_logits(
            edge_logits[cross_mask],
            torch.zeros_like(edge_logits[cross_mask]),
            reduction="none",
        )
        if abs(cross_w - 1.0) > 1e-6:
            lc = lc * cross_w
        parts.append(lc)

    if not parts:
        return edge_logits.new_zeros(())
    cat = torch.cat(parts)
    return torch.where(torch.isfinite(cat), cat, torch.zeros_like(cat)).mean()


def _gnn_edge_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    args,
) -> torch.Tensor:
    """Per-edge BCE or focal loss on selected logits (RetinaNet-style focal on logits)."""
    gts = labels.to(dtype=logits.dtype)
    neg_w = float(getattr(args, "gnn_neg_weight", 1.0) or 1.0)
    loss_type = str(getattr(args, "gnn_edge_loss_type", "bce")).lower().strip()
    if loss_type == "focal":
        ce = F.binary_cross_entropy_with_logits(logits, gts, reduction="none")
        pt = torch.exp(-ce)
        alpha = float(getattr(args, "gnn_focal_alpha", 0.75))
        gamma = float(getattr(args, "gnn_focal_gamma", 2.0))
        alpha_t = torch.where(
            gts > 0.5,
            torch.full_like(gts, alpha),
            torch.full_like(gts, 1.0 - alpha),
        )
        focal = alpha_t * ((1.0 - pt).clamp(min=0.0) ** gamma) * ce
        pos_boost = float(getattr(args, "gnn_pos_weight", 1.0) or 1.0)
        if abs(pos_boost - 1.0) > 1e-6:
            boost = torch.where(gts > 0.5, torch.full_like(gts, pos_boost), torch.ones_like(gts))
            focal = focal * boost
        if abs(neg_w - 1.0) > 1e-6:
            nboost = torch.where(gts > 0.5, torch.ones_like(gts), torch.full_like(gts, neg_w))
            focal = focal * nboost
        return torch.where(torch.isfinite(focal), focal, torch.zeros_like(focal)).mean()
    pos_w = torch.tensor(
        float(getattr(args, "gnn_pos_weight", 1.0)), device=logits.device, dtype=logits.dtype,
    )
    bce = F.binary_cross_entropy_with_logits(logits, gts, pos_weight=pos_w, reduction="none")
    if abs(neg_w - 1.0) > 1e-6:
        nboost = torch.where(gts > 0.5, torch.ones_like(gts), torch.full_like(gts, neg_w))
        bce = bce * nboost
    return bce.mean()


def _resolve_seg_inst_vidx(
    e2e_out: Dict[str, torch.Tensor],
    e2e_gt_pack: Optional[Dict[str, torch.Tensor]],
    matched_supervision: Optional[Dict[str, torch.Tensor]],
    args,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return (seg_inst, seg_vidx, padded, inst_mask, pt_mask) for GNN supervision."""
    device = e2e_out["edge_logits"].device
    node_mid = e2e_out["node_mid_px"]
    node_valid = e2e_out["node_valid"]

    use_matched = bool(getattr(args, "gnn_use_matched_gt_from_geom", False))
    seg_inst = None
    seg_vidx = None
    if use_matched and isinstance(matched_supervision, dict):
        ms_valid = matched_supervision.get("node_gt_valid", None)
        ms_inst = matched_supervision.get("node_gt_instance", None)
        if isinstance(ms_valid, torch.Tensor) and isinstance(ms_inst, torch.Tensor):
            seg_inst = ms_inst.to(device=device, dtype=torch.long)
            ms_valid = ms_valid.to(device=device, dtype=torch.bool)
            seg_inst = torch.where(ms_valid, seg_inst, torch.full_like(seg_inst, -1))
            seg_inst = torch.where(node_valid, seg_inst, torch.full_like(seg_inst, -1))
            seg_vidx = torch.full_like(seg_inst, -1)
            if int((seg_inst >= 0).sum().item()) == 0:
                seg_inst, seg_vidx = None, None

    padded = inst_mask = pt_mask = None
    if e2e_gt_pack is not None:
        padded = e2e_gt_pack["padded"].to(device)
        inst_mask = e2e_gt_pack["inst_mask"].to(device)
        pt_mask = e2e_gt_pack["pt_mask"].to(device)

    if seg_inst is None or seg_vidx is None:
        if padded is None:
            return None, None, None, None, None
        seg_inst, seg_vidx = _assign_nodes_to_instances_and_vertices(
            node_mid=node_mid,
            node_valid=node_valid,
            padded=padded,
            inst_mask=inst_mask,
            pt_mask=pt_mask,
            radius_px=float(getattr(args, "gnn_node_assign_radius_px", 24.0)),
        )
    return seg_inst, seg_vidx, padded, inst_mask, pt_mask


def _gnn_topology_views(e2e_out: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (loss_logits, loss_neighbors, loss_valid, conn_logits, conn_neighbors, conn_valid).

    ``directional2_ctx`` stores full ``gat`` tensors in ``edge_logits`` / ``neighbors`` for BCE,
    and ``line_*`` for connection graph (CC, viz, degree, RW).
    """
    loss_logits = e2e_out["edge_logits"]
    loss_neighbors = e2e_out["neighbors"]
    loss_valid = e2e_out["neigh_valid"]
    line_n = e2e_out.get("line_neighbors")
    line_v = e2e_out.get("line_neigh_valid")
    if isinstance(line_n, torch.Tensor) and isinstance(line_v, torch.Tensor):
        k_line = int(line_n.shape[-1])
        conn_logits = loss_logits[:, :, :k_line]
        return loss_logits, loss_neighbors, loss_valid, conn_logits, line_n, line_v
    return loss_logits, loss_neighbors, loss_valid, loss_logits, loss_neighbors, loss_valid


def compute_gnn_e2e_loss(
    e2e_out: Dict[str, torch.Tensor],
    e2e_gt_pack: Optional[Dict[str, torch.Tensor]],
    img_h: int,
    img_w: int,
    epoch: int,
    args,
    matched_supervision: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """BCE (+ optional strict next-hop, degree, Chamfer)."""
    edge_logits, neighbors, neigh_valid, conn_logits, conn_neighbors, conn_neigh_valid = _gnn_topology_views(
        e2e_out,
    )
    node_mid = e2e_out["node_mid_px"]
    node_valid = e2e_out["node_valid"]

    device = edge_logits.device
    dtype = edge_logits.dtype

    lam = _effective_gnn_lambda(epoch, args)
    zero_aux = (edge_logits.sum() + e2e_out["node_feat"].sum()) * 0.0

    out_d: Dict[str, torch.Tensor] = {
        "total": zero_aux,
        "bce": edge_logits.new_zeros(()),
        "lam": float(lam),
        "n_pos": 0,
        "n_neg": 0,
        "n_cross": 0,
        "n_ignore": 0,
        "n_kept": 0,
        "mean_edge_prob": float(0.0),
        "frac_pos_edges": float(0.0),
        "n_nodes_fg": 0,
        "n_nodes_total": int(node_valid.sum().item()),
        "degree_loss": edge_logits.new_zeros(()),
        "rw_topology_loss": edge_logits.new_zeros(()),
        "chamfer": edge_logits.new_zeros(()),
        "instance_iou_mean": float(0.0),
        "n_pos_strict": 0,
        "frac_pos_strict": float(0.0),
    }
    if neigh_valid.any():
        with torch.no_grad():
            mp = torch.sigmoid(edge_logits[neigh_valid]).float().mean()
        out_d["mean_edge_prob"] = float(mp.item())

    with torch.no_grad():
        seg_inst, seg_vidx, padded, inst_mask, pt_mask = _resolve_seg_inst_vidx(
            e2e_out, e2e_gt_pack, matched_supervision, args,
        )
    if seg_inst is None or seg_vidx is None:
        return out_d

    with torch.no_grad():
        cc_thresh = float(getattr(args, "gnn_cc_edge_thresh", 0.3) or 0.3)
        out_d["instance_iou_mean"] = instance_iou_mean(
            conn_logits,
            conn_neighbors,
            conn_neigh_valid,
            seg_inst,
            node_valid,
            edge_thresh=cc_thresh,
        )

    if lam <= 0.0:
        return out_d

    b, n, k = neighbors.shape
    seg_inst_neigh = torch.gather(
        seg_inst.unsqueeze(1).expand(-1, n, -1), 2, neighbors,
    )
    seg_vidx_neigh = torch.gather(
        seg_vidx.unsqueeze(1).expand(-1, n, -1), 2, neighbors,
    )

    src_fg = seg_inst.unsqueeze(-1) >= 0
    dst_fg = seg_inst_neigh >= 0
    same_inst = (seg_inst.unsqueeze(-1) == seg_inst_neigh) & src_fg & dst_fg
    base_mask = _supervision_endpoint_mask(seg_inst, seg_inst_neigh, neigh_valid, args)
    with torch.no_grad():
        out_d["n_nodes_matched"] = int((seg_inst >= 0).sum().item())
        out_d["n_nodes_unmatched"] = int(((seg_inst < 0) & node_valid).sum().item())
        legacy_orphan = neigh_valid & (src_fg ^ dst_fg)
        out_d["n_edges_orphan_half_gt"] = int(legacy_orphan.sum().item())

    edge_sup = str(getattr(args, "gnn_edge_supervision", "matched_instance")).lower().strip()
    strict = bool(getattr(args, "gnn_use_strict_next_hop", False))
    if edge_sup == "polyline_adjacent":
        node_ea = e2e_out.get("node_end_a_px")
        node_eb = e2e_out.get("node_end_b_px")
        end_px = float(getattr(args, "gnn_polyline_adjacent_end_px", 12.0) or 12.0)
        if isinstance(node_ea, torch.Tensor) and isinstance(node_eb, torch.Tensor):
            wire_adj = _wire_endpoint_adjacent_mask(node_ea, node_eb, neighbors, neigh_valid, end_px)
            pos_mask = base_mask & same_inst & wire_adj
        else:
            pos_mask = base_mask & same_inst
    elif strict and not torch.all(seg_vidx < 0) and pt_mask is not None:
        T_counts = pt_mask.long().sum(dim=-1)
        T_src = T_counts.gather(1, seg_inst.clamp(min=0))
        T_exp = T_src.unsqueeze(-1).expand(b, n, k)
        T_exp = torch.where((seg_inst.unsqueeze(-1) >= 0) & (seg_inst_neigh >= 0), T_exp, torch.ones_like(T_exp))
        adj = _vertex_adjacent(
            seg_vidx.unsqueeze(-1).expand(b, n, k),
            seg_vidx_neigh,
            T_exp,
            bool(getattr(args, "gnn_next_hop_allow_closed", False)),
        )
        pos_mask = neigh_valid & same_inst & adj
    else:
        pos_mask = base_mask & same_inst

    cross_mask = base_mask & ~same_inst
    ignore_mask = base_mask & same_inst & ~pos_mask
    loss_type = str(getattr(args, "gnn_edge_loss_type", "bce")).lower().strip()
    if loss_type == "cross_ignore":
        neg_mask = cross_mask
    else:
        neg_mask = base_mask & ~pos_mask

    eval_mask = base_mask
    for thr in _parse_eval_edge_threshs(args):
        tag = str(thr).replace(".", "")
        p, r, f1 = _edge_prf_at_threshold(edge_logits, pos_mask, eval_mask, thr)
        out_d[f"edge_prec_{tag}"] = float(p)
        out_d[f"edge_rec_{tag}"] = float(r)
        out_d[f"edge_f1_{tag}"] = float(f1)

    n_pos = int(pos_mask.sum().item())
    n_neg = int(neg_mask.sum().item())
    n_cross = int(cross_mask.sum().item())
    n_ignore = int(ignore_mask.sum().item())
    remote_pos_mask = None
    if bool(getattr(args, "gnn_same_inst_ignore_as_pos", False)) and loss_type == "cross_ignore":
        remote_pos_mask = ignore_mask
        n_ignore = 0
    n_pos_remote = int(remote_pos_mask.sum().item()) if remote_pos_mask is not None else 0
    out_d["n_pos"] = n_pos
    out_d["n_neg"] = n_neg
    out_d["n_cross"] = n_cross
    out_d["n_ignore"] = n_ignore
    out_d["n_pos_remote"] = n_pos_remote
    out_d["n_nodes_fg"] = int((seg_inst >= 0).sum().item())
    out_d["n_pos_strict"] = n_pos
    if base_mask.any():
        with torch.no_grad():
            out_d["frac_pos_strict"] = float(
                pos_mask.float().sum().item() / max(float(base_mask.float().sum().item()), 1.0)
            )

    if n_pos == 0 and n_neg == 0 and (remote_pos_mask is None or n_pos_remote == 0):
        return out_d
    # If there are no positives (e.g. early training: no node within GT assign radius),
    # training only on random negatives pushes the trivial all-zero solution and can
    # dominate the still-untrained edge head. Skip GNN loss contribution for such batches.
    if n_pos == 0 and (remote_pos_mask is None or n_pos_remote == 0):
        return out_d

    neg_keep_mask = neg_mask
    ratio = int(getattr(args, "gnn_neg_per_pos", 0) or 0)
    if ratio > 0 and n_neg > 0:
        per_b_pos = pos_mask.view(b, -1).sum(dim=1)
        per_b_neg = neg_mask.view(b, -1).sum(dim=1)
        neg_floor = int(getattr(args, "gnn_neg_min_kept", 8) or 0)
        if neg_floor > 0:
            target = torch.maximum(per_b_pos * ratio, torch.full_like(per_b_pos, neg_floor))
        else:
            target = per_b_pos * ratio
        keep_prob = (target.float() / per_b_neg.float().clamp(min=1.0)).clamp(max=1.0)
        keep_prob = keep_prob.view(b, 1, 1)
        rand = torch.rand_like(edge_logits)
        neg_keep_mask = neg_mask & (rand < keep_prob)

    final_mask = pos_mask | neg_keep_mask
    n_kept = int(final_mask.sum().item())
    out_d["n_kept"] = n_kept
    if n_kept == 0:
        return out_d

    if loss_type == "cross_ignore":
        edge_cls = _gnn_edge_cross_ignore_loss(
            edge_logits, pos_mask, neg_keep_mask, args,
            remote_pos_mask=remote_pos_mask,
        )
    else:
        labels = pos_mask.to(dtype)
        logits_sel = edge_logits[final_mask]
        labels_sel = labels[final_mask]
        edge_cls = _gnn_edge_classification_loss(logits_sel, labels_sel, args)
    total = lam * edge_cls + zero_aux

    deg_w = float(getattr(args, "gnn_degree_loss_weight", 0.0) or 0.0)
    if deg_w > 0.0:
        b_c, n_c, k_c = conn_logits.shape
        seg_inst_conn = torch.gather(
            seg_inst.unsqueeze(1).expand(-1, n_c, -1), 2, conn_neighbors,
        )
        conn_base = _supervision_endpoint_mask(seg_inst, seg_inst_conn, conn_neigh_valid, args)
        deg = _degree_soft_penalty(
            conn_logits, conn_neighbors, conn_neigh_valid,
            float(getattr(args, "gnn_degree_row_cap", 1.0)),
            float(getattr(args, "gnn_degree_col_cap", 1.0)),
            edge_mask=conn_base,
        )
        out_d["degree_loss"] = deg.detach()
        total = total + deg_w * deg

    rw_w = float(getattr(args, "gnn_rw_topology_weight", 0.0) or 0.0)
    rw_start = int(getattr(args, "gnn_rw_start_epoch", 0) or 0)
    if int(epoch) < rw_start:
        rw_w = 0.0
    if rw_w > 0.0:
        b_c, n_c, k_c = conn_logits.shape
        seg_inst_conn = torch.gather(
            seg_inst.unsqueeze(1).expand(-1, n_c, -1), 2, conn_neighbors,
        )
        conn_base = _supervision_endpoint_mask(seg_inst, seg_inst_conn, conn_neigh_valid, args)
        p_edge = torch.sigmoid(conn_logits) * conn_neigh_valid.float()
        if bool(getattr(args, "gnn_supervise_matched_nodes_only", True)):
            p_edge = p_edge * conn_base.float()
        t_mat = build_transition_matrix(p_edge, conn_neighbors, conn_neigh_valid, n_c)
        rw = random_walk_topology_loss(
            t_mat,
            seg_inst,
            node_valid,
            steps=int(getattr(args, "gnn_rw_steps", 6) or 6),
            pos_weight=float(getattr(args, "gnn_rw_pos_weight", 20.0) or 20.0),
        )
        out_d["rw_topology_loss"] = rw.detach()
        total = total + rw_w * rw

    ch_w = _effective_chamfer_weight(int(epoch), args)
    tier = str(getattr(args, "gnn_assembly_tier", "soft_chain")).lower()
    if ch_w > 0.0 and tier == "soft_chain" and padded is not None and inst_mask is not None and pt_mask is not None:
        ch = compute_soft_chain_chamfer_loss(
            edge_logits=conn_logits,
            neighbors=conn_neighbors,
            neigh_valid=conn_neigh_valid,
            node_mid=node_mid,
            padded=padded,
            inst_mask=inst_mask,
            pt_mask=pt_mask,
            seg_inst=seg_inst,
            max_instances=int(getattr(args, "gnn_assembly_max_instances", 8)),
            max_nodes=int(getattr(args, "gnn_assembly_max_nodes_per_inst", 32)),
        )
        out_d["chamfer"] = ch.detach()
        total = total + ch_w * ch
    elif ch_w > 0.0 and tier == "soft_rw" and padded is not None and inst_mask is not None and pt_mask is not None:
        ch = compute_soft_rw_chamfer_loss(
            edge_logits=conn_logits,
            neighbors=conn_neighbors,
            neigh_valid=conn_neigh_valid,
            node_mid=node_mid,
            padded=padded,
            inst_mask=inst_mask,
            pt_mask=pt_mask,
            seg_inst=seg_inst,
            max_instances=int(getattr(args, "gnn_assembly_max_instances", 8)),
            max_nodes=int(getattr(args, "gnn_assembly_max_nodes_per_inst", 32)),
            rw_steps=int(getattr(args, "gnn_rw_steps", 6) or 6),
        )
        out_d["chamfer"] = ch.detach()
        total = total + ch_w * ch

    out_d["total"] = total
    out_d["bce"] = edge_cls.detach()
    out_d["edge_loss_type"] = loss_type
    if final_mask.any():
        with torch.no_grad():
            if loss_type == "cross_ignore":
                n_pos_kept = int(pos_mask.sum().item())
                frac_pos = float(n_pos_kept) / max(float(n_kept), 1.0)
            else:
                labels_sel = pos_mask[final_mask].to(dtype)
                frac_pos = labels_sel.float().mean().item()
        out_d["frac_pos_edges"] = float(frac_pos)

    # Optional: mean normalized min-endpoint distance on valid edges (from head cache).
    ef = e2e_out.get("edge_feat", None)
    if ef is not None and isinstance(ef, torch.Tensor) and ef.shape[-1] >= 9:
        with torch.no_grad():
            d_end_norm = ef[..., 6]
            m = neigh_valid.float()
            denom = m.sum().clamp(min=1.0)
            out_d["edge_feat_d_end_mean"] = (d_end_norm * m).sum() / denom

    return out_d
