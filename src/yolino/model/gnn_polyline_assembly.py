# SPDX-License-Identifier: GPL-3.0-or-later
"""Soft assembly + Chamfer for GNN edge logits (soft_chain | soft_rw)."""
from __future__ import annotations

import torch

from yolino.model.gnn_topology_loss import build_transition_matrix


def _chamfer_xy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer between two point sets ``[Na,2]``, ``[Nb,2]`` (mean of mins)."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return a.new_zeros(())
    d = torch.cdist(a.float(), b.float(), p=2)
    return d.min(dim=1)[0].mean() + d.min(dim=0)[0].mean()


def compute_soft_chain_chamfer_loss(
    edge_logits: torch.Tensor,
    neighbors: torch.Tensor,
    neigh_valid: torch.Tensor,
    node_mid: torch.Tensor,
    padded: torch.Tensor,
    inst_mask: torch.Tensor,
    pt_mask: torch.Tensor,
    seg_inst: torch.Tensor,
    max_instances: int,
    max_nodes: int,
) -> torch.Tensor:
    """Differentiable proxy: per foreground GT instance, ``Q = row_norm(P_sub) @ mids``, Chamfer vs GT vertices.

    ``P_sub`` is the restriction of ``sigmoid(logit)*valid`` to edges whose source and dest lie on the same
    instance (from ``seg_inst`` hard assignment). Row-normalization yields a soft convex combination of neighbour
    mids per source node; ``Q`` has one row per node in the instance subset (order = ascending node index).
    """
    b, n, _ = node_mid.shape
    device = edge_logits.device
    dtype = edge_logits.dtype
    ni = padded.shape[1]
    p_edge = torch.sigmoid(edge_logits) * neigh_valid.float()

    total = edge_logits.sum() * 0.0
    count = 0

    for bi in range(b):
        dense = torch.zeros(n, n, device=device, dtype=dtype)
        dense.scatter_add_(1, neighbors[bi].clamp(0, n - 1), p_edge[bi])

        for ii in range(min(ni, int(max_instances))):
            if not bool(inst_mask[bi, ii].item()):
                continue
            inst_idx = torch.where(seg_inst[bi] == ii)[0]
            if inst_idx.numel() == 0:
                continue
            if inst_idx.numel() > max_nodes:
                inst_idx = inst_idx[:max_nodes]
            s = int(inst_idx.numel())
            idx = inst_idx.long()
            sub = dense[idx][:, idx]
            mids = node_mid[bi, idx]
            row_sum = sub.sum(dim=1, keepdim=True).clamp(min=1e-6)
            q = (sub / row_sum) @ mids

            valid = pt_mask[bi, ii].bool()
            gt = padded[bi, ii][valid].float()
            if gt.shape[0] == 0:
                continue
            total = total + _chamfer_xy(q, gt)
            count += 1

    if count == 0:
        return total
    return total / float(count)


def compute_soft_rw_chamfer_loss(
    edge_logits: torch.Tensor,
    neighbors: torch.Tensor,
    neigh_valid: torch.Tensor,
    node_mid: torch.Tensor,
    padded: torch.Tensor,
    inst_mask: torch.Tensor,
    pt_mask: torch.Tensor,
    seg_inst: torch.Tensor,
    max_instances: int,
    max_nodes: int,
    rw_steps: int = 6,
) -> torch.Tensor:
    """RW soft assembly: ``soft_pts = Tk[inst_nodes] @ mids`` (full N), Chamfer vs GT vertices."""
    b, n, _ = node_mid.shape
    device = edge_logits.device
    dtype = edge_logits.dtype
    ni = padded.shape[1]
    steps = max(int(rw_steps), 1)
    p_edge = torch.sigmoid(edge_logits) * neigh_valid.float()

    total = edge_logits.sum() * 0.0
    count = 0

    for bi in range(b):
        t = build_transition_matrix(p_edge[bi : bi + 1], neighbors[bi : bi + 1], neigh_valid[bi : bi + 1], n)[0]
        tk = t
        for _ in range(steps - 1):
            tk = tk @ t
        mids_all = node_mid[bi]

        for ii in range(min(ni, int(max_instances))):
            if not bool(inst_mask[bi, ii].item()):
                continue
            inst_idx = torch.where(seg_inst[bi] == ii)[0]
            if inst_idx.numel() == 0:
                continue
            if inst_idx.numel() > max_nodes:
                inst_idx = inst_idx[:max_nodes]
            idx = inst_idx.long()
            soft_pts = tk[idx] @ mids_all

            valid = pt_mask[bi, ii].bool()
            gt = padded[bi, ii][valid].float()
            if gt.shape[0] == 0:
                continue
            total = total + _chamfer_xy(soft_pts, gt)
            count += 1

    if count == 0:
        return total
    return total / float(count)
