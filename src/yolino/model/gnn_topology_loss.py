# SPDX-License-Identifier: GPL-3.0-or-later
"""Random-walk topology loss on GNN edge probabilities."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def build_transition_matrix(
    edge_probs: torch.Tensor,
    neighbors: torch.Tensor,
    neigh_valid: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """Row-stochastic transition matrix ``T[b,i,j]`` from sparse edge probs.

    Args:
        edge_probs: ``[B, N, K]`` in ``[0, 1]``.
        neighbors: ``[B, N, K]`` destination indices.
        neigh_valid: ``[B, N, K]`` bool mask.
    """
    b, n, _ = edge_probs.shape
    t = edge_probs.new_zeros((b, n, n))
    t.scatter_add_(2, neighbors.clamp(0, n_nodes - 1), edge_probs * neigh_valid.float())
    row_sum = t.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return t / row_sum


def random_walk_topology_loss(
    transition: torch.Tensor,
    seg_inst: torch.Tensor,
    node_valid: torch.Tensor,
    *,
    steps: int = 6,
    pos_weight: float = 20.0,
) -> torch.Tensor:
    """Weighted BCE between ``T^k`` and same-instance pairs.

    Args:
        transition: ``[B, N, N]`` row-stochastic.
        seg_inst: ``[B, N]`` GT instance id per node (-1 = background).
        node_valid: ``[B, N]`` bool.
    """
    steps = max(int(steps), 1)
    tk = transition
    for _ in range(steps - 1):
        tk = torch.bmm(tk, transition)

    b, n, _ = tk.shape
    device = tk.device
    eye = torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0)
    valid_pair = node_valid.unsqueeze(2) & node_valid.unsqueeze(1) & ~eye

    same_inst = (
        (seg_inst.unsqueeze(2) == seg_inst.unsqueeze(1))
        & (seg_inst.unsqueeze(2) >= 0)
        & (seg_inst.unsqueeze(1) >= 0)
        & valid_pair
    )
    target = same_inst.float()
    pred = tk.clamp(1e-6, 1.0 - 1e-6)

    pw = float(pos_weight)
    weight = torch.where(same_inst, pred.new_tensor(pw), pred.new_ones(()))
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    masked = valid_pair
    if not bool(masked.any()):
        return (tk.sum() * 0.0)
    num = (bce * weight)[masked].sum()
    den = weight[masked].sum().clamp(min=1.0)
    return num / den
