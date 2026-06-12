# SPDX-License-Identifier: GPL-3.0-or-later
"""Soft geometric prior over directed segment pairs (lateral × direction)."""
from __future__ import annotations

from typing import Dict, Tuple

import torch


def compute_edge_prior(
    node_mid: torch.Tensor,
    node_dnorm: torch.Tensor,
    node_valid: torch.Tensor,
    *,
    sigma_lat_px: float = 32.0,
    dir_floor: float = 0.5,
    prior_eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Directed soft prior ``edge_prior[b,i,j]`` in ``(0, 1]``.

    Args:
        node_mid: ``[B, N, 2]`` pixel midpoints.
        node_dnorm: ``[B, N, 2]`` unit direction vectors.
        node_valid: ``[B, N]`` bool — valid nodes.
    Returns:
        prior ``[B, N, N]`` and scalar stats for logging.
    """
    b, n, _ = node_mid.shape
    device = node_mid.device

    mid_i = node_mid.unsqueeze(2).expand(b, n, n, 2)
    mid_j = node_mid.unsqueeze(1).expand(b, n, n, 2)
    delta_ij = mid_j - mid_i

    dn_i = node_dnorm.unsqueeze(2).expand(b, n, n, 2)
    dn_j = node_dnorm.unsqueeze(1).expand(b, n, n, 2)

    d_perp = (delta_ij[..., 0] * dn_i[..., 1] - delta_ij[..., 1] * dn_i[..., 0]).abs()
    dir_dot = (dn_i * dn_j).sum(dim=-1).abs()

    sigma = max(float(sigma_lat_px), 1e-6)
    lateral_score = torch.exp(-d_perp / sigma)
    dir_score = (dir_dot - float(dir_floor)).relu()
    prior = (lateral_score * dir_score).clamp(min=float(prior_eps), max=1.0)

    eye = torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0)
    pair_ok = node_valid.unsqueeze(2) & node_valid.unsqueeze(1) & ~eye
    prior = prior * pair_ok.float()

    with torch.no_grad():
        if bool(pair_ok.any()):
            pv = prior[pair_ok]
            stats = {
                "mean_prior": pv.mean(),
                "frac_prior_gt_0.1": (pv > 0.1).float().mean(),
            }
        else:
            z = prior.new_zeros(())
            stats = {"mean_prior": z, "frac_prior_gt_0.1": z}

    return prior, stats
