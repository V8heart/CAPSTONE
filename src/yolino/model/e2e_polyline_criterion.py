# Copyright 2023 Karlsruhe Institute of Technology, Institute for Measurement
# and Control Systems
#
# E2E polyline set criterion (Hungarian matching + curve loss). Matching uses scipy
# and is non-differentiable; losses run on matched pairs only.
#
# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


def chamfer_distance_l1(
    pred_pts: torch.Tensor,
    gt_pts: torch.Tensor,
) -> torch.Tensor:
    """
    Symmetric Chamfer with L1 on point sets (differentiable).

    pred_pts: [Np, 2], gt_pts: [Ng, 2]
    """
    if pred_pts.numel() == 0 or gt_pts.numel() == 0:
        return pred_pts.new_zeros(())
    d = torch.cdist(pred_pts.unsqueeze(0), gt_pts.unsqueeze(0), p=1.0).squeeze(0)  # [Np, Ng]
    return d.min(dim=1)[0].mean() + d.min(dim=0)[0].mean()


def hungarian_cost_matrix(
    pred_ctrl: torch.Tensor,
    gt_ctrl: torch.Tensor,
    pred_conf: Optional[torch.Tensor],
    gt_conf: Optional[torch.Tensor],
    w_endpoint: float = 1.0,
    w_conf: float = 0.5,
) -> np.ndarray:
    """
    pred_ctrl: [M, K, 2], gt_ctrl: [N, K, 2]
    Costs are L1 between concatenated endpoints + optional |conf_i - conf_j|.

    Returns cost matrix [M, N] (numpy). Pad with dummy columns if M > N using constant cost.
    """
    m, k, _ = pred_ctrl.shape
    n = gt_ctrl.shape[0]
    pe = torch.cat([pred_ctrl[:, 0, :], pred_ctrl[:, -1, :]], dim=1)  # [M, 4]
    ge = torch.cat([gt_ctrl[:, 0, :], gt_ctrl[:, -1, :]], dim=1)  # [N, 4]
    # [M, N]
    c_geom = torch.cdist(pe, ge, p=1.0)
    if pred_conf is not None and gt_conf is not None:
        c_conf = torch.abs(pred_conf.view(-1, 1) - gt_conf.view(1, -1))
        c = w_endpoint * c_geom + w_conf * c_conf
    else:
        c = w_endpoint * c_geom
    return c.detach().cpu().numpy()


@dataclass
class E2EMatchResult:
    pred_indices: List[int]
    gt_indices: List[int]


class E2EPolylineSetCriterion(nn.Module):
    """
    DETR-style helper: build a cost matrix, run Hungarian assignment, apply
    Smooth L1 on matched Bézier control points and/or Chamfer on sampled polylines.

    Intended for use when GT polylines are available as tensors (future dataset hook).
    """

    def __init__(
        self,
        curve_loss: str = "smooth_l1_ctrl",
        w_ctrl: float = 1.0,
        w_chamfer: float = 0.0,
        w_endpoint: float = 1.0,
        w_conf_cost: float = 0.5,
        beta: float = 1.0,
    ):
        super().__init__()
        if curve_loss not in ("smooth_l1_ctrl", "chamfer", "both"):
            raise ValueError("curve_loss must be smooth_l1_ctrl|chamfer|both")
        self.curve_loss = curve_loss
        self.w_ctrl = float(w_ctrl)
        self.w_chamfer = float(w_chamfer)
        self.w_endpoint = float(w_endpoint)
        self.w_conf_cost = float(w_conf_cost)
        self.beta = float(beta)

    def hungarian_match(
        self,
        pred_ctrl: torch.Tensor,
        gt_ctrl: torch.Tensor,
        pred_conf: Optional[torch.Tensor] = None,
        gt_conf: Optional[torch.Tensor] = None,
    ) -> E2EMatchResult:
        if linear_sum_assignment is None:
            raise ImportError("scipy is required for Hungarian matching in E2EPolylineSetCriterion.")
        m, n = pred_ctrl.shape[0], gt_ctrl.shape[0]
        if m == 0 or n == 0:
            return E2EMatchResult([], [])
        cost = hungarian_cost_matrix(
            pred_ctrl, gt_ctrl, pred_conf, gt_conf,
            w_endpoint=self.w_endpoint, w_conf=self.w_conf_cost,
        )
        r, c = linear_sum_assignment(cost)
        return E2EMatchResult([int(x) for x in r], [int(x) for x in c])

    def forward(
        self,
        pred_curve_px: torch.Tensor,
        pred_ctrl_px: torch.Tensor,
        gt_curve_px: torch.Tensor,
        gt_ctrl_px: torch.Tensor,
        pred_conf: Optional[torch.Tensor] = None,
        gt_conf: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, E2EMatchResult]:
        """
        pred_curve_px: [M, T, 2], pred_ctrl_px: [M, K, 2]
        gt_curve_px:   [N, Tg, 2], gt_ctrl_px: [N, K, 2]
        pred_conf: [M], gt_conf: [N] optional
        """
        match = self.hungarian_match(pred_ctrl_px, gt_ctrl_px, pred_conf, gt_conf)
        if len(match.pred_indices) == 0:
            z = pred_curve_px.new_zeros(())
            return z, match

        pi = torch.tensor(match.pred_indices, device=pred_curve_px.device, dtype=torch.long)
        gi = torch.tensor(match.gt_indices, device=pred_curve_px.device, dtype=torch.long)
        pc = pred_ctrl_px.index_select(0, pi)
        gc = gt_ctrl_px.index_select(0, gi)
        pcr = pred_curve_px.index_select(0, pi)
        gcr = gt_curve_px.index_select(0, gi)

        loss = pred_curve_px.new_zeros(())
        if self.curve_loss in ("smooth_l1_ctrl", "both"):
            loss = loss + self.w_ctrl * F.smooth_l1_loss(pc, gc, beta=self.beta, reduction="mean")
        if self.curve_loss in ("chamfer", "both"):
            ch = []
            for i in range(pcr.shape[0]):
                ch.append(chamfer_distance_l1(pcr[i], gcr[i]))
            loss = loss + self.w_chamfer * torch.stack(ch).mean()
        return loss, match
