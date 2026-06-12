# SPDX-License-Identifier: GPL-3.0-or-later
"""Val-only GNN topology metrics (instance IoU via connected components)."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


def _union_find_components(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for u, v in edges:
        union(u, v)

    groups: dict = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())


def _instance_iou_one_image(
    seg_inst: np.ndarray,
    node_valid: np.ndarray,
    pred_labels: np.ndarray,
) -> Tuple[float, int]:
    """Mean IoU over foreground GT instances with ≥1 assigned node."""
    ious = []
    gt_ids = np.unique(seg_inst[(seg_inst >= 0) & node_valid])
    for gid in gt_ids:
        g_nodes = np.where((seg_inst == gid) & node_valid)[0]
        if g_nodes.size == 0:
            continue
        g_set = set(int(x) for x in g_nodes)
        best_iou = 0.0
        pred_ids = np.unique(pred_labels[g_nodes])
        for pid in pred_ids:
            if pid < 0:
                continue
            c_nodes = np.where(pred_labels == pid)[0]
            c_set = set(int(x) for x in c_nodes)
            inter = len(g_set & c_set)
            union = len(g_set | c_set)
            if union > 0:
                best_iou = max(best_iou, float(inter) / float(union))
        ious.append(best_iou)
    if not ious:
        return 0.0, 0
    return float(np.mean(ious)), len(ious)


def instance_iou_mean(
    edge_logits: torch.Tensor,
    neighbors: torch.Tensor,
    neigh_valid: torch.Tensor,
    seg_inst: torch.Tensor,
    node_valid: torch.Tensor,
    *,
    edge_thresh: float = 0.3,
) -> float:
    """Mean instance-level IoU (val / no_grad only).

    Builds an undirected graph from edges with ``sigmoid(logit) >= edge_thresh``,
    union-find CCs, then matches each GT instance to best-overlap pred CC.
    """
    with torch.no_grad():
        probs = torch.sigmoid(edge_logits)
        b, n, k = edge_logits.shape
        scores = []
        n_gt = 0
        for bi in range(b):
            si = seg_inst[bi].detach().cpu().numpy()
            nv = node_valid[bi].detach().cpu().numpy().astype(bool)
            neigh = neighbors[bi].detach().cpu().numpy()
            valid = neigh_valid[bi].detach().cpu().numpy().astype(bool)
            p = probs[bi].detach().cpu().numpy()

            edges: List[Tuple[int, int]] = []
            for i in range(n):
                if not nv[i]:
                    continue
                for kk in range(k):
                    if not valid[i, kk]:
                        continue
                    j = int(neigh[i, kk])
                    if j < 0 or j >= n or not nv[j] or i >= j:
                        continue
                    if float(p[i, kk]) >= float(edge_thresh):
                        edges.append((i, j))

            comps = _union_find_components(n, edges)
            pred_labels = np.full(n, -1, dtype=np.int64)
            for ci, nodes in enumerate(comps):
                for nd in nodes:
                    if nv[nd]:
                        pred_labels[nd] = ci

            iou, cnt = _instance_iou_one_image(si, nv, pred_labels)
            if cnt > 0:
                scores.append(iou)
                n_gt += cnt

        if not scores:
            return 0.0
        return float(np.mean(scores))
