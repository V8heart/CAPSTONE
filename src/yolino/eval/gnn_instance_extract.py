# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract GNN connected-component instances as segment groups for polyline IoU eval."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from yolino.utils.enums import Variables

SegmentXY = Tuple[np.ndarray, np.ndarray]


def _uv_row_to_cv2_segment(row: np.ndarray, geom_pos: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """UV_SPLIT row ``[V0,H0,V1,H1]`` → OpenCV segment endpoints ``(x,y)=(col,row)``."""
    g = [int(x) for x in geom_pos[:4]]
    # geom stores [V, H, V, H]; cv2 / GNN node_end use (H, V) = (col, row).
    return (
        (float(row[g[1]]), float(row[g[0]])),
        (float(row[g[3]]), float(row[g[2]])),
    )


def _uv_start_endpoints(lv: np.ndarray, geom_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start/end points in UV [V,H] order (matches ``fit_lines`` adjacency)."""
    g = [int(x) for x in geom_pos[:4]]
    startpoints = lv[:, [g[0], g[1]]]
    endpoints = lv[:, [g[2], g[3]]]
    return startpoints, endpoints


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def extract_gnn_cc_segment_groups(
    gnn_single: Dict,
    args,
    *,
    node_conf_thresh: float,
    edge_thresh: float | None = None,
) -> List[List[SegmentXY]]:
    """Return one segment group per GNN CC (each group = list of (end_a, end_b) segments).

    Uses ``line_neighbors`` when present (``directional2_ctx`` CC path), else full ``neighbors``.
    """
    node_valid = _to_numpy(gnn_single["node_valid"]).astype(bool)
    node_conf = gnn_single.get("node_conf")
    if node_conf is not None:
        node_conf = _to_numpy(node_conf).astype(np.float32)
        node_valid = node_valid & (node_conf >= float(node_conf_thresh))

    line_n = gnn_single.get("line_neighbors")
    if line_n is not None:
        neighbors = _to_numpy(line_n).astype(np.int64)
        neigh_valid = _to_numpy(gnn_single["line_neigh_valid"]).astype(bool)
        k_line = int(neighbors.shape[-1])
        el = gnn_single["edge_logits"]
        edge_logits = _to_numpy(el)[..., :k_line] if isinstance(el, torch.Tensor) else np.asarray(el)[:, :k_line]
    else:
        neighbors = _to_numpy(gnn_single["neighbors"]).astype(np.int64)
        neigh_valid = _to_numpy(gnn_single["neigh_valid"]).astype(bool)
        edge_logits = _to_numpy(gnn_single["edge_logits"])

    node_ea = _to_numpy(gnn_single.get("node_end_a_px"))
    node_eb = _to_numpy(gnn_single.get("node_end_b_px"))
    if node_ea is None or node_eb is None:
        return []

    if not np.any(node_valid):
        return []

    thresh = float(edge_thresh if edge_thresh is not None else getattr(args, "gnn_cc_edge_thresh", 0.3))
    probs = 1.0 / (1.0 + np.exp(-edge_logits.astype(np.float64)))
    n_nodes, k = neighbors.shape

    edge_pairs: set = set()
    for ni in range(n_nodes):
        if not bool(node_valid[ni]):
            continue
        for kj in range(k):
            if not bool(neigh_valid[ni, kj]):
                continue
            if float(probs[ni, kj]) < thresh:
                continue
            nj = int(neighbors[ni, kj])
            if nj < 0 or nj >= n_nodes or not bool(node_valid[nj]):
                continue
            a, b = (ni, nj) if ni < nj else (nj, ni)
            edge_pairs.add((a, b))

    if not edge_pairs:
        return []

    parent = np.arange(n_nodes, dtype=np.int32)

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    in_graph = np.zeros(n_nodes, dtype=bool)
    for a, b in edge_pairs:
        in_graph[a] = True
        in_graph[b] = True
        _union(a, b)

    groups: Dict[int, List[int]] = {}
    for i in range(n_nodes):
        if not bool(node_valid[i]) or not bool(in_graph[i]):
            continue
        r = _find(i)
        groups.setdefault(r, []).append(i)

    out: List[List[SegmentXY]] = []
    for nodes in groups.values():
        segs: List[SegmentXY] = []
        for idx in nodes:
            ea = node_ea[idx]
            eb = node_eb[idx]
            if np.any(np.isnan(ea)) or np.any(np.isnan(eb)):
                continue
            segs.append((ea.copy(), eb.copy()))
        if segs:
            out.append(segs)
    return out


def extract_post_cc_segment_groups(
    lines_uv: np.ndarray,
    coords,
    args,
) -> List[List[SegmentXY]]:
    """Post-process CC segment groups **before spline fitting** (adjacency merge only).

    Expects the same UV_SPLIT table as ``fit_lines`` input (e.g. from GNN nodes).
    """
    from yolino.postprocessing.line_fit import get_adjacency_list, get_connected_components

    if lines_uv is None or lines_uv.size == 0:
        return []
    lv = np.asarray(lines_uv, dtype=np.float64)
    if lv.ndim == 1:
        lv = lv.reshape(1, -1)
    geom_pos = np.asarray(coords.get_position_within_prediction(Variables.GEOMETRY)).ravel()
    if geom_pos.size < 4:
        raise ValueError("geometry columns required for post CC extract")

    segments = [_uv_row_to_cv2_segment(row, geom_pos) for row in lv]
    startpoints, endpoints = _uv_start_endpoints(lv, geom_pos)
    confidences = np.ones((lv.shape[0],), dtype=np.float64)

    adjacency, reversed_adjacency = get_adjacency_list(
        float(args.adjacency_threshold), endpoints, startpoints,
    )
    roots = [node for node in adjacency.keys() if not adjacency[node]]
    _, seg_id_groups = get_connected_components(
        confidences, "post_cc", np.zeros((1, 1, 3), dtype=np.uint8),
        int(args.min_segments_for_polyline), args.paths, reversed_adjacency,
        roots, segments, write_debug_images=False,
    )

    out: List[List[SegmentXY]] = []
    for seg_ids in seg_id_groups:
        group: List[SegmentXY] = []
        for sid in seg_ids:
            p0, p1 = segments[int(sid)]
            group.append((np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)))
        if group:
            out.append(group)
    return out


def splines_to_polylines_xy(splines: Sequence) -> List[np.ndarray]:
    """Convert ``fit_lines`` spline output to list of ``[N,2]`` (x, y) polylines."""
    out: List[np.ndarray] = []
    for inst in splines:
        if inst is None:
            continue
        arr = np.asarray(inst)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        row_px, col_px = arr[0], arr[1]
        poly = np.stack([col_px, row_px], axis=-1).astype(np.float64)
        if len(poly) >= 2:
            out.append(poly)
    return out
