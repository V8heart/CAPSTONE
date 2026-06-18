#!/usr/bin/env python3
"""Compare GNN segment counts: legacy (node+connector) vs ISQ fix (nodes only)."""
from __future__ import annotations

import os
import pickle
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PIXEL = os.path.join(BASE, "eval_pixel_f1")
sys.path.insert(0, EVAL_PIXEL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_pixel_f1 import (  # noqa: E402
    DATASET_ROOT,
    EXP55_CKPT,
    EXP55_CONFIG,
    GNN_EDGE_THRESH,
    GNN_NODE_CONF,
    SPLIT,
    extract_gnn_segments,
    pred_polylines_to_segment_tuples,
    setup_run,
    run_gnn_predictions,
    load_records,
)
from isq_core import gnn_segments_to_polylines, pred_polylines_to_segment_tuples as isq_tuples  # noqa: E402

STEMS = ["04_3420_L", "09_9180_R", "105_4350_R", "107_2505_R", "114_210_R"]


def count_legacy_connectors(gnn_single: dict) -> tuple[int, int, int]:
    """Returns (n_nodes_in_graph, n_edges, n_segments_if_adding_connectors)."""
    import numpy as np

    node_valid = gnn_single["node_valid"]
    if hasattr(node_valid, "detach"):
        node_valid = node_valid.detach().cpu().numpy().astype(bool)
    node_conf = gnn_single.get("node_conf")
    if node_conf is not None:
        nc = node_conf.detach().cpu().numpy() if hasattr(node_conf, "detach") else node_conf
        node_valid = node_valid & (nc >= GNN_NODE_CONF)

    neighbors = gnn_single["neighbors"]
    neigh_valid = gnn_single["neigh_valid"]
    edge_logits = gnn_single["edge_logits"]
    if hasattr(neighbors, "detach"):
        neighbors = neighbors.detach().cpu().numpy().astype(int)
        neigh_valid = neigh_valid.detach().cpu().numpy().astype(bool)
        edge_logits = edge_logits.detach().cpu().float().numpy()
    probs = 1.0 / (1.0 + np.exp(-edge_logits))
    n_nodes = neighbors.shape[0]
    edge_pairs = set()
    for ni in range(n_nodes):
        if not node_valid[ni]:
            continue
        for kj in range(neighbors.shape[1]):
            if not neigh_valid[ni, kj] or probs[ni, kj] < GNN_EDGE_THRESH:
                continue
            nj = int(neighbors[ni, kj])
            if nj < 0 or nj >= n_nodes or not node_valid[nj]:
                continue
            a, b = (ni, nj) if ni < nj else (nj, ni)
            edge_pairs.add((a, b))
    in_graph = {i for a, b in edge_pairs for i in (a, b)}
    n_node_segs = int(node_valid.sum())  # includes lonely + in-graph
    return len(in_graph), len(edge_pairs), n_node_segs + len(edge_pairs)


def main() -> None:
    import torch

    use_gpu = torch.cuda.is_available()
    records = load_records(DATASET_ROOT, STEMS)
    gnn_raw: dict = {}
    _a, fwd, ds, loader, dev = setup_run(EXP55_CONFIG, EXP55_CKPT, DATASET_ROOT, use_gpu)
    run_gnn_predictions(fwd, ds, loader, dev, records, gnn_raw)

    print("stem | geom(extract flat) | legacy flat | ISQ polylines | in_graph | edges")
    print("-" * 75)
    for stem in STEMS:
        g = gnn_raw[stem]
        flat_legacy = extract_gnn_segments(g, include_connectors=True)
        flat_nodes = extract_gnn_segments(g, include_connectors=False)
        polys = gnn_segments_to_polylines(flat_legacy, g)
        n_isq = sum(len(p) for p in polys)
        ig, ne, legacy_est = count_legacy_connectors(g)
        n_geom = len(pickle.load(open(os.path.join(os.path.dirname(__file__), "pred_geom_quick.pkl"), "rb"))[stem]) if os.path.isfile(os.path.join(os.path.dirname(__file__), "pred_geom_quick.pkl")) else -1
        print(
            "%s | geom_pkl=%d | flat_legacy=%d | ISQ_fixed=%d | nodes~%d edges=%d"
            % (stem, n_geom, len(flat_legacy), n_isq, len(flat_nodes), ne)
        )


if __name__ == "__main__":
    main()
