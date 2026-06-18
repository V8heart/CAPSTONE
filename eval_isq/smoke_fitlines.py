#!/usr/bin/env python3
"""Smoke test: union-find (legacy) vs YOLinO fit_lines CC geom grouping on 5 val images."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("YOLINO_IGNORE_DIRTY", "1")

EVAL_ISQ_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(EVAL_ISQ_DIR)
sys.path.insert(0, EVAL_ISQ_DIR)
sys.path.insert(0, os.path.join(BASE, "eval_pixel_f1"))
sys.path.insert(0, os.path.join(BASE, "yolino", "CAPSTONE", "src"))

import torch  # noqa: E402

import eval_pixel_f1 as epf  # noqa: E402
from eval_pixel_f1 import EXP19_CKPT, EXP19_CONFIG, run_geom_predictions  # noqa: E402
from isq_core import (  # noqa: E402
    DEFAULT_ADJACENCY_THRESHOLD,
    DEFAULT_MIN_SEGMENTS_FOR_POLYLINE,
    Segment,
    _geom_segments_to_fitlines_cc,
    compute_isq_image,
    geom_polylines_to_pred_polylines,
    gt_polylines_to_segments,
    segment_tuples_to_pred_polylines,
    summarize_isq,
)
from quick_compare import _load_isq_records, _setup_run_subset, _val_stems  # noqa: E402
from yolino.visualize_ttpla_gt_instances import load_ttpla_label_with_ids  # noqa: E402


def _legacy_union_find(segments: list[Segment]) -> list[list[Segment]]:
    """Old ISQ grouping (endpoint union-find, sqrt(768) px)."""
    import numpy as np
    from collections import defaultdict

    n = len(segments)
    if n == 0:
        return []
    parent = list(range(n))
    thresh2 = 768.0

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            connected = False
            for pi in (0, 1):
                for pj in (0, 1):
                    a = np.asarray(segments[i][pi], dtype=float)
                    b = np.asarray(segments[j][pj], dtype=float)
                    d2 = float(np.sum((a - b) ** 2))
                    if d2 <= thresh2:
                        connected = True
                        break
                if connected:
                    break
            if connected:
                union(i, j)

    groups: dict[int, list[Segment]] = defaultdict(list)
    for i, seg in enumerate(segments):
        groups[find(i)].append(seg)
    return list(groups.values())


def _raw_segments(geom_polylines) -> list[Segment]:
    segs: list[Segment] = []
    for poly in geom_polylines:
        if len(poly) < 2:
            continue
        segs.append(
            ((float(poly[0][0]), float(poly[0][1])), (float(poly[1][0]), float(poly[1][1])))
        )
    return segs


def _eval_grouping(records, stems, pred_fn, label: str) -> dict:
    per = []
    for stem in stems:
        h, w, polylines, instance_ids = _load_isq_records(epf.DATASET_ROOT, [stem])[stem]
        gt = gt_polylines_to_segments(polylines, instance_ids, image_size=max(h, w))
        raw = _raw_segments(records[stem].geom_polylines)
        pred = pred_fn(raw)
        m = compute_isq_image(gt, pred)
        n_polys = len(pred)
        n_segs = sum(len(p) for p in pred)
        m["stem"] = stem
        m["n_pred_polylines"] = n_polys
        m["n_pred_segments"] = n_segs
        per.append(m)
        print(
            "  %-6s  polylines=%4d  segs=%4d  F1=%.4f  OS=%.2f  UM=%.2f"
            % (stem, n_polys, n_segs, m["f1"], m["over_split"], m["under_merge"])
        )
    s = summarize_isq(per)
    print(
        "%-12s summary: polylines/img=%.1f  F1=%.4f  P=%.4f  R=%.4f"
        % (
            label,
            sum(x["n_pred_polylines"] for x in per) / len(per),
            s["f1"],
            s["precision"],
            s["recall"],
        )
    )
    return s


def main() -> None:
    dataset_root = epf.DATASET_ROOT
    stems_all = _val_stems(dataset_root, 0)
    stems: list[str] = []
    label_dir = os.path.join(dataset_root, "labels", "val")
    for stem in stems_all:
        pl, ids, _ = load_ttpla_label_with_ids(os.path.join(label_dir, stem + ".npy"))
        if len(ids) >= 3:
            stems.append(stem)
        if len(stems) >= 5:
            break
    print("[INFO] smoke stems:", stems)

    use_gpu = torch.cuda.is_available()
    geom_args, fwd, ds, loader, dev = _setup_run_subset(
        EXP19_CONFIG, EXP19_CKPT, dataset_root, use_gpu, stems
    )
    records = epf.load_records(dataset_root, stems)
    run_geom_predictions(fwd, ds, loader, dev, records)

    adj = float(geom_args.adjacency_threshold)
    min_seg = int(geom_args.min_segments_for_polyline)
    print("[INFO] adjacency_threshold=%s min_segments_for_polyline=%d" % (adj, min_seg))

    print("\n=== Legacy union-find ===")
    _eval_grouping(records, stems, _legacy_union_find, "union-find")

    print("\n=== YOLinO fit_lines CC ===")
    _eval_grouping(
        records,
        stems,
        lambda segs: _geom_segments_to_fitlines_cc(
            segs, adjacency_threshold=adj, min_segments_for_polyline=min_seg
        ),
        "fit_lines",
    )

    print("\n[OK] smoke test done — proceed to full nohup runs if grouping looks reasonable.")


if __name__ == "__main__":
    main()
