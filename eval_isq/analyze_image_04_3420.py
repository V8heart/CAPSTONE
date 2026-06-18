#!/usr/bin/env python3
"""Per-polyline ISQ breakdown for 04_3420_L (geom vs GNN)."""
from __future__ import annotations

import os
import pickle
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PIXEL = os.path.join(BASE, "eval_pixel_f1")
sys.path.insert(0, EVAL_PIXEL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "yolino", "CAPSTONE", "src")
sys.path.insert(0, SRC)

from yolino.visualize_ttpla_gt_instances import load_ttpla_label_with_ids  # noqa: E402
from isq_core import (  # noqa: E402
    MIN_SEGMENTS_PER_PRED_POLYLINE,
    build_gt_spatial_index,
    geom_polylines_to_pred_polylines,
    gnn_segments_to_polylines,
    gt_polylines_to_segments,
    match_pred_segment_to_gt,
    segment_tuples_to_pred_polylines,
)

STEM = "04_3420_L"
DATASET = os.path.join(BASE, "yolino", "ttpla_yolino_dataset_1024x1024_smallset")
GEOM_PKL = os.path.join(os.path.dirname(__file__), "pred_geom_quick.pkl")
GNN_PKL = os.path.join(os.path.dirname(__file__), "pred_gnn_quick.pkl")


def analyze_polylines(
    gt_segments,
    pred_polylines: List,
    label: str,
) -> Tuple[dict, List[dict]]:
    gt_index = build_gt_spatial_index(gt_segments)
    gt_by_idx = {g.seg_idx: g for g in gt_segments}
    gt_seg_by_id: Dict[int, List[int]] = defaultdict(list)
    for g in gt_segments:
        gt_seg_by_id[g.polyline_id].append(g.seg_idx)

    matches = []
    for pi, poly in enumerate(pred_polylines):
        for si, seg in enumerate(poly):
            gt_idx, gt_pid, dist = match_pred_segment_to_gt(seg, gt_segments, gt_index=gt_index)
            matches.append(
                {
                    "pi": pi,
                    "si": si,
                    "gt_idx": gt_idx,
                    "gt_pid": gt_pid,
                    "dist": dist,
                }
            )

    eval_indices = [pi for pi, p in enumerate(pred_polylines) if len(p) >= MIN_SEGMENTS_PER_PRED_POLYLINE]
    dominant: Dict[int, Optional[int]] = {}
    id_counts_per_poly: Dict[int, Counter] = {}
    for pi in eval_indices:
        ids = [m["gt_pid"] for m in matches if m["pi"] == pi and m["gt_pid"] is not None]
        id_counts_per_poly[pi] = Counter(ids)
        dominant[pi] = id_counts_per_poly[pi].most_common(1)[0][0] if ids else None

    claimed_gt: set = set()
    per_poly_rows: List[dict] = []
    total_tp = total_fp = 0

    for pi in sorted(set(range(len(pred_polylines))) | set(eval_indices)):
        poly_matches = [m for m in matches if m["pi"] == pi]
        n_seg = len(pred_polylines[pi])
        in_eval = pi in dominant
        dom = dominant.get(pi)
        tp = fp = 0
        fp_no_match = fp_wrong_id = fp_dup_gt = fp_not_eval = 0

        for m in poly_matches:
            if not in_eval:
                fp_not_eval += 1
                continue
            if dom is None or m["gt_pid"] != dom or m["gt_idx"] is None:
                fp += 1
                if m["gt_idx"] is None:
                    fp_no_match += 1
                elif dom is None:
                    fp_wrong_id += 1
                else:
                    fp_wrong_id += 1
                continue
            if m["gt_idx"] in claimed_gt:
                fp += 1
                fp_dup_gt += 1
                continue
            claimed_gt.add(m["gt_idx"])
            tp += 1

        total_tp += tp
        total_fp += fp
        matched_ids = sorted(set(m["gt_pid"] for m in poly_matches if m["gt_pid"] is not None))
        gt_hits_by_id = defaultdict(set)
        for m in poly_matches:
            if m["gt_idx"] is not None and m["gt_pid"] is not None:
                gt_hits_by_id[m["gt_pid"]].add(m["gt_idx"])

        per_poly_rows.append(
            {
                "poly_idx": pi,
                "n_segments": n_seg,
                "in_eval": in_eval,
                "dominant_id": dom,
                "id_vote_counts": dict(id_counts_per_poly.get(pi, {})),
                "matched_gt_ids": matched_ids,
                "gt_hits_n_segs": {gid: len(s) for gid, s in sorted(gt_hits_by_id.items())},
                "tp": tp,
                "fp": fp,
                "fp_no_match": fp_no_match,
                "fp_wrong_id": fp_wrong_id,
                "fp_dup_gt": fp_dup_gt,
                "fp_not_eval": fp_not_eval,
            }
        )

    fn = sum(1 for g in gt_segments if g.seg_idx not in claimed_gt)
    summary = {
        "label": label,
        "n_polylines": len(pred_polylines),
        "n_eval_polylines": len(eval_indices),
        "n_gt_segments": len(gt_segments),
        "n_gt_instances": len(gt_seg_by_id),
        "tp": total_tp,
        "fp": total_fp,
        "fn": fn,
        "precision": total_tp / (total_tp + total_fp + 1e-8),
        "recall": total_tp / (total_tp + fn + 1e-8),
        "claimed_gt_instances": sorted({gt_by_idx[i].polyline_id for i in claimed_gt}),
    }
    return summary, per_poly_rows


def print_report(summary: dict, rows: List[dict]) -> None:
    print("\n=== %s ===" % summary["label"])
    print(
        "polylines=%d eval=%d | GT segs=%d instances=%d | TP=%d FP=%d FN=%d | P=%.3f R=%.3f"
        % (
            summary["n_polylines"],
            summary["n_eval_polylines"],
            summary["n_gt_segments"],
            summary["n_gt_instances"],
            summary["tp"],
            summary["fp"],
            summary["fn"],
            summary["precision"],
            summary["recall"],
        )
    )
    print("claimed GT instance ids:", summary["claimed_gt_instances"])
    print(
        "\n%-4s %5s %5s %12s %-30s %-20s %4s %4s  FP breakdown"
        % ("poly", "segs", "eval?", "dom_id", "id_votes", "gt_hits(#segs)", "TP", "FP")
    )
    print("-" * 100)
    for r in rows:
        votes = ", ".join("%d:%d" % (k, v) for k, v in sorted(r["id_vote_counts"].items(), key=lambda x: -x[1])[:6])
        hits = ", ".join("%d:%d" % (k, v) for k, v in r["gt_hits_n_segs"].items())
        print(
            "%-4d %5d %5s %12s %-30s %-20s %4d %4d  nomatch=%d wrong_id=%d dup=%d skip=%d"
            % (
                r["poly_idx"],
                r["n_segments"],
                "Y" if r["in_eval"] else "N",
                str(r["dominant_id"]),
                votes or "-",
                hits or "-",
                r["tp"],
                r["fp"],
                r["fp_no_match"],
                r["fp_wrong_id"],
                r["fp_dup_gt"],
                r["fp_not_eval"],
            )
        )


def main() -> None:
    label_path = os.path.join(DATASET, "labels", "val", STEM + ".npy")
    polylines, instance_ids, _ = load_ttpla_label_with_ids(label_path)
    gt_segments = gt_polylines_to_segments(polylines, instance_ids, image_size=1024)

    print("GT instance ids:", sorted(set(instance_ids)))
    gt_by_inst = Counter()
    for g in gt_segments:
        gt_by_inst[g.polyline_id] += 1
    print("GT segments per instance:", dict(sorted(gt_by_inst.items())))

    with open(GEOM_PKL, "rb") as f:
        geom_pkl = pickle.load(f)
    with open(GNN_PKL, "rb") as f:
        gnn_pkl = pickle.load(f)

    geom_polys = segment_tuples_to_pred_polylines(geom_pkl[STEM])
    gnn_polys = segment_tuples_to_pred_polylines(gnn_pkl[STEM])

    geom_sum, geom_rows = analyze_polylines(gt_segments, geom_polys, "Pred-1 geom")
    gnn_sum, gnn_rows = analyze_polylines(gt_segments, gnn_polys, "Pred-2 GNN")

    print_report(geom_sum, geom_rows)
    print_report(gnn_sum, gnn_rows)

    # Under-merge check for GNN eval polylines
    print("\n=== GNN under-merge check (eval polylines with 2+ GT instances hit) ===")
    for r in gnn_rows:
        if not r["in_eval"]:
            continue
        n_gt_inst = len(r["gt_hits_n_segs"])
        if n_gt_inst >= 2:
            print(
                "  poly %d: dominant=%s hits %s -> TP=%d FP=%d (wrong_id FP likely from non-dominant GT)"
                % (r["poly_idx"], r["dominant_id"], r["gt_hits_n_segs"], r["tp"], r["fp"])
            )

    # GT instances never claimed by GNN
    gnn_claimed = set()
    for g in gt_segments:
        for m in gnn_rows:
            pass
    claimed_inst_gnn = set(geom_sum["claimed_gt_instances"])  # wrong
    from isq_core import compute_isq_image

    gnn_m = compute_isq_image(gt_segments, gnn_polys)
    geom_m = compute_isq_image(gt_segments, geom_polys)
    print("\n=== compute_isq_image cross-check ===")
    print("geom", geom_m["tp"], geom_m["fp"], geom_m["fn"])
    print("gnn ", gnn_m["tp"], gnn_m["fp"], gnn_m["fn"])

    all_gt_ids = set(gt_by_inst.keys())
    gnn_claimed_ids = set()
    for r in gnn_rows:
        if r["tp"] > 0:
            gnn_claimed_ids.add(r["dominant_id"])
    print("\nGT instances (%d total):" % len(all_gt_ids), sorted(all_gt_ids))
    print("GNN dominant ids with any TP:", sorted(gnn_claimed_ids))
    print("Never dominant+TP (missed instances):", sorted(all_gt_ids - gnn_claimed_ids))


if __name__ == "__main__":
    main()
