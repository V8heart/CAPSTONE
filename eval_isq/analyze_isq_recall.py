#!/usr/bin/env python3
"""Diagnose low ISQ recall: GT segment counts, match rates, polyline stats, TP funnel."""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "yolino", "CAPSTONE", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from yolino.visualize_ttpla_gt_instances import load_ttpla_label_with_ids  # noqa: E402

from isq_core import (  # noqa: E402
    MIN_SEGMENTS_PER_PRED_POLYLINE,
    compute_isq_image,
    gt_polylines_to_segments,
    match_pred_segment_to_gt,
    segment_tuples_to_pred_polylines,
)

DATASET_ROOT = os.path.join(BASE, "yolino", "ttpla_yolino_dataset_1024x1024_smallset")
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
QUICK_GEOM = os.path.join(EVAL_DIR, "pred_geom_quick.pkl")
QUICK_GNN = os.path.join(EVAL_DIR, "pred_gnn_quick.pkl")
FULL_GEOM = os.path.join(EVAL_DIR, "pred_geom.pkl")
FULL_GNN = os.path.join(EVAL_DIR, "pred_gnn.pkl")
OUT_JSON = os.path.join(EVAL_DIR, "results", "isq_recall_diagnosis.json")


def diagnose_image(
    stem: str,
    polylines,
    instance_ids,
    pred_by_stem: Dict,
    image_size: int,
) -> dict:
    gt_segments = gt_polylines_to_segments(polylines, instance_ids, image_size=image_size)
    pred_polylines = segment_tuples_to_pred_polylines(pred_by_stem[stem])

    n_pred_segs = sum(len(p) for p in pred_polylines)
    n_poly_2plus = sum(1 for p in pred_polylines if len(p) >= MIN_SEGMENTS_PER_PRED_POLYLINE)
    seg_per_poly = [len(p) for p in pred_polylines]

    # Raw matching (step 2) before dominant_id / TP rules
    matched_any = 0
    matched_wrong_id = 0
    dists_matched = []
    for pi, poly in enumerate(pred_polylines):
        for seg in poly:
            gt_idx, gt_pid, dist = match_pred_segment_to_gt(seg, gt_segments)
            if gt_idx is not None:
                matched_any += 1
                dists_matched.append(dist)

    # Dominant_id funnel
    from isq_core import build_gt_spatial_index

    gt_index = build_gt_spatial_index(gt_segments)
    matches = []
    for pi, poly in enumerate(pred_polylines):
        for si, seg in enumerate(poly):
            gt_idx, gt_pid, dist = match_pred_segment_to_gt(seg, gt_segments, gt_index=gt_index)
            matches.append((pi, si, gt_idx, gt_pid, dist))

    eval_indices = [pi for pi, p in enumerate(pred_polylines) if len(p) >= MIN_SEGMENTS_PER_PRED_POLYLINE]
    dominant: Dict[int, Optional[int]] = {}
    for pi in eval_indices:
        ids = [m[3] for m in matches if m[0] == pi and m[3] is not None]
        dominant[pi] = Counter(ids).most_common(1)[0][0] if ids else None

    in_eval = sum(1 for m in matches if m[0] in dominant)
    dom_match = sum(
        1 for m in matches if m[0] in dominant and m[3] == dominant[m[0]] and m[2] is not None
    )
    claimed = set()
    tp = fp = 0
    fp_reasons = Counter()
    for m in matches:
        pi, _si, gt_idx, gt_pid, _d = m
        if pi not in dominant:
            continue
        dom = dominant[pi]
        if dom is None or gt_pid != dom or gt_idx is None:
            fp += 1
            if gt_idx is None:
                fp_reasons["no_gt_match"] += 1
            elif dom is None:
                fp_reasons["no_dominant_id"] += 1
            else:
                fp_reasons["wrong_gt_instance"] += 1
            continue
        if gt_idx in claimed:
            fp += 1
            fp_reasons["duplicate_gt_seg"] += 1
            continue
        claimed.add(gt_idx)
        tp += 1

    fn = len(gt_segments) - len(claimed)
    n_gt_pts = sum(len(p) for p in polylines)

    return {
        "stem": stem,
        "n_gt_polylines": len(polylines),
        "n_gt_polyline_points": int(n_gt_pts),
        "n_gt_grid_segments": len(gt_segments),
        "gt_segments_per_instance": len(gt_segments) / max(len(polylines), 1),
        "n_pred_polylines": len(pred_polylines),
        "n_pred_segments": n_pred_segs,
        "n_pred_polylines_ge2_segs": n_poly_2plus,
        "pred_segments_per_polyline_max": max(seg_per_poly) if seg_per_poly else 0,
        "pred_segments_per_polyline_mean": float(np.mean(seg_per_poly)) if seg_per_poly else 0.0,
        "pred_match_rate_any_gt": matched_any / max(n_pred_segs, 1),
        "n_pred_matched_to_gt": matched_any,
        "match_dist_mean_px": float(np.mean(dists_matched)) if dists_matched else None,
        "n_pred_in_eval_polylines": in_eval,
        "n_pred_dom_match": dom_match,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fp_reasons": dict(fp_reasons),
        "recall": tp / (tp + fn + 1e-8),
        "precision": tp / (tp + fp + 1e-8),
        "ratio_pred_gt_segments": n_pred_segs / max(len(gt_segments), 1),
        "micro_recall_if_all_matches_were_tp": matched_any / max(len(gt_segments), 1),
    }


def aggregate(rows: List[dict]) -> dict:
    keys = [
        "n_gt_grid_segments",
        "n_pred_segments",
        "n_pred_polylines_ge2_segs",
        "pred_match_rate_any_gt",
        "ratio_pred_gt_segments",
        "micro_recall_if_all_matches_were_tp",
        "recall",
        "precision",
    ]
    out = {"n_images": len(rows)}
    for k in keys:
        out["mean_" + k] = float(np.mean([r[k] for r in rows]))
    out["total_gt_segments"] = int(sum(r["n_gt_grid_segments"] for r in rows))
    out["total_pred_segments"] = int(sum(r["n_pred_segments"] for r in rows))
    out["total_pred_matched"] = int(sum(r["n_pred_matched_to_gt"] for r in rows))
    out["total_tp"] = int(sum(r["tp"] for r in rows))
    out["total_fn"] = int(sum(r["fn"] for r in rows))
    out["micro_recall_pooled"] = out["total_tp"] / (out["total_tp"] + out["total_fn"] + 1e-8)
    out["micro_match_rate"] = out["total_pred_matched"] / max(out["total_pred_segments"], 1)
    fp_all = Counter()
    for r in rows:
        fp_all.update(r.get("fp_reasons", {}))
    out["fp_reasons_pooled"] = dict(fp_all)
    return out


def run_pack(geom_pkl: str, gnn_pkl: str, stems: List[str]) -> dict:
    with open(geom_pkl, "rb") as f:
        geom = pickle.load(f)
    with open(gnn_pkl, "rb") as f:
        gnn = pickle.load(f)
    label_dir = os.path.join(DATASET_ROOT, "labels", "val")
    img_dir = os.path.join(DATASET_ROOT, "images", "val")

    geom_rows, gnn_rows = [], []
    for stem in stems:
        import cv2

        img = cv2.imread(os.path.join(img_dir, stem + ".png"))
        h, w = img.shape[:2]
        pl, ids, _ = load_ttpla_label_with_ids(os.path.join(label_dir, stem + ".npy"))
        geom_rows.append(diagnose_image(stem, pl, ids, geom, max(h, w)))
        gnn_rows.append(diagnose_image(stem, pl, ids, gnn, max(h, w)))

    return {
        "stems": stems,
        "Pred-1_geom": {"per_image": geom_rows, "aggregate": aggregate(geom_rows)},
        "Pred-2_gnn": {"per_image": gnn_rows, "aggregate": aggregate(gnn_rows)},
    }


def main() -> None:
    # Prefer quick_compare pickles (5 images); fall back to full val.
    if os.path.isfile(QUICK_GEOM):
        geom_pkl, gnn_pkl = QUICK_GEOM, QUICK_GNN
        with open(geom_pkl, "rb") as f:
            stems = sorted(pickle.load(f).keys())
    else:
        geom_pkl, gnn_pkl = FULL_GEOM, FULL_GNN
        with open(geom_pkl, "rb") as f:
            stems = sorted(pickle.load(f).keys())

    report = run_pack(geom_pkl, gnn_pkl, stems)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for model in ("Pred-1_geom", "Pred-2_gnn"):
        a = report[model]["aggregate"]
        print("\n=== %s (%d images) ===" % (model, a["n_images"]))
        print("1) GT grid segments: mean=%.0f  total=%d  (per GT instance ~%.0f segs)"
              % (a["mean_n_gt_grid_segments"], a["total_gt_segments"],
                 a["mean_n_gt_grid_segments"] / max(
                     np.mean([r["n_gt_polylines"] for r in report[model]["per_image"]]), 1)))
        print("2) Pred→GT match rate: %.1f%%  (micro: %d/%d pred segments)"
              % (100 * a["micro_match_rate"], a["total_pred_matched"], a["total_pred_segments"]))
        print("   If every match counted as TP: micro recall=%.4f" % a["micro_recall_pooled"])
        print("3) Pred polylines with ≥2 segments: mean=%.1f / %.1f polylines"
              % (a["mean_n_pred_polylines_ge2_segs"], np.mean([r["n_pred_polylines"] for r in report[model]["per_image"]])))
        print("4) Actual TP funnel: TP=%d  FN=%d  micro_recall=%.4f  macro_recall=%.4f"
              % (a["total_tp"], a["total_fn"], a["micro_recall_pooled"], a["mean_recall"]))
        print("   FP reasons:", a["fp_reasons_pooled"])
        print("   Pred/GT segment ratio: %.4f" % a["mean_ratio_pred_gt_segments"])

    print("\n[OK]", OUT_JSON)


if __name__ == "__main__":
    main()
