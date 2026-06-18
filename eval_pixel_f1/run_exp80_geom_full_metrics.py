#!/usr/bin/env python3
"""Run all eval_pixel_f1 metric suites for exp80 geom (best_model) on YOLinO_benchmark."""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_BASE, "eval_isq"))
sys.path.insert(0, os.path.join(_BASE, "CAPSTONE", "src"))

import eval_pixel_f1 as ep1  # noqa: E402

ep1.CAPSTONE_ROOT = os.path.join(_BASE, "CAPSTONE")
ep1.BASE = _BASE
ep1.DATASET_ROOT = os.path.join(_BASE, "YOLinO_benchmark")
ep1.ORIG_JSON_DIR = os.path.join(_BASE, "data_original_size_v1", "data_original_size")
ep1.SPLIT = "test"
ep1.GT_A_STRETCH = True
ep1.GT_A_CROP_AWARE = False

import torch  # noqa: E402

CAPSTONE = ep1.CAPSTONE_ROOT
EXP80_CONFIG = os.path.join(CAPSTONE, "configs", "experiments", "exp80_ttpla_512512_scale16.yaml")
EXP80_CKPT = os.path.join(
    CAPSTONE,
    "ttpla_train_exp",
    "log",
    "checkpoints",
    "exp80_ttpla_512512_scale16",
    "best_model.pth",
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT_JSON = os.path.join(OUT_DIR, "exp80_best_model_pixel_f1_full.json")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    use_gpu = torch.cuda.is_available()
    print("[INFO] device:", "cuda" if use_gpu else "cpu")
    print("[INFO] checkpoint:", EXP80_CKPT)
    print("[INFO] dataset:", ep1.DATASET_ROOT, "split:", ep1.SPLIT)

    img_dir = os.path.join(ep1.DATASET_ROOT, "images", ep1.SPLIT)
    file_names = sorted(
        os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith(".png")
    )
    print("[INFO] images:", len(file_names))

    records = ep1.load_records(ep1.DATASET_ROOT, file_names)
    _args, forward, dataset, loader, device = ep1.setup_run(
        EXP80_CONFIG, EXP80_CKPT, ep1.DATASET_ROOT, use_gpu
    )
    ep1.run_geom_predictions(forward, dataset, loader, device, records)

    results = {
        "model": "exp80_ttpla_512512_scale16",
        "checkpoint": EXP80_CKPT,
        "config": EXP80_CONFIG,
        "pred_kind": "geom",
        "settings": {
            "dataset_root": ep1.DATASET_ROOT,
            "split": ep1.SPLIT,
            "geom_confidence": ep1.GEOM_CONF,
            "gt_a_stretch": ep1.GT_A_STRETCH,
            "gt_a_crop_aware": ep1.GT_A_CROP_AWARE,
            "orig_json_dir": ep1.ORIG_JSON_DIR,
            "lsnetv2_beta_sq": ep1.LSNETV2_BETA_SQ,
        },
        "metrics": {},
    }

    # 1) GT-B thickness sweep (2, 5, 10 px)
    gt_b = {}
    for thickness in ep1.THICKNESS_VALUES:
        block = ep1.evaluate_gt_b_vs_pred(records, "geom", thickness)
        gt_b[str(thickness)] = block["summary"]
        s = block["summary"]
        print(
            "[GT-B t=%d] P=%.4f R=%.4f F1=%.4f IoU=%.4f"
            % (thickness, s["precision"], s["recall"], s["f1"], s["iou"])
        )
    results["metrics"]["gt_b_thickness_sweep"] = gt_b

    # 2) LSNetv2-style GT-B (line widths 2, 4 px)
    lsnetv2 = {}
    for line_width in ep1.LSNETV2_LINE_WIDTHS:
        block = ep1.evaluate_lsnetv2_gt_b(records, "geom", line_width)
        lsnetv2[str(line_width)] = block["summary"]
        s = block["summary"]
        print(
            "[LSNetv2 GT-B lw=%d] APR=%.4f ARR=%.4f F1=%.4f Fβ=%.4f"
            % (line_width, s["APR"], s["ARR"], s["F1"], s["F_beta"])
        )
    results["metrics"]["lsnetv2_gt_b"] = lsnetv2

    # 3) GT-A polygon thickness sweep (4, 5, 8, 10 px)
    gt_a_thick = {}
    for thickness in ep1.GTA_THICKNESS_SWEEP:
        block = ep1.evaluate_gt_a_thickness(records, "geom", thickness)
        gt_a_thick[str(thickness)] = block["summary"]
        s = block["summary"]
        print(
            "[GT-A t=%d] P=%.4f R=%.4f F1=%.4f Fβ=%.4f"
            % (thickness, s["precision"], s["recall"], s["f1"], s["f_beta"])
        )
    results["metrics"]["gt_a_thickness_sweep"] = gt_a_thick

    # 4) GT-A PLGAN-style relaxation (2px @ 512²)
    relax_specs = [(2, "2px_512_reference")]
    relax_out = {}
    for relax_px, label in relax_specs:
        block = ep1.evaluate_gt_a_relaxed(
            records, "geom", relax_px, pred_thickness=ep1.RELAX_PRED_THICKNESS
        )
        relax_out[label] = block["summary"]
        s = block["summary"]
        print(
            "[GT-A relax %s] correctness=%.4f completeness=%.4f quality=%.4f "
            "relaxed_F1=%.4f strict_F1=%.4f IoU=%.4f"
            % (
                label,
                s["correctness"],
                s["completeness"],
                s["quality"],
                s["relaxed_f1"],
                s["f1"],
                s["iou"],
            )
        )
    results["metrics"]["gt_a_relaxation"] = relax_out

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("[OK] Wrote", OUT_JSON)


if __name__ == "__main__":
    main()
