#!/usr/bin/env python3
"""ISQ eval for geom + GNN exp55 + GNN exp65 (val 65, GT-centric over-split)."""
from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PIXEL = os.path.join(BASE, "eval_pixel_f1")
EVAL_ISQ_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (EVAL_PIXEL, EVAL_ISQ_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("YOLINO_IGNORE_DIRTY", "1")
logging.basicConfig(level=logging.WARNING)

import torch  # noqa: E402

from eval_pixel_f1 import (  # noqa: E402
    DATASET_ROOT,
    EXP19_CKPT,
    EXP19_CONFIG,
    EXP55_CKPT,
    EXP55_CONFIG,
    SPLIT,
    build_geom_pkl,
    build_gnn_pkl,
    load_records,
    run_geom_predictions,
    run_gnn_predictions,
)
from isq_core import MEANINGFUL_COVERAGE_THRESH  # noqa: E402
from quick_compare import (  # noqa: E402
    _eval_isq,
    _load_isq_records,
    _setup_run_subset,
    _val_stems,
)

EXP65_CONFIG = os.path.join(
    BASE,
    "yolino",
    "CAPSTONE",
    "configs",
    "experiments",
    "exp65_gnn_polyline_adjacent_directional8_lateral5_1024.yaml",
)
EXP65_CKPT = os.path.join(
    BASE,
    "yolino",
    "CAPSTONE",
    "ttpla_train_exp",
    "log",
    "checkpoints",
    "exp65_gnn_polyline_adjacent_directional8_lateral5_1024_smallset",
    "best_model.pth",
)

OUT_JSON = os.path.join(EVAL_ISQ_DIR, "results", "isq_results_oversplit.json")
PKL_GEOM = os.path.join(EVAL_ISQ_DIR, "pred_geom_oversplit.pkl")
PKL_GNN55 = os.path.join(EVAL_ISQ_DIR, "pred_gnn_exp55_oversplit.pkl")
PKL_GNN65 = os.path.join(EVAL_ISQ_DIR, "pred_gnn_exp65_oversplit.pkl")


def _run_gnn(stems, dataset_root, use_gpu, config, ckpt, out_pkl, label):
    records = load_records(dataset_root, stems)
    gnn_raw: dict = {}
    t0 = time.time()
    print("[%s] GNN inference (%d images)..." % (label, len(stems)))
    _a, fwd, ds, loader, dev = _setup_run_subset(config, ckpt, dataset_root, use_gpu, stems)
    run_gnn_predictions(fwd, ds, loader, dev, records, gnn_raw)
    del fwd, _a, ds, loader
    if use_gpu:
        torch.cuda.empty_cache()
    gnn_pkl = build_gnn_pkl(records, gnn_raw)
    with open(out_pkl, "wb") as f:
        pickle.dump(gnn_pkl, f)
    print("      %s done in %.1fs -> %s" % (label, time.time() - t0, out_pkl))
    return gnn_pkl


def main() -> None:
    dataset_root = os.path.abspath(DATASET_ROOT)
    stems = _val_stems(dataset_root, 65)
    use_gpu = torch.cuda.is_available()
    print("[INFO] device:", "cuda" if use_gpu else "cpu")
    print("[INFO] dataset:", dataset_root)
    print("[INFO] val images:", len(stems))
    print("[INFO] MEANINGFUL_COVERAGE_THRESH:", MEANINGFUL_COVERAGE_THRESH)

    for p in (EXP19_CONFIG, EXP55_CONFIG, EXP65_CONFIG, EXP19_CKPT, EXP55_CKPT, EXP65_CKPT):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    records = load_records(dataset_root, stems)
    t0 = time.time()
    print("[geom] exp19 inference...")
    _a19, fwd19, ds19, loader19, dev19 = _setup_run_subset(
        EXP19_CONFIG, EXP19_CKPT, dataset_root, use_gpu, stems
    )
    run_geom_predictions(fwd19, ds19, loader19, dev19, records)
    del fwd19, _a19, ds19, loader19
    if use_gpu:
        torch.cuda.empty_cache()
    geom_pkl = build_geom_pkl(records)
    with open(PKL_GEOM, "wb") as f:
        pickle.dump(geom_pkl, f)
    print("      geom done in %.1fs -> %s" % (time.time() - t0, PKL_GEOM))

    gnn55_pkl = _run_gnn(stems, dataset_root, use_gpu, EXP55_CONFIG, EXP55_CKPT, PKL_GNN55, "exp55")
    gnn65_pkl = _run_gnn(stems, dataset_root, use_gpu, EXP65_CONFIG, EXP65_CKPT, PKL_GNN65, "exp65")

    isq_meta = _load_isq_records(dataset_root, stems)
    results = {
        "settings": {
            "dataset_root": dataset_root,
            "split": SPLIT,
            "n_images": len(stems),
            "stems": stems,
            "meaningful_coverage_thresh": MEANINGFUL_COVERAGE_THRESH,
            "over_split_definition": (
                "GT instance is over-split when >=2 pred polylines have "
                "dominant_id==GT and coverage (distinct matched GT segs / n_gt) >= thresh"
            ),
            "exp19_checkpoint": EXP19_CKPT,
            "exp55_checkpoint": EXP55_CKPT,
            "exp65_checkpoint": EXP65_CKPT,
            "pred_geom_pkl": PKL_GEOM,
            "pred_gnn_exp55_pkl": PKL_GNN55,
            "pred_gnn_exp65_pkl": PKL_GNN65,
        },
        "Pred-1_geom_exp19": _eval_isq(isq_meta, geom_pkl),
        "Pred-2_gnn_exp55": _eval_isq(isq_meta, gnn55_pkl),
        "Pred-3_gnn_exp65": _eval_isq(isq_meta, gnn65_pkl),
    }

    for label, block in (
        ("Pred-1 geom (exp19)", results["Pred-1_geom_exp19"]),
        ("Pred-2 GNN (exp55)", results["Pred-2_gnn_exp55"]),
        ("Pred-3 GNN (exp65)", results["Pred-3_gnn_exp65"]),
    ):
        s = block["summary"]
        print(
            "\n[ISQ] %s  P=%.4f  R=%.4f  F1=%.4f  over_split_count=%d (mean/img=%.2f)  under_merge=%.2f"
            % (
                label,
                s["precision"],
                s["recall"],
                s["f1"],
                s["over_split_count"],
                s["over_split"],
                s["under_merge"],
            )
        )

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n[OK] Wrote", OUT_JSON)


if __name__ == "__main__":
    main()
