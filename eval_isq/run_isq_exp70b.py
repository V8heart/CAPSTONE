#!/usr/bin/env python3
"""ISQ evaluation for exp70b GNN on TTPLA smallset val (65 images).

Writes new artifacts only:
  pred_geom_exp70b.pkl, pred_gnn_exp70b.pkl
  results/isq_results_exp70b.json
  results/isq_summary_exp70b.md
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
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
    GEOM_CONF,
    GNN_EDGE_THRESH,
    GNN_NODE_CONF,
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

EXP70B_CONFIG = os.path.join(
    BASE,
    "yolino",
    "CAPSTONE",
    "configs",
    "experiments",
    "exp70b_gnn_directional2_ctx_cross_ignore_cross15_1024.yaml",
)
EXP70B_CKPT = os.path.join(
    BASE,
    "yolino",
    "CAPSTONE",
    "ttpla_train_exp",
    "log",
    "checkpoints",
    "exp70b_gnn_directional2_ctx_cross_ignore_cross15_1024_smallset",
    "best_model.pth",
)

PKL_GEOM = os.path.join(EVAL_ISQ_DIR, "pred_geom_exp70b.pkl")
PKL_GNN = os.path.join(EVAL_ISQ_DIR, "pred_gnn_exp70b.pkl")
OUT_JSON = os.path.join(EVAL_ISQ_DIR, "results", "isq_results_exp70b.json")
OUT_MD = os.path.join(EVAL_ISQ_DIR, "results", "isq_summary_exp70b.md")

# Reuse identical exp19 geom cache if present (read-only).
GEOM_REUSE_CANDIDATES = (
    os.path.join(EVAL_ISQ_DIR, "pred_geom_exp65.pkl"),
    os.path.join(EVAL_ISQ_DIR, "pred_geom_full.pkl"),
    os.path.join(EVAL_ISQ_DIR, "pred_geom.pkl"),
)


def _ensure_geom_pkl(stems: list[str], dataset_root: str, use_gpu: bool) -> dict:
    if os.path.isfile(PKL_GEOM):
        with open(PKL_GEOM, "rb") as f:
            geom_pkl = pickle.load(f)
        print("[geom] reuse existing", PKL_GEOM)
        return geom_pkl

    for src in GEOM_REUSE_CANDIDATES:
        if os.path.isfile(src):
            shutil.copy2(src, PKL_GEOM)
            with open(PKL_GEOM, "rb") as f:
                geom_pkl = pickle.load(f)
            if all(stem in geom_pkl for stem in stems):
                print("[geom] copied from", src, "->", PKL_GEOM)
                return geom_pkl
            os.remove(PKL_GEOM)

    records = load_records(dataset_root, stems)
    t0 = time.time()
    print("[geom] exp19 inference (%d images)..." % len(stems))
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
    return geom_pkl


def _run_exp70b_gnn(stems: list[str], dataset_root: str, use_gpu: bool) -> dict:
    if os.path.isfile(PKL_GNN):
        with open(PKL_GNN, "rb") as f:
            gnn_pkl = pickle.load(f)
        print("[exp70b] reuse existing", PKL_GNN)
        return gnn_pkl

    records = load_records(dataset_root, stems)
    gnn_raw: dict = {}
    t0 = time.time()
    print("[exp70b] GNN inference (%d images, %s)..." % (len(stems), "cuda" if use_gpu else "cpu"))
    _a, fwd, ds, loader, dev = _setup_run_subset(
        EXP70B_CONFIG, EXP70B_CKPT, dataset_root, use_gpu, stems
    )
    run_gnn_predictions(fwd, ds, loader, dev, records, gnn_raw)
    del fwd, _a, ds, loader
    if use_gpu:
        torch.cuda.empty_cache()
    gnn_pkl = build_gnn_pkl(records, gnn_raw)
    with open(PKL_GNN, "wb") as f:
        pickle.dump(gnn_pkl, f)
    print("      exp70b done in %.1fs -> %s" % (time.time() - t0, PKL_GNN))
    return gnn_pkl


def _load_baseline_summary(path: str, key: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    block = data.get(key) or data.get("isq", {}).get(key)
    if block is None:
        return None
    return block.get("summary")


def _write_summary_md(results: dict) -> None:
    baselines = {
        "exp55": _load_baseline_summary(
            os.path.join(EVAL_ISQ_DIR, "results", "isq_results_oversplit.json"),
            "Pred-2_gnn_exp55",
        ),
        "exp65": _load_baseline_summary(
            os.path.join(EVAL_ISQ_DIR, "results", "isq_results_oversplit.json"),
            "Pred-3_gnn_exp65",
        ),
    }
    s_geom = results["Pred-1_geom_exp19"]["summary"]
    s_gnn = results["Pred-2_gnn_exp70b"]["summary"]
    settings = results["settings"]

    lines = [
        "# ISQ Results — exp70b (TTPLA smallset val)",
        "",
        "## Setup",
        "",
        "- **Dataset**: `%s` (val **65** images)" % settings["dataset_root"],
        "- **Geom baseline**: exp19 (`confidence=%.1f`)" % settings["geom_confidence"],
        "- **GNN model**: exp70b (`edge_thresh=%.1f`, `node_conf=%.1f`)"
        % (settings["gnn_edge_thresh"], settings["gnn_node_conf"]),
        "- **Config**: `%s`" % settings["exp70b_config"],
        "- **Checkpoint**: `%s`" % settings["exp70b_checkpoint"],
        "- **ISQ**: grid cell=%d, match dist=%d px, angle=%d deg, coverage thresh=%.1f"
        % (
            settings["cell_size"],
            settings["match_max_dist_px"],
            settings["match_max_angle_deg"],
            settings["meaningful_coverage_thresh"],
        ),
        "",
        "## Summary (macro mean over 65 val images)",
        "",
        "| Model | Precision | Recall | F1 | over_split (mean/img) | under_merge (mean/img) |",
        "|-------|-----------|--------|-----|------------------------|-------------------------|",
        "| Pred-1 geom (exp19) | %.4f | %.4f | %.4f | %.2f | %.2f |"
        % (s_geom["precision"], s_geom["recall"], s_geom["f1"], s_geom["over_split"], s_geom["under_merge"]),
        "| **Pred-2 GNN (exp70b)** | **%.4f** | **%.4f** | **%.4f** | **%.2f** | **%.2f** |"
        % (s_gnn["precision"], s_gnn["recall"], s_gnn["f1"], s_gnn["over_split"], s_gnn["under_merge"]),
    ]
    if baselines["exp55"]:
        b = baselines["exp55"]
        lines.append(
            "| Pred-2 GNN (exp55, ref) | %.4f | %.4f | %.4f | %.2f | %.2f |"
            % (b["precision"], b["recall"], b["f1"], b["over_split"], b["under_merge"])
        )
    if baselines["exp65"]:
        b = baselines["exp65"]
        lines.append(
            "| Pred-3 GNN (exp65, ref) | %.4f | %.4f | %.4f | %.2f | %.2f |"
            % (b["precision"], b["recall"], b["f1"], b["over_split"], b["under_merge"])
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- **over_split**: mean per-image count of GT instances split across multiple pred polylines.",
            "- **under_merge**: mean per-image count of pred polylines merging multiple GT instances.",
            "- exp55/exp65 reference rows are from `results/isq_results_oversplit.json` (same smallset val).",
            "",
            "Artifacts:",
            "- `%s`" % settings["pred_geom_pkl"],
            "- `%s`" % settings["pred_gnn_pkl"],
            "- `%s`" % settings["results_json"],
        ]
    )
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    dataset_root = os.path.abspath(DATASET_ROOT)
    stems = _val_stems(dataset_root, 0)
    use_gpu = torch.cuda.is_available()

    print("[INFO] device:", "cuda" if use_gpu else "cpu")
    print("[INFO] dataset:", dataset_root)
    print("[INFO] val images:", len(stems))
    for p in (EXP19_CONFIG, EXP19_CKPT, EXP70B_CONFIG, EXP70B_CKPT, dataset_root):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    geom_pkl = _ensure_geom_pkl(stems, dataset_root, use_gpu)
    gnn_pkl = _run_exp70b_gnn(stems, dataset_root, use_gpu)

    print("[ISQ] computing metrics (CPU)...")
    t0 = time.time()
    isq_meta = _load_isq_records(dataset_root, stems)
    results = {
        "settings": {
            "dataset_root": dataset_root,
            "split": SPLIT,
            "n_images": len(stems),
            "stems": stems,
            "cell_size": 32,
            "match_max_dist_px": 24,
            "match_max_angle_deg": 15,
            "meaningful_coverage_thresh": MEANINGFUL_COVERAGE_THRESH,
            "geom_confidence": GEOM_CONF,
            "gnn_edge_thresh": GNN_EDGE_THRESH,
            "gnn_node_conf": GNN_NODE_CONF,
            "exp19_checkpoint": EXP19_CKPT,
            "exp70b_config": EXP70B_CONFIG,
            "exp70b_checkpoint": EXP70B_CKPT,
            "pred_geom_pkl": PKL_GEOM,
            "pred_gnn_pkl": PKL_GNN,
            "results_json": OUT_JSON,
            "device_inference": "cuda" if use_gpu else "cpu",
        },
        "Pred-1_geom_exp19": _eval_isq(isq_meta, geom_pkl),
        "Pred-2_gnn_exp70b": _eval_isq(isq_meta, gnn_pkl),
    }
    print("      ISQ done in %.1fs" % (time.time() - t0))

    for label, block in (
        ("Pred-1 geom (exp19)", results["Pred-1_geom_exp19"]),
        ("Pred-2 GNN (exp70b)", results["Pred-2_gnn_exp70b"]),
    ):
        s = block["summary"]
        print(
            "\n[ISQ] %s  P=%.4f  R=%.4f  F1=%.4f  over_split=%.2f  under_merge=%.2f  (n=%d)"
            % (
                label,
                s["precision"],
                s["recall"],
                s["f1"],
                s["over_split"],
                s["under_merge"],
                s["n_images"],
            )
        )

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _write_summary_md(results)
    print("\n[OK] Wrote", OUT_JSON)
    print("[OK] Wrote", OUT_MD)


if __name__ == "__main__":
    main()
