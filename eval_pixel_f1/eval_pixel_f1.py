#!/usr/bin/env python3
"""Pixel-level Precision / Recall / F1 for YOLinO geom (exp19) and GNN (exp55) on TTPLA smallset val.

GT-B and predictions are rasterized at multiple line thicknesses (2, 5, 10) for sensitivity analysis.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

# Trusted local checkpoints (PyTorch >=2.6 defaults weights_only=True).
_torch_load = torch.load


def _torch_load_trusted(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_trusted  # noqa: PLW0603

CAPSTONE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolino", "CAPSTONE")
SRC_ROOT = os.path.join(CAPSTONE_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from yolino.dataset.dataset_factory import DatasetFactory  # noqa: E402
from yolino.grid.grid_factory import GridFactory  # noqa: E402
from yolino.model.model_factory import load_checkpoint  # noqa: E402
from yolino.runner.forward_runner import ForwardRunner  # noqa: E402
from yolino.utils.enums import CoordinateSystem, TaskType  # noqa: E402
from yolino.utils.general_setup import general_setup  # noqa: E402
from yolino.visualize_ttpla_gt_instances import (  # noqa: E402
    build_binary_mask,
    load_ttpla_label_with_ids,
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ISQ_CORE = os.path.join(BASE, "eval_isq")
if _ISQ_CORE not in sys.path:
    sys.path.insert(0, _ISQ_CORE)
from isq_core import (  # noqa: E402
    geom_polylines_to_pred_polylines,
    gnn_segments_to_polylines,
    pred_polylines_to_segment_tuples,
)
DATASET_ROOT = os.path.join(BASE, "yolino", "ttpla_yolino_dataset_1024x1024_smallset")
ORIG_JSON_DIR = os.path.join(BASE, "maptr", "data_ttpla_1600x900", "annotations")
EXP19_CONFIG = os.path.join(
    CAPSTONE_ROOT, "configs", "experiments", "exp19_fpn_bottomup_p4_num_predictors4_1024.yaml"
)
EXP55_CONFIG = os.path.join(
    CAPSTONE_ROOT, "configs", "experiments", "exp55_gnn_soft_geom_rw_topology_1024.yaml"
)
EXP19_CKPT = os.path.join(
    CAPSTONE_ROOT,
    "ttpla_train_exp",
    "log",
    "checkpoints",
    "exp19_fpn_bottomup_p4_num_predictors4_1024",
    "best_model.pth",
)
EXP55_CKPT = os.path.join(
    CAPSTONE_ROOT,
    "ttpla_train_exp",
    "log",
    "checkpoints",
    "exp55_gnn_soft_geom_rw_topology_1024_smallset",
    "best_model.pth",
)
EXP76_CONFIG = os.path.join(
    CAPSTONE_ROOT, "configs", "experiments", "exp76_gnn_ttpla_full.yaml"
)
EXP76_CKPT = os.path.join(
    CAPSTONE_ROOT,
    "ttpla_train_exp",
    "log",
    "checkpoints",
    "exp76_gnn_ttpla_full",
    "ep0005_model.pth",
)
FULL_DATASET_ROOT = os.path.join(BASE, "yolino", "ttpla_yolino_dataset_1024x1024")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PRED_GEOM_PKL = os.path.join(BASE, "eval_isq", "pred_geom.pkl")
PRED_GNN_PKL = os.path.join(BASE, "eval_isq", "pred_gnn.pkl")
RESULTS_JSON = os.path.join(OUT_DIR, "pixel_f1_results.json")
RESULTS_RELAXATION_JSON = os.path.join(OUT_DIR, "pixel_f1_results_relaxation.json")
RESULTS_LSNETV2_JSON = os.path.join(OUT_DIR, "lsnetv2_comparison.json")
RESULTS_GTA_THICKNESS_SWEEP_JSON = os.path.join(OUT_DIR, "gta_thickness_sweep.json")
VIZ_DIR = os.path.join(OUT_DIR, "viz")

THICKNESS_VALUES = (2, 5, 10)
# LSNetv2 TTPLA protocol: GT-B polyline raster, line width 2px / 4px, macro APR/ARR/F1/Fβ.
LSNETV2_LINE_WIDTHS = (2, 4)
LSNETV2_BETA_SQ = 0.3
LSNETV2_BASELINE = {
    "line_width_px": 2,
    "resolution": "512x512",
    "APR": 0.714,
    "ARR": 0.560,
    "F1": 0.628,
    "F_beta": 0.671,
    "source": "LSNetv2 paper TTPLA (reported)",
}
GTA_THICKNESS_SWEEP = (4, 5, 8, 10)
PLGAN_BASELINE_905_217 = {
    "resolution": "512x512",
    "split": "905/217",
    "gt_type": "GT-A_polygon_fill",
    "HAWP": {"F1": 0.485, "F_beta": 0.532},
    "LCNN": {"F1": 0.498, "F_beta": 0.519},
    "AFM": {"F1": 0.457, "F_beta": 0.498},
}
VIZ_THICKNESS = 2
RELAX_PRED_THICKNESS = 2
# PLGAN-style tolerance: 2px @ 512² reference, 4px @ 1024² reference.
RELAXATION_SPECS = (
    (2, "2px_512_reference"),
    (4, "4px_1024_reference"),
)
GT_A_CROP_AWARE = True
GT_A_STRETCH = False
GEOM_CONF = 0.7
GNN_EDGE_THRESH = 0.2
GNN_NODE_CONF = 0.7
SPLIT = "val"
NUM_VIZ = 5

CABLE_LABEL_KEYWORDS = ("cable", "power line", "power_line", "powerline")


@dataclass
class ImageRecord:
    stem: str
    height: int
    width: int
    json_path: str = ""
    label_polylines: List = field(default_factory=list)
    geom_polylines: Optional[List] = None
    gnn_segments: Optional[List] = None


def stem_to_json_base(stem: str) -> str:
    if "_" in stem and stem.rsplit("_", 1)[-1] in ("L", "R", "T", "B"):
        return stem.rsplit("_", 1)[0]
    return stem


def stem_to_crop_tag(stem: str) -> str:
    if "_" in stem:
        tag = stem.rsplit("_", 1)[-1]
        if tag in ("L", "R", "T", "B"):
            return tag
    return "L"


def _resize_short_to(img_w: int, img_h: int, target: int) -> tuple[int, int, float]:
    if img_w <= img_h:
        scale = target / float(img_w)
    else:
        scale = target / float(img_h)
    return int(round(img_w * scale)), int(round(img_h * scale)), scale


def _crop_boxes(new_w: int, new_h: int, target: int) -> list[tuple[str, tuple[int, int, int, int]]]:
    if new_w >= new_h:
        return [
            ("L", (0, 0, target, target)),
            ("R", (new_w - target, 0, new_w, target)),
        ]
    return [
        ("T", (0, 0, target, target)),
        ("B", (0, new_h - target, target, new_h)),
    ]


def is_cable_label(label: str) -> bool:
    low = (label or "").lower()
    return any(kw in low for kw in CABLE_LABEL_KEYWORDS)


def build_gt_a_mask(json_path: str, height: int, width: int) -> np.ndarray:
    """GT-A: cable/powerline polygons filled (legacy full-frame coords, no crop)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if not os.path.isfile(json_path):
        return mask
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for shape in data.get("shapes", []):
        if not is_cable_label(shape.get("label", "")):
            continue
        pts = shape.get("points")
        if not pts or len(pts) < 3:
            continue
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        arr = np.round(arr[:, :2]).astype(np.int32)
        cv2.fillPoly(mask, [arr], 255)
    return mask


def build_gt_a_mask_cropped(
    json_path: str, stem: str, tile_size: int = 1024
) -> np.ndarray:
    """GT-A polygon fill aligned to 1024 dual-crop tiles (PLGAN-style, crop-aware)."""
    mask_full = np.zeros((tile_size, tile_size), dtype=np.uint8)
    if not os.path.isfile(json_path):
        return mask_full
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    img_w = int(data.get("imageWidth") or 0)
    img_h = int(data.get("imageHeight") or 0)
    if img_w <= 0 or img_h <= 0:
        return mask_full

    nw, nh, scale = _resize_short_to(img_w, img_h, tile_size)
    resized_mask = np.zeros((nh, nw), dtype=np.uint8)
    for shape in data.get("shapes", []):
        if not is_cable_label(shape.get("label", "")):
            continue
        pts = shape.get("points")
        if not pts or len(pts) < 3:
            continue
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        arr = arr * float(scale)
        arr = np.round(arr[:, :2]).astype(np.int32)
        cv2.fillPoly(resized_mask, [arr], 255)

    crop_tag = stem_to_crop_tag(stem)
    box_map = {tag: box for tag, box in _crop_boxes(nw, nh, tile_size)}
    if crop_tag not in box_map:
        return mask_full
    x0, y0, x1, y1 = box_map[crop_tag]
    return resized_mask[y0:y1, x0:x1].copy()


def build_gt_a_mask_stretched(json_path: str, height: int, width: int) -> np.ndarray:
    """GT-A polygon fill with independent sx/sy stretch (PLGAN 512² benchmark style)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if not os.path.isfile(json_path):
        return mask
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    img_w = int(data.get("imageWidth") or 0)
    img_h = int(data.get("imageHeight") or 0)
    if img_w <= 0 or img_h <= 0:
        return mask
    sx = width / float(img_w)
    sy = height / float(img_h)
    for shape in data.get("shapes", []):
        if not is_cable_label(shape.get("label", "")):
            continue
        pts = shape.get("points")
        if not pts or len(pts) < 3:
            continue
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        arr[:, 0] *= sx
        arr[:, 1] *= sy
        arr = np.round(arr[:, :2]).astype(np.int32)
        cv2.fillPoly(mask, [arr], 255)
    return mask


def dilate_binary(mask_bool: np.ndarray, radius_px: int) -> np.ndarray:
    """Morphological dilation (ellipse) by ``radius_px`` pixels."""
    if radius_px <= 0:
        return mask_bool.astype(bool)
    k = 2 * int(radius_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1)
    return dil > 0


def gt_a_mask(rec: ImageRecord) -> np.ndarray:
    if GT_A_STRETCH:
        return build_gt_a_mask_stretched(rec.json_path, rec.height, rec.width)
    if GT_A_CROP_AWARE:
        return build_gt_a_mask_cropped(rec.json_path, rec.stem, tile_size=rec.height)
    return build_gt_a_mask(rec.json_path, rec.height, rec.width)


def rasterize_polylines_mask(
    height: int, width: int, polylines: Sequence[Sequence[Sequence[float]]], thickness: int
) -> np.ndarray:
    return build_binary_mask(height, width, polylines, thickness)


def rasterize_line_segments_mask(
    height: int,
    width: int,
    segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
    thickness: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for p0, p1 in segments:
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        if np.any(np.isnan((x0, y0, x1, y1))):
            continue
        cv2.line(
            mask,
            (int(round(x0)), int(round(y0))),
            (int(round(x1)), int(round(y1))),
            255,
            max(1, int(thickness)),
            lineType=cv2.LINE_8,
        )
    return mask


def gt_b_mask(rec: ImageRecord, thickness: int) -> np.ndarray:
    return rasterize_polylines_mask(rec.height, rec.width, rec.label_polylines, thickness)


def pred_geom_mask(rec: ImageRecord, thickness: int) -> np.ndarray:
    if rec.geom_polylines is None:
        raise ValueError("Missing geom polylines for %s" % rec.stem)
    return rasterize_polylines_mask(rec.height, rec.width, rec.geom_polylines, thickness)


def pred_gnn_mask(rec: ImageRecord, thickness: int) -> np.ndarray:
    if rec.gnn_segments is None:
        raise ValueError("Missing GNN segments for %s" % rec.stem)
    return rasterize_line_segments_mask(rec.height, rec.width, rec.gnn_segments, thickness)


def geom_uv_to_polylines(pred_uv: np.ndarray) -> List[List[List[float]]]:
    if pred_uv.ndim == 3:
        rows = pred_uv[0]
    else:
        rows = pred_uv
    polys = []
    for row in rows:
        if row.shape[0] < 4 or np.any(np.isnan(row[:4])):
            continue
        y1, x1, y2, x2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        polys.append([[x1, y1], [x2, y2]])
    return polys


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def extract_gnn_segments(
    gnn_single: dict,
    edge_thresh: float = GNN_EDGE_THRESH,
    node_conf_thresh: float = GNN_NODE_CONF,
    *,
    include_connectors: bool = True,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    node_mid = _to_numpy(gnn_single["node_mid_px"])
    node_valid = _to_numpy(gnn_single["node_valid"]).astype(bool)
    neighbors = _to_numpy(gnn_single["neighbors"]).astype(np.int64)
    neigh_valid = _to_numpy(gnn_single["neigh_valid"]).astype(bool)
    edge_logits = _to_numpy(gnn_single["edge_logits"]).astype(np.float32)
    node_ea = gnn_single.get("node_end_a_px")
    node_eb = gnn_single.get("node_end_b_px")
    node_conf = gnn_single.get("node_conf")
    if node_ea is not None:
        node_ea = _to_numpy(node_ea)
    if node_eb is not None:
        node_eb = _to_numpy(node_eb)
    if node_conf is not None:
        node_conf = _to_numpy(node_conf).astype(np.float32)
        node_valid = node_valid & (node_conf >= float(node_conf_thresh))

    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    if not np.any(node_valid) or node_ea is None or node_eb is None:
        return segments

    n_nodes, k = neighbors.shape
    probs = 1.0 / (1.0 + np.exp(-edge_logits))
    edge_pairs = set()
    for ni in range(n_nodes):
        if not bool(node_valid[ni]):
            continue
        for kj in range(k):
            if not bool(neigh_valid[ni, kj]):
                continue
            if float(probs[ni, kj]) < float(edge_thresh):
                continue
            nj = int(neighbors[ni, kj])
            if nj < 0 or nj >= n_nodes or not bool(node_valid[nj]):
                continue
            a, b = (ni, nj) if ni < nj else (nj, ni)
            edge_pairs.add((a, b))

    in_graph = np.zeros((n_nodes,), dtype=bool)
    for a, b in edge_pairs:
        in_graph[a] = True
        in_graph[b] = True

    for i in range(n_nodes):
        if not bool(node_valid[i]) or not bool(in_graph[i]):
            continue
        ea, eb = node_ea[i], node_eb[i]
        segments.append(((float(ea[0]), float(ea[1])), (float(eb[0]), float(eb[1]))))

    if include_connectors:
        for a, b in edge_pairs:
            ma, mb = node_mid[a], node_mid[b]
            segments.append(((float(ma[0]), float(ma[1])), (float(mb[0]), float(mb[1]))))
    return segments


def f_beta_score(precision: float, recall: float, beta_sq: float = LSNETV2_BETA_SQ) -> float:
    """Fβ = (1 + β²) × P × R / (β² × P + R), LSNetv2 uses β² = 0.3."""
    return float((1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-8))


def pixel_prf1(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    pred_b = pred_mask > 0
    gt_b = gt_mask > 0
    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, np.logical_not(gt_b)).sum())
    fn = int(np.logical_and(np.logical_not(pred_b), gt_b).sum())
    union = int(np.logical_or(pred_b, gt_b).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (union + 1e-8)
    f_beta = f_beta_score(float(precision), float(recall))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f_beta": float(f_beta),
        "iou": float(iou),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def summarize_lsnetv2(per_image: List[Dict[str, float]]) -> Dict[str, float]:
    """Macro APR/ARR; F1 and Fβ from macro precision/recall (LSNetv2-style)."""
    if not per_image:
        return {"APR": 0.0, "ARR": 0.0, "F1": 0.0, "F_beta": 0.0, "n_images": 0}
    apr = float(np.mean([m["precision"] for m in per_image]))
    arr = float(np.mean([m["recall"] for m in per_image]))
    f1 = 2.0 * apr * arr / (apr + arr + 1e-8)
    f_beta = f_beta_score(apr, arr)
    return {"APR": apr, "ARR": arr, "F1": float(f1), "F_beta": float(f_beta), "n_images": len(per_image)}


def pixel_metrics_gt_a_relaxed(
    pred_mask: np.ndarray, gt_mask: np.ndarray, relaxation_px: int
) -> Dict[str, float]:
    """PLGAN-style CCQ with tolerance + strict P/R/F1/IoU.

    Relaxed (dilation) rules:
      - Correctness: pred pixels inside dilate(GT, r) count as true positives.
      - Completeness: GT pixels inside dilate(Pred, r) count as covered.
      - Quality: harmonic mean of correctness and completeness.
    """
    pred_b = pred_mask > 0
    gt_b = gt_mask > 0
    r = int(relaxation_px)
    gt_dil = dilate_binary(gt_b, r)
    pred_dil = dilate_binary(pred_b, r)

    pred_n = int(pred_b.sum())
    gt_n = int(gt_b.sum())

    tp_correctness = int(np.logical_and(pred_b, gt_dil).sum())
    correctness = tp_correctness / (pred_n + 1e-8)

    tp_completeness = int(np.logical_and(gt_b, pred_dil).sum())
    completeness = tp_completeness / (gt_n + 1e-8)

    quality = 2.0 * correctness * completeness / (correctness + completeness + 1e-8)
    relaxed_f1 = float(quality)
    relaxed_f_beta = f_beta_score(float(correctness), float(completeness))

    strict = pixel_prf1(pred_mask, gt_mask)
    return {
        "correctness": float(correctness),
        "completeness": float(completeness),
        "quality": float(quality),
        "relaxed_precision": float(correctness),
        "relaxed_recall": float(completeness),
        "relaxed_f1": relaxed_f1,
        "relaxed_f_beta": float(relaxed_f_beta),
        "precision": strict["precision"],
        "recall": strict["recall"],
        "f1": strict["f1"],
        "f_beta": strict["f_beta"],
        "iou": strict["iou"],
        "pred_pixels": pred_n,
        "gt_pixels": gt_n,
        "tp_correctness": tp_correctness,
        "tp_completeness": tp_completeness,
        "strict_tp": strict["tp"],
        "strict_fp": strict["fp"],
        "strict_fn": strict["fn"],
    }


METRIC_KEYS = (
    "correctness",
    "completeness",
    "quality",
    "relaxed_precision",
    "relaxed_recall",
    "relaxed_f1",
    "relaxed_f_beta",
    "precision",
    "recall",
    "f1",
    "f_beta",
    "iou",
)


def mean_metrics(per_image: List[Dict[str, float]], keys: Sequence[str] = ("precision", "recall", "f1")) -> Dict[str, float]:
    if not per_image:
        out = {k: 0.0 for k in keys}
        out["n_images"] = 0
        return out
    out = {k: float(np.mean([m[k] for m in per_image])) for k in keys}
    out["n_images"] = len(per_image)
    return out


def evaluate_gt_b_vs_pred(
    records: Dict[str, ImageRecord],
    pred_kind: str,
    thickness: int,
) -> Dict[str, object]:
    per_image = []
    for stem, rec in sorted(records.items()):
        gt = gt_b_mask(rec, thickness)
        if pred_kind == "geom":
            pred = pred_geom_mask(rec, thickness)
        elif pred_kind == "gnn":
            pred = pred_gnn_mask(rec, thickness)
        else:
            raise ValueError(pred_kind)
        m = pixel_prf1(pred, gt)
        m["stem"] = stem
        per_image.append(m)
    return {"summary": mean_metrics(per_image, keys=("precision", "recall", "f1", "iou")), "per_image": per_image}


def evaluate_lsnetv2_gt_b(
    records: Dict[str, ImageRecord],
    pred_kind: str,
    line_width: int,
) -> Dict[str, object]:
    """GT-B vs pred at ``line_width``; report APR, ARR, F1, Fβ (β²=0.3)."""
    per_image = []
    for stem, rec in sorted(records.items()):
        gt = gt_b_mask(rec, line_width)
        if pred_kind == "geom":
            pred = pred_geom_mask(rec, line_width)
        elif pred_kind == "gnn":
            pred = pred_gnn_mask(rec, line_width)
        else:
            raise ValueError(pred_kind)
        m = pixel_prf1(pred, gt)
        m["stem"] = stem
        per_image.append(m)
    return {"summary": summarize_lsnetv2(per_image), "per_image": per_image}


def run_lsnetv2_comparison(records: Dict[str, ImageRecord]) -> Dict[str, object]:
    """LSNetv2-comparable GT-B pixel metrics at line widths 2px and 4px."""
    out: Dict[str, object] = {
        "settings": {
            "dataset_root": DATASET_ROOT,
            "split": SPLIT,
            "gt_type": "GT-B_polyline_raster",
            "line_widths_px": list(LSNETV2_LINE_WIDTHS),
            "beta_sq": LSNETV2_BETA_SQ,
            "f_beta_formula": "(1 + beta_sq) * APR * ARR / (beta_sq * APR + ARR)",
            "apr_definition": "macro mean of per-image pixel precision",
            "arr_definition": "macro mean of per-image pixel recall",
            "geom_confidence": GEOM_CONF,
            "gnn_edge_thresh": GNN_EDGE_THRESH,
            "gnn_node_conf": GNN_NODE_CONF,
            "exp19_checkpoint": EXP19_CKPT,
            "exp55_checkpoint": EXP55_CKPT,
        },
        "lsnetv2_baseline_ttpla": LSNETV2_BASELINE,
        "comparison_table": [],
        "by_line_width": {},
    }
    model_rows = (("geom", "Geom", "Pred-1 exp19"), ("gnn", "GNN", "Pred-2 exp55"))
    print("\n=== LSNetv2-style GT-B comparison (APR / ARR / F1 / Fβ, β²=%.1f) ===" % LSNETV2_BETA_SQ)
    print("| line_width | model | APR | ARR | F1 | Fβ |")
    print("|-----------|-------|-----|-----|-----|-----|")
    for line_width in LSNETV2_LINE_WIDTHS:
        lw_key = str(line_width)
        out["by_line_width"][lw_key] = {}
        for pred_kind, model_label, _desc in model_rows:
            block = evaluate_lsnetv2_gt_b(records, pred_kind, line_width)
            out["by_line_width"][lw_key][model_label] = block
            s = block["summary"]
            row = {
                "line_width": "%dpx" % line_width,
                "line_width_px": line_width,
                "model": model_label,
                "pred_kind": pred_kind,
                "APR": s["APR"],
                "ARR": s["ARR"],
                "F1": s["F1"],
                "F_beta": s["F_beta"],
                "n_images": s["n_images"],
            }
            out["comparison_table"].append(row)
            print(
                "| %s | %s | %.4f | %.4f | %.4f | %.4f |"
                % (row["line_width"], model_label, s["APR"], s["ARR"], s["F1"], s["F_beta"])
            )
    b = LSNETV2_BASELINE
    print(
        "\n[LSNetv2 baseline @ %dpx %s] APR=%.3f ARR=%.3f F1=%.3f Fβ=%.3f"
        % (b["line_width_px"], b["resolution"], b["APR"], b["ARR"], b["F1"], b["F_beta"])
    )
    return out


def evaluate_gt_a_thickness(
    records: Dict[str, ImageRecord],
    pred_kind: str,
    thickness: int,
) -> Dict[str, object]:
    per_image = []
    for stem, rec in sorted(records.items()):
        gt = gt_a_mask(rec)
        if pred_kind == "geom":
            pred = pred_geom_mask(rec, thickness)
        elif pred_kind == "gnn":
            pred = pred_gnn_mask(rec, thickness)
        else:
            raise ValueError(pred_kind)
        m = pixel_prf1(pred, gt)
        m["stem"] = stem
        per_image.append(m)
    return {"summary": mean_metrics(per_image, keys=("precision", "recall", "f1", "f_beta")), "per_image": per_image}


def run_gt_a_thickness_sweep(records: Dict[str, ImageRecord]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "settings": {
            "dataset_root": DATASET_ROOT,
            "split": SPLIT,
            "gt_type": "GT-A_polygon_fill",
            "orig_json_dir": ORIG_JSON_DIR,
            "thickness_values": list(GTA_THICKNESS_SWEEP),
            "beta_sq": LSNETV2_BETA_SQ,
            "f_beta_formula": "(1 + beta_sq) * Precision * Recall / (beta_sq * Precision + Recall)",
            "geom_confidence": GEOM_CONF,
            "gnn_edge_thresh": GNN_EDGE_THRESH,
            "gnn_node_conf": GNN_NODE_CONF,
            "exp19_checkpoint": EXP19_CKPT,
            "exp55_checkpoint": EXP55_CKPT,
        },
        "plgan_reported_baselines": PLGAN_BASELINE_905_217,
        "comparison_table": [],
        "by_thickness": {},
    }
    model_rows = (("geom", "Geom"), ("gnn", "GNN"))
    print("\n=== GT-A polygon | thickness sweep (P / R / F1 / Fβ, β²=%.1f) ===" % LSNETV2_BETA_SQ)
    print("| thickness | model | Precision | Recall | F1 | Fβ |")
    print("|-----------|-------|-----------|--------|-----|-----|")
    for thickness in GTA_THICKNESS_SWEEP:
        t_key = str(thickness)
        out["by_thickness"][t_key] = {}
        for pred_kind, model_label in model_rows:
            block = evaluate_gt_a_thickness(records, pred_kind, thickness)
            out["by_thickness"][t_key][model_label] = block
            s = block["summary"]
            row = {
                "thickness": thickness,
                "model": model_label,
                "pred_kind": pred_kind,
                "precision": s["precision"],
                "recall": s["recall"],
                "f1": s["f1"],
                "f_beta": s["f_beta"],
                "n_images": s["n_images"],
            }
            out["comparison_table"].append(row)
            print(
                "| %d | %s | %.4f | %.4f | %.4f | %.4f |"
                % (thickness, model_label, s["precision"], s["recall"], s["f1"], s["f_beta"])
            )
    return out


def evaluate_gt_a_relaxed(
    records: Dict[str, ImageRecord],
    pred_kind: str,
    relaxation_px: int,
    pred_thickness: int = RELAX_PRED_THICKNESS,
) -> Dict[str, object]:
    per_image = []
    for stem, rec in sorted(records.items()):
        gt = gt_a_mask(rec)
        if pred_kind == "geom":
            pred = pred_geom_mask(rec, pred_thickness)
        elif pred_kind == "gnn":
            pred = pred_gnn_mask(rec, pred_thickness)
        else:
            raise ValueError(pred_kind)
        m = pixel_metrics_gt_a_relaxed(pred, gt, relaxation_px)
        m["stem"] = stem
        per_image.append(m)
    return {"summary": mean_metrics(per_image, keys=METRIC_KEYS), "per_image": per_image}


def run_gt_a_relaxation_eval(
    records: Dict[str, ImageRecord],
    relaxation_specs: Sequence[Tuple[int, str]] | None = None,
) -> Dict[str, object]:
    specs = list(relaxation_specs or RELAXATION_SPECS)
    out = {
        "settings": {
            "dataset_root": DATASET_ROOT,
            "split": SPLIT,
            "gt_type": "GT-A_polygon_fill",
            "orig_json_dir": ORIG_JSON_DIR,
            "pred_raster_line_thickness": RELAX_PRED_THICKNESS,
            "relaxation_specs": [
                {"relaxation_px": r, "label": label} for r, label in specs
            ],
            "geom_confidence": GEOM_CONF,
            "gnn_edge_thresh": GNN_EDGE_THRESH,
            "gnn_node_conf": GNN_NODE_CONF,
            "exp19_checkpoint": EXP19_CKPT,
            "exp55_checkpoint": EXP55_CKPT,
            "gt_a_crop_aware": GT_A_CROP_AWARE,
            "gt_a_stretch": GT_A_STRETCH,
            "metric_notes": {
                "correctness": "relaxed precision: |Pred ∩ dilate(GT,r)| / |Pred|",
                "completeness": "relaxed recall: |GT ∩ dilate(Pred,r)| / |GT|",
                "quality": "2 * correctness * completeness / (correctness + completeness)",
                "relaxed_f1": "harmonic mean of relaxed precision and relaxed recall",
                "relaxed_f_beta": "(1 + beta_sq) * relaxed_P * relaxed_R / (beta_sq * relaxed_P + relaxed_R)",
                "precision_recall_f1_iou": "strict pixel overlap (no tolerance)",
            },
        },
        "by_relaxation": {},
    }
    for relax_px, relax_label in specs:
        print(
            "\n=== GT-A polygon GT | relaxation=%dpx (%s) | pred raster t=%d ==="
            % (relax_px, relax_label, RELAX_PRED_THICKNESS)
        )
        out["by_relaxation"][relax_label] = {}
        for pred_kind, comp_label in (("geom", "GT-A_vs_Pred-1_geom"), ("gnn", "GT-A_vs_Pred-2_gnn")):
            block = evaluate_gt_a_relaxed(records, pred_kind, relax_px, RELAX_PRED_THICKNESS)
            out["by_relaxation"][relax_label][comp_label] = block
            s = block["summary"]
            print(
                "[RELAX] r=%d  %s  Correctness=%.4f  Completeness=%.4f  Quality=%.4f  "
                "relaxed_F1=%.4f  relaxed_Fβ=%.4f  strict_P=%.4f  strict_R=%.4f  strict_F1=%.4f  (n=%d)"
                % (
                    relax_px,
                    comp_label,
                    s["correctness"],
                    s["completeness"],
                    s["quality"],
                    s["relaxed_f1"],
                    s["relaxed_f_beta"],
                    s["precision"],
                    s["recall"],
                    s["f1"],
                    s["n_images"],
                )
            )
    return out


def setup_run(config_path: str, checkpoint_path: str, dataset_root: str, use_gpu: bool):
    os.environ["DATASET_TTPLA"] = os.path.abspath(dataset_root)
    os.environ.setdefault("YOLINO_IGNORE_DIRTY", "1")
    dvc = os.path.join(CAPSTONE_ROOT, "ttpla_train_exp")
    default_cfg = os.path.join(dvc, "default_params.yaml")
    ckpt = os.path.abspath(checkpoint_path)
    alt = [
        "-c",
        os.path.abspath(config_path),
        "--root",
        CAPSTONE_ROOT,
        "--dvc",
        dvc,
        "--split",
        SPLIT,
        "--explicit_model",
        ckpt,
        "--run_name",
        "eval_pixel_f1",
        "--loggers",
        "file",
        "--batch_size",
        "1",
        "--loading_workers",
        "0",
    ]
    if use_gpu:
        alt.extend(["--gpu", "--gpu_id", "0"])
    args = general_setup(
        "eval_pixel_f1",
        task_type=TaskType.TEST,
        config_file=os.path.abspath(config_path),
        ignore_cmd_args=True,
        alternative_args=alt,
        default_config=default_cfg,
    )
    args.explicit_model = ckpt
    args.paths.pretrain_model = ckpt
    if use_gpu and torch.cuda.is_available():
        args.gpu = True
        args.cuda = "cuda:0"
        device = torch.device("cuda:0")
    else:
        args.gpu = False
        args.cuda = "cpu"
        device = torch.device("cpu")
    coords = DatasetFactory.get_coords(SPLIT, args)
    model, _, _ = load_checkpoint(args, coords, allow_failure=False, load_best=False)
    model.eval().to(device)
    net = model.module if hasattr(model, "module") else model
    if getattr(net, "e2e_head", None) is not None and hasattr(net.e2e_head, "node_conf_thresh"):
        net.e2e_head.node_conf_thresh = float(GNN_NODE_CONF)
    forward = ForwardRunner(args, preloaded_model=model, coords=coords)
    dataset, _ = DatasetFactory.get(
        args.dataset,
        only_available=True,
        split=SPLIT,
        args=args,
        shuffle=False,
        augment=False,
        ignore_duplicates=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    return args, forward, dataset, loader, device


def load_records(dataset_root: str, file_names: Sequence[str]) -> Dict[str, ImageRecord]:
    img_dir = os.path.join(dataset_root, "images", SPLIT)
    label_dir = os.path.join(dataset_root, "labels", SPLIT)
    records: Dict[str, ImageRecord] = {}
    for stem in file_names:
        img_path = os.path.join(img_dir, stem + ".png")
        label_path = os.path.join(label_dir, stem + ".npy")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)
        h, w = img.shape[:2]
        json_path = os.path.join(ORIG_JSON_DIR, stem_to_json_base(stem) + ".json")
        polylines, _, _ = load_ttpla_label_with_ids(label_path)
        records[stem] = ImageRecord(
            stem=stem,
            height=h,
            width=w,
            json_path=json_path,
            label_polylines=polylines,
        )
    return records


def run_geom_predictions(
    forward: ForwardRunner,
    dataset,
    loader: DataLoader,
    device: torch.device,
    records: Dict[str, ImageRecord],
) -> None:
    for images, _grid_tensor, fileinfo, *_ in loader:
        stem = str(fileinfo[0] if isinstance(fileinfo, (list, tuple)) else fileinfo)
        if stem not in records:
            continue
        images = images.to(device)
        geom_preds, _, _, _e2e = forward(images, is_train=False, epoch=None)
        ih = int(images.shape[-2])
        pred_grid, _ = GridFactory.get(
            torch.unsqueeze(geom_preds[0].detach().cpu(), dim=0),
            [],
            CoordinateSystem.CELL_SPLIT,
            args=forward.args,
            input_coords=dataset.coords,
            only_train_vars=True,
            anchors=dataset.anchors,
        )
        pred_uv = pred_grid.get_image_lines(
            coords=dataset.coords,
            image_height=ih,
            is_training_data=True,
            confidence_threshold=float(GEOM_CONF),
        )
        records[stem].geom_polylines = geom_uv_to_polylines(np.asarray(pred_uv))


def run_gnn_predictions(
    forward: ForwardRunner,
    dataset,
    loader: DataLoader,
    device: torch.device,
    records: Dict[str, ImageRecord],
    gnn_raw: Dict[str, dict],
) -> None:
    for images, _grid_tensor, fileinfo, *_ in loader:
        stem = str(fileinfo[0] if isinstance(fileinfo, (list, tuple)) else fileinfo)
        if stem not in records:
            continue
        images = images.to(device)
        _geom_preds, _, _, e2e_out = forward(images, is_train=False, epoch=None)
        if e2e_out is None or "edge_logits" not in e2e_out:
            raise RuntimeError("GNN e2e output missing; check exp55 config (e2e_mode=gnn).")
        gnn_single = {}
        for k, v in e2e_out.items():
            if k in ("soft_nms_debug",):
                continue
            if isinstance(v, torch.Tensor) and v.shape[0] == 1:
                gnn_single[k] = v[0]
            else:
                gnn_single[k] = v
        gnn_raw[stem] = gnn_single
        records[stem].gnn_segments = extract_gnn_segments(
            gnn_single, include_connectors=False
        )


def build_geom_pkl(
    records: Dict[str, ImageRecord],
    *,
    adjacency_threshold: float | None = None,
    min_segments_for_polyline: int | None = None,
) -> Dict[str, List[Tuple[float, float, float, float, int]]]:
    from isq_core import DEFAULT_ADJACENCY_THRESHOLD, DEFAULT_MIN_SEGMENTS_FOR_POLYLINE

    adj = float(DEFAULT_ADJACENCY_THRESHOLD if adjacency_threshold is None else adjacency_threshold)
    min_seg = int(
        DEFAULT_MIN_SEGMENTS_FOR_POLYLINE
        if min_segments_for_polyline is None
        else min_segments_for_polyline
    )
    out: Dict[str, List[Tuple[float, float, float, float, int]]] = {}
    for stem, rec in sorted(records.items()):
        if rec.geom_polylines is None:
            raise ValueError("Missing geom polylines for %s" % stem)
        pred_polylines = geom_polylines_to_pred_polylines(
            rec.geom_polylines,
            adjacency_threshold=adj,
            min_segments_for_polyline=min_seg,
        )
        out[stem] = pred_polylines_to_segment_tuples(pred_polylines)
    return out


def build_gnn_pkl(
    records: Dict[str, ImageRecord],
    gnn_raw: Dict[str, dict],
) -> Dict[str, List[Tuple[float, float, float, float, int]]]:
    out: Dict[str, List[Tuple[float, float, float, float, int]]] = {}
    for stem, rec in sorted(records.items()):
        if rec.gnn_segments is None:
            raise ValueError("Missing GNN segments for %s" % stem)
        if stem not in gnn_raw:
            raise ValueError("Missing GNN raw output for %s" % stem)
        pred_polylines = gnn_segments_to_polylines(
            rec.gnn_segments,
            gnn_raw[stem],
            edge_thresh=GNN_EDGE_THRESH,
            node_conf_thresh=GNN_NODE_CONF,
        )
        out[stem] = pred_polylines_to_segment_tuples(pred_polylines)
    return out


def save_isq_pred_pkls(records: Dict[str, ImageRecord], gnn_raw: Dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(PRED_GEOM_PKL), exist_ok=True)
    geom_pkl = build_geom_pkl(records)
    gnn_pkl = build_gnn_pkl(records, gnn_raw)
    with open(PRED_GEOM_PKL, "wb") as f:
        pickle.dump(geom_pkl, f)
    with open(PRED_GNN_PKL, "wb") as f:
        pickle.dump(gnn_pkl, f)
    print("[OK] Wrote ISQ pred pickles:", PRED_GEOM_PKL, PRED_GNN_PKL)


def _mask_panel(title: str, mask: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    cv2.putText(
        rgb,
        title,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return rgb


def save_viz(records: Dict[str, ImageRecord], stems: Sequence[str], thickness: int = VIZ_THICKNESS) -> None:
    """GT-B | Pred-1 (geom) | Pred-2 (GNN) at a fixed line thickness."""
    os.makedirs(VIZ_DIR, exist_ok=True)
    for stem in stems:
        rec = records[stem]
        gt = gt_b_mask(rec, thickness)
        p1 = pred_geom_mask(rec, thickness)
        p2 = pred_gnn_mask(rec, thickness)
        panels = [
            _mask_panel("GT-B (t=%d)" % thickness, gt),
            _mask_panel("Pred-1 geom (t=%d)" % thickness, p1),
            _mask_panel("Pred-2 GNN (t=%d)" % thickness, p2),
        ]
        row = np.concatenate(panels, axis=1)
        out_path = os.path.join(VIZ_DIR, "%s_gtB_pred1_pred2_t%d.png" % (stem, thickness))
        cv2.imwrite(out_path, row)


def _pkl_paths(tag: str) -> tuple[str, str]:
    suffix = "" if not tag else "_%s" % tag
    eval_isq = os.path.join(BASE, "eval_isq")
    return (
        os.path.join(eval_isq, "pred_geom%s.pkl" % suffix),
        os.path.join(eval_isq, "pred_gnn%s.pkl" % suffix),
    )


def _load_geom_polylines_from_pkl(
    records: Dict[str, ImageRecord], geom_pkl: Dict[str, list], stems: Sequence[str]
) -> None:
    for stem in stems:
        if stem not in geom_pkl:
            raise ValueError("Geom pickle missing stem %s" % stem)
        records[stem].geom_polylines = [
            [[float(x1), float(y1)], [float(x2), float(y2)]]
            for x1, y1, x2, y2, _ in geom_pkl[stem]
        ]


def save_gnn_pkl_only(records: Dict[str, ImageRecord], gnn_raw: Dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(PRED_GNN_PKL), exist_ok=True)
    gnn_pkl = build_gnn_pkl(records, gnn_raw)
    with open(PRED_GNN_PKL, "wb") as f:
        pickle.dump(gnn_pkl, f)
    print("[OK] Wrote GNN pred pickle:", PRED_GNN_PKL)


def main() -> None:
    import argparse

    global SPLIT, DATASET_ROOT, ORIG_JSON_DIR, PRED_GEOM_PKL, PRED_GNN_PKL
    global RESULTS_LSNETV2_JSON, RESULTS_GTA_THICKNESS_SWEEP_JSON, RESULTS_RELAXATION_JSON
    global EXP55_CONFIG, EXP55_CKPT, LSNETV2_LINE_WIDTHS, GT_A_CROP_AWARE, GT_A_STRETCH

    parser = argparse.ArgumentParser(description="Pixel F1 / LSNetv2-style eval on TTPLA")
    parser.add_argument("--split", type=str, default=SPLIT, help="Dataset split (default: val)")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DATASET_ROOT,
        help="TTPLA dataset root (default: smallset)",
    )
    parser.add_argument("--geom-config", type=str, default=EXP19_CONFIG)
    parser.add_argument("--geom-ckpt", type=str, default=EXP19_CKPT)
    parser.add_argument("--gnn-config", type=str, default=EXP55_CONFIG)
    parser.add_argument("--gnn-ckpt", type=str, default=EXP55_CKPT)
    parser.add_argument(
        "--pkl-tag",
        type=str,
        default="",
        help="Pickle suffix pred_geom_<tag>.pkl (empty = default paths)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="LSNetv2 comparison JSON path (default: results/lsnetv2_comparison.json)",
    )
    parser.add_argument(
        "--lsnetv2-only",
        action="store_true",
        help="Run inference + LSNetv2 APR/ARR/F1/Fb only (skip thickness sweep, relaxation, viz)",
    )
    parser.add_argument(
        "--gta-only",
        action="store_true",
        help="GT-A polygon-fill eval only (requires --from-pkl; skips inference/LSNetv2)",
    )
    parser.add_argument(
        "--orig-json-dir",
        type=str,
        default=None,
        help="LabelMe JSON dir for GT-A polygons (default: maptr 1600x900 annotations)",
    )
    parser.add_argument(
        "--gta-output-json",
        type=str,
        default=None,
        help="GT-A thickness sweep JSON path (default: results/gta_thickness_sweep.json)",
    )
    parser.add_argument(
        "--relaxation-output-json",
        type=str,
        default=None,
        help="GT-A relaxation / PLGAN JSON path (default: results/pixel_f1_results_relaxation.json)",
    )
    parser.add_argument(
        "--plgan-only",
        action="store_true",
        help="With --gta-only: skip thickness sweep; run PLGAN relaxation eval only",
    )
    parser.add_argument(
        "--gt-a-stretch",
        action="store_true",
        help="GT-A: stretch JSON polygons to tile size (512² PLGAN benchmark)",
    )
    parser.add_argument(
        "--no-gt-a-crop",
        action="store_true",
        help="Disable 1024 dual-crop GT-A alignment (use with --gt-a-stretch for 512²)",
    )
    parser.add_argument(
        "--relaxation-px",
        type=int,
        nargs="+",
        default=None,
        help="Tolerance radius list in px (default: 2 for --gt-a-stretch, else 2 and 4)",
    )
    parser.add_argument(
        "--line-widths",
        type=int,
        nargs="+",
        default=None,
        help="LSNetv2 GT-B line widths in px (default: 2 4)",
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization PNGs")
    parser.add_argument(
        "--from-pkl",
        action="store_true",
        help="Skip inference; load cached prediction pickles (must match split/stems)",
    )
    parser.add_argument(
        "--gnn-only",
        action="store_true",
        help="Load geom from pickle, re-run GNN inference + eval only",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Limit images (0 = all)")
    args = parser.parse_args()

    SPLIT = args.split
    DATASET_ROOT = os.path.abspath(args.dataset_root)
    geom_config = os.path.abspath(args.geom_config)
    geom_ckpt = os.path.abspath(args.geom_ckpt)
    gnn_config = os.path.abspath(args.gnn_config)
    gnn_ckpt = os.path.abspath(args.gnn_ckpt)
    if args.pkl_tag:
        PRED_GEOM_PKL, PRED_GNN_PKL = _pkl_paths(args.pkl_tag)
    if args.orig_json_dir:
        ORIG_JSON_DIR = os.path.abspath(args.orig_json_dir)
    RESULTS_LSNETV2_JSON = os.path.abspath(
        args.output_json or os.path.join(OUT_DIR, "lsnetv2_comparison.json")
    )
    RESULTS_GTA_THICKNESS_SWEEP_JSON = os.path.abspath(
        args.gta_output_json or os.path.join(OUT_DIR, "gta_thickness_sweep.json")
    )
    RESULTS_RELAXATION_JSON = os.path.abspath(
        args.relaxation_output_json or os.path.join(OUT_DIR, "pixel_f1_results_relaxation.json")
    )
    if args.line_widths:
        LSNETV2_LINE_WIDTHS = tuple(int(x) for x in args.line_widths)
    GT_A_STRETCH = bool(args.gt_a_stretch)
    GT_A_CROP_AWARE = not args.no_gt_a_crop and not GT_A_STRETCH
    if args.relaxation_px:
        relax_specs = [(int(r), "%dpx" % int(r)) for r in args.relaxation_px]
    elif GT_A_STRETCH:
        relax_specs = [(2, "2px_512_reference")]
    else:
        relax_specs = list(RELAXATION_SPECS)

    if args.gta_only and not args.from_pkl:
        raise ValueError("--gta-only requires --from-pkl (use cached predictions)")

    use_gpu = torch.cuda.is_available()
    print("[INFO] device:", "cuda" if use_gpu else "cpu")
    print("[INFO] split:", SPLIT)
    print("[INFO] dataset_root:", DATASET_ROOT)
    print("[INFO] GT-A json dir:", ORIG_JSON_DIR)
    path_checks = [DATASET_ROOT, ORIG_JSON_DIR]
    if args.gta_only:
        if not args.pkl_tag:
            raise ValueError("--gta-only requires --pkl-tag")
        path_checks.extend([PRED_GEOM_PKL, PRED_GNN_PKL])
    elif args.gnn_only:
        if not args.pkl_tag:
            raise ValueError("--gnn-only requires --pkl-tag (geom pickle prefix)")
        path_checks.extend([gnn_config, gnn_ckpt, PRED_GEOM_PKL])
    elif args.from_pkl:
        path_checks.extend([PRED_GEOM_PKL, PRED_GNN_PKL])
    else:
        path_checks.extend([geom_config, geom_ckpt, gnn_config, gnn_ckpt])
    for p in path_checks:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    os.makedirs(OUT_DIR, exist_ok=True)
    if not args.no_viz and os.path.isdir(VIZ_DIR):
        for f in os.listdir(VIZ_DIR):
            if f.endswith(".png"):
                os.remove(os.path.join(VIZ_DIR, f))

    img_dir = os.path.join(DATASET_ROOT, "images", SPLIT)
    file_names = sorted(
        os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith(".png")
    )
    if args.max_images > 0:
        file_names = file_names[: args.max_images]
    print("[INFO] %s images:" % SPLIT, len(file_names))

    if args.from_pkl:
        if not os.path.exists(PRED_GEOM_PKL) or not os.path.exists(PRED_GNN_PKL):
            raise FileNotFoundError("Pickles not found: %s / %s" % (PRED_GEOM_PKL, PRED_GNN_PKL))
        records = load_records(DATASET_ROOT, file_names)
        with open(PRED_GEOM_PKL, "rb") as f:
            geom_pkl = pickle.load(f)
        with open(PRED_GNN_PKL, "rb") as f:
            gnn_pkl = pickle.load(f)
        for stem in file_names:
            if stem not in geom_pkl or stem not in gnn_pkl:
                raise ValueError("Pickles missing stem %s (wrong split/tag?)" % stem)
            rec = records[stem]
            rec.geom_polylines = [
                [[float(x1), float(y1)], [float(x2), float(y2)]]
                for x1, y1, x2, y2, _ in geom_pkl[stem]
            ]
            rec.gnn_segments = [
                ((float(x1), float(y1)), (float(x2), float(y2)))
                for x1, y1, x2, y2, _ in gnn_pkl[stem]
            ]
        print("[INFO] Loaded predictions from pickles:", PRED_GEOM_PKL, PRED_GNN_PKL)
    elif args.gnn_only:
        records = load_records(DATASET_ROOT, file_names)
        with open(PRED_GEOM_PKL, "rb") as f:
            geom_pkl = pickle.load(f)
        _load_geom_polylines_from_pkl(records, geom_pkl, file_names)
        print("[INFO] Loaded geom from pickle (skip inference):", PRED_GEOM_PKL)
        gnn_raw = {}
        print("[INFO] Loading GNN (Pred-2) only...")
        _a55, fwd55, ds55, loader55, dev55 = setup_run(
            gnn_config, gnn_ckpt, DATASET_ROOT, use_gpu
        )
        run_gnn_predictions(fwd55, ds55, loader55, dev55, records, gnn_raw)
        save_gnn_pkl_only(records, gnn_raw)
    else:
        records = load_records(DATASET_ROOT, file_names)
        gnn_raw: Dict[str, dict] = {}

        print("[INFO] Loading geom (Pred-1)...")
        _a19, fwd19, ds19, loader19, dev19 = setup_run(
            geom_config, geom_ckpt, DATASET_ROOT, use_gpu
        )
        run_geom_predictions(fwd19, ds19, loader19, dev19, records)
        del fwd19, _a19, ds19, loader19
        if use_gpu:
            torch.cuda.empty_cache()

        print("[INFO] Loading GNN (Pred-2)...")
        _a55, fwd55, ds55, loader55, dev55 = setup_run(
            gnn_config, gnn_ckpt, DATASET_ROOT, use_gpu
        )
        run_gnn_predictions(fwd55, ds55, loader55, dev55, records, gnn_raw)
        save_isq_pred_pkls(records, gnn_raw)

    if args.gta_only:
        if not args.plgan_only:
            gta_sweep_results = run_gt_a_thickness_sweep(records)
            gta_sweep_results["settings"]["geom_checkpoint"] = geom_ckpt
            gta_sweep_results["settings"]["gnn_checkpoint"] = gnn_ckpt
            gta_sweep_results["settings"]["gt_a_crop_aware"] = GT_A_CROP_AWARE
            gta_sweep_results["settings"]["gt_a_stretch"] = GT_A_STRETCH
            with open(RESULTS_GTA_THICKNESS_SWEEP_JSON, "w", encoding="utf-8") as f:
                json.dump(gta_sweep_results, f, indent=2)
            print("[OK] Wrote", RESULTS_GTA_THICKNESS_SWEEP_JSON)
        relaxation_results = run_gt_a_relaxation_eval(records, relaxation_specs=relax_specs)
        relaxation_results["settings"]["geom_checkpoint"] = geom_ckpt
        relaxation_results["settings"]["gnn_checkpoint"] = gnn_ckpt
        with open(RESULTS_RELAXATION_JSON, "w", encoding="utf-8") as f:
            json.dump(relaxation_results, f, indent=2)
        print("[OK] Wrote", RESULTS_RELAXATION_JSON)
        return

    if not args.lsnetv2_only:
        results = {
            "settings": {
                "dataset_root": DATASET_ROOT,
                "split": SPLIT,
                "line_thickness_values": list(THICKNESS_VALUES),
                "viz_line_thickness": VIZ_THICKNESS,
                "geom_confidence": GEOM_CONF,
                "gnn_edge_thresh": GNN_EDGE_THRESH,
                "gnn_node_conf": GNN_NODE_CONF,
                "geom_checkpoint": geom_ckpt,
                "gnn_checkpoint": gnn_ckpt,
            },
            "by_thickness": {},
        }

        for thickness in THICKNESS_VALUES:
            print(
                "\n=== line_thickness=%d (GT-B and preds rasterized at t=%d) ==="
                % (thickness, thickness)
            )
            t_key = str(thickness)
            results["by_thickness"][t_key] = {}
            for pred_kind, label in (("geom", "GT-B_vs_Pred-1_geom"), ("gnn", "GT-B_vs_Pred-2_gnn")):
                block = evaluate_gt_b_vs_pred(records, pred_kind, thickness)
                results["by_thickness"][t_key][label] = block
                s = block["summary"]
                print(
                    "[RESULT] t=%d  %s  P=%.4f  R=%.4f  F1=%.4f  IoU=%.4f  (n=%d)"
                    % (thickness, label, s["precision"], s["recall"], s["f1"], s["iou"], s["n_images"])
                )

        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("\n[OK] Wrote", RESULTS_JSON)

        relaxation_results = run_gt_a_relaxation_eval(records)
        with open(RESULTS_RELAXATION_JSON, "w", encoding="utf-8") as f:
            json.dump(relaxation_results, f, indent=2)
        print("[OK] Wrote", RESULTS_RELAXATION_JSON)

    lsnetv2_results = run_lsnetv2_comparison(records)
    lsnetv2_results["settings"]["geom_checkpoint"] = geom_ckpt
    lsnetv2_results["settings"]["gnn_checkpoint"] = gnn_ckpt
    lsnetv2_results["settings"]["dataset_root"] = DATASET_ROOT
    with open(RESULTS_LSNETV2_JSON, "w", encoding="utf-8") as f:
        json.dump(lsnetv2_results, f, indent=2)
    print("[OK] Wrote", RESULTS_LSNETV2_JSON)

    if not args.lsnetv2_only:
        gta_sweep_results = run_gt_a_thickness_sweep(records)
        with open(RESULTS_GTA_THICKNESS_SWEEP_JSON, "w", encoding="utf-8") as f:
            json.dump(gta_sweep_results, f, indent=2)
        print("[OK] Wrote", RESULTS_GTA_THICKNESS_SWEEP_JSON)

    if not args.no_viz and not args.lsnetv2_only:
        viz_stems = file_names[: min(NUM_VIZ, len(file_names))]
        save_viz(records, viz_stems, thickness=VIZ_THICKNESS)
        print(
            "[OK] Wrote %d viz PNGs (GT-B | Pred-1 | Pred-2, t=%d) under %s"
            % (len(viz_stems), VIZ_THICKNESS, VIZ_DIR)
        )


if __name__ == "__main__":
    main()
