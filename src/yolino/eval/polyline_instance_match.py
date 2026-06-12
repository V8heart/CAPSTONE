# SPDX-License-Identifier: GPL-3.0-or-later
"""Polyline-level instance matching via mask IoU + Hungarian assignment."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


PolylineXY = np.ndarray  # [N, 2] float, (x, y) pixel coords
SegmentXY = Tuple[np.ndarray, np.ndarray]  # end_a, end_b each [2]


def load_gt_polylines_xy(label_path: str) -> List[PolylineXY]:
    """Load annotation polylines from a TTPLA-style ``.npy`` label (x, y) pixels."""
    raw = np.load(label_path, allow_pickle=True)
    payload = raw.item() if (isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ()) else raw
    polylines: List[PolylineXY] = []
    if isinstance(payload, dict):
        if "instances" in payload:
            for inst in payload.get("instances", []):
                pts = inst.get("points", [])
                if len(pts) >= 2:
                    polylines.append(np.asarray(pts, dtype=np.float64))
        else:
            raw_polys = payload.get("polylines", [])
            for p in raw_polys:
                pts = p.get("points", []) if isinstance(p, dict) else p
                if len(pts) >= 2:
                    polylines.append(np.asarray(pts, dtype=np.float64))
    else:
        for p in payload:
            if p is not None and len(p) >= 2:
                polylines.append(np.asarray(p, dtype=np.float64))
    return polylines


def polyline_to_mask(
    poly_xy: PolylineXY,
    height: int,
    width: int,
    line_width: int = 10,
) -> np.ndarray:
    """Rasterize an open polyline to a bool mask ``[H, W]``."""
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    if poly_xy is None or len(poly_xy) < 2:
        return mask.astype(bool)
    pts = np.round(np.asarray(poly_xy, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
    pts[..., 0] = np.clip(pts[..., 0], 0, width - 1)
    pts[..., 1] = np.clip(pts[..., 1], 0, height - 1)
    cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=int(line_width), lineType=cv2.LINE_AA)
    return mask.astype(bool)


def segments_to_mask(
    segments: Sequence[SegmentXY],
    height: int,
    width: int,
    line_width: int = 10,
) -> np.ndarray:
    """Rasterize line segments (union) to a bool mask."""
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    for ea, eb in segments:
        if ea is None or eb is None:
            continue
        p0 = (int(round(float(ea[0]))), int(round(float(ea[1]))))
        p1 = (int(round(float(eb[0]))), int(round(float(eb[1]))))
        if any(np.isnan(p0 + p1)):
            continue
        p0 = (np.clip(p0[0], 0, width - 1), np.clip(p0[1], 0, height - 1))
        p1 = (np.clip(p1[0], 0, width - 1), np.clip(p1[1], 0, height - 1))
        cv2.line(mask, p0, p1, 1, thickness=int(line_width), lineType=cv2.LINE_AA)
    return mask.astype(bool)


def compute_mask_iou(m0: np.ndarray, m1: np.ndarray) -> float:
    inter = int(np.logical_and(m0, m1).sum())
    union = int(np.logical_or(m0, m1).sum())
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _pred_to_mask(
    pred: Union[PolylineXY, Sequence[SegmentXY]],
    height: int,
    width: int,
    line_width: int,
) -> np.ndarray:
    if isinstance(pred, np.ndarray):
        return polyline_to_mask(pred, height, width, line_width)
    return segments_to_mask(pred, height, width, line_width)


@dataclass
class BipartiteMatchResult:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    n_pred: int
    n_gt: int
    iou_matrix: np.ndarray = field(repr=False)
    matched_pairs: List[Tuple[int, int, float]] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "tp": int(self.tp),
            "fp": int(self.fp),
            "fn": int(self.fn),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "n_pred": int(self.n_pred),
            "n_gt": int(self.n_gt),
            "matched_pairs": [(int(p), int(g), float(i)) for p, g, i in self.matched_pairs],
        }


def bipartite_polyline_match(
    pred_polylines: Sequence[Union[PolylineXY, Sequence[SegmentXY]]],
    gt_polylines: Sequence[PolylineXY],
    height: int,
    width: int,
    *,
    line_width: int = 10,
    iou_threshold: float = 0.5,
) -> BipartiteMatchResult:
    """Hungarian match pred↔GT by mask IoU; TP when IoU ≥ threshold."""
    n_pred = len(pred_polylines)
    n_gt = len(gt_polylines)
    if n_pred == 0 and n_gt == 0:
        return BipartiteMatchResult(0, 0, 0, 1.0, 1.0, 1.0, 0, 0, np.zeros((0, 0)))

    pred_masks = [_pred_to_mask(p, height, width, line_width) for p in pred_polylines]
    gt_masks = [polyline_to_mask(g, height, width, line_width) for g in gt_polylines]

    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float64)
    for i in range(n_pred):
        for j in range(n_gt):
            iou_matrix[i, j] = compute_mask_iou(pred_masks[i], gt_masks[j])

    if n_pred == 0:
        return BipartiteMatchResult(
            0, 0, n_gt, 0.0, 0.0, 0.0, 0, n_gt, iou_matrix,
        )
    if n_gt == 0:
        return BipartiteMatchResult(
            0, n_pred, 0, 0.0, 0.0, 0.0, n_pred, 0, iou_matrix,
        )

    pred_idx, gt_idx = linear_sum_assignment(-iou_matrix)
    matched_pred: set = set()
    matched_gt: set = set()
    pairs: List[Tuple[int, int, float]] = []
    tp = 0
    for p, g in zip(pred_idx.tolist(), gt_idx.tolist()):
        iou = float(iou_matrix[p, g])
        pairs.append((int(p), int(g), iou))
        if iou >= float(iou_threshold):
            tp += 1
            matched_pred.add(int(p))
            matched_gt.add(int(g))

    fp = n_pred - len(matched_pred)
    fn = n_gt - len(matched_gt)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return BipartiteMatchResult(
        tp, fp, fn, precision, recall, f1, n_pred, n_gt, iou_matrix, pairs,
    )


def audit_assignment_one_to_one(result: BipartiteMatchResult) -> Dict[str, bool | int]:
    """Verify Hungarian assignment uses at most one pred per GT (and vice versa)."""
    gt_ids = [g for _p, g, _i in result.matched_pairs]
    pred_ids = [p for p, _g, _i in result.matched_pairs]
    return {
        "hungarian_pairs": len(result.matched_pairs),
        "unique_gt_in_assignment": len(set(gt_ids)) == len(gt_ids),
        "unique_pred_in_assignment": len(set(pred_ids)) == len(pred_ids),
        "tp_pairs": int(result.tp),
        "n_pred": int(result.n_pred),
        "n_gt": int(result.n_gt),
    }


def aggregate_match_results(results: Sequence[BipartiteMatchResult]) -> Dict[str, float]:
    """Micro-average TP/FP/FN across images."""
    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    n_pred = sum(r.n_pred for r in results)
    n_gt = sum(r.n_gt for r in results)
    macro_f1 = float(np.mean([r.f1 for r in results])) if results else 0.0
    return {
        "micro_tp": float(tp),
        "micro_fp": float(fp),
        "micro_fn": float(fn),
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "micro_f1": float(f1),
        "macro_f1": macro_f1,
        "total_pred_instances": float(n_pred),
        "total_gt_instances": float(n_gt),
        "mean_pred_per_image": float(n_pred / max(len(results), 1)),
        "mean_gt_per_image": float(n_gt / max(len(results), 1)),
        "n_images": float(len(results)),
    }
