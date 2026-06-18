#!/usr/bin/env python3
"""ISQ metric validation harness (reuses isq_core.compute_isq_image)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

EVAL_ISQ_DIR = Path(__file__).resolve().parent
BASE = EVAL_ISQ_DIR.parent
sys.path.insert(0, str(EVAL_ISQ_DIR))
sys.path.insert(0, str(BASE / "yolino" / "CAPSTONE" / "src"))

from isq_core import (  # noqa: E402
    CELL_SIZE,
    GtSegment,
    Segment,
    compute_isq_image,
    gt_polylines_to_segments,
    summarize_isq,
)
from yolino.visualize_ttpla_gt_instances import load_ttpla_label_with_ids  # noqa: E402

DATASET_ROOT = BASE / "yolino" / "ttpla_yolino_dataset_1024x1024_smallset"
OUT_JSON = EVAL_ISQ_DIR / "results" / "isq_validation.json"
SPLIT = "val"
MIN_GT_INSTANCES = 3
N_IMAGES = 5


@dataclass
class TestResult:
    name: str
    precision: float
    recall: float
    f1: float
    over_split: float
    over_split_count: int
    under_merge: float
    passed: bool
    detail: str
    extra: Optional[Dict] = None


def _fast_gt_segments(
    polylines: Sequence, instance_ids: Sequence[int], image_size: int
) -> List[GtSegment]:
    """Same segments as gt_polylines_to_segments, with bbox-pruned cell iteration."""
    all_segs: List[GtSegment] = []
    next_idx = 0
    for poly, pid in zip(polylines, instance_ids):
        pts = [p for p in poly if len(p) >= 2]
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        n = image_size // CELL_SIZE
        r0 = max(0, int(min(ys)) // CELL_SIZE)
        r1 = min(n - 1, int(max(ys)) // CELL_SIZE)
        c0 = max(0, int(min(xs)) // CELL_SIZE)
        c1 = min(n - 1, int(max(xs)) // CELL_SIZE)
        from isq_core import polyline_segment_in_cell

        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                seg = polyline_segment_in_cell(poly, row, col, cell_size=CELL_SIZE)
                if seg is None:
                    continue
                (x1, y1), (x2, y2) = seg
                all_segs.append(
                    GtSegment(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        polyline_id=int(pid),
                        seg_idx=next_idx,
                    )
                )
                next_idx += 1
    return all_segs


def _load_image_cache(stem: str) -> Dict[str, object]:
    label_path = DATASET_ROOT / "labels" / SPLIT / f"{stem}.npy"
    img_path = DATASET_ROOT / "images" / SPLIT / f"{stem}.png"
    import cv2

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    image_size = max(h, w)
    polylines, instance_ids, _ = load_ttpla_label_with_ids(str(label_path))
    gt_segments = _fast_gt_segments(polylines, instance_ids, image_size)
    return {
        "polylines": polylines,
        "instance_ids": list(instance_ids),
        "gt_segments": gt_segments,
        "by_pid": _segments_by_polyline(gt_segments),
        "image_size": image_size,
    }


_IMAGE_CACHE: Dict[str, Dict[str, object]] = {}


def _get_image(stem: str) -> Dict[str, object]:
    if stem not in _IMAGE_CACHE:
        _IMAGE_CACHE[stem] = _load_image_cache(stem)
    return _IMAGE_CACHE[stem]


def _segments_by_polyline(gt_segments: Sequence[GtSegment]) -> Dict[int, List[Segment]]:
    by_pid: Dict[int, List[Segment]] = defaultdict(list)
    for g in gt_segments:
        by_pid[g.polyline_id].append(g.as_endpoints())
    return dict(by_pid)


def _eval_all(
    stems: List[str],
    pred_fn: Callable[[List[GtSegment], Dict[int, List[Segment]]], List[List[Segment]]],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    per_image = []
    for stem in stems:
        rec = _get_image(stem)
        gt_segments: List[GtSegment] = rec["gt_segments"]  # type: ignore[assignment]
        by_pid: Dict[int, List[Segment]] = rec["by_pid"]  # type: ignore[assignment]
        pred = pred_fn(gt_segments, by_pid)
        m = compute_isq_image(gt_segments, pred)
        m["stem"] = stem
        per_image.append(m)
    return summarize_isq(per_image), per_image


def _select_stems() -> List[str]:
    img_dir = DATASET_ROOT / "images" / SPLIT
    stems = sorted(p.stem for p in img_dir.glob("*.png"))
    chosen: List[str] = []
    for stem in stems:
        label_path = DATASET_ROOT / "labels" / SPLIT / f"{stem}.npy"
        polylines, instance_ids, _ = load_ttpla_label_with_ids(str(label_path))
        n_inst = len(instance_ids)
        if n_inst >= MIN_GT_INSTANCES:
            chosen.append(stem)
        if len(chosen) >= N_IMAGES:
            break
    if len(chosen) < N_IMAGES:
        raise RuntimeError(
            f"Need {N_IMAGES} val images with >={MIN_GT_INSTANCES} GT instances; found {len(chosen)}"
        )
    return chosen


def _metrics_from_summary(summary: Dict[str, object]) -> Dict[str, float]:
    return {
        "precision": float(summary["precision"]),
        "recall": float(summary["recall"]),
        "f1": float(summary["f1"]),
        "over_split": float(summary["over_split"]),
        "over_split_count": int(summary.get("over_split_count", 0)),
        "under_merge": float(summary["under_merge"]),
    }


def _make_under_merge_pred(by_pid: Dict[int, List[Segment]]) -> List[List[Segment]]:
    pids = sorted(by_pid)
    if len(pids) < 2:
        return [by_pid[pids[0]]] if pids else []
    merged = by_pid[pids[0]] + by_pid[pids[1]]
    pred = [merged]
    for pid in pids[2:]:
        pred.append(by_pid[pid])
    return pred


def _make_over_split_pred(by_pid: Dict[int, List[Segment]]) -> List[List[Segment]]:
    """Split longest GT polyline: two preds overlap in coverage (over-split) but omit tail segs (F1 drop)."""
    pids = sorted(by_pid, key=lambda p: -len(by_pid[p]))
    split_pid = pids[0]
    segs = by_pid[split_pid]
    n = len(segs)
    mid = max(2, n // 2)
    first = segs[:mid]
    # Second pred covers ~35% of GT (>=0.3 thresh) but leaves tail segments unmatched → FN.
    second_len = max(2, int(round(n * 0.35)))
    second = segs[mid : mid + second_len]
    pred: List[List[Segment]] = [first, second]
    for pid in pids[1:]:
        pred.append(by_pid[pid])
    return pred


def _make_partial_pred(by_pid: Dict[int, List[Segment]]) -> List[List[Segment]]:
    pids = sorted(by_pid)
    keep = pids[: max(1, len(pids) // 2)]
    return [by_pid[pid] for pid in keep]


def _add_noise_to_pred(
    pred: List[List[Segment]], sigma_px: float, rng: np.random.Generator
) -> List[List[Segment]]:
    if sigma_px <= 0:
        return pred
    out: List[List[Segment]] = []
    for poly in pred:
        noisy: List[Segment] = []
        for (x1, y1), (x2, y2) in poly:
            dx1, dy1 = rng.normal(0, sigma_px, 2)
            dx2, dy2 = rng.normal(0, sigma_px, 2)
            noisy.append(
                ((float(x1 + dx1), float(y1 + dy1)), (float(x2 + dx2), float(y2 + dy2)))
            )
        out.append(noisy)
    return out


def _print_table(results: List[TestResult]) -> None:
    header = f"{'Test':<28} {'P':>6} {'R':>6} {'F1':>6} {'OS':>5} {'UM':>5}  {'Result'}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<28} {r.precision:6.3f} {r.recall:6.3f} {r.f1:6.3f} "
            f"{r.over_split:5.2f} {r.under_merge:5.2f}  {('PASS' if r.passed else 'FAIL'):>4}  {r.detail}"
        )
    n_pass = sum(1 for r in results if r.passed)
    print("-" * len(header))
    print(f"Summary: {n_pass}/{len(results)} PASS, {len(results) - n_pass}/{len(results)} FAIL")


def main() -> None:
    stems = _select_stems()
    print(f"[INFO] dataset: {DATASET_ROOT}")
    print(f"[INFO] validation stems ({len(stems)}): {stems}")

    results: List[TestResult] = []
    test1_f1: Optional[float] = None
    rng = np.random.default_rng(42)

    # Test 1: Perfect prediction
    s1, _ = _eval_all(stems, lambda _gt, by_pid: [by_pid[p] for p in sorted(by_pid)])
    m1 = _metrics_from_summary(s1)
    test1_f1 = m1["f1"]
    t1_pass = m1["f1"] >= 0.95
    results.append(
        TestResult(
            name="T1 Perfect prediction",
            precision=m1["precision"],
            recall=m1["recall"],
            f1=m1["f1"],
            over_split=m1["over_split"],
            over_split_count=m1["over_split_count"],
            under_merge=m1["under_merge"],
            passed=t1_pass,
            detail=f"F1>={0.95:.2f} expected; OS=0 UM=0 expected",
        )
    )

    # Test 2: Empty prediction
    s2, _ = _eval_all(stems, lambda _gt, _by_pid: [])
    m2 = _metrics_from_summary(s2)
    t2_pass = m2["f1"] <= 0.01
    results.append(
        TestResult(
            name="T2 Empty prediction",
            precision=m2["precision"],
            recall=m2["recall"],
            f1=m2["f1"],
            over_split=m2["over_split"],
            over_split_count=m2["over_split_count"],
            under_merge=m2["under_merge"],
            passed=t2_pass,
            detail="F1<=0.01 expected",
        )
    )

    # Test 3: Under-merge
    s3, _ = _eval_all(stems, lambda _gt, by_pid: _make_under_merge_pred(by_pid))
    m3 = _metrics_from_summary(s3)
    t3_pass = m3["under_merge"] >= 1.0 and m3["f1"] < test1_f1
    results.append(
        TestResult(
            name="T3 Under-merge",
            precision=m3["precision"],
            recall=m3["recall"],
            f1=m3["f1"],
            over_split=m3["over_split"],
            over_split_count=m3["over_split_count"],
            under_merge=m3["under_merge"],
            passed=t3_pass,
            detail=f"UM>=1 & F1<{test1_f1:.3f}",
        )
    )

    # Test 4: Over-split
    s4, _ = _eval_all(stems, lambda _gt, by_pid: _make_over_split_pred(by_pid))
    m4 = _metrics_from_summary(s4)
    t4_pass = m4["over_split_count"] >= 1 and m4["f1"] < test1_f1
    results.append(
        TestResult(
            name="T4 Over-split",
            precision=m4["precision"],
            recall=m4["recall"],
            f1=m4["f1"],
            over_split=m4["over_split"],
            over_split_count=m4["over_split_count"],
            under_merge=m4["under_merge"],
            passed=t4_pass,
            detail=f"OS_count>=1 & F1<{test1_f1:.3f}",
        )
    )

    # Test 5: Partial miss
    s5, _ = _eval_all(stems, lambda _gt, by_pid: _make_partial_pred(by_pid))
    m5 = _metrics_from_summary(s5)
    t5_pass = 0.3 <= m5["recall"] <= 0.7 and m5["f1"] < test1_f1
    results.append(
        TestResult(
            name="T5 Partial miss",
            precision=m5["precision"],
            recall=m5["recall"],
            f1=m5["f1"],
            over_split=m5["over_split"],
            over_split_count=m5["over_split_count"],
            under_merge=m5["under_merge"],
            passed=t5_pass,
            detail=f"R in [0.3,0.7] & F1<{test1_f1:.3f}",
        )
    )

    # Test 6: Noise monotonicity
    noise_levels = [0, 5, 10, 20, 30]
    f1_by_noise: Dict[int, float] = {}
    noise_summaries: Dict[str, object] = {}
    for nv in noise_levels:
        s_n, _ = _eval_all(
            stems,
            lambda _gt, by_pid: _add_noise_to_pred(
                [by_pid[p] for p in sorted(by_pid)], float(nv), rng
            ),
        )
        noise_summaries[str(nv)] = s_n
        f1_by_noise[nv] = float(s_n["f1"])

    t6_pass = (
        f1_by_noise[0] > f1_by_noise[5] > f1_by_noise[20] > f1_by_noise[30]
    )
    s6_rep = noise_summaries["0"]
    m6 = _metrics_from_summary(s6_rep)  # report 0px as primary row
    results.append(
        TestResult(
            name="T6 Noise monotonicity",
            precision=m6["precision"],
            recall=m6["recall"],
            f1=m6["f1"],
            over_split=m6["over_split"],
            over_split_count=m6["over_split_count"],
            under_merge=m6["under_merge"],
            passed=t6_pass,
            detail=(
                "F1(0)>F1(5)>F1(20)>F1(30); "
                + ", ".join(f"{k}px={f1_by_noise[k]:.3f}" for k in noise_levels)
            ),
            extra={"f1_by_noise_px": {str(k): f1_by_noise[k] for k in noise_levels}},
        )
    )

    _print_table(results)

    n_pass = sum(1 for r in results if r.passed)
    payload = {
        "settings": {
            "dataset_root": str(DATASET_ROOT),
            "split": SPLIT,
            "min_gt_instances": MIN_GT_INSTANCES,
            "n_images": N_IMAGES,
            "stems": stems,
        },
        "tests": [
            {
                "name": r.name,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "over_split": r.over_split,
                "over_split_count": r.over_split_count,
                "under_merge": r.under_merge,
                "passed": r.passed,
                "detail": r.detail,
                **({"extra": r.extra} if r.extra else {}),
            }
            for r in results
        ],
        "summary": {
            "n_pass": n_pass,
            "n_fail": len(results) - n_pass,
            "n_total": len(results),
            "all_passed": n_pass == len(results),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[OK] Wrote {OUT_JSON}")

    if not payload["summary"]["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
