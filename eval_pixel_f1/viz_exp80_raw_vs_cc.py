#!/usr/bin/env python3
"""3-panel viz: RGB | raw geom segments | CC-grouped polylines (+ GT overlay)."""
from __future__ import annotations

import os
import pickle
import sys
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "yolino", "CAPSTONE", "src")
sys.path.insert(0, SRC)

from yolino.visualize_ttpla_gt_instances import load_ttpla_label_with_ids  # noqa: E402

DATASET_ROOT = os.path.join(BASE, "yolino", "YOLinO_benchmark")
RAW_PKL = os.path.join(BASE, "eval_isq", "pred_geom_raw_exp80_512_test.pkl")
CC_PKL = os.path.join(BASE, "eval_isq", "pred_geom_exp80_512_test.pkl")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "viz_exp80_raw_vs_cc")

# Complex scenes: high raw segment count + multiple CC polylines.
SELECTED_STEMS = [
    "46_01191",
    "32_6285",
    "71_3440",
    "26_1050",
    "04_585",
]

LINE_THICKNESS = 2
GT_COLOR = (0, 255, 0)  # BGR green


def _color_for_id(idx: int) -> tuple[int, int, int]:
    cmap = plt.get_cmap("tab20")
    c = cmap(idx % 20)
    return (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))


def _draw_gt(overlay: np.ndarray, polylines) -> None:
    for poly in polylines:
        pts = np.asarray(poly, dtype=np.float32)
        if len(pts) < 2:
            continue
        pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            overlay, [pts_i], False, GT_COLOR, thickness=1, lineType=cv2.LINE_AA
        )


def _draw_raw_segments(overlay: np.ndarray, segments) -> None:
    for i, seg in enumerate(segments):
        if len(seg) == 4:
            x1, y1, x2, y2 = seg
        else:
            x1, y1, x2, y2, _ = seg
        color = _color_for_id(i)
        cv2.line(
            overlay,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color,
            LINE_THICKNESS,
            cv2.LINE_AA,
        )


def _chain_by_pid(segments_with_pid) -> list[list[tuple[float, float]]]:
    """Group CC pickle segments by polyline id; greedy chain endpoints per group."""
    by_pid: dict[int, list[tuple]] = defaultdict(list)
    for row in segments_with_pid:
        x1, y1, x2, y2, pid = row
        by_pid[int(pid)].append(((float(x1), float(y1)), (float(x2), float(y2))))

    polylines: list[list[tuple[float, float]]] = []
    tol2 = 4.0

    def d2(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    for pid in sorted(by_pid):
        segs = by_pid[pid]
        if not segs:
            continue
        if len(segs) == 1:
            a, b = segs[0]
            polylines.append([a, b])
            continue
        used = [False] * len(segs)
        start = 0
        used[start] = True
        a, b = segs[start]
        path = [a, b]
        changed = True
        while changed:
            changed = False
            for end in ("tail", "head"):
                ref = path[-1] if end == "tail" else path[0]
                best_j, best_d, best_attach = -1, tol2 + 1, None
                for j, (p0, p1) in enumerate(segs):
                    if used[j]:
                        continue
                    for q, other in ((p0, p1), (p1, p0)):
                        dd = d2(q, ref)
                        if dd <= tol2 and dd < best_d:
                            best_j, best_d, best_attach = j, dd, other
                if best_j >= 0:
                    used[best_j] = True
                    if end == "tail":
                        path.append(best_attach)
                    else:
                        path.insert(0, best_attach)
                    changed = True
        for j, (p0, p1) in enumerate(segs):
            if not used[j]:
                path.extend([p0, p1])
        polylines.append(path)
    return polylines


def _draw_cc_polylines(overlay: np.ndarray, segments_with_pid) -> None:
    polylines = _chain_by_pid(segments_with_pid)
    for i, pts in enumerate(polylines):
        if len(pts) < 2:
            continue
        color = _color_for_id(i)
        arr = np.round(np.asarray(pts, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [arr], False, color, LINE_THICKNESS, cv2.LINE_AA)


def _title_panel(img: np.ndarray, title: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(
        out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
    )
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RAW_PKL, "rb") as f:
        raw_pkl = pickle.load(f)
    with open(CC_PKL, "rb") as f:
        cc_pkl = pickle.load(f)

    label_dir = os.path.join(DATASET_ROOT, "labels", "test")
    img_dir = os.path.join(DATASET_ROOT, "images", "test")

    for stem in SELECTED_STEMS:
        img_path = os.path.join(img_dir, stem + ".png")
        lbl_path = os.path.join(label_dir, stem + ".npy")
        rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(img_path)
        polylines, _, _ = load_ttpla_label_with_ids(lbl_path)

        raw_segs = raw_pkl[stem]
        cc_segs = cc_pkl[stem]
        n_raw = len(raw_segs)
        n_cc_poly = len(set(int(s[4]) for s in cc_segs))

        panel_rgb = _title_panel(rgb, "RGB")

        panel_raw = rgb.copy()
        _draw_gt(panel_raw, polylines)
        _draw_raw_segments(panel_raw, raw_segs)
        panel_raw = _title_panel(panel_raw, "Raw segments (n=%d) + GT green" % n_raw)

        panel_cc = rgb.copy()
        _draw_gt(panel_cc, polylines)
        _draw_cc_polylines(panel_cc, cc_segs)
        panel_cc = _title_panel(panel_cc, "CC polylines (n=%d) + GT green" % n_cc_poly)

        row = np.concatenate([panel_rgb, panel_raw, panel_cc], axis=1)
        out_path = os.path.join(OUT_DIR, "%s_raw_vs_cc.png" % stem)
        cv2.imwrite(out_path, row)
        print(
            "[OK] %s  raw=%d segs  cc=%d polylines (%d segs in pkl)"
            % (stem, n_raw, n_cc_poly, len(cc_segs))
        )

    print("[DONE] wrote %d images to %s" % (len(SELECTED_STEMS), OUT_DIR))


if __name__ == "__main__":
    main()
