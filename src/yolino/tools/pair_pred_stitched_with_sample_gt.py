#!/usr/bin/env python3
"""
Pair existing ``*_pred_stitched.png`` (full-frame cyan predictions on source RGB) with a
GT panel: original image from Sample + polylines from CVAT-style XML.

Default annotation source is ``Sample/merged_annotations.xml`` (root ``<annotations>`` /
``<image name=\"...jpg\">`` / ``<polyline points=\"...\">``). Restrict to validation basenames
via ``sample_yolino_guide/val.txt``.

Does not modify files under ``--pred-dir``; writes only under ``--out-dir``.

Example:

  cd CAPSTONE && python src/yolino/tools/pair_pred_stitched_with_sample_gt.py \\
    --pred-dir ttpla_train_exp/pred_stitched/exp21_val_conf0.8 \\
    --out-dir ttpla_train_exp/pred_stitched/exp21_val_conf0.8_with_sample_gt
"""
from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def concat_paired_bgr(left: np.ndarray, right: np.ndarray, gap_px: int = 8) -> np.ndarray:
    if left.shape[0] != right.shape[0]:
        scale = left.shape[0] / float(right.shape[0])
        nw = int(round(right.shape[1] * scale))
        right = cv2.resize(right, (nw, left.shape[0]), interpolation=cv2.INTER_AREA)
    elif left.shape[1] != right.shape[1]:
        nw = left.shape[1]
        nh = left.shape[0]
        right = cv2.resize(right, (nw, nh), interpolation=cv2.INTER_AREA)
    g = max(4, int(gap_px))
    gutter = np.full((left.shape[0], g, 3), 32, dtype=np.uint8)
    return np.concatenate([left, gutter, right], axis=1)


def load_val_basenames(val_txt: str) -> set[str]:
    out = set()
    with open(val_txt, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.add(s)
    return out


def parse_cvat_polyline_points(s: str) -> np.ndarray | None:
    """Return Nx1x2 int32 for cv2.polylines, or None if empty."""
    pairs = []
    for part in str(s).split(";"):
        part = part.strip()
        if not part:
            continue
        xy = part.split(",")
        if len(xy) != 2:
            continue
        pairs.append([float(xy[0].strip()), float(xy[1].strip())])
    if len(pairs) < 2:
        return None
    pts = np.asarray(pairs, dtype=np.float32)
    return pts.astype(np.int32).reshape(-1, 1, 2)


def build_image_to_polylines(annotations_xml: str) -> dict[str, list[np.ndarray]]:
    """
    Key: basename with extension exactly as in XML ``name``, e.g. ``foo_frame_000000.jpg``.
    Value: list of Nx1x2 polyline arrays.
    """
    tree = ET.parse(annotations_xml)
    root = tree.getroot()
    out: dict[str, list[np.ndarray]] = {}

    # CVAT/CVAT-like: images as direct children named "image"
    for img_el in root.iter("image"):
        fname = img_el.get("name")
        if not fname:
            continue
        polylines: list[np.ndarray] = []
        for pl in img_el.findall("polyline"):
            pts_s = pl.get("points")
            if pts_s is None:
                continue
            arr = parse_cvat_polyline_points(pts_s)
            if arr is not None:
                polylines.append(arr)
        out[fname] = polylines
    return out


def draw_gt(canvas_bgr: np.ndarray, polylines: list[np.ndarray], color, thickness: int) -> None:
    for pts in polylines:
        cv2.polylines(canvas_bgr, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def find_source_copy(sample_images_dir: str, base_no_ext: str) -> np.ndarray | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        p = os.path.join(sample_images_dir, base_no_ext + ext)
        if os.path.isfile(p):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                return None
            return img
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Horizontally pair pred_stitched PNGs with GT from Sample XML.")
    ap.add_argument("--pred-dir", required=True, help="Folder containing *_pred_stitched.png")
    ap.add_argument("--out-dir", required=True, help="New folder for *_pred_gt_paired.png (created if missing)")
    ap.add_argument(
        "--annotations-xml",
        default="/home/work/caps_drone/yolino/Sample/merged_annotations.xml",
        help="CVAT-export style XML with <image name=\"...\"><polyline points=\"\"/></image>",
    )
    ap.add_argument(
        "--sample-images-dir",
        default="/home/work/caps_drone/yolino/Sample/Sample",
        help="Directory of original JPG/PNG frames (basename matches *_pred_stitched stem)",
    )
    ap.add_argument(
        "--val-list",
        default="/home/work/caps_drone/yolino/Sample/sample_yolino_guide/val.txt",
        help="Basenames without extension; only these are processed. Empty path = process all preds in folder.",
    )
    ap.add_argument("--gap-pixels", type=int, default=8)
    ap.add_argument("--gt-thickness", type=int, default=2)
    ap.add_argument(
        "--gt-color-bgr",
        default="255,0,0",
        help='GT polyline color "B,G,R" integers, default pure blue in OpenCV BGR.',
    )
    cli = ap.parse_args()

    pred_dir = os.path.abspath(os.path.expanduser(cli.pred_dir))
    out_dir = os.path.abspath(os.path.expanduser(cli.out_dir))
    if not os.path.isdir(pred_dir):
        print("[ERROR] --pred-dir is not a directory: %s" % pred_dir, file=sys.stderr)
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(cli.annotations_xml):
        print("[ERROR] annotations XML not found: %s" % cli.annotations_xml, file=sys.stderr)
        sys.exit(1)

    val_set: set[str] | None
    vl = str(cli.val_list).strip()
    if not vl:
        val_set = None
    elif not os.path.isfile(vl):
        print("[WARN] --val-list not found, processing all preds: %s" % vl)
        val_set = None
    else:
        val_set = load_val_basenames(vl)

    try:
        b, g, r = [int(x.strip()) for x in cli.gt_color_bgr.split(",")]
        gt_color = (b, g, r)
    except ValueError:
        print("[ERROR] --gt-color-bgr must be like 255,255,255", file=sys.stderr)
        sys.exit(1)

    image_polys = build_image_to_polylines(cli.annotations_xml)
    preds = sorted(
        f for f in os.listdir(pred_dir)
        if f.endswith("_pred_stitched.png") and os.path.isfile(os.path.join(pred_dir, f))
    )
    if not preds:
        print("[ERROR] No *_pred_stitched.png under %s" % pred_dir, file=sys.stderr)
        sys.exit(1)

    n_ok = 0
    for fname in preds:
        stem_full = fname[: -len("_pred_stitched.png")]
        if val_set is not None and stem_full not in val_set:
            continue

        xml_key = stem_full + ".jpg"
        if xml_key not in image_polys:
            alt = stem_full + ".png"
            if alt in image_polys:
                xml_key = alt
            else:
                print("[SKIP] no XML entry for %s (.jpg/.png)" % stem_full)
                continue

        poly_list = image_polys.get(xml_key) or []
        pred_path = os.path.join(pred_dir, fname)
        left = cv2.imread(pred_path, cv2.IMREAD_COLOR)
        if left is None:
            print("[SKIP] cannot read pred: %s" % pred_path)
            continue

        src = find_source_copy(cli.sample_images_dir, stem_full)
        if src is None:
            print("[SKIP] no source image for base %s in %s" % (stem_full, cli.sample_images_dir))
            continue

        gt_panel = src.copy()
        if poly_list:
            draw_gt(gt_panel, poly_list, gt_color, int(cli.gt_thickness))

        combo = concat_paired_bgr(left, gt_panel, gap_px=cli.gap_pixels)
        out_path = os.path.join(out_dir, "%s_pred_gt_paired.png" % stem_full)
        cv2.imwrite(out_path, combo)
        n_ok += 1
        print("[OK] %s" % out_path)

    print("[DONE] wrote %d paired image(s) to %s" % (n_ok, out_dir))
    if n_ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
