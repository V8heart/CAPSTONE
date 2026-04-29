#!/usr/bin/env python3
"""
Assign per-source global instance IDs to tiled TTPLA labels.

This script upgrades tile label files from legacy format:
  npy -> [polyline0, polyline1, ...]
to ID-aware format:
  npy -> {
      "polylines": [...],
      "instance_ids": [...],  # global within source frame
      "source_file": "<basename>",
      "tile_offset_xy": [x0, y0],
      "format_version": 2
  }

Global ID definition:
  For each source frame JSON, each cable-like shape is assigned a stable ID by
  its order in that JSON (1..N). Tiles inheriting pieces of that shape keep the same ID.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from collections import defaultdict

import cv2
import numpy as np
from scipy.spatial import cKDTree

# Reuse project extraction logic from root script.
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocess_ttpla import extract_continuous_line_from_polygon


def parse_source_name(tile_file: str) -> str:
    base = os.path.splitext(os.path.basename(tile_file))[0]
    if "_tile" not in base:
        raise ValueError(f"Unexpected tile filename: {base}")
    return base.split("_tile")[0]


def load_source_json(json_path: str):
    with open(json_path, "r") as f:
        return json.load(f)


def decode_source_image_bgr(data: dict):
    image_data = data.get("imageData", None)
    if not image_data:
        return None
    buff = np.frombuffer(base64.b64decode(image_data), dtype=np.uint8)
    img = cv2.imdecode(buff, cv2.IMREAD_COLOR)
    return img


def is_cable_label(label: str) -> bool:
    lbl = (label or "").lower()
    return ("cable" in lbl) or ("power line" in lbl) or ("power_line" in lbl)


def build_source_instances(source_json: dict, source_img_bgr):
    h, w = source_img_bgr.shape[:2]
    instances = []
    gid = 1
    for shape in source_json.get("shapes", []):
        if not is_cable_label(shape.get("label", "")):
            continue
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        poly = extract_continuous_line_from_polygon((h, w), points)
        if len(poly) < 2:
            continue
        arr = np.asarray(poly, dtype=np.float32)  # [x,y]
        instances.append({"global_id": gid, "polyline_xy": arr})
        gid += 1
    return instances


def find_tile_offset_xy(source_bgr, tile_bgr):
    src_gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
    tile_gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(src_gray, tile_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    # max_loc is (x, y)
    return max_loc, float(max_val)


def chamfer_distance(a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    if len(a_xy) == 0 or len(b_xy) == 0:
        return 1e9
    tree_a = cKDTree(a_xy)
    tree_b = cKDTree(b_xy)
    da, _ = tree_b.query(a_xy, k=1)
    db, _ = tree_a.query(b_xy, k=1)
    return float(np.mean(da) + np.mean(db))


def best_match_global_id(global_poly_xy: np.ndarray, source_instances, tile_rect_xywh):
    x0, y0, w, h = tile_rect_xywh
    x1, y1 = x0 + w, y0 + h

    # Cheap candidate pruning by bbox overlap.
    candidates = []
    for inst in source_instances:
        p = inst["polyline_xy"]
        xmin, ymin = float(np.min(p[:, 0])), float(np.min(p[:, 1]))
        xmax, ymax = float(np.max(p[:, 0])), float(np.max(p[:, 1]))
        overlap = not (xmax < x0 or xmin > x1 or ymax < y0 or ymin > y1)
        if overlap:
            candidates.append(inst)
    if not candidates:
        candidates = source_instances

    best_id = None
    best_d = 1e18
    for inst in candidates:
        d = chamfer_distance(global_poly_xy, inst["polyline_xy"])
        if d < best_d:
            best_d = d
            best_id = inst["global_id"]
    return best_id, best_d


def load_legacy_polylines(label_npy_path: str):
    raw = np.load(label_npy_path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        payload = raw.item()
        if isinstance(payload, dict):
            # Already upgraded format
            return payload.get("polylines", []), payload.get("instance_ids", [])
    if isinstance(raw, np.ndarray):
        return [np.asarray(p, dtype=np.float32) for p in raw], None
    return [np.asarray(p, dtype=np.float32) for p in raw], None


def process_split(dataset_root, source_json_root, split, max_files=-1):
    img_dir = os.path.join(dataset_root, "images", split)
    lbl_dir = os.path.join(dataset_root, "labels", split)
    tile_imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
    if max_files > 0:
        tile_imgs = tile_imgs[:max_files]

    grouped = defaultdict(list)
    for tp in tile_imgs:
        grouped[parse_source_name(tp)].append(tp)

    updated = 0
    skipped = 0
    for src_name, tiles in grouped.items():
        json_path = os.path.join(source_json_root, f"{src_name}.json")
        if not os.path.exists(json_path):
            skipped += len(tiles)
            continue

        src_json = load_source_json(json_path)
        src_img = decode_source_image_bgr(src_json)
        if src_img is None:
            skipped += len(tiles)
            continue
        source_instances = build_source_instances(src_json, src_img)
        if len(source_instances) == 0:
            skipped += len(tiles)
            continue

        for tile_path in tiles:
            tile_img = cv2.imread(tile_path, cv2.IMREAD_COLOR)
            if tile_img is None:
                skipped += 1
                continue
            (x0, y0), score = find_tile_offset_xy(src_img, tile_img)
            h, w = tile_img.shape[:2]

            tile_base = os.path.splitext(os.path.basename(tile_path))[0]
            label_path = os.path.join(lbl_dir, f"{tile_base}.npy")
            if not os.path.exists(label_path):
                skipped += 1
                continue

            polys, maybe_ids = load_legacy_polylines(label_path)
            # Skip if already upgraded with same length IDs.
            if maybe_ids is not None and len(maybe_ids) == len(polys):
                skipped += 1
                continue

            assigned = []
            distances = []
            for poly in polys:
                if len(poly) < 2:
                    assigned.append(-1)
                    distances.append(1e9)
                    continue
                global_poly = np.copy(poly)
                global_poly[:, 0] += x0  # x
                global_poly[:, 1] += y0  # y
                gid, d = best_match_global_id(global_poly, source_instances, (x0, y0, w, h))
                assigned.append(int(gid) if gid is not None else -1)
                distances.append(float(d))

            payload = {
                "polylines": [p.tolist() for p in polys],
                "instance_ids": assigned,
                "source_file": src_name,
                "tile_offset_xy": [int(x0), int(y0)],
                "template_score": score,
                "match_chamfer_mean": float(np.mean(distances) if len(distances) > 0 else 0.0),
                "format_version": 2,
            }
            np.save(label_path, payload, allow_pickle=True)
            updated += 1

    print(
        f"[{split}] updated={updated}, skipped={skipped}, "
        f"tile_files_considered={len(tile_imgs)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="/home/work/caps_drone/yolino/TTPLA_YOLinO_Dataset",
        help="Tiled dataset root with images/<split> and labels/<split>",
    )
    parser.add_argument(
        "--source-json-root",
        default="/home/work/caps_drone/yolino/data_original_size_v1/data_original_size",
        help="Original TTPLA JSON directory",
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--max-files", type=int, default=-1, help="Optional cap per split")
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for s in splits:
        process_split(args.dataset_root, args.source_json_root, s, max_files=args.max_files)


if __name__ == "__main__":
    main()
