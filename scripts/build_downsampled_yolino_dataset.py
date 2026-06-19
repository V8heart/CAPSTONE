import argparse
import json
import os
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

from preprocess_ttpla import extract_continuous_line_from_polygon


def is_cable_label(label):
    lowered = (label or "").lower()
    return (
        ("cable" in lowered)
        or ("power line" in lowered)
        or ("power_line" in lowered)
        or ("powerline" in lowered)
    )


def load_split_basenames(guide_dir, split):
    txt_path = os.path.join(guide_dir, f"{split}.txt")
    if not os.path.exists(txt_path):
        return []
    with open(txt_path, "r", encoding="utf-8") as f:
        return [os.path.splitext(line.strip())[0] for line in f if line.strip()]


def find_image_file(base_dir, basename):
    for ext in [".jpg", ".png", ".JPG", ".PNG"]:
        p = os.path.join(base_dir, f"{basename}{ext}")
        if os.path.exists(p):
            return p
    return None


def scale_polyline(polyline_xy, sx, sy):
    arr = np.asarray(polyline_xy, dtype=np.float32).copy()
    arr[:, 0] *= sx  # x
    arr[:, 1] *= sy  # y
    return arr


def convert_one(source_json, source_img, target_width, min_points):
    with open(source_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    with Image.open(source_img) as im:
        src_w, src_h = im.size
        target_height = int(round(src_h * (target_width / float(src_w))))
        resized = im.resize((target_width, target_height), Image.Resampling.LANCZOS).convert("RGB")

    sx = target_width / float(src_w)
    sy = target_height / float(src_h)

    polylines = []
    instance_ids = []
    next_id = 1

    for shape in data.get("shapes", []):
        if not is_cable_label(shape.get("label", "")):
            continue
        pts = shape.get("points", [])
        if len(pts) < 2:
            continue
        line = extract_continuous_line_from_polygon((src_h, src_w), pts)
        if len(line) < min_points:
            continue
        scaled = scale_polyline(line, sx=sx, sy=sy)
        polylines.append(scaled.tolist())
        instance_ids.append(next_id)
        next_id += 1

    return resized, polylines, instance_ids, target_width, target_height


def main():
    parser = argparse.ArgumentParser(
        description="Build YOLinO dataset from raw TTPLA by aspect-ratio-preserving downsampling (no tiling)."
    )
    parser.add_argument("--input-dir", required=True, help="Raw directory containing *.jpg and *.json pairs.")
    parser.add_argument("--guide-dir", required=True, help="Directory containing train/val/test txt files.")
    parser.add_argument("--output-root", required=True, help="Output YOLinO dataset root.")
    parser.add_argument("--target-width", type=int, default=1024, help="Downsample target width (keep aspect ratio).")
    parser.add_argument("--min-points", type=int, default=2, help="Minimum points to keep a line.")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    for split in ["train", "val", "test"]:
        basenames = load_split_basenames(args.guide_dir, split)
        out_img_dir = os.path.join(args.output_root, "images", split)
        out_lbl_dir = os.path.join(args.output_root, "labels", split)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        missing_json = 0
        missing_img = 0
        saved = 0

        for b in tqdm(basenames, desc=f"build {split}"):
            json_path = os.path.join(args.input_dir, f"{b}.json")
            if not os.path.exists(json_path):
                missing_json += 1
                continue

            img_path = find_image_file(args.input_dir, b)
            if img_path is None:
                missing_img += 1
                continue

            resized, polylines, instance_ids, tw, th = convert_one(
                source_json=json_path,
                source_img=img_path,
                target_width=args.target_width,
                min_points=args.min_points,
            )

            # Keep all guide images, even with empty polylines, to preserve split determinism.
            out_img = os.path.join(out_img_dir, f"{b}.png")
            out_npy = os.path.join(out_lbl_dir, f"{b}.npy")
            resized.save(out_img, format="PNG")

            payload = {
                "polylines": polylines,
                "instance_ids": instance_ids,
                "source_file": b,
                "image_size_wh": [tw, th],
                "format_version": 3,
            }
            np.save(out_npy, payload, allow_pickle=True)
            saved += 1

        print(f"[{split}] saved={saved} missing_json={missing_json} missing_img={missing_img}")


if __name__ == "__main__":
    main()
