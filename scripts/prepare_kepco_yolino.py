#!/usr/bin/env python3
"""
mini_data_001 + mini_data_002_part1 → Yolino_KEPCO (Sample → Sample_YOLinO 와 동일 파이프라인).

각 소스: merged_annotations.xml + 이미지 폴더
  - mini_data_001/images
  - mini_data_002_part1/sample

전체 basename 80:20 train/val → build_ttpla_style_dataset cvat-xml (짧은변=1024, L/R·T/B dual crop).
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import numpy as np
from tqdm import tqdm

from build_downsampled_yolino_dataset import find_image_file
from build_ttpla_style_dataset import (
    convert_cvat_dual_square_crops,
    parse_cvat_xml,
)

# (cvat_xml relative to repo, image_dir relative to repo)
DEFAULT_SOURCES: list[tuple[str, str]] = [
    ("mini_data_001/merged_annotations.xml", "mini_data_001/images"),
    ("mini_data_002_part1/merged_annotations.xml", "mini_data_002_part1/sample"),
]


def collect_entries(repo_root: str, sources: list[tuple[str, str]]) -> dict[str, tuple[str, str, list]]:
    """basename -> (abs_xml, abs_image_dir, polys_meta)."""
    entries: dict[str, tuple[str, str, list]] = {}
    for xml_rel, img_rel in sources:
        xml_path = os.path.join(repo_root, xml_rel)
        img_dir = os.path.join(repo_root, img_rel)
        if not os.path.isfile(xml_path):
            raise SystemExit(f"Missing XML: {xml_path}")
        if not os.path.isdir(img_dir):
            raise SystemExit(f"Missing image dir: {img_dir}")
        by_base = parse_cvat_xml(xml_path)
        for b in sorted(by_base.keys()):
            if find_image_file(img_dir, b) is None:
                continue
            if b in entries:
                raise SystemExit(f"Duplicate basename across sources: {b}")
            entries[b] = (xml_path, img_dir, by_base[b])
    return entries


def build_dataset(
    entries: dict[str, tuple[str, str, list]],
    guide_dir: str,
    out_root: str,
    target_width: int,
    min_points: int,
) -> None:
    os.makedirs(out_root, exist_ok=True)
    for sub in ("images", "labels"):
        p = os.path.join(out_root, sub)
        if os.path.isdir(p):
            shutil.rmtree(p)

    for split in ["train", "val", "test"]:
        guide_path = os.path.join(guide_dir, f"{split}.txt")
        if not os.path.isfile(guide_path):
            basenames: list[str] = []
        else:
            with open(guide_path, encoding="utf-8") as f:
                basenames = [ln.strip() for ln in f if ln.strip()]

        out_img_dir = os.path.join(out_root, "images", split)
        out_lbl_dir = os.path.join(out_root, "labels", split)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        missing = missing_img = saved = skipped_empty = 0
        for b in tqdm(basenames, desc=f"kepco {split}"):
            if b not in entries:
                missing += 1
                continue
            xml_path, img_dir, polys_meta = entries[b]
            img_path = find_image_file(img_dir, b)
            if img_path is None:
                missing_img += 1
                continue

            crops = convert_cvat_dual_square_crops(
                polys_meta, img_path, target=target_width, min_points=min_points
            )
            skipped_empty += max(0, 2 - len(crops))
            for tag, resized, polylines, instance_ids, tw, th in crops:
                stub = f"{b}_{tag}"
                out_img = os.path.join(out_img_dir, f"{stub}.png")
                out_npy = os.path.join(out_lbl_dir, f"{stub}.npy")
                resized.save(out_img, format="PNG")
                payload = {
                    "polylines": polylines,
                    "instance_ids": instance_ids,
                    "source_file": stub,
                    "source_image_basename": b,
                    "crop_tag": tag,
                    "square_dual_crops": True,
                    "image_size_wh": [tw, th],
                    "format_version": 3,
                    "source_cvat_xml": xml_path,
                }
                np.save(out_npy, payload, allow_pickle=True)
                saved += 1
        print(f"[{split}] saved={saved} skipped_empty_crops={skipped_empty} not_in_entries={missing} missing_img={missing_img}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Yolino_KEPCO from mini_data CVAT exports.")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--repo-root", default=here)
    ap.add_argument("--output-root", default="Yolino_KEPCO")
    ap.add_argument("--guide-dir", default="Yolino_KEPCO/kepco_yolino_guide")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-width", type=int, default=1024)
    ap.add_argument("--min-points", type=int, default=2)
    args = ap.parse_args()

    root = os.path.abspath(args.repo_root)
    os.chdir(root)

    out_root = args.output_root if os.path.isabs(args.output_root) else os.path.join(root, args.output_root)
    guide_dir = args.guide_dir if os.path.isabs(args.guide_dir) else os.path.join(root, args.guide_dir)

    entries = collect_entries(root, DEFAULT_SOURCES)
    basenames = sorted(entries.keys())
    if not basenames:
        raise SystemExit("No images with CVAT polyline + file on disk.")

    random.seed(args.seed)
    random.shuffle(basenames)
    n = len(basenames)
    n_train = int(round(args.train_ratio * n))
    train, val = basenames[:n_train], basenames[n_train:]

    os.makedirs(guide_dir, exist_ok=True)
    with open(os.path.join(guide_dir, "train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train) + ("\n" if train else ""))
    with open(os.path.join(guide_dir, "val.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val) + ("\n" if val else ""))
    with open(os.path.join(guide_dir, "test.txt"), "w", encoding="utf-8"):
        pass

    print(f"sources: {len(DEFAULT_SOURCES)}  images={n}  train={len(train)}  val={len(val)}")
    print(f"guide: {guide_dir}")
    print(f"output: {out_root}")

    build_dataset(entries, guide_dir, out_root, args.target_width, args.min_points)
    print("\nDone. Set: export DATASET_TTPLA=%s" % out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
