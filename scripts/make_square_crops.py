#!/usr/bin/env python3
"""
TTPLA downsampled dataset -> 1024x1024 square-crop dataset.

Input label format (current pipeline):
  .npy (dict) with keys:
    - polylines: list[list[[x, y], ...]]
    - instance_ids: list[int]
    - image_size_wh: [W, H]

Per image:
  1) Resize with aspect ratio preserved so short side becomes target (default: 1024)
  2) Make two square crops from long side (L/R or T/B)
  3) Scale + clip polylines to crop box and shift crop origin to (0,0)
  4) Save image (.png) and label (.npy dict)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


Point = np.ndarray  # shape (2,) -> [x, y]
Polyline = np.ndarray  # shape (N, 2)
Box = Tuple[float, float, float, float]  # x0, y0, x1, y1


def resize_short_to(img_w: int, img_h: int, target: int) -> Tuple[int, int, float]:
    if img_w <= img_h:
        scale = target / float(img_w)
    else:
        scale = target / float(img_h)
    return int(round(img_w * scale)), int(round(img_h * scale)), scale


def crop_boxes(new_w: int, new_h: int, target: int) -> List[Tuple[str, Box]]:
    if new_w >= new_h:
        return [
            ("L", (0, 0, target, target)),
            ("R", (new_w - target, 0, new_w, target)),
        ]
    return [
        ("T", (0, 0, target, target)),
        ("B", (0, new_h - target, target, new_h)),
    ]


def liang_barsky_segment(p: Point, q: Point, box: Box):
    x0, y0, x1, y1 = box
    px, py = float(p[0]), float(p[1])
    qx, qy = float(q[0]), float(q[1])
    dx, dy = qx - px, qy - py

    t0, t1 = 0.0, 1.0
    tests = [
        (-dx, px - x0),
        (dx, x1 - px),
        (-dy, py - y0),
        (dy, y1 - py),
    ]
    for p_test, q_test in tests:
        if abs(p_test) < 1e-12:
            if q_test < 0:
                return None
            continue
        t = q_test / p_test
        if p_test < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None

    a = np.array([px + t0 * dx, py + t0 * dy], dtype=np.float32)
    b = np.array([px + t1 * dx, py + t1 * dy], dtype=np.float32)
    return a, b, t0, t1


def inside_box(p: Point, box: Box) -> bool:
    x0, y0, x1, y1 = box
    return (x0 <= p[0] <= x1) and (y0 <= p[1] <= y1)


def clip_polyline_to_box(poly: Polyline, box: Box) -> List[Polyline]:
    if len(poly) < 2:
        return []

    out: List[Polyline] = []
    current: List[Point] = []

    for i in range(len(poly) - 1):
        p, q = poly[i], poly[i + 1]
        p_in, q_in = inside_box(p, box), inside_box(q, box)

        if p_in and q_in:
            if not current:
                current.append(p.astype(np.float32))
            current.append(q.astype(np.float32))
            continue

        clipped = liang_barsky_segment(p, q, box)
        if clipped is None:
            if len(current) >= 2:
                out.append(np.array(current, dtype=np.float32))
            current = []
            continue

        a, b, t_enter, t_exit = clipped
        if p_in and (not q_in):
            if not current:
                current.append(p.astype(np.float32))
            current.append(b)
            if len(current) >= 2:
                out.append(np.array(current, dtype=np.float32))
            current = []
        elif (not p_in) and q_in:
            current = [a, q.astype(np.float32)]
        else:
            # both out but segment crosses the box
            if t_exit - t_enter > 1e-6:
                out.append(np.array([a, b], dtype=np.float32))
            if len(current) >= 2:
                out.append(np.array(current, dtype=np.float32))
            current = []

    if len(current) >= 2:
        out.append(np.array(current, dtype=np.float32))

    cleaned: List[Polyline] = []
    for p in out:
        seg_len = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) >= 2 else 0.0
        if len(p) >= 2 and seg_len >= 1.0:
            cleaned.append(p)
    return cleaned


def load_label(label_path: Path):
    if not label_path.exists():
        return {"polylines": [], "instance_ids": [], "source_file": "", "image_size_wh": [0, 0], "format_version": 1}
    data = np.load(str(label_path), allow_pickle=True).item()
    # Normalize keys
    data.setdefault("polylines", [])
    data.setdefault("instance_ids", [])
    data.setdefault("source_file", str(label_path))
    data.setdefault("image_size_wh", [0, 0])
    data.setdefault("format_version", 1)
    return data


def save_label(path: Path, polylines: Sequence[Polyline], instance_ids: Sequence[int], source_file: str, target: int):
    payload = {
        "polylines": [p.astype(np.float32).tolist() for p in polylines],
        "instance_ids": list(instance_ids),
        "source_file": source_file,
        "image_size_wh": [int(target), int(target)],
        "format_version": 1,
    }
    np.save(str(path), payload, allow_pickle=True)


def transform_label(data, scale: float, box: Box):
    x0, y0, _, _ = box
    out_polylines: List[Polyline] = []
    out_ids: List[int] = []

    instance_ids = data.get("instance_ids", [])
    polylines = data.get("polylines", [])
    for idx, pts in enumerate(polylines):
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
            continue
        scaled = arr * float(scale)
        clipped_chunks = clip_polyline_to_box(scaled, box)
        for chunk in clipped_chunks:
            shifted = chunk.copy()
            shifted[:, 0] -= float(x0)
            shifted[:, 1] -= float(y0)
            out_polylines.append(shifted)
            out_ids.append(int(instance_ids[idx]) if idx < len(instance_ids) else idx)

    return out_polylines, out_ids


def iter_images(folder: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in exts and p.is_file():
            yield p


def process_split(src_root: Path, dst_root: Path, split: str, target: int):
    src_img = src_root / "images" / split
    src_lbl = src_root / "labels" / split
    dst_img = dst_root / "images" / split
    dst_lbl = dst_root / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
    for ip in iter_images(src_img):
        n_in += 1
        with Image.open(ip) as im:
            im = im.convert("RGB")
            w, h = im.size
            nw, nh, scale = resize_short_to(w, h, target=target)
            resized = im.resize((nw, nh), Image.BILINEAR)

        lbl_data = load_label(src_lbl / f"{ip.stem}.npy")
        for tag, box in crop_boxes(nw, nh, target=target):
            cropped = resized.crop(tuple(map(int, box)))
            polylines, ids = transform_label(lbl_data, scale=scale, box=box)
            stem = f"{ip.stem}_{tag}"
            cropped.save(dst_img / f"{stem}.png")
            save_label(
                dst_lbl / f"{stem}.npy",
                polylines=polylines,
                instance_ids=ids,
                source_file=str(ip),
                target=target,
            )
            n_out += 1
    print(f"[{split}] {n_in} -> {n_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source dataset root")
    ap.add_argument("--dst", required=True, help="destination dataset root")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--target", type=int, default=1024)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    for split in args.splits:
        process_split(src, dst, split=split, target=args.target)
    print("done")


if __name__ == "__main__":
    main()
