#!/usr/bin/env python3
"""
Build TTPLA-style YOLinO folders (same layout as ttpla_yolino_dataset_1024_downsample):

  output_root/images/{train,val,test}/<id>.png
  output_root/labels/{train,val,test}/<id>.npy

Each .npy uses format_version 3 with parallel lists:
  polylines[i]   — list of [x,y] in pixel coords AFTER resize (same as training image)
  instance_ids[i]— GLOBAL instance id for that polyline on the full image

YOLinO splits lines across grid cells at train time; GridFactory passes the same
var_payload (including INSTANCE id) into every cell slice, so the same physical
wire keeps one id across cells — provided instance_ids[] here are correct per wire.

Modes:
  labelme_json — LabelMe polygon (cable) → skeleton(centerline). 폴리곤 전용.
  cvat_xml     — CVAT polyline 그대로; 기본은 make_square_crops.py 와 동일한
                 짧은변=target 후 L/R 또는 T/B 의 1024² 두 장 (--no-square-dual-crops 로 단일 출력).

Optional CVAT polyline attribute group_id: shapes sharing group_id share one instance id.
Otherwise each polyline gets a sequential id (stable XML order).

Example:
  python build_ttpla_style_dataset.py cvat-xml \\
    --cvat-xml Sample/merged_annotations.xml \\
    --image-dir /path/to/images \\
    --guide-dir data_guide \\
    --output-root ttpla_new_1024 \\
    --target-width 1024

  python build_ttpla_style_dataset.py labelme \\
    --input-dir raw_json_jpg \\
    --guide-dir data_guide \\
    --output-root ttpla_new_1024 \\
    --target-width 1024
"""
from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from tqdm import tqdm

from build_downsampled_yolino_dataset import (
    convert_one,
    find_image_file,
    is_cable_label,
    load_split_basenames,
    scale_polyline,
)
from make_square_crops import crop_boxes, resize_short_to, transform_label


def parse_cvat_polyline_points(points_str: str) -> list[list[float]]:
    out: list[list[float]] = []
    if not points_str or not points_str.strip():
        return out
    for part in points_str.split(";"):
        part = part.strip()
        if not part:
            continue
        xy = part.split(",")
        if len(xy) != 2:
            continue
        out.append([float(xy[0]), float(xy[1])])
    return out


def assign_ids_from_group_or_sequential(rows: list[dict]) -> list[int]:
    """rows: each dict may have 'group_id' str|None."""
    gid_to_iid: dict[str, int] = {}
    next_iid = 1
    ids: list[int] = []
    for row in rows:
        gid = row.get("group_id")
        if gid is not None and str(gid).strip() != "":
            key = str(gid).strip()
            if key not in gid_to_iid:
                gid_to_iid[key] = next_iid
                next_iid += 1
            ids.append(gid_to_iid[key])
        else:
            ids.append(next_iid)
            next_iid += 1
    return ids


def parse_cvat_xml(xml_path: str) -> dict[str, dict]:
    """Map basename (no ext) -> { polylines_xy, meta for id assignment }."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    by_base: dict[str, list[dict]] = {}
    for img_el in root.findall(".//image"):
        name = img_el.get("name")
        if not name:
            continue
        basename = os.path.splitext(os.path.basename(name))[0]
        polys: list[dict] = []
        for tag in ("polyline", "points"):
            for pl in img_el.findall(tag):
                label = pl.get("label", "")
                if not is_cable_label(label):
                    continue
                pts = parse_cvat_polyline_points(pl.get("points", ""))
                if len(pts) < 2:
                    continue
                polys.append(
                    {
                        "points": pts,
                        "group_id": pl.get("group_id"),
                    }
                )
        if polys:
            by_base[basename] = polys
    return by_base


def crop_polylines_meta(polys_meta: list[dict], box_ltrb: tuple[float, float, float, float]) -> list[dict]:
    """Keep polyline vertices inside [l,r) × [t,b); shift to crop coordinates."""
    l, t, r, b = box_ltrb
    out: list[dict] = []
    for row in polys_meta:
        pts_in: list[list[float]] = []
        for x, y in row["points"]:
            if l <= x < r and t <= y < b:
                pts_in.append([x - l, y - t])
        if len(pts_in) >= 2:
            out.append({"points": pts_in, "group_id": row.get("group_id")})
    return out


def resize_or_letterbox(im: Image.Image, target_width: int, target_height: int | None):
    """Return RGB image and mapper polyline_xy -> list[list[float]] in output pixels."""
    src_w, src_h = im.size
    im = im.convert("RGB")

    if target_height is None:
        th = int(round(src_h * (target_width / float(src_w))))
        out = im.resize((target_width, th), Image.Resampling.LANCZOS)
        sx = target_width / float(src_w)
        sy = th / float(src_h)

        def map_poly(pts):
            return scale_polyline(pts, sx=sx, sy=sy).tolist()

        return out, map_poly, target_width, th

    tw, th = target_width, target_height
    rscale = min(tw / src_w, th / src_h)
    nw, nh = int(round(src_w * rscale)), int(round(src_h * rscale))
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tw, th), (0, 0, 0))
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    canvas.paste(resized, (ox, oy))

    def map_poly(pts):
        arr = scale_polyline(pts, sx=rscale, sy=rscale)
        arr[:, 0] += ox
        arr[:, 1] += oy
        return arr.tolist()

    return canvas, map_poly, tw, th


def convert_cvat_entry(
    polys_meta: list[dict],
    source_img_path: str,
    target_width: int,
    min_points: int,
    target_height: int | None = None,
    crop_box_ltrb: tuple[int, int, int, int] | None = None,
):
    """Load image, optional crop, resize or letterbox; map polylines; assign instance ids.

    Returns:
        (image, polylines, instance_ids, tw, th) or None if nothing to save.
    """
    with Image.open(source_img_path) as im:
        if crop_box_ltrb is not None:
            l, t, r, b = crop_box_ltrb
            l = max(0, int(l))
            t = max(0, int(t))
            r = min(im.size[0], int(r))
            b = min(im.size[1], int(b))
            if r <= l or b <= t:
                return None
            polys_meta = crop_polylines_meta(polys_meta, (float(l), float(t), float(r), float(b)))
            if not polys_meta:
                return None
            im = im.crop((l, t, r, b))

        canvas, map_poly, tw, th = resize_or_letterbox(im, target_width, target_height)

    instance_ids = assign_ids_from_group_or_sequential(polys_meta)
    polylines_out: list[list[list[float]]] = []
    ids_out: list[int] = []

    for row, iid in zip(polys_meta, instance_ids):
        mapped = map_poly(row["points"])
        if len(mapped) < min_points:
            continue
        polylines_out.append(mapped)
        ids_out.append(int(iid))

    if not polylines_out:
        return None

    return canvas, polylines_out, ids_out, tw, th


def convert_cvat_dual_square_crops(
    polys_meta: list[dict],
    source_img_path: str,
    target: int,
    min_points: int,
):
    """Same two-crop logic as make_square_crops.py (ttpla_yolino_dataset_1024_downsample 스타일).

    1) 짧은 변이 ``target`` 이 되도록 비율 유지 리사이즈
    2) 긴 변 방향으로 두 개의 ``target×target`` 정사각 크롭 (가로형: L/R, 세로형: T/B)
    3) 폴리라인은 스케일 후 사각형에 Liang–Barsky로 클립, 크롭 원점 기준 좌표

    Returns:
        list of (tag, PIL.Image, polylines, instance_ids, tw, th); tw == th == target
    """
    im = Image.open(source_img_path).convert("RGB")
    w, h = im.size
    nw, nh, scale = resize_short_to(w, h, target=target)
    resized = im.resize((nw, nh), Image.Resampling.BILINEAR)

    instance_ids = assign_ids_from_group_or_sequential(polys_meta)
    lbl = {
        "polylines": [row["points"] for row in polys_meta],
        "instance_ids": instance_ids,
    }

    results: list[tuple] = []
    for tag, box in crop_boxes(nw, nh, target=target):
        cropped = resized.crop(tuple(map(int, box)))
        polys_np, ids = transform_label(lbl, scale=scale, box=box)
        polylines_out: list[list[list[float]]] = []
        ids_out: list[int] = []
        for p_arr, iid in zip(polys_np, ids):
            if len(p_arr) >= min_points:
                polylines_out.append(p_arr.astype(np.float32).tolist())
                ids_out.append(int(iid))
        if not polylines_out:
            continue
        results.append((tag, cropped, polylines_out, ids_out, target, target))
    return results


def convert_labelme_entry(
    json_path: str,
    img_path: str,
    target_width: int,
    min_points: int,
):
    """Reuse polygon→skeleton pipeline; instance id = stable order among cable shapes."""
    return convert_one(
        source_json=json_path,
        source_img=img_path,
        target_width=target_width,
        min_points=min_points,
    )


def cmd_labelme(args):
    os.makedirs(args.output_root, exist_ok=True)
    for split in ["train", "val", "test"]:
        basenames = load_split_basenames(args.guide_dir, split)
        out_img_dir = os.path.join(args.output_root, "images", split)
        out_lbl_dir = os.path.join(args.output_root, "labels", split)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        missing_json = missing_img = saved = 0
        for b in tqdm(basenames, desc=f"labelme {split}"):
            json_path = os.path.join(args.input_dir, f"{b}.json")
            if not os.path.exists(json_path):
                missing_json += 1
                continue
            img_path = find_image_file(args.input_dir, b)
            if img_path is None:
                missing_img += 1
                continue

            resized, polylines, instance_ids, tw, th = convert_labelme_entry(
                json_path, img_path, args.target_width, args.min_points
            )
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


def cmd_cvat_xml(args):
    os.makedirs(args.output_root, exist_ok=True)
    by_base = parse_cvat_xml(args.cvat_xml)
    square_dual = bool(getattr(args, "square_dual_crops", True))
    target_height = getattr(args, "target_height", None)
    if target_height is not None:
        target_height = int(target_height)

    for split in ["train", "val", "test"]:
        basenames = load_split_basenames(args.guide_dir, split)
        out_img_dir = os.path.join(args.output_root, "images", split)
        out_lbl_dir = os.path.join(args.output_root, "labels", split)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        missing_xml_entry = missing_img = saved = 0
        for b in tqdm(basenames, desc=f"cvat {split}"):
            if b not in by_base:
                missing_xml_entry += 1
                continue
            img_path = find_image_file(args.image_dir, b)
            if img_path is None:
                missing_img += 1
                continue
            polys_meta = by_base[b]

            if square_dual:
                for tag, resized, polylines, instance_ids, tw, th in convert_cvat_dual_square_crops(
                    polys_meta, img_path, target=int(args.target_width), min_points=args.min_points
                ):
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
                        "source_cvat_xml": os.path.abspath(args.cvat_xml),
                    }
                    np.save(out_npy, payload, allow_pickle=True)
                    saved += 1
                continue

            entry = convert_cvat_entry(
                polys_meta,
                img_path,
                args.target_width,
                args.min_points,
                target_height=target_height,
                crop_box_ltrb=None,
            )
            if entry is None:
                continue
            resized, polylines, instance_ids, tw, th = entry
            out_img = os.path.join(out_img_dir, f"{b}.png")
            out_npy = os.path.join(out_lbl_dir, f"{b}.npy")
            resized.save(out_img, format="PNG")
            payload = {
                "polylines": polylines,
                "instance_ids": instance_ids,
                "source_file": b,
                "image_size_wh": [tw, th],
                "format_version": 3,
                "source_cvat_xml": os.path.abspath(args.cvat_xml),
            }
            np.save(out_npy, payload, allow_pickle=True)
            saved += 1
        print(f"[{split}] saved={saved} not_in_xml={missing_xml_entry} missing_img={missing_img}")


def main():
    parser = argparse.ArgumentParser(description="Build TTPLA-style YOLinO dataset (global instance ids).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_lm = sub.add_parser("labelme", help="LabelMe JSON + images (polygon cable GT, skeletonize).")
    p_lm.add_argument("--input-dir", required=True)
    p_lm.add_argument("--guide-dir", required=True)
    p_lm.add_argument("--output-root", required=True)
    p_lm.add_argument("--target-width", type=int, default=1024)
    p_lm.add_argument("--min-points", type=int, default=2)
    p_lm.set_defaults(func=cmd_labelme)

    p_cv = sub.add_parser("cvat-xml", help="CVAT merged XML + separate image folder.")
    p_cv.add_argument("--cvat-xml", required=True)
    p_cv.add_argument("--image-dir", required=True, help="Folder containing images named as in XML.")
    p_cv.add_argument("--guide-dir", required=True)
    p_cv.add_argument("--output-root", required=True)
    p_cv.add_argument("--target-width", type=int, default=1024)
    p_cv.add_argument(
        "--target-height",
        type=int,
        default=None,
        help="Only when --no-square-dual-crops: letterbox to target_width×target_height; "
             "omit for width-only scale.",
    )
    p_cv.add_argument(
        "--no-square-dual-crops",
        dest="square_dual_crops",
        action="store_false",
        help="단일 출력 (--target-width 및 선택적 --target-height). 기본은 TTPLA 방식 이중 크롭.",
    )
    p_cv.add_argument("--min-points", type=int, default=2)
    p_cv.set_defaults(func=cmd_cvat_xml, square_dual_crops=True)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
