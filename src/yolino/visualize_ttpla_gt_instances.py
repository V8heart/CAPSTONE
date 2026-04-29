import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize TTPLA GT polyline instances and connected components."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Path to TTPLA dataset root. If omitted, DATASET_TTPLA env var is used.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to inspect.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="Maximum number of images to visualize.",
    )
    parser.add_argument(
        "--id_aware_only",
        action="store_true",
        help="If set, visualize only labels that already contain explicit instance_ids payload.",
    )
    parser.add_argument(
        "--line_thickness",
        type=int,
        default=1,
        help="Line thickness when rasterizing GT into a binary mask.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: <dataset_root>/debug/ttpla_gt_instance_check/<split>",
    )
    return parser.parse_args()


def resolve_paths(args):
    dataset_root = args.dataset_root or os.getenv("DATASET_TTPLA")
    if not dataset_root:
        raise ValueError("Please set --dataset_root or DATASET_TTPLA environment variable.")

    dataset_root = os.path.abspath(dataset_root)
    img_dir = os.path.join(dataset_root, "images", args.split)
    label_dir = os.path.join(dataset_root, "labels", args.split)

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image dir not found: {img_dir}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Label dir not found: {label_dir}")

    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
    else:
        out_dir = os.path.join(dataset_root, "debug", "ttpla_gt_instance_check", args.split)

    os.makedirs(out_dir, exist_ok=True)
    return dataset_root, img_dir, label_dir, out_dir


def color_for_id(idx):
    cmap = plt.get_cmap("tab20")
    c = cmap(idx % 20)
    return (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))


def draw_instance_overlay(image_bgr, polylines, instance_ids):
    overlay = image_bgr.copy()
    for inst_idx, poly in zip(instance_ids, polylines):
        pts = np.asarray(poly, dtype=np.float32)
        if len(pts) < 2:
            continue
        pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts_i], False, color_for_id(inst_idx), thickness=2, lineType=cv2.LINE_AA)
        x0, y0 = int(pts_i[0, 0, 0]), int(pts_i[0, 0, 1])
        cv2.putText(
            overlay,
            str(inst_idx),
            (x0, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color_for_id(inst_idx),
            1,
            cv2.LINE_AA,
        )
    return overlay


def build_binary_mask(height, width, polylines, thickness):
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polylines:
        pts = np.asarray(poly, dtype=np.float32)
        if len(pts) < 2:
            continue
        pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [pts_i], False, 255, thickness=max(1, thickness), lineType=cv2.LINE_8)
    return mask


def connected_component_color(labels):
    h, w = labels.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    unique_ids = np.unique(labels)
    for comp_id in unique_ids:
        if comp_id == 0:
            continue
        color[labels == comp_id] = color_for_id(int(comp_id))
    return color


def load_ttpla_label_with_ids(label_path):
    raw = np.load(label_path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        payload = raw.item()
        if isinstance(payload, dict):
            polylines = payload.get("polylines", [])
            instance_ids = payload.get("instance_ids", None)
            if instance_ids is None or len(instance_ids) != len(polylines):
                instance_ids = list(range(1, len(polylines) + 1))
            return polylines, instance_ids, payload
    # legacy format
    polylines = raw.tolist() if isinstance(raw, np.ndarray) else list(raw)
    instance_ids = list(range(1, len(polylines) + 1))
    return polylines, instance_ids, None


def main():
    args = parse_args()
    _, img_dir, label_dir, out_dir = resolve_paths(args)

    img_files = sorted(Path(img_dir).glob("*.png"))
    if len(img_files) == 0:
        raise FileNotFoundError(f"No images found in {img_dir}")

    limit = len(img_files) if args.max_samples < 0 else min(args.max_samples, len(img_files))
    summary = []
    checked = 0

    for i in range(len(img_files)):
        if len(summary) >= limit:
            break
        img_path = str(img_files[i])
        stem = Path(img_path).stem
        label_path = os.path.join(label_dir, stem + ".npy")
        if not os.path.isfile(label_path):
            continue
        checked += 1

        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue

        polylines, instance_ids, payload = load_ttpla_label_with_ids(label_path)
        if args.id_aware_only and payload is None:
            continue
        num_instances = len(polylines)
        unique_global_ids = len(set(instance_ids))

        overlay = draw_instance_overlay(image_bgr, polylines, instance_ids)
        mask = build_binary_mask(image_bgr.shape[0], image_bgr.shape[1], polylines, args.line_thickness)
        cc_count, cc_labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
        cc_fg = int(max(0, cc_count - 1))
        cc_vis = connected_component_color(cc_labels)

        fig = plt.figure(figsize=(16, 5))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        ax1.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original image")
        ax1.axis("off")

        ax2.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax2.set_title(f"GT instances (lines={num_instances}, global_ids={unique_global_ids})")
        ax2.axis("off")

        ax3.imshow(cv2.cvtColor(cc_vis, cv2.COLOR_BGR2RGB))
        ax3.set_title(f"Connected components on rasterized GT (count={cc_fg})")
        ax3.axis("off")

        fig.suptitle(stem)
        fig.tight_layout()
        out_img = os.path.join(out_dir, f"{stem}_gt_instance_check.png")
        fig.savefig(out_img, dpi=170)
        plt.close(fig)

        summary.append(
            {
                "file": stem,
                "label_instances": int(num_instances),
                "unique_global_instance_ids": int(unique_global_ids),
                "connected_components": int(cc_fg),
                "image_path": img_path,
                "label_path": label_path,
                "viz_path": out_img,
                "source_file": None if payload is None else payload.get("source_file"),
            }
        )

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": args.split,
                "line_thickness": args.line_thickness,
                "num_visualized": len(summary),
                "items": summary,
            },
            f,
            indent=2,
        )

    print(f"Saved {len(summary)} visualizations to: {out_dir}")
    print(f"Summary JSON: {summary_path}")
    print(f"Scanned files: {checked}")


if __name__ == "__main__":
    main()
