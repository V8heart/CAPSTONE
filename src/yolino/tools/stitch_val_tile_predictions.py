"""Stitch tiled val predictions onto full-frame source images.

Use ``--output-mode paired`` for one wide PNG per frame: left = predictions (cyan), right = GT (blue),
same layout idea as ``val_pred_gt_overlay`` (``{base}_paired.png``).
"""
import argparse
import os
import re
from collections import defaultdict
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from yolino.dataset.dataset_factory import DatasetFactory
from yolino.grid.grid_factory import GridFactory
from yolino.runner.forward_runner import ForwardRunner
from yolino.utils.enums import CoordinateSystem
from yolino.utils.general_setup import general_setup


TILE_RE = re.compile(r"^(?P<base>.+)_tile_x(?P<x>\d+)_y(?P<y>\d+)$")


def pick_latest_checkpoint(ckpt_dir: str) -> str:
    files = []
    for name in os.listdir(ckpt_dir):
        if not name.startswith("ep") or not name.endswith("_model.pth"):
            continue
        # ep0001_model.pth
        epoch_str = name[2:].split("_", 1)[0]
        if epoch_str.isdigit():
            files.append((int(epoch_str), os.path.join(ckpt_dir, name)))
    if len(files) == 0:
        raise FileNotFoundError(f"No epoch checkpoints found in {ckpt_dir}")
    files.sort(key=lambda x: x[0])
    return files[-1][1]


def parse_tile_name(stem: str):
    m = TILE_RE.match(stem)
    if m is None:
        return None
    return m.group("base"), int(m.group("x")), int(m.group("y"))


def find_source_image(source_dir: str, base_name: str) -> str | None:
    exts = ("png", "jpg", "jpeg", "webp", "bmp")
    for ext in exts:
        p = os.path.join(source_dir, f"{base_name}.{ext}")
        if os.path.isfile(p):
            return p
    matches = []
    for ext in exts:
        matches.extend(glob(os.path.join(source_dir, f"{base_name}*.{ext}")))
    return sorted(matches)[0] if matches else None


def load_tile_meta(dataset_root: str, tile_name: str):
    label_path = os.path.join(dataset_root, "labels", "val", f"{tile_name}.npy")
    if not os.path.isfile(label_path):
        return None
    payload = np.load(label_path, allow_pickle=True).item()
    base = payload.get("source_image_basename")
    box = payload.get("tile_box_ltrb", None)
    if base is None or box is None or len(box) != 4:
        return None
    l, t, r, b = [int(v) for v in box]
    return {"base": str(base), "left": l, "top": t, "right": r, "bottom": b}


def concat_paired_bgr(left: np.ndarray, right: np.ndarray, gap_px: int = 8) -> np.ndarray:
    """BGR images, same height; horizontal concat with gray gutter."""
    if left.shape[0] != right.shape[0]:
        scale = left.shape[0] / float(right.shape[0])
        nw = int(round(right.shape[1] * scale))
        right = cv2.resize(right, (nw, left.shape[0]), interpolation=cv2.INTER_AREA)
    g = max(4, int(gap_px))
    gutter = np.full((left.shape[0], g, 3), 32, dtype=np.uint8)
    return np.concatenate([left, gutter, right], axis=1)


def draw_lines(canvas: np.ndarray, lines: np.ndarray, x_off: int, y_off: int, color=(0, 255, 255), thickness=2):
    for p in lines:
        # y1, x1, y2, x2, conf, ...
        if np.any(np.isnan(p[:4])):
            continue
        y1, x1, y2, x2 = p[:4]
        pt1 = (int(round(x1 + x_off)), int(round(y1 + y_off)))
        pt2 = (int(round(x2 + x_off)), int(round(y2 + y_off)))
        cv2.arrowedLine(canvas, pt1, pt2, color=color, thickness=thickness, tipLength=0.12)


def main():
    parser = argparse.ArgumentParser(description="Infer on val tiles and stitch predictions per original frame.")
    parser.add_argument("--config", required=True, help="Experiment yaml path")
    parser.add_argument("--dataset-root", required=True, help="Tiled dataset root")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing epXXXX_model.pth")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional explicit checkpoint file path. If set, this is used instead of latest in --checkpoint-dir.")
    parser.add_argument("--confidence", type=float, default=None, help="Override visualization confidence threshold")
    parser.add_argument("--out-dir", required=True, help="Output directory for stitched overlays")
    parser.add_argument("--source-image-dir", default="/home/work/caps_drone/yolino/Sample/Sample",
                        help="Directory of original (pre-tiled) images for projection.")
    parser.add_argument("--max-groups", type=int, default=-1, help="Optional debug limit for number of original frames")
    parser.add_argument("--loading-workers", type=int, default=0,
                        help="DataLoader workers for this script. Keep 0 to avoid shm bus errors.")
    parser.add_argument("--gpu", action="store_true",
                        help="Use CUDA for forward pass when available.")
    parser.add_argument(
        "--output-mode",
        choices=["pred_stitched", "paired", "both"],
        default="pred_stitched",
        help="pred_stitched: cyan preds on source (legacy). paired: pred|GT horizontal single PNG "
             "({base}_paired.png). both: write both.",
    )
    parser.add_argument("--pair-gap-pixels", type=int, default=8, help="Gutter width between pred and GT panes.")
    cli, _ = parser.parse_known_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    default_cfg = os.path.join(project_root, "ttpla_train_exp", "default_params.yaml")

    alt_args = ["-c", cli.config, "--root", project_root, "--dvc", project_root]
    if cli.gpu:
        alt_args += ["--gpu", "--gpu_id", "0"]

    args = general_setup(
        "Stitch val tile predictions",
        task_type=None,
        config_file=cli.config,
        ignore_cmd_args=True,
        alternative_args=alt_args,
        default_config=default_cfg,
    )
    args.dvc = cli.dataset_root
    if cli.checkpoint is not None and str(cli.checkpoint).strip() != "":
        args.explicit_model = os.path.abspath(os.path.expanduser(cli.checkpoint))
    else:
        args.explicit_model = pick_latest_checkpoint(cli.checkpoint_dir)
    # general_setup already computed args.paths from old/yaml explicit_model.
    # Keep runtime checkpoint path in sync so ForwardRunner loads the selected latest checkpoint.
    args.paths.pretrain_model = args.explicit_model
    args.loading_workers = int(cli.loading_workers)
    if cli.gpu:
        if torch.cuda.is_available():
            args.gpu = True
            args.gpu_id = 0 if getattr(args, "gpu_id", -1) < 0 else int(args.gpu_id)
            args.cuda = f"cuda:{args.gpu_id}"
            torch.cuda.set_device(args.gpu_id)
        else:
            print("[WARN] --gpu was requested but CUDA is not available. Falling back to CPU.")
    if cli.confidence is not None:
        args.confidence = float(cli.confidence)

    os.makedirs(cli.out_dir, exist_ok=True)

    dataset, _ = DatasetFactory.get(
        args.dataset, only_available=True, split="val", args=args, shuffle=False, augment=False, ignore_duplicates=False
    )
    # IMPORTANT: keep all validation tiles for stitching; do not drop the last partial batch.
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.loading_workers,
        pin_memory=args.gpu,
    )
    model = ForwardRunner(args=args, coords=dataset.coords, load_best=False)

    # filename -> filtered predicted uv lines; GT uv lines per tile for paired layout
    preds_per_tile = {}
    gt_per_tile = {}
    conf_thr = float(args.confidence)
    need_gt = cli.output_mode in ("paired", "both")

    for images, grid_tensor, fileinfo, *_ in loader:
        gt_cpu = grid_tensor.detach().cpu() if need_gt else None
        with torch.no_grad():
            geom_preds, _, _ = model(images, is_train=False, epoch=None)
        geom_preds_cpu = geom_preds.detach().cpu()

        for i, fname in enumerate(fileinfo):
            ih = images[i].shape[1]
            pred_grid, _ = GridFactory.get(
                geom_preds_cpu[[i]],
                [],
                CoordinateSystem.CELL_SPLIT,
                args=args,
                input_coords=dataset.coords,
                only_train_vars=True,
                anchors=dataset.anchors,
            )
            uv = np.asarray(
                pred_grid.get_image_lines(coords=dataset.coords, image_height=ih, is_training_data=True)
            )[0]
            if uv.ndim == 1:
                uv = np.expand_dims(uv, axis=0)
            keep = uv[:, 4] >= conf_thr if uv.shape[1] > 4 else np.ones((len(uv),), dtype=bool)
            preds_per_tile[fname] = uv[keep]

            if need_gt:
                gt_grid, _ = GridFactory.get(
                    gt_cpu[[i]],
                    [],
                    CoordinateSystem.CELL_SPLIT,
                    args=args,
                    input_coords=dataset.coords,
                    only_train_vars=False,
                    anchors=dataset.anchors,
                )
                gt_uv = np.asarray(
                    gt_grid.get_image_lines(coords=dataset.coords, image_height=ih, is_training_data=True)
                )[0]
                if gt_uv.ndim == 1:
                    gt_uv = np.expand_dims(gt_uv, axis=0)
                gt_per_tile[fname] = gt_uv

    groups = defaultdict(list)
    for tile_name in preds_per_tile.keys():
        meta = load_tile_meta(cli.dataset_root, tile_name)
        if meta is None:
            continue
        groups[meta["base"]].append((tile_name, meta))

    processed = 0
    for base_name, items in sorted(groups.items()):
        if cli.max_groups > 0 and processed >= cli.max_groups:
            break

        src_path = find_source_image(cli.source_image_dir, base_name)
        if src_path is None:
            continue
        canvas = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if canvas is None:
            continue

        gt_thickness = max(1, 2 - 1)

        if cli.output_mode in ("pred_stitched", "both"):
            pred_only = canvas.copy()
            for tile_name, meta in items:
                lines = preds_per_tile.get(tile_name, np.zeros((0, 5), dtype=np.float32))
                draw_lines(pred_only, lines, x_off=meta["left"], y_off=meta["top"], color=(0, 255, 255), thickness=2)
            out_path = os.path.join(cli.out_dir, f"{base_name}_pred_stitched.png")
            cv2.imwrite(out_path, pred_only)

        if cli.output_mode in ("paired", "both"):
            pred_pane = canvas.copy()
            gt_pane = canvas.copy()
            for tile_name, meta in items:
                plines = preds_per_tile.get(tile_name, np.zeros((0, 5), dtype=np.float32))
                glines = gt_per_tile.get(tile_name, np.zeros((0, 5), dtype=np.float32))
                draw_lines(pred_pane, plines, x_off=meta["left"], y_off=meta["top"], color=(0, 255, 255), thickness=2)
                draw_lines(gt_pane, glines, x_off=meta["left"], y_off=meta["top"], color=(255, 0, 0),
                           thickness=gt_thickness)
            combo = concat_paired_bgr(pred_pane, gt_pane, gap_px=cli.pair_gap_pixels)
            out_paired = os.path.join(cli.out_dir, f"{base_name}_paired.png")
            cv2.imwrite(out_paired, combo)

        processed += 1

    print(f"[DONE] processed_groups={processed} out_dir={cli.out_dir} output_mode={cli.output_mode}")
    print(f"[INFO] checkpoint={args.explicit_model}")
    print(f"[INFO] confidence={conf_thr}")
    print(f"[INFO] device={args.cuda}")


if __name__ == "__main__":
    main()
