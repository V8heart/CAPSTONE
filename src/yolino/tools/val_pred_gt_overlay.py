#!/usr/bin/env python3
"""
Validation prediction + GT visualization (one or more PNGs).

Default ``--layout paired`` follows training ``plot_val_summary`` / ``plot_debug_images``
(default ``training_vars_only=False``): de-normalized RGB, cell grid (optional),
pred filtering via ``--confidence``, **two panes** — prediction | GT — in **one** PNG.
(TensorBoard used a multi-tile grid; this tool writes a single horizontal image.)

Checkpoint path is relative to ``--dvc``:

  export DATASET_TTPLA=/path/to/ttpla_yolino_dataset_1024x1024
  python src/yolino/tools/val_pred_gt_overlay.py \\
    --config configs/experiments/exp19_fpn_bottomup_p4_num_predictors4_1024.yaml \\
    --dvc "$(pwd)/ttpla_train_exp" \\
    --checkpoint log/checkpoints/.../ep0041_model.pth \\
    --out res/exp19_val.png --sample-index 0 --num-images 1 --gpu

  # Legacy: pred and GT drawn on the **same** image
  python ... --layout overlay --no-show-grid ...

  # Five frames: out_00.png … (see --out rules in --help)
  python ... --out res/batch.png --num-images 5
"""
import argparse
import os
import re
import sys

import numpy as np
import torch

try:
    import cv2
except ImportError:
    cv2 = None

from yolino.dataset.dataset_factory import DatasetFactory
from yolino.grid.grid_factory import GridFactory
from yolino.runner.forward_runner import ForwardRunner
from yolino.utils.enums import ColorStyle
from yolino.utils.enums import CoordinateSystem
from yolino.utils.enums import TaskType
from yolino.utils.general_setup import general_setup
from yolino.viz.plot import plot


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _sanitize_stem(name: str) -> str:
    base = os.path.basename(str(name))
    base = re.sub(r"[^0-9A-Za-z_.-]+", "_", base)
    return base[:120] if len(base) > 120 else base


def _denormalize_for_viz(image: torch.Tensor, dataset) -> torch.Tensor:
    """Same as TrainHandler._denormalize_for_viz — input must be CxHxW float."""
    if not isinstance(image, torch.Tensor):
        return image
    if image.ndim != 3 or image.shape[0] > 3:
        return image
    augmentor = getattr(dataset, "augmentor", None)
    if augmentor is None:
        return image
    mean = torch.tensor(augmentor.norm_mean, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor(augmentor.norm_std, dtype=image.dtype, device=image.device).view(3, 1, 1)
    return torch.clamp(image * std + mean, 0.0, 1.0)


def _out_path_for_batch(cli_out: str, batch_idx_in_run: int, num_images: int, file_stem_for_name: str) -> str:
    """num_images==1 -> cli_out; else stem_NN.png or dir/NN_stem.png."""
    expanded = os.path.abspath(os.path.expanduser(cli_out))
    if num_images == 1:
        return expanded

    trimmed = cli_out.strip()
    looks_like_dir = (
        trimmed.endswith(os.sep)
        or trimmed.endswith("/")
        or os.path.isdir(expanded.rstrip(os.sep))
    )

    if looks_like_dir:
        d = expanded.rstrip(os.sep)
        name = "%02d_%s.png" % (batch_idx_in_run, _sanitize_stem(file_stem_for_name))
        return os.path.join(d, name)

    stem, _ext = os.path.splitext(expanded)
    return "%s_%02d.png" % (stem, batch_idx_in_run)


def _concat_paired_bgr(left: np.ndarray, right: np.ndarray, gap_px: int) -> np.ndarray:
    if left.shape[0] != right.shape[0]:
        scale = left.shape[0] / float(right.shape[0])
        nw = int(round(right.shape[1] * scale))
        if cv2 is not None:
            right = cv2.resize(right, (nw, left.shape[0]), interpolation=cv2.INTER_AREA)
        else:
            raise RuntimeError("cv2 required for resize; install opencv-python")
    g = max(1, int(gap_px))
    gutter = np.full((left.shape[0], g, 3), 32, dtype=np.uint8)
    return np.concatenate([left, gutter, right], axis=1)


def _render_frame(
        imgs_row,
        grids_row,
        args,
        dataset,
        forward,
        device,
        draw_conf,
        layout: str,
        show_grid: bool,
        thickness: int,
        out_png: str,
) -> None:
    if cv2 is None:
        raise RuntimeError("cv2 (opencv-python) is required to write PNGs.")

    imgs = torch.as_tensor(imgs_row)
    grids = torch.as_tensor(grids_row)

    geom_preds, _, _ = forward(imgs.to(device), epoch=-1, is_train=False)
    geom_preds = geom_preds.detach()

    ih = int(imgs.shape[-2])
    # Match TrainHandler.plot_val_summary: grids from CELL_SPLIT preds / GT labels.
    # Confidence cut at Grid build (same as TrainHandler.on_data_loaded preview path):
    # __grid_from_prediction__ only skips low-confidence predictors when threshold > 0.
    # Plotting alone is not enough: plot_val_summary omits GridFactory.threshold, so everything
    # enters the Grid; passing draw_conf here matches on_data_loaded (threshold=args.confidence).
    pred_grid, _ = GridFactory.get(torch.unsqueeze(geom_preds[0].cpu(), dim=0), [],
                                   CoordinateSystem.CELL_SPLIT, args,
                                   input_coords=dataset.coords,
                                   threshold=float(draw_conf),
                                   only_train_vars=True,
                                   anchors=dataset.anchors)
    gt_grid, _ = GridFactory.get(torch.unsqueeze(grids[0].cpu(), dim=0), [],
                                 CoordinateSystem.CELL_SPLIT, args,
                                 input_coords=dataset.coords,
                                 only_train_vars=False,
                                 anchors=dataset.anchors)

    pred_uv = pred_grid.get_image_lines(
        coords=dataset.coords,
        image_height=ih,
        confidence_threshold=float(draw_conf),
    )
    gt_uv = gt_grid.get_image_lines(coords=dataset.coords, image_height=ih)

    viz0 = _denormalize_for_viz(imgs[0], dataset)
    if hasattr(viz0, "is_cuda") and viz0.is_cuda:
        viz0 = viz0.detach().cpu()

    gt_thickness = max(thickness - 1, 1)

    if layout == "overlay":
        overlay, _ = plot(
            pred_uv, name="", image=viz0, coords=dataset.coords,
            show_grid=show_grid, cell_size=args.cell_size,
            threshold=draw_conf, coordinates=CoordinateSystem.UV_SPLIT,
            tag="pred", training_vars_only=False, anchors=dataset.anchors,
            colorstyle=ColorStyle.ORIENTATION, thickness=thickness)
        _, _ = plot(
            gt_uv, name=out_png, image=overlay, coords=dataset.coords,
            show_grid=False, cell_size=args.cell_size,
            threshold=0.0, coordinates=CoordinateSystem.UV_SPLIT,
            tag="gt", anchors=dataset.anchors,
            colorstyle=ColorStyle.UNIFORM,
            color=(255, 0, 0), thickness=gt_thickness)
        return

    # paired: prediction | GT. training_vars_only=False: get_image_lines(..., is_training_data=False)
    # packs rows at get_position_of(); True would read get_position_within_prediction and misalign
    # if INSTANCE (etc.) sits between CLASS and CONF — same default as plot_val_summary.
    pred_bgr, _ = plot(
        pred_uv, name="", image=viz0.clone(), coords=dataset.coords,
        show_grid=show_grid, cell_size=args.cell_size,
        threshold=draw_conf, coordinates=CoordinateSystem.UV_SPLIT,
        tag="pred", training_vars_only=False, anchors=dataset.anchors,
        colorstyle=ColorStyle.ORIENTATION, thickness=thickness)
    gt_bgr, _ = plot(
        gt_uv, name="", image=viz0.clone(), coords=dataset.coords,
        show_grid=False, cell_size=args.cell_size,
        threshold=0.0, coordinates=CoordinateSystem.UV_SPLIT,
        tag="gt", anchors=dataset.anchors,
        colorstyle=ColorStyle.UNIFORM,
        color=(255, 0, 0), thickness=gt_thickness)

    combo = _concat_paired_bgr(pred_bgr, gt_bgr, gap_px=max(4, thickness * 2))
    cv2.imwrite(out_png, combo)


def main():
    p = argparse.ArgumentParser(description="Save PNG(s): pred overlay + GT on val image(s).")
    p.add_argument("--config", required=True, help="Experiment yaml (same as training).")
    p.add_argument(
        "--dvc",
        default=None,
        help="Experiment folder with log/checkpoints/... (default: <CAPSTONE>/ttpla_train_exp).",
    )
    p.add_argument(
        "--dataset-root",
        default=None,
        help="Overrides DATASET_TTPLA for this run before loading config.",
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path RELATIVE to --dvc e.g. log/checkpoints/exp19.../ep0041_model.pth",
    )
    p.add_argument(
        "--out",
        required=True,
        help="If --num-images 1: output .png path. If >1 and path ends with .png: stem_00.png …; "
             "if path is a directory (or ends with /): files inside as 00_<name>.png.",
    )
    p.add_argument("--sample-index", type=int, default=0, help="First index into val split (default 0).")
    p.add_argument("--num-images", type=int, default=1,
                   help="Number of consecutive val samples to export (default 1).")
    p.add_argument("--gpu", action="store_true")
    p.add_argument(
        "--layout",
        choices=["paired", "overlay"],
        default="paired",
        help="paired: left=pred only, right=GT only (single PNG, like val summary). "
             "overlay: pred then GT on same image (legacy).",
    )
    p.add_argument(
        "--show-grid",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Cell grid on pred pane (and overlay mode). Default: on for paired, off for overlay.",
    )
    p.add_argument("--run-name", type=str, default="val_pred_gt_overlay", help="Run id for paths/logging.")
    p.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Override yaml --confidence for drawing preds only (default: use experiment yaml). "
             "Drawing keeps a segment iff conf > this value.",
    )
    cli = p.parse_args()

    if cli.show_grid is None:
        show_grid = cli.layout == "paired"
    else:
        show_grid = bool(cli.show_grid)

    if cli.num_images < 1:
        print("[ERROR] --num-images must be >= 1", file=sys.stderr)
        sys.exit(1)

    if cli.dataset_root:
        os.environ["DATASET_TTPLA"] = os.path.abspath(os.path.expanduser(cli.dataset_root))

    project_root = _project_root()
    dvc = cli.dvc or os.path.join(project_root, "ttpla_train_exp")
    dvc = os.path.abspath(os.path.expanduser(dvc))
    default_cfg = os.path.join(dvc, "default_params.yaml")
    cfg_path = cli.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(project_root, cfg_path)

    alt = [
        "-c", cfg_path,
        "--root", project_root,
        "--dvc", dvc,
        "--split", "val",
        "--log_dir", "ttpla_overlay_tool",
        "--explicit_model", cli.checkpoint,
        "--run_name", cli.run_name,
        "--loggers", "file",
        "--batch_size", "1",
        "--loading_workers", "0",
    ]
    if cli.gpu:
        alt.extend(["--gpu", "--gpu_id", "0"])

    args = general_setup(
        "Val pred+GT overlay",
        task_type=TaskType.TEST,
        config_file=cfg_path,
        ignore_cmd_args=True,
        alternative_args=alt,
        default_config=default_cfg if os.path.isfile(default_cfg) else os.path.join(project_root, "ttpla_train_exp", "default_params.yaml"),
    )

    draw_conf = float(args.confidence) if cli.confidence is None else float(cli.confidence)
    if cli.confidence is not None:
        args.confidence = draw_conf

    if cli.gpu and torch.cuda.is_available():
        args.gpu = True
        gid = int(getattr(args, "gpu_id", 0) or 0)
        args.gpu_id = gid
        args.cuda = f"cuda:{gid}"
        torch.cuda.set_device(gid)
    else:
        args.gpu = False
        args.cuda = "cpu"

    ckpt = args.explicit_model
    if not os.path.isfile(ckpt):
        print("[ERROR] Checkpoint not found: %s" % ckpt, file=sys.stderr)
        sys.exit(1)

    dataset, loader = DatasetFactory.get(
        args.dataset,
        only_available=True,
        split="val",
        args=args,
        shuffle=False,
        augment=False,
        ignore_duplicates=False,
    )

    forward = ForwardRunner(args=args, coords=dataset.coords, load_best=False)
    device = next(forward.model.parameters()).device

    print(
        "[INFO] layout=%s show_grid=%s | pred conf cut: GridFactory+get_image_lines (keep conf >= %.4f); "
        "plot draw_line still uses conf > %.4f; GT pane unfiltered; num_predictors/cell=%d scale=%s grid=%s"
        % (cli.layout, str(show_grid), draw_conf, draw_conf, args.num_predictors, args.scale,
           getattr(args, "grid_shape", "?"))
    )
    print("[INFO] Writing %d val sample(s) starting at index %d." % (cli.num_images, cli.sample_index))

    saved = 0
    loader_idx = 0
    for _, data in enumerate(loader):
        if loader_idx < cli.sample_index:
            loader_idx += 1
            continue
        if saved >= cli.num_images:
            break

        images, grid_tensor, fileinfo, _, _ = data
        fname = fileinfo[0] if hasattr(fileinfo, "__getitem__") else fileinfo
        out_png = _out_path_for_batch(cli.out, saved, cli.num_images, fname)

        od = os.path.dirname(out_png)
        if od:
            os.makedirs(od, exist_ok=True)
        thickness = max(1, images.shape[-1] // 300)

        _render_frame(
            images,
            grid_tensor,
            args,
            dataset,
            forward,
            device,
            draw_conf,
            cli.layout,
            show_grid,
            thickness,
            out_png,
        )

        print("[OK] val_index=%d file=%s -> %s" % (loader_idx, fname, out_png))
        saved += 1
        loader_idx += 1

    if saved < cli.num_images:
        print(
            "[ERROR] Needed %d image(s) from val starting at %d, only got %d (loader too short)."
            % (cli.num_images, cli.sample_index, saved),
            file=sys.stderr,
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
