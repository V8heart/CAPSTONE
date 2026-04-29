#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser(description="Debug single tile forward/loss/backward stability.")
    parser.add_argument("--tile", default="25_00475_tile0", help="Tile basename without extension")
    parser.add_argument("--config", default="ttpla_train_exp/params.yaml", help="Training config path")
    parser.add_argument("--disable-cudnn", action="store_true", help="Disable cuDNN to test backend-specific failures.")
    args_ns = parser.parse_args()

    if args_ns.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled for this debug run")

    project_root = "/home/work/caps_drone/yolino/YOLinO"
    os.chdir(project_root)
    os.environ.setdefault("DATASET_TTPLA", "/home/work/caps_drone/yolino/TTPLA_YOLinO_Dataset")
    os.environ.setdefault("YOLINO_IGNORE_DIRTY", "1")
    os.environ.setdefault("YOLINO_DEBUG_ANOMALY", "1")

    from yolino.runner.trainer import TrainHandler
    from yolino.utils.enums import TaskType, Variables
    from yolino.utils.general_setup import general_setup

    cli_override = [
        "-c", args_ns.config,
        "--root", project_root,
        "--dvc", os.path.join(project_root, "ttpla_train_exp"),
        "--log_dir", "ttpla_single_tile_debug",
        "--run_name", "ttpla_single_tile_debug",
        "--split", "train",
        "--loading_workers", "0",
        "--gpu",
        "--gpu_id", "0",
        "--amp", "False",
    ]
    args = general_setup(
        "SingleTileDebug",
        config_file=args_ns.config,
        default_config="ttpla_train_exp/default_params.yaml",
        ignore_cmd_args=True,
        alternative_args=cli_override,
        setup_logging=False,
        task_type=TaskType.TRAIN,
    )

    trainer = TrainHandler(args)
    ds = trainer.dataset
    if args_ns.tile not in ds.file_names:
        raise ValueError(f"Tile not found in train split: {args_ns.tile}")
    idx = ds.file_names.index(args_ns.tile)

    image, grid_tensor, name, _, _ = ds[idx]
    print(f"tile={name} image_shape={tuple(image.shape)} grid_shape={tuple(grid_tensor.shape)}")

    # Basic GT sanity checks around geometry/instance channels.
    geom_pos = ds.coords.get_position_of(Variables.GEOMETRY)
    inst_pos = ds.coords.get_position_of(Variables.INSTANCE)
    geom_gt = grid_tensor[:, :, geom_pos]
    print(f"geom_gt finite_any={torch.isfinite(geom_gt).any().item()} nan_all={torch.isnan(geom_gt).all().item()}")
    if len(inst_pos) > 0:
        inst_gt = grid_tensor[:, :, inst_pos[0]]
        valid_inst = torch.isfinite(inst_gt) & (inst_gt > 0)
        print(
            f"inst_valid_slots={int(valid_inst.sum().item())} "
            f"unique_inst={torch.unique(inst_gt[valid_inst]).tolist() if valid_inst.any() else []}"
        )

    images = image.unsqueeze(0)
    gts = grid_tensor.unsqueeze(0)
    trainer.optimizer.zero_grad(set_to_none=True)

    with torch.autograd.detect_anomaly():
        geom_preds, embed_preds = trainer.forward(images, is_train=True, epoch=0, first_run=False)
    print(f"geom_preds shape={tuple(geom_preds.shape)} finite={torch.isfinite(geom_preds).all().item()}")
    print(f"embed_preds shape={tuple(embed_preds.shape)} finite={torch.isfinite(embed_preds).all().item()}")
    with torch.autograd.detect_anomaly():
        losses, sum_loss, mean_losses = trainer.loss(
            gts.to(args.cuda), geom_preds, embed_preds, [name], epoch=0, tag="debug"
        )
    print(f"sum_loss={sum_loss.detach().item()} finite={torch.isfinite(sum_loss).item()}")
    print(f"loss_components={losses}")
    print(f"mean_components={mean_losses}")

    # Backward in anomaly mode to pinpoint invalid op.
    trainer.backward(sum_loss, epoch=0)
    print("backward finished without exception")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
