#!/usr/bin/env python3
"""
Scan TTPLA val samples and report whether grid_tensor carries INSTANCE IDs (>0).

Usage (from YOLinO/):
  export DATASET_TTPLA=/path/to/TTPLA_YOLinO_Dataset   # optional if unset below
  PYTHONPATH=src python scripts/verify_ttpla_instance_grid.py [--max N]
"""
from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLINO_ROOT = os.path.dirname(SCRIPT_DIR)
if YOLINO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(YOLINO_ROOT, "src"))

# Default dataset root for this workspace (override with DATASET_TTPLA)
os.environ.setdefault("DATASET_TTPLA", "/home/work/caps_drone/yolino/TTPLA_YOLinO_Dataset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=25, help="Max val samples to scan")
    args_ns = parser.parse_args()

    os.chdir(YOLINO_ROOT)
    os.environ.setdefault("YOLINO_IGNORE_DIRTY", "1")

    import torch

    from yolino.dataset.dataset_factory import DatasetFactory
    from yolino.utils.enums import TaskType, Variables
    from yolino.utils.general_setup import general_setup

    cfg = os.path.join(YOLINO_ROOT, "ttpla_train_exp", "params.yaml")
    defaults = os.path.join(YOLINO_ROOT, "ttpla_train_exp", "default_params.yaml")
    args = general_setup(
        "verify_instance",
        config_file=cfg,
        default_config=defaults,
        ignore_cmd_args=True,
        alternative_args=[
            "-c",
            cfg,
            "--root",
            YOLINO_ROOT,
            "--split",
            "val",
            "--loading_workers",
            "0",
            "--log_dir",
            "verify_instance_tmp",
        ],
        setup_logging=False,
        task_type=TaskType.TRAIN,
    )

    ds, _ = DatasetFactory.get(
        args.dataset,
        only_available=True,
        split="val",
        args=args,
        shuffle=False,
        augment=False,
    )
    coords = ds.coords
    inst_idx = coords.get_position_of(Variables.INSTANCE)
    if len(inst_idx) == 0:
        print("INSTANCE has zero width in coords layout; nothing to check.")
        return 1

    # Primary ID channel (matches DiscriminativeEmbeddingLoss which uses pos_inst[0])
    inst_col = int(inst_idx[0])
    L = coords.get_length(one_hot=True)

    print("grid_tensor layout: (cells, num_predictors, coord_len)")
    print(f"coord_len={L}, INSTANCE column index={inst_col}")
    print(f"coords.train_vars()={coords.train_vars()}")
    print("---")

    n = min(args_ns.max, len(ds))
    ok = 0
    for i in range(n):
        sample = ds[i]
        grid_tensor = sample[1]
        name = sample[2]
        if grid_tensor.dim() != 3:
            print(f"[{i}] {name}: unexpected grid_tensor shape {tuple(grid_tensor.shape)}")
            continue
        gts = grid_tensor[:, :, inst_col].reshape(-1)
        valid = torch.isfinite(gts) & (gts > 0)
        cnt = int(valid.sum().item())
        if cnt > 0:
            ok += 1
            u = torch.unique(gts[valid])
            print(
                f"[{i}] {name}: instance-positive slots={cnt}, "
                f"unique_ids={int(u.numel())}, id_min={float(u.min()):.4f}, id_max={float(u.max()):.4f}"
            )
        else:
            print(f"[{i}] {name}: NO INSTANCE>0 in grid_tensor (column {inst_col})")

    print("---")
    print(f"summary: {ok}/{n} val samples have at least one INSTANCE>0 in grid_tensor")
    return 0 if ok > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
