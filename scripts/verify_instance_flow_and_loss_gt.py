#!/usr/bin/env python3
"""
Verify instance-id flow and discriminative-loss GT readiness on TTPLA.

Checks:
1) ID flow in GT->grid conversion:
   - IDs are generated as contiguous local IDs [1..N] per image.
   - After slicing into local cell GT (grid_tensor), positive INSTANCE IDs are still present.
2) Discriminative-loss GT readiness:
   - Count valid slots used by loss (finite geometry + INSTANCE>0).
   - Count unique instance IDs per image among valid slots.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLINO_ROOT = os.path.dirname(SCRIPT_DIR)
if YOLINO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(YOLINO_ROOT, "src"))

os.environ.setdefault("DATASET_TTPLA", "/home/work/caps_drone/yolino/TTPLA_YOLinO_Dataset")
os.environ.setdefault("YOLINO_IGNORE_DIRTY", "1")


def _build_args():
    from yolino.utils.enums import TaskType
    from yolino.utils.general_setup import general_setup

    cfg = os.path.join(YOLINO_ROOT, "ttpla_train_exp", "params.yaml")
    defaults = os.path.join(YOLINO_ROOT, "ttpla_train_exp", "default_params.yaml")
    return general_setup(
        "verify_instance_flow",
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
            "verify_instance_flow_tmp",
        ],
        setup_logging=False,
        task_type=TaskType.TRAIN,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=20, help="Number of val samples to inspect")
    args_ns = parser.parse_args()
    os.chdir(YOLINO_ROOT)

    from yolino.dataset.dataset_factory import DatasetFactory
    from yolino.utils.enums import Variables

    args = _build_args()
    ds, _ = DatasetFactory.get(
        args.dataset, only_available=True, split="val", args=args, shuffle=False, augment=False
    )
    coords = ds.coords
    pos_geom = coords.get_position_of(Variables.GEOMETRY)
    pos_inst = coords.get_position_of(Variables.INSTANCE)
    if len(pos_inst) == 0:
        print("INSTANCE slot is not present in coords; cannot verify discriminative GT.")
        return 2
    inst_col = int(pos_inst[0])

    print("coords.train_vars() =", coords.train_vars())
    print("geom_idx =", list(pos_geom), "inst_col =", inst_col)
    print("---")

    n = min(args_ns.max, len(ds))
    has_positive = 0
    contiguous_ok = 0
    loss_valid_ok = 0
    multi_instance_ok = 0

    for i in range(n):
        # Pull intermediate (pre-grid) for traceability.
        lines, _instance_ids = ds.__get_labels__(i)
        img_t = ds.__make_torch__(ds.__load_image__(i))
        _, lines_aug, _ = ds.__augment__(i, img_t, lines)
        local_expected_n = int(lines_aug.shape[0])
        expected_ids = torch.arange(1, local_expected_n + 1, dtype=torch.float32)

        _, grid_tensor, fname, _, _ = ds[i]
        inst_vals = grid_tensor[:, :, inst_col].reshape(-1)
        geom_vals = grid_tensor[:, :, pos_geom]
        geom_valid = torch.isfinite(geom_vals).all(dim=-1).reshape(-1)
        inst_valid = torch.isfinite(inst_vals) & (inst_vals > 0)
        valid_for_disc = geom_valid & inst_valid

        pos_ids = torch.unique(inst_vals[inst_valid]).sort().values
        disc_ids = torch.unique(inst_vals[valid_for_disc]).sort().values

        has_pos = int(inst_valid.sum().item()) > 0
        contiguous = bool(
            len(pos_ids) == local_expected_n
            and torch.allclose(pos_ids.cpu(), expected_ids[: len(pos_ids)])
        ) if local_expected_n > 0 else True
        loss_ready = int(valid_for_disc.sum().item()) > 0
        multi_ready = len(disc_ids) >= 2

        has_positive += int(has_pos)
        contiguous_ok += int(contiguous)
        loss_valid_ok += int(loss_ready)
        multi_instance_ok += int(multi_ready)

        print(
            f"[{i}] {fname}: raw_instances={local_expected_n}, "
            f"grid_pos_slots={int(inst_valid.sum().item())}, "
            f"grid_unique_ids={len(pos_ids)}, contiguous_ids={contiguous}, "
            f"disc_valid_slots={int(valid_for_disc.sum().item())}, disc_unique_ids={len(disc_ids)}"
        )

    print("---")
    print(f"positive INSTANCE in grid_tensor: {has_positive}/{n}")
    print(f"contiguous local IDs preserved [1..N]: {contiguous_ok}/{n}")
    print(f"discriminative-loss valid slots exist: {loss_valid_ok}/{n}")
    print(f"discriminative-loss push-term possible (>=2 ids): {multi_instance_ok}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
