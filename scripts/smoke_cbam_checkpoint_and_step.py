#!/usr/bin/env python3
"""
Step 3–4 smoke tests:
  (3) Load older-style weights into cbam_shared YolinoNet (strict=False); report missing/unexpected keys.
      If --checkpoint is given, load that file; else use a fresh sa_embed_only state_dict as pseudo-checkpoint.
  (4) One training batch: TrainHandler, first train step only (forward + backward).

Run from YOLinO/:
  YOLINO_IGNORE_DIRTY=1 DATASET_TTPLA=/path/to/dataset PYTHONPATH=src python scripts/smoke_cbam_checkpoint_and_step.py
  YOLINO_IGNORE_DIRTY=1 PYTHONPATH=src python scripts/smoke_cbam_checkpoint_and_step.py --checkpoint /path/to/model.pth
"""
from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLINO_ROOT = os.path.dirname(SCRIPT_DIR)
if YOLINO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(YOLINO_ROOT, "src"))

os.environ.setdefault("DATASET_TTPLA", "/home/work/caps_drone/yolino/TTPLA_YOLinO_Dataset")
os.environ.setdefault("YOLINO_IGNORE_DIRTY", "1")


def _base_args(extra_cli=None):
    cfg = os.path.join(YOLINO_ROOT, "ttpla_train_exp", "params.yaml")
    defaults = os.path.join(YOLINO_ROOT, "ttpla_train_exp", "default_params.yaml")
    from yolino.utils.enums import TaskType
    from yolino.utils.general_setup import general_setup

    alt = [
        "-c",
        cfg,
        "--root",
        YOLINO_ROOT,
        "--split",
        "train",
        "--loading_workers",
        "0",
        "--log_dir",
        "smoke_cbam_tmp",
        "--retrain",
        "--max_n",
        "2",
        "--batch_size",
        "1",
        "--epoch",
        "1",
    ]
    if extra_cli:
        alt.extend(extra_cli)
    return general_setup(
        "smoke_cbam",
        config_file=cfg,
        default_config=defaults,
        ignore_cmd_args=True,
        alternative_args=alt,
        setup_logging=False,
        task_type=TaskType.TRAIN,
    )


def step3_state_dict_transfer(checkpoint_path: str | None) -> None:
    import torch
    from yolino.dataset.dataset_factory import DatasetFactory
    from yolino.model.model_factory import get_model

    args_sa = _base_args(["--feature_refine", "sa_embed_only"])
    coords = DatasetFactory.get_coords("train", args_sa)
    args_cb = _base_args(["--feature_refine", "cbam_shared"])

    m_sa = get_model(args_sa, coords)
    m_cb = get_model(args_cb, coords)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        print("Step3: loading checkpoint from", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        inc = m_cb.load_state_dict(sd, strict=False)
    else:
        print("Step3: no --checkpoint; using sa_embed_only model weights as pseudo old checkpoint")
        sd = m_sa.state_dict()
        inc = m_cb.load_state_dict(sd, strict=False)

    print("  missing_keys:", len(inc.missing_keys))
    if inc.missing_keys[:12]:
        print("  missing_keys (first 12):", inc.missing_keys[:12])
    print("  unexpected_keys:", len(inc.unexpected_keys))
    if inc.unexpected_keys[:12]:
        print("  unexpected_keys (first 12):", inc.unexpected_keys[:12])


def step4_one_train_batch(feature_refine: str) -> None:
    from yolino.runner.trainer import TrainHandler

    base = _base_args(["--feature_refine", feature_refine])

    print("Step4: TrainHandler one batch, feature_refine=%s" % feature_refine)
    trainer = TrainHandler(base)
    trainer.model.train(True)
    data = next(iter(trainer.loader))
    images, grid_tensor, fileinfo, duplicate_info, params = data
    for j, f in enumerate(fileinfo):
        trainer.dataset.params_per_file[f] = {}
        for k, v in params.items():
            trainer.dataset.params_per_file[f].update({k: v[j].item()})

    loss_val, preds = trainer(
        fileinfo,
        images,
        grid_tensor,
        epoch=0,
        image_idx_in_batch=0,
        first_run=True,
        is_train=True,
    )
    # TrainHandler.__call__ already runs backward when is_train=True
    print("  batch_loss:", float(loss_val))
    print("  preds type:", type(preds).__name__, "geom shape:", preds[0].shape if isinstance(preds, tuple) else preds.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional .pth (full checkpoint dict)")
    parser.add_argument("--skip-step3", action="store_true")
    parser.add_argument("--skip-step4", action="store_true")
    parser.add_argument(
        "--step4-mode",
        choices=["none", "sa_embed_only", "sa_shared", "cbam_shared"],
        default="cbam_shared",
    )
    ns = parser.parse_args()

    os.chdir(YOLINO_ROOT)

    if not ns.skip_step3:
        step3_state_dict_transfer(ns.checkpoint)
    if not ns.skip_step4:
        step4_one_train_batch(ns.step4_mode)

    print("smoke_cbam_checkpoint_and_step: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
