import argparse
import os
from collections import defaultdict

import numpy as np
import torch

from yolino.dataset.dataset_factory import DatasetFactory
from yolino.runner.forward_runner import ForwardRunner
from yolino.utils.enums import Variables
from yolino.utils.general_setup import general_setup
from yolino.utils.logger import Log


def qstats(values: np.ndarray):
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "high@0.5": float(np.mean(values >= 0.5)),
        "high@0.7": float(np.mean(values >= 0.7)),
        "high@0.8": float(np.mean(values >= 0.8)),
    }


def print_block(title: str, stats: dict):
    print(f"\n[{title}]")
    if stats.get("count", 0) == 0:
        print("count=0")
        return
    print(
        "count={count} mean={mean:.4f} std={std:.4f} p01={p01:.4f} p10={p10:.4f} "
        "p50={p50:.4f} p90={p90:.4f} p99={p99:.4f} high@0.5={high05:.4f} high@0.7={high07:.4f} high@0.8={high08:.4f}".format(
            count=stats["count"],
            mean=stats["mean"],
            std=stats["std"],
            p01=stats["p01"],
            p10=stats["p10"],
            p50=stats["p50"],
            p90=stats["p90"],
            p99=stats["p99"],
            high05=stats["high@0.5"],
            high07=stats["high@0.7"],
            high08=stats["high@0.8"],
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictor confidence distribution for a checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--loading-workers", type=int, default=0)
    parser.add_argument("--gpu", action="store_true")
    cli, _ = parser.parse_known_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    default_cfg = os.path.join(project_root, "ttpla_train_exp", "default_params.yaml")
    alt_args = ["-c", cli.config, "--root", project_root, "--dvc", project_root]
    if cli.gpu:
        alt_args += ["--gpu", "--gpu_id", "0"]

    args = general_setup(
        "Confidence distribution eval",
        task_type=None,
        config_file=cli.config,
        ignore_cmd_args=True,
        alternative_args=alt_args,
        default_config=default_cfg,
    )
    args.dvc = cli.dataset_root
    args.explicit_model = cli.checkpoint
    args.paths.pretrain_model = args.explicit_model
    args.loading_workers = int(cli.loading_workers)
    if cli.gpu and torch.cuda.is_available():
        args.gpu = True
        args.gpu_id = 0 if getattr(args, "gpu_id", -1) < 0 else int(args.gpu_id)
        args.cuda = f"cuda:{args.gpu_id}"
        torch.cuda.set_device(args.gpu_id)

    dataset, loader = DatasetFactory.get(
        args.dataset,
        only_available=True,
        split=cli.split,
        args=args,
        shuffle=False,
        augment=False,
        ignore_duplicates=False,
    )
    model = ForwardRunner(args=args, coords=dataset.coords, load_best=False)

    conf_pos_pred = dataset.coords.get_position_within_prediction(Variables.CONF)[0]
    conf_pos_gt = dataset.coords.get_position_of(Variables.CONF)[0]
    geom_pos_gt = dataset.coords.get_position_of(Variables.GEOMETRY)
    num_predictors = int(args.num_predictors)

    all_conf = []
    matched_conf = []
    unmatched_conf = []
    by_predictor = defaultdict(list)

    for images, grid_tensor, fileinfo, *_ in loader:
        with torch.no_grad():
            geom_preds, _, _ = model(images, is_train=False, epoch=None)

        pred_conf = geom_preds[..., conf_pos_pred].detach().cpu().numpy()  # [B, cells, P]
        gt_conf = grid_tensor[..., conf_pos_gt].detach().cpu().numpy()  # [B, cells, P]
        gt_geom = grid_tensor[..., geom_pos_gt].detach().cpu().numpy()  # [B, cells, P, geom_vars]

        gt_conf = np.nan_to_num(gt_conf, nan=0.0)
        matched_mask = ~np.any(np.isnan(gt_geom), axis=-1)  # [B, cells, P]
        unmatched_mask = ~matched_mask

        all_conf.append(pred_conf.reshape(-1))
        matched_conf.append(pred_conf[matched_mask])
        unmatched_conf.append(pred_conf[unmatched_mask])

        # predictor slot based on fixed predictor axis
        for p in range(num_predictors):
            by_predictor[p].append(pred_conf[:, :, p].reshape(-1))

    all_conf = np.concatenate(all_conf) if all_conf else np.array([], dtype=np.float32)
    matched_conf = np.concatenate(matched_conf) if matched_conf else np.array([], dtype=np.float32)
    unmatched_conf = np.concatenate(unmatched_conf) if unmatched_conf else np.array([], dtype=np.float32)

    print(f"[INFO] checkpoint={args.explicit_model}")
    print(f"[INFO] split={cli.split} device={args.cuda}")
    print_block("pred/all", qstats(all_conf))
    print_block("pred/matched(gt-exists)", qstats(matched_conf))
    print_block("pred/unmatched(gt-empty)", qstats(unmatched_conf))

    for p in range(num_predictors):
        vals = np.concatenate(by_predictor[p]) if by_predictor[p] else np.array([], dtype=np.float32)
        print_block(f"pred/p{p}", qstats(vals))


if __name__ == "__main__":
    # no grads and no graph retention
    with torch.no_grad():
        main()
