"""
Aggregate GT geometry stats from TTPLA tiled labels after grid rasterization (same path as training).

- Per grid cell: number of GT line slots filled (0 .. num_predictors).
- Per GT segment (cell-local md): direction angle atan2(dy, dx) and slope dy/dx when |dx| is large enough.

Uses the same config/coords/grid as training (default: exp15 tiled yaml).
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from yolino.dataset.dataset_factory import DatasetFactory
from yolino.utils.enums import Dataset, Variables
from yolino.utils.general_setup import general_setup


def _norm_angle_deg(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Map direction to (-90, 90] degrees (undirected line tilt vs +x in cell coords)."""
    rad = np.arctan2(dy.astype(np.float64), dx.astype(np.float64))
    deg = np.degrees(rad)
    deg = ((deg + 90.0) % 180.0) - 90.0
    return deg.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/exp15_tile_finetune_sample_yolino_from_exp10_1024.yaml",
        help="YAML that defines grid scale, linerep, num_predictors (must match training).",
    )
    parser.add_argument(
        "--dataset-root",
        default="/home/work/caps_drone/yolino/Sample_YOLinO_tiled_1024",
        help="Overrides DATASET_TTPLA before setup if set.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--splits", default="train,val", help="Comma-separated: train,val,test")
    parser.add_argument("--max-batches", type=int, default=0, help="0 = full split")
    args_cli = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    default_cfg = os.path.join(project_root, "ttpla_train_exp", "default_params.yaml")
    cfg_path = args_cli.config if os.path.isabs(args_cli.config) else os.path.join(project_root, args_cli.config)

    if args_cli.dataset_root:
        os.environ["DATASET_TTPLA"] = os.path.abspath(os.path.expanduser(args_cli.dataset_root))

    alternative_args = [
        "-c",
        cfg_path,
        "--root",
        project_root,
        "--dvc",
        os.path.join(project_root, "ttpla_train_exp"),
        "--epoch",
        "1",
        "--batch_size",
        str(args_cli.batch_size),
        "--loading_workers",
        "0",
    ]

    setup_args = general_setup(
        "Tiled grid GT distribution",
        task_type=None,
        config_file=cfg_path,
        ignore_cmd_args=True,
        alternative_args=alternative_args,
        default_config=default_cfg,
        setup_logging=False,
    )
    setup_args.gpu = False
    setup_args.cuda = "cpu"

    splits = [s.strip() for s in args_cli.splits.split(",") if s.strip()]

    for split in splits:
        dataset, _ = DatasetFactory.get(
            Dataset.TTPLA,
            only_available=True,
            split=split,
            args=setup_args,
            shuffle=False,
            augment=False,
            ignore_duplicates=False,
        )
        coords = dataset.coords
        geom_ix = coords.get_position_of(Variables.GEOMETRY, one_hot=True)
        assert len(geom_ix) >= 4, "Expected md geometry (mx,my,dx,dy)"

        loader = DataLoader(
            dataset,
            batch_size=int(args_cli.batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False,
        )

        p_max = int(setup_args.num_predictors)
        n_bins_count = p_max + 1
        count_hist = np.zeros(n_bins_count, dtype=np.int64)
        angles_all = []
        slopes_finite = []

        total_cells = 0
        total_lines = 0
        batches = 0

        for batch in loader:
            _, grid_tensor, *_ = batch
            # [B, cells, P, vars]
            g = grid_tensor[..., geom_ix[:4]].numpy()
            valid = np.all(np.isfinite(g), axis=-1)
            per_cell = valid.sum(axis=-1).astype(np.int64)  # [B, cells]

            flat_counts = per_cell.reshape(-1)
            total_cells += flat_counts.size
            for c in range(n_bins_count):
                count_hist[c] += int(np.sum(flat_counts == c))
            total_lines += int(valid.sum())

            dx = g[..., 2][valid]
            dy = g[..., 3][valid]
            if dx.size > 0:
                angles_all.append(_norm_angle_deg(dx, dy))
                m = np.abs(dx) > 1e-4
                if np.any(m):
                    slopes_finite.append((dy[m] / dx[m]).astype(np.float32))

            batches += 1
            if args_cli.max_batches and batches >= args_cli.max_batches:
                break

        angles = np.concatenate(angles_all) if angles_all else np.array([], dtype=np.float32)
        slopes = np.concatenate(slopes_finite) if slopes_finite else np.array([], dtype=np.float32)

        print("\n" + "=" * 72)
        print("split=%s  images≈%d  grid_cells_total=%d  gt_segments_total=%d" % (
            split, len(dataset), total_cells, total_lines))
        print("num_predictors=%d  linerep=%s  cell_shape=%s" % (
            p_max, setup_args.linerep, list(setup_args.grid_shape)))

        print("\n--- GT count per grid cell (how many predictor slots hold a line) ---")
        probs = count_hist.astype(np.float64) / max(total_cells, 1)
        for k in range(n_bins_count):
            print("  count=%d  cells=%d (%.2f%%)" % (k, count_hist[k], 100.0 * probs[k]))
        # recompute mean from histogram
        mean_gt = sum(k * count_hist[k] for k in range(n_bins_count)) / max(total_cells, 1)
        var_gt = sum((k - mean_gt) ** 2 * count_hist[k] for k in range(n_bins_count)) / max(total_cells, 1)
        print("  mean(lines/cell)=%.4f  std=%.4f" % (mean_gt, np.sqrt(var_gt)))

        print("\n--- GT direction in cell coords (atan2(dy,dx), mapped to (-90,90] deg) ---")
        if angles.size == 0:
            print("  (no segments)")
        else:
            qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            pct = np.percentile(angles, qs)
            print("  n=%d  mean=%.2f°  std=%.2f°" % (angles.size, float(np.mean(angles)), float(np.std(angles))))
            print("  percentiles " + ", ".join(["p%d=%.2f°" % (q, pct[i]) for i, q in enumerate(qs)]))
            hist_a, edges = np.histogram(angles, bins=18, range=(-90, 90))
            print("  histogram [-90,90] 18 bins (counts):")
            print("    " + " ".join("%d" % h for h in hist_a))

        print("\n--- Slope dy/dx (cell coords), only |dx|>1e-4 ---")
        if slopes.size == 0:
            print("  (none)")
        else:
            qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            pct = np.percentile(slopes, qs)
            print("  n=%d  mean=%.4f  std=%.4f" % (slopes.size, float(np.mean(slopes)), float(np.std(slopes))))
            print("  percentiles " + ", ".join(["p%d=%.4f" % (q, pct[i]) for i, q in enumerate(qs)]))


if __name__ == "__main__":
    main()
