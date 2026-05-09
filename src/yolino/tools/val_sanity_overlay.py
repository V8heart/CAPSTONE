import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from yolino.dataset.dataset_factory import DatasetFactory
from yolino.grid.grid_factory import GridFactory
from yolino.runner.forward_runner import ForwardRunner
from yolino.utils.enums import CoordinateSystem
from yolino.utils.general_setup import general_setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--explicit-model", default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--out", required=True)
    args_in, _ = parser.parse_known_args()

    args = general_setup("Validation sanity overlay", task_type=None, config_file=args_in.config,
                         ignore_cmd_args=True)
    args.dvc = args_in.dataset_root
    if args_in.explicit_model is not None:
        args.explicit_model = args_in.explicit_model
    if args_in.confidence is not None:
        args.confidence = float(args_in.confidence)

    dataset, loader = DatasetFactory.get(args.dataset, only_available=True, split="val", args=args, shuffle=False,
                                         augment=False, ignore_duplicates=False)
    model = ForwardRunner(args=args, coords=dataset.coords, load_best=True)

    images, grid_tensor, fileinfo, *_ = next(iter(loader))
    geom_preds, _, _ = model(images, is_train=False, epoch=None)
    pred_grid, _ = GridFactory.get(geom_preds[[0]], [], CoordinateSystem.CELL_SPLIT, args, input_coords=dataset.coords,
                                   only_train_vars=True, anchors=dataset.anchors)
    gt_grid, _ = GridFactory.get(grid_tensor[[0]], [], CoordinateSystem.CELL_SPLIT, args, input_coords=dataset.coords,
                                 only_train_vars=False, anchors=dataset.anchors)
    best_uv = np.asarray(pred_grid.get_image_lines(coords=dataset.coords, image_height=images[0].shape[1]))[0]
    gt_uv = np.asarray(gt_grid.get_image_lines(coords=dataset.coords, image_height=images[0].shape[1]))[0]

    img = images[0].permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)

    pred_draw_mask = best_uv[:, 4] >= float(args.confidence)
    shown_uv = best_uv[pred_draw_mask]
    for p in shown_uv:
        a = float(np.clip(0.15 + 0.85 * p[4], 0.15, 1.0))
        ax.plot([p[1], p[3]], [p[0], p[2]], color="yellow", linewidth=1, alpha=a)

    for g in gt_uv:
        if np.any(np.isnan(g[:4])):
            continue
        ax.plot([g[1], g[3]], [g[0], g[2]], color="white", linewidth=1, alpha=0.8)

    ax.set_title(f"{fileinfo[0]} | conf={args.confidence} | shown={len(shown_uv)}/{len(best_uv)}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(args_in.out, dpi=150)


if __name__ == "__main__":
    main()
