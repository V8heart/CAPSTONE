import argparse
import os

import pandas as pd
import torch

from yolino.dataset.dataset_factory import DatasetFactory
from yolino.runner.evaluator import Evaluator
from yolino.runner.forward_runner import ForwardRunner
from yolino.utils.general_setup import general_setup


def run_full_val(args, loader, evaluator, conf):
    args.confidence = float(conf)
    evaluator.scores = {}
    for i, data in enumerate(loader):
        images, grid_tensor, fileinfo, dupl, _ = data
        num_duplicates = int(sum(dupl["total_duplicates_in_image"]).item())
        evaluator(images=images, grid_tensor=grid_tensor, idx=i, filenames=fileinfo, epoch=None,
                  num_duplicates=num_duplicates, tag=f"sweep_{conf}")
    return evaluator.publish_scores(epoch=None, tag=f"sweep_{conf}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--explicit-model", required=True)
    parser.add_argument("--conf-values", nargs="+", type=float, default=[0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-viz", type=int, default=5)
    cli = parser.parse_args()

    os.makedirs(cli.out_dir, exist_ok=True)
    args = general_setup("Confidence sweep eval", task_type=None, config_file=cli.config, ignore_cmd_args=True)
    args.dvc = cli.dataset_root
    args.explicit_model = cli.explicit_model
    args.plot = True

    dataset, loader = DatasetFactory.get(args.dataset, only_available=True, split="val", args=args, shuffle=False,
                                         augment=False, ignore_duplicates=False)
    evaluator = Evaluator(args=args, anchors=dataset.anchors, coords=dataset.coords, prepare_forward=False)
    evaluator.forward = ForwardRunner(args=args, coords=dataset.coords, load_best=False)

    rows = []
    for conf in cli.conf_values:
        scores = run_full_val(args=args, loader=loader, evaluator=evaluator, conf=conf)
        scores["confidence"] = conf
        rows.append(scores)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cli.out_dir, "confidence_sweep_metrics.csv"), index=False)
    print(df[["confidence"] + [c for c in df.columns if "f1" in c or c in ("precision", "recall")]])


if __name__ == "__main__":
    with torch.no_grad():
        main()
