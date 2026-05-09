import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


METRICS = ["f1", "precision", "recall", "accuracy"]


def _load_metric_series(run_dir: Path, metric: str):
    """Return dict: step -> mean(value across event files)."""
    tag_dir = run_dir / "val_cell_metrics" / metric / "val"
    if not tag_dir.exists():
        raise FileNotFoundError(f"Missing TB metric directory: {tag_dir}")

    by_step = defaultdict(list)
    event_files = sorted(tag_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No event files in {tag_dir}")

    for ef in event_files:
        try:
            ea = event_accumulator.EventAccumulator(str(ef), size_guidance={"scalars": 0})
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            if not tags:
                continue
            # Each per-tag dir usually has one scalar tag; use first.
            scalar_events = ea.Scalars(tags[0])
            for ev in scalar_events:
                by_step[int(ev.step)].append(float(ev.value))
        except Exception:
            # Skip broken/incomplete files safely.
            continue

    if not by_step:
        raise RuntimeError(f"No scalar values parsed for {metric} in {tag_dir}")

    return {s: float(np.mean(vs)) for s, vs in by_step.items()}


def _best_step_from_f1(run_dir: Path):
    f1_series = _load_metric_series(run_dir, "f1")
    best_step = max(f1_series.keys(), key=lambda s: f1_series[s])
    return best_step, f1_series[best_step], f1_series


def _metric_at_step(run_dir: Path, metric: str, step: int):
    series = _load_metric_series(run_dir, metric)
    if step in series:
        return series[step]
    # Fallback: nearest step if exact step absent.
    nearest = min(series.keys(), key=lambda s: abs(s - step))
    return series[nearest]


def summarize_run(run_dir: Path):
    best_step, best_f1, _ = _best_step_from_f1(run_dir)
    out = {"best_step": best_step, "f1": best_f1}
    for m in ["precision", "recall", "accuracy"]:
        out[m] = _metric_at_step(run_dir, m, best_step)
    return out


def save_pair_hist(out_dir: Path, left_name: str, left_vals: dict, right_name: str, right_vals: dict, title: str, fname: str):
    xs = np.arange(len(METRICS))
    width = 0.36
    left_y = [left_vals[m] for m in METRICS]
    right_y = [right_vals[m] for m in METRICS]

    plt.figure(figsize=(9, 5))
    plt.bar(xs - width / 2, left_y, width=width, label=left_name)
    plt.bar(xs + width / 2, right_y, width=width, label=right_name)
    plt.xticks(xs, [m.upper() for m in METRICS])
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    # Annotate bars
    for i, v in enumerate(left_y):
        plt.text(i - width / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(right_y):
        plt.text(i + width / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    out_path = out_dir / fname
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def save_multi_hist(out_dir: Path, labels: list, values_by_label: dict, title: str, fname: str):
    xs = np.arange(len(METRICS))
    n = len(labels)
    width = 0.8 / max(n, 1)

    plt.figure(figsize=(11, 5))
    for i, lab in enumerate(labels):
        ys = [values_by_label[lab][m] for m in METRICS]
        offs = (i - (n - 1) / 2.0) * width
        plt.bar(xs + offs, ys, width=width, label=lab)
        for j, v in enumerate(ys):
            plt.text(j + offs, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    plt.xticks(xs, [m.upper() for m in METRICS])
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend(fontsize=8, ncol=2)
    plt.grid(axis="y", alpha=0.25)
    out_path = out_dir / fname
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Compare experiment metrics from TensorBoard logs.")
    parser.add_argument("--runs-root", default="/home/work/caps_drone/yolino/CAPSTONE/ttpla_train_exp/runs")
    parser.add_argument("--out-dir", default="/home/work/caps_drone/yolino/CAPSTONE/ttpla_train_exp/analysis/metric_histograms")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Requested comparison chain:
    # 1) exp09 -> exp05(convnext)
    # 2) exp05 -> exp10(fpn+bottomup p4)
    # 3) exp10 -> exp19(num_predictors=4)
    # 4) fine-tune runs only (on Sample_YOLinO_tiled_1024)
    run_map = {
        "exp09_darknet_1024": runs_root / "exp09_darknet_1024",
        "exp05_convnext_baseline_1024": runs_root / "train50_exp05_convnext_baseline_1024",
        "exp10_fpn_bottomup_p4_1024": runs_root / "exp10_fpn_bottomup_p4_1024",
        "exp19_fpn_bottomup_p4_num_predictors4_1024": runs_root / "exp19_fpn_bottomup_p4_num_predictors4_1024",
        "exp15_tile_finetune_sample_e40": runs_root / "exp15_tile_finetune_sample_e40",
        "exp16_tile_finetune_sample_aug_relaxed_e80": runs_root / "exp16_tile_finetune_sample_aug_relaxed_e80",
        "exp17_tile_finetune_sample_equal_direction_e80": runs_root / "exp17_tile_finetune_sample_equal_direction_e80",
        "exp18_tile_finetune_sample_matching_relaxed_e80": runs_root / "exp18_tile_finetune_sample_matching_relaxed_e80",
        "exp21_tile_finetune_sample_from_exp19_ep41_e80": runs_root / "exp21_tile_finetune_sample_from_exp19_ep41_e80",
    }

    summaries = {}
    for name, path in run_map.items():
        summaries[name] = summarize_run(path)

    # Save CSV summary
    csv_path = out_dir / "best_f1_epoch_metrics_summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("run,best_step,f1,precision,recall,accuracy\n")
        for name in run_map.keys():
            s = summaries[name]
            f.write(f"{name},{s['best_step']},{s['f1']:.6f},{s['precision']:.6f},{s['recall']:.6f},{s['accuracy']:.6f}\n")

    pair_defs = [
        ("exp09_darknet_1024", "exp05_convnext_baseline_1024", "Baseline change: Darknet -> ConvNeXt Tiny", "01_darknet_to_convnext.png"),
        ("exp05_convnext_baseline_1024", "exp10_fpn_bottomup_p4_1024", "Architecture change: ConvNeXt -> ConvNeXt+FPN+BU(P4)", "02_convnext_to_fpn_bottomup.png"),
        ("exp10_fpn_bottomup_p4_1024", "exp19_fpn_bottomup_p4_num_predictors4_1024", "Head capacity change: num_predictors 8 -> 4", "03_exp10_to_exp19_predictor_change.png"),
    ]

    pngs = []
    for left, right, title, fname in pair_defs:
        p = save_pair_hist(
            out_dir=out_dir,
            left_name=left,
            left_vals=summaries[left],
            right_name=right,
            right_vals=summaries[right],
            title=title,
            fname=fname,
        )
        pngs.append(p)

    finetune_labels = [
        "exp15_tile_finetune_sample_e40",
        "exp16_tile_finetune_sample_aug_relaxed_e80",
        "exp17_tile_finetune_sample_equal_direction_e80",
        "exp18_tile_finetune_sample_matching_relaxed_e80",
        "exp21_tile_finetune_sample_from_exp19_ep41_e80",
    ]
    p = save_multi_hist(
        out_dir=out_dir,
        labels=finetune_labels,
        values_by_label=summaries,
        title="Fine-tune only comparison (best-F1 epoch metrics)",
        fname="04_finetune_only_comparison.png",
    )
    pngs.append(p)

    print("[DONE] Saved metric comparison histograms:")
    for p in pngs:
        print(f" - {p}")
    print(f"[DONE] CSV summary: {csv_path}")


if __name__ == "__main__":
    main()

