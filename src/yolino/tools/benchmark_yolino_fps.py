#!/usr/bin/env python3
"""
Measure YOLinO inference FPS on a single GPU (batch_size=1).

Scopes:
  geom_only  — backbone + FPN/refine + geom/embed heads (stops before GNN / other E2E head)
  geom_post  — ``geom_only`` forward + activations + grid decode + built-in ``fit_lines`` (per tile)
  geom_post_chain — same as ``geom_post`` but ``fit_lines(..., skip_spline=True)`` (graph/BFS/merge only)
  geom_gnn   — full forward including ``e2e_mode=gnn`` head (use exp43-style config)

Each timed sample is one dataset image (1024×1024 tile).
Images are **preloaded** to GPU first; reported latencies are **forward-only** (wall-clock,
CUDA-synchronized), excluding DataLoader I/O.

Reports mean / median latency (ms) and FPS over N images.

Example (A100, one GPU)::

  cd CAPSTONE && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src YOLINO_IGNORE_DIRTY=1 \\
    python src/yolino/tools/benchmark_yolino_fps.py \\
      --scope geom_only \\
      --config configs/experiments/exp21_tile_finetune_sample_from_exp19_ep41_1024.yaml \\
      --dataset-root /home/work/caps_drone/yolino/Sample_YOLinO_tiled_1024 \\
      --checkpoint-dir ttpla_train_exp/log/checkpoints/exp21_tile_finetune_sample_from_exp19_ep41_e80 \\
      --num-images 100 --gpu

  # TTPLA full 1024×1024 val (all 218 tiles), geom + GNN, GPU 0
  cd CAPSTONE && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src YOLINO_IGNORE_DIRTY=1 \\
    python -m yolino.tools.benchmark_yolino_fps \\
      --scope geom_gnn \\
      --config configs/experiments/exp76_gnn_ttpla_full.yaml \\
      --dataset-root /home/work/caps_drone/yolino/ttpla_yolino_dataset_1024x1024 \\
      --checkpoint ttpla_train_exp/log/checkpoints/exp76_gnn_ttpla_full/best_model.pth \\
      --split val --all-images --gpu --gpu-id 0
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from yolino.dataset.dataset_factory import DatasetFactory
from yolino.grid.grid_factory import GridFactory
from yolino.model.model_factory import load_checkpoint
from yolino.eval.gnn_instance_extract import extract_post_cc_segment_groups
from yolino.postprocessing.geom_segments import geom_act_to_uv_lines
from yolino.postprocessing.line_fit import fit_lines
from yolino.runner.forward_runner import ForwardRunner
from yolino.tools.stitch_val_tile_predictions import pick_latest_checkpoint, resolve_line_fit_kwargs
from yolino.utils.enums import CoordinateSystem, Variables
from yolino.utils.general_setup import general_setup


def _unwrap(net: torch.nn.Module) -> torch.nn.Module:
    return net.module if hasattr(net, "module") else net


@torch.inference_mode()
def forward_geom_only(net: torch.nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Same path as :meth:`YolinoNet.forward` up to geom/embed heads (no ``e2e_head``)."""
    m = _unwrap(net)
    feats = m.backbone(images)
    if m.use_std:
        x = feats["P3"]
    else:
        x = feats[m.head_level]

    mode = m.feature_refine
    if mode == "none":
        x_geom, x_embed = x, x
    elif mode == "sa_embed_only":
        x_geom, x_embed = x, m.attention(x)
    elif mode == "sa_shared":
        x_ref = m.attention(x)
        x_geom, x_embed = x_ref, x_ref
    elif mode == "cbam_shared":
        x_ref = m.cbam(x)
        x_geom, x_embed = x_ref, x_ref
    else:
        raise ValueError("Unknown feature_refine=%r" % mode)

    if m.use_std:
        feat_std = m.std_head.unshuffle(x_geom)
        if m.std_skip_geom_head:
            hf, wf = feat_std.shape[-2:]
            geom_raw = torch.zeros(
                feat_std.shape[0], m.num_predictors * m.vars_train, hf, wf,
                device=feat_std.device, dtype=feat_std.dtype,
            )
            geom_pred = m.reshape_prediction(geom_raw)
            _, embed_raw = m.std_head.forward_from_feat(feat_std, geom=False, embed=True)
            if embed_raw is None:
                embed_raw = torch.zeros(
                    feat_std.shape[0], m.num_predictors * m.embed_dim, hf, wf,
                    device=feat_std.device, dtype=feat_std.dtype,
                )
            embed_pred = m.reshape_embedding(embed_raw)
        else:
            geom_raw, embed_raw = m.std_head.forward_from_feat(feat_std, geom=True, embed=True)
            geom_pred = m.reshape_prediction(geom_raw)
            embed_pred = m.reshape_embedding(embed_raw)
    else:
        geom_pred = m.reshape_prediction(m.yolo(x_geom))
        embed_pred = m.reshape_embedding(m.embed_head(x_embed))
    return geom_pred, embed_pred


@torch.inference_mode()
def forward_full(net: torch.nn.Module, images: torch.Tensor):
    """Full ``YolinoNet.forward`` (geom + embed + E2E/GNN when configured)."""
    return _unwrap(net)(images)


def _geom_preds_to_uv_lines(
    geom_preds_cpu: torch.Tensor,
    image_height: int,
    args,
    dataset,
    conf_thr: float,
    *,
    segment_source: str = "grid",
) -> np.ndarray:
    """``segment_source``: ``grid`` = GridFactory (legacy); ``tensor`` = direct MID_DIR decode."""
    if segment_source == "tensor":
        img_h = int(args.img_size[0])
        img_w = int(args.img_size[1])
        return geom_act_to_uv_lines(
            geom_preds_cpu,
            dataset.coords,
            tuple(args.grid_shape),
            float(args.scale),
            img_h,
            img_w,
            float(conf_thr),
        )
    if segment_source != "grid":
        raise ValueError("segment_source must be 'grid' or 'tensor', got %r" % segment_source)

    pred_grid, _ = GridFactory.get(
        geom_preds_cpu,
        [],
        CoordinateSystem.CELL_SPLIT,
        args=args,
        input_coords=dataset.coords,
        only_train_vars=True,
        anchors=dataset.anchors,
    )
    uv = np.asarray(
        pred_grid.get_image_lines(
            coords=dataset.coords,
            image_height=image_height,
            is_training_data=True,
        )
    )[0]
    if uv.ndim == 1:
        uv = np.expand_dims(uv, axis=0)
    if uv.shape[0] == 0:
        return uv
    conf_pos = dataset.coords.get_position_within_prediction(Variables.CONF)
    if conf_pos is not None and len(conf_pos) > 0:
        conf_i = int(np.asarray(conf_pos).ravel()[0])
        keep = uv[:, conf_i] >= conf_thr
        return uv[keep]
    return uv


def make_geom_post_fn(
    forward: ForwardRunner,
    args,
    dataset,
    line_fit_kw: dict,
    *,
    skip_spline: bool = False,
    segment_source: str = "grid",
):
    """Per-tile: GPU forward (with activations) + grid decode + ``fit_lines`` (CPU)."""

    def _run(images: torch.Tensor) -> list:
        ih = int(images.shape[-2])
        geom_preds, _, _, _ = forward(images, is_train=False, epoch=None)
        geom_cpu = geom_preds.detach().cpu()
        if geom_cpu.ndim == 3:
            geom_cpu = geom_cpu.unsqueeze(0)
        uv = _geom_preds_to_uv_lines(
            geom_cpu, ih, args, dataset, float(line_fit_kw["confidence"]),
            segment_source=segment_source,
        )
        if uv.ndim == 1:
            uv = np.expand_dims(uv, axis=0)
        if uv.ndim == 2:
            lines_uv = np.expand_dims(uv.astype(np.float32), axis=0)
        else:
            lines_uv = uv.astype(np.float32)
        if lines_uv.shape[1] == 0:
            return []
        return fit_lines(
            lines_uv=lines_uv,
            coords=dataset.coords,
            confidence_threshold=float(line_fit_kw["confidence"]),
            adjacency_threshold=float(line_fit_kw["adjacency_threshold"]),
            grid_shape=args.grid_shape,
            min_segments_for_polyline=args.min_segments_for_polyline,
            cell_size=args.cell_size,
            image=None,
            file_name="fps_bench",
            paths=args.paths,
            args=args,
            split=args.split,
            write_debug_images=False,
            spline_s=float(line_fit_kw["spline_s"]),
            angle_thr_deg=line_fit_kw.get("angle_thr_deg"),
            collinear_thr_px=line_fit_kw.get("collinear_thr_px"),
            second_pass_merge=bool(line_fit_kw.get("second_pass_merge", True)),
            second_pass_gap_px=line_fit_kw.get("second_pass_gap_px"),
            skip_spline=skip_spline,
        )

    return _run


def make_geom_post_cc_fn(
    forward: ForwardRunner,
    args,
    dataset,
    confidence: float,
    *,
    min_segments_for_cc: int = 1,
    seg_counts: list | None = None,
):
    """Per-tile: GPU forward + conf-filtered tensor decode + adjacency CC (pre-spline).

    Fair post baseline vs GNN grouping: same segment pool semantics as
    ``extract_post_cc_segment_groups`` (no smoothing / spline / second-pass merge).
  """

    def _run(images: torch.Tensor) -> list:
        ih = int(images.shape[-2])
        iw = int(images.shape[-1])
        geom_preds, _, _, _ = forward(images, is_train=False, epoch=None)
        geom_cpu = geom_preds.detach().cpu()
        if geom_cpu.ndim == 3:
            geom_cpu = geom_cpu.unsqueeze(0)
        uv = geom_act_to_uv_lines(
            geom_cpu,
            dataset.coords,
            tuple(args.grid_shape),
            float(args.scale),
            ih,
            iw,
            float(confidence),
        )
        if seg_counts is not None:
            seg_counts.append(int(uv.shape[0]))
        if uv.shape[0] == 0:
            return []
        prev_min = int(args.min_segments_for_polyline)
        args.min_segments_for_polyline = int(min_segments_for_cc)
        try:
            return extract_post_cc_segment_groups(uv, dataset.coords, args)
        finally:
            args.min_segments_for_polyline = prev_min

    return _run


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _preload_images(loader: DataLoader, num_images: int, device: torch.device) -> list[torch.Tensor]:
    batches: list[torch.Tensor] = []
    for data in loader:
        images = data[0]
        if images.shape[0] != 1:
            raise ValueError("Expected batch_size=1, got %d" % images.shape[0])
        batches.append(images.to(device, non_blocking=True))
        if len(batches) >= num_images:
            break
    if len(batches) < num_images:
        print(
            "[WARN] requested %d images but loader yielded %d" % (num_images, len(batches)),
            file=sys.stderr,
        )
    return batches


def _bench_forward(
    fn,
    batches: list[torch.Tensor],
    warmup: int,
    device: torch.device,
) -> list[float]:
    """Return per-image latencies in seconds (GPU-synchronized)."""
    # Warmup on first tensor (cycle if short dataset)
    n_w = max(1, warmup)
    for i in range(n_w):
        fn(batches[i % len(batches)])
    _sync(device)

    latencies: list[float] = []
    for images in batches:
        _sync(device)
        t0 = time.perf_counter()
        fn(images)
        _sync(device)
        latencies.append(time.perf_counter() - t0)
    return latencies


def _stats_from_latencies(scope: str, latencies: list[float]) -> dict:
    mean_s = statistics.mean(latencies)
    median_s = statistics.median(latencies)
    total_s = sum(latencies)
    out = {
        "scope": scope,
        "n_images": len(latencies),
        "mean_ms": mean_s * 1000.0,
        "median_ms": median_s * 1000.0,
        "fps_mean": 1.0 / mean_s,
        "fps_throughput": len(latencies) / total_s,
        "total_s": total_s,
    }
    if len(latencies) >= 2:
        out["std_ms"] = statistics.stdev(latencies) * 1000.0
    return out


def _report(scope: str, latencies: list[float], num_images: int) -> None:
    if not latencies:
        print("[ERROR] no timings collected", file=sys.stderr)
        return
    st = _stats_from_latencies(scope, latencies)
    print("")
    print("========== YOLinO FPS benchmark (%s) ==========" % scope)
    print("  images timed     : %d" % st["n_images"])
    print("  mean latency     : %.3f ms  (%.2f FPS per image)" % (st["mean_ms"], st["fps_mean"]))
    print("  median latency   : %.3f ms" % st["median_ms"])
    print("  throughput       : %.2f FPS  (N / sum(latency))" % st["fps_throughput"])
    print("  total forward    : %.3f s" % st["total_s"])
    if "std_ms" in st:
        print("  std latency      : %.3f ms" % st["std_ms"])
    print("==============================================")


def run_benchmark(
    *,
    scope: str,
    config: str,
    dataset_root: str,
    checkpoint: str | None = None,
    checkpoint_dir: str | None = None,
    split: str = "val",
    num_images: int = 100,
    all_images: bool = False,
    warmup: int = 20,
    gpu: bool = True,
    gpu_id: int = 0,
    loading_workers: int = 0,
    linefit_preset: str = "",
    segment_source: str = "grid",
    post_confidence: float | None = None,
    verbose: bool = True,
) -> dict:
    """Preload images, time forward-only latencies, return stats dict."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    default_cfg = os.path.join(project_root, "ttpla_train_exp", "default_params.yaml")
    dataset_root = os.path.abspath(os.path.expanduser(dataset_root))
    os.environ["DATASET_TTPLA"] = dataset_root

    alt = ["-c", config, "--root", project_root, "--dvc", project_root, "--batch_size", "1"]
    if gpu:
        alt += ["--gpu", "--gpu_id", str(int(gpu_id))]

    args = general_setup(
        "YOLinO FPS benchmark",
        task_type=None,
        config_file=config,
        ignore_cmd_args=True,
        alternative_args=alt,
        default_config=default_cfg,
    )
    args.dvc = dataset_root
    args.batch_size = 1
    args.loading_workers = int(loading_workers)

    if gpu and torch.cuda.is_available():
        gid = int(gpu_id)
        args.gpu = True
        args.gpu_id = gid
        args.cuda = "cuda:%d" % gid
        torch.cuda.set_device(gid)
        device = torch.device(args.cuda)
    else:
        args.gpu = False
        args.cuda = "cpu"
        device = torch.device("cpu")

    if checkpoint and str(checkpoint).strip():
        args.explicit_model = os.path.abspath(os.path.expanduser(checkpoint))
    elif checkpoint_dir:
        args.explicit_model = pick_latest_checkpoint(os.path.abspath(os.path.expanduser(checkpoint_dir)))
    else:
        raise ValueError("checkpoint or checkpoint_dir required")
    args.paths.pretrain_model = args.explicit_model

    coords = DatasetFactory.get_coords(split, args)
    model, _, _ = load_checkpoint(args, coords, allow_failure=False, load_best=False)
    model.eval()
    model.to(device)
    forward_runner = ForwardRunner(args, preloaded_model=model, coords=coords)

    dataset, _ = DatasetFactory.get(
        args.dataset, only_available=True, split=split, args=args,
        shuffle=False, augment=False, ignore_duplicates=False,
    )
    n_request = len(dataset) if all_images else int(num_images)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False,
        num_workers=args.loading_workers, pin_memory=device.type == "cuda",
    )
    batches = _preload_images(loader, n_request, device)
    if not batches:
        raise RuntimeError("no images loaded")

    net = _unwrap(model)
    if verbose:
        print("[INFO] device=%s scope=%s checkpoint=%s" % (device, scope, args.explicit_model))
        print("[INFO] dataset=%s split=%s (n=%d timed=%d)" % (dataset_root, split, len(dataset), len(batches)))

    seg_counts: list[int] = []
    if scope == "geom_only":
        fn = lambda im: forward_geom_only(model, im)
    elif scope == "geom_post_cc":
        conf = float(post_confidence if post_confidence is not None else args.confidence)
        fn = make_geom_post_cc_fn(
            forward_runner, args, dataset, conf, seg_counts=seg_counts,
        )
    elif scope in ("geom_post", "geom_post_chain"):
        line_fit_kw = resolve_line_fit_kwargs(str(linefit_preset).strip() or None, args)
        fn = make_geom_post_fn(
            forward_runner, args, dataset, line_fit_kw,
            skip_spline=(scope == "geom_post_chain"),
            segment_source=segment_source,
        )
    elif scope == "geom_gnn":
        if net.e2e_head is None:
            raise RuntimeError("scope=geom_gnn but model has no e2e_head")
        fn = lambda im: forward_runner(im, is_train=False, epoch=None)
    else:
        raise ValueError("unknown scope %r" % scope)

    latencies = _bench_forward(fn, batches, int(warmup), device)
    if verbose:
        _report(scope, latencies, len(batches))
    stats = _stats_from_latencies(scope, latencies)
    if scope == "geom_post_cc" and seg_counts:
        stats["mean_n_segments"] = float(statistics.mean(seg_counts))
        stats["median_n_segments"] = float(statistics.median(seg_counts))
        if verbose:
            print("  mean segments/img : %.1f (conf-filtered tensor decode)" % stats["mean_n_segments"])
    stats.update({
        "config": config,
        "checkpoint": args.explicit_model,
        "dataset_root": dataset_root,
        "split": split,
        "warmup": warmup,
        "device": str(device),
    })
    return stats


def main() -> None:
    os.environ.setdefault("YOLINO_IGNORE_DIRTY", "1")
    ap = argparse.ArgumentParser(description="YOLinO geom / geom+GNN FPS on one GPU, batch=1.")
    ap.add_argument(
        "--scope",
        choices=["geom_only", "geom_post", "geom_post_chain", "geom_post_cc", "geom_gnn"],
        required=True,
        help="geom_only: raw geom+embed forward. geom_post: forward+activations+grid+fit_lines per tile. "
             "geom_post_chain: geom_post without B-spline (adjacency/BFS/merge only). "
             "geom_post_cc: forward + conf-filtered tensor segments + adjacency CC only (pre-spline, fair vs GNN). "
             "geom_gnn: full forward incl. GNN head.",
    )
    ap.add_argument(
        "--linefit-preset",
        default="",
        help="fit_lines preset (e.g. adj150_s0p01_conf0p6). Empty = yaml confidence/adjacency + spline_s=0.05.",
    )
    ap.add_argument(
        "--segment-source",
        choices=["grid", "tensor"],
        default="grid",
        help="geom_post*: 'grid' = GridFactory+get_image_lines (legacy); "
             "'tensor' = direct MID_DIR geom_act→segments (no grid decode).",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--num-images", type=int, default=100,
                    help="Number of images to time (ignored when --all-images).")
    ap.add_argument("--all-images", action="store_true",
                    help="Time every image in --split (e.g. full TTPLA val).")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument(
        "--post-confidence",
        type=float,
        default=None,
        help="Confidence cut for geom_post_cc (default: yaml confidence).",
    )
    ap.add_argument("--gpu", action="store_true", help="Use cuda:0 (respects CUDA_VISIBLE_DEVICES).")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--loading-workers", type=int, default=0)
    cli = ap.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    config = cli.config if os.path.isabs(cli.config) else os.path.join(project_root, cli.config)

    run_benchmark(
        scope=cli.scope,
        config=config,
        dataset_root=cli.dataset_root,
        checkpoint=cli.checkpoint,
        checkpoint_dir=cli.checkpoint_dir,
        split=cli.split,
        num_images=cli.num_images,
        all_images=cli.all_images,
        warmup=cli.warmup,
        gpu=cli.gpu,
        gpu_id=cli.gpu_id,
        loading_workers=cli.loading_workers,
        linefit_preset=cli.linefit_preset,
        segment_source=cli.segment_source,
        post_confidence=cli.post_confidence,
        verbose=True,
    )


if __name__ == "__main__":
    main()
