#!/usr/bin/env python3
"""Fast geom (exp19) vs GNN (exp55) on a few val images — for presentation."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Quiet YOLinO before imports
os.environ.setdefault("YOLINO_IGNORE_DIRTY", "1")
logging.basicConfig(level=logging.WARNING)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PIXEL = os.path.join(BASE, "eval_pixel_f1")
if EVAL_PIXEL not in sys.path:
    sys.path.insert(0, EVAL_PIXEL)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_pixel_f1 as epf  # noqa: E402
from eval_pixel_f1 import (  # noqa: E402
    DATASET_ROOT,
    EXP19_CKPT,
    EXP19_CONFIG,
    EXP55_CKPT,
    EXP55_CONFIG,
    GEOM_CONF,
    GNN_EDGE_THRESH,
    GNN_NODE_CONF,
    SPLIT,
    VIZ_THICKNESS,
    build_geom_pkl,
    build_gnn_pkl,
    evaluate_gt_b_vs_pred,
    load_records as load_pixel_records,
    run_geom_predictions,
    run_gnn_predictions,
    save_viz,
)
import torch  # noqa: E402
from isq_core import (  # noqa: E402
    compute_isq_image,
    gt_polylines_to_segments,
    segment_tuples_to_pred_polylines,
    summarize_isq,
)
from yolino.visualize_ttpla_gt_instances import load_ttpla_label_with_ids  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS_JSON = os.path.join(OUT_DIR, "quick_compare.json")
EVAL_ISQ_DIR = os.path.dirname(os.path.abspath(__file__))
THICKNESS = 2


def _pkl_paths(tag: str) -> tuple[str, str]:
    suffix = "" if not tag else "_%s" % tag
    return (
        os.path.join(EVAL_ISQ_DIR, "pred_geom%s.pkl" % suffix),
        os.path.join(EVAL_ISQ_DIR, "pred_gnn%s.pkl" % suffix),
    )


def _load_isq_records(dataset_root: str, stems: list[str]) -> dict:
    img_dir = os.path.join(dataset_root, "images", SPLIT)
    label_dir = os.path.join(dataset_root, "labels", SPLIT)
    records = {}
    for stem in stems:
        img = __import__("cv2").imread(os.path.join(img_dir, stem + ".png"))
        h, w = img.shape[:2]
        polylines, instance_ids, _ = load_ttpla_label_with_ids(
            os.path.join(label_dir, stem + ".npy")
        )
        records[stem] = (h, w, polylines, list(instance_ids))
    return records


def _eval_isq_one(stem: str, meta: tuple, pred_segs: list) -> dict:
    h, w, polylines, instance_ids = meta
    gt_segments = gt_polylines_to_segments(
        polylines, instance_ids, cell_size=32, image_size=max(h, w)
    )
    pred_polylines = segment_tuples_to_pred_polylines(pred_segs)
    n_pred_segs = sum(len(p) for p in pred_polylines)
    m = compute_isq_image(gt_segments, pred_polylines)
    m["stem"] = stem
    m["n_gt_segments"] = len(gt_segments)
    m["n_pred_segments"] = n_pred_segs
    return m


def _eval_isq(records_meta: dict, pred_by_stem: dict, workers: int = 1) -> dict:
    stems = sorted(records_meta)
    if workers <= 1:
        per_image = [
            _eval_isq_one(stem, records_meta[stem], pred_by_stem[stem]) for stem in stems
        ]
    else:
        per_image: list = []
        tasks = [(stem, records_meta[stem], pred_by_stem[stem]) for stem in stems]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_eval_isq_one, *task): task[0] for task in tasks}
            done = 0
            for fut in as_completed(futures):
                per_image.append(fut.result())
                done += 1
                if done % 50 == 0 or done == len(stems):
                    print("[ISQ] %d / %d images" % (done, len(stems)), flush=True)
        per_image.sort(key=lambda m: m["stem"])
    return {"summary": summarize_isq(per_image), "per_image": per_image}


def _val_stems(dataset_root: str, max_images: int) -> list[str]:
    img_dir = os.path.join(dataset_root, "images", SPLIT)
    names = sorted(
        os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith(".png")
    )
    if max_images > 0:
        names = names[:max_images]
    return names


def _setup_run_subset(
    config_path: str, checkpoint_path: str, dataset_root: str, use_gpu: bool, stems: list[str]
):
    """Like eval_pixel_f1.setup_run but only loads ``stems`` (avoids 65-image dataloader loop)."""
    os.environ["DATASET_TTPLA"] = os.path.abspath(dataset_root)
    os.environ.setdefault("YOLINO_IGNORE_DIRTY", "1")
    dvc = os.path.join(epf.CAPSTONE_ROOT, "ttpla_train_exp")
    default_cfg = os.path.join(dvc, "default_params.yaml")
    ckpt = os.path.abspath(checkpoint_path)
    alt = [
        "-c",
        os.path.abspath(config_path),
        "--root",
        epf.CAPSTONE_ROOT,
        "--dvc",
        dvc,
        "--split",
        SPLIT,
        "--explicit_model",
        ckpt,
        "--run_name",
        "eval_quick_compare",
        "--loggers",
        "file",
        "--batch_size",
        "1",
        "--loading_workers",
        "0",
        "--explicit",
        *stems,
    ]
    if use_gpu:
        alt.extend(["--gpu", "--gpu_id", "0"])
    from yolino.runner.forward_runner import ForwardRunner  # noqa: E402
    from yolino.utils.enums import TaskType  # noqa: E402
    from yolino.utils.general_setup import general_setup  # noqa: E402
    from torch.utils.data import DataLoader  # noqa: E402
    from yolino.dataset.dataset_factory import DatasetFactory  # noqa: E402
    from yolino.model.model_factory import load_checkpoint  # noqa: E402

    args = general_setup(
        "eval_quick_compare",
        task_type=TaskType.TEST,
        config_file=os.path.abspath(config_path),
        ignore_cmd_args=True,
        alternative_args=alt,
        default_config=default_cfg,
    )
    args.explicit_model = ckpt
    args.paths.pretrain_model = ckpt
    if use_gpu and torch.cuda.is_available():
        args.gpu = True
        args.cuda = "cuda:0"
        device = torch.device("cuda:0")
    else:
        args.gpu = False
        args.cuda = "cpu"
        device = torch.device("cpu")
    coords = DatasetFactory.get_coords(SPLIT, args)
    model, _, _ = load_checkpoint(args, coords, allow_failure=False, load_best=False)
    model.eval().to(device)
    net = model.module if hasattr(model, "module") else model
    if getattr(net, "e2e_head", None) is not None and hasattr(net.e2e_head, "node_conf_thresh"):
        net.e2e_head.node_conf_thresh = float(GNN_NODE_CONF)
    forward = ForwardRunner(args, preloaded_model=model, coords=coords)
    dataset, _ = DatasetFactory.get(
        args.dataset,
        only_available=True,
        split=SPLIT,
        args=args,
        shuffle=False,
        augment=False,
        ignore_duplicates=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    return args, forward, dataset, loader, device


def run_inference(
    stems: list[str],
    use_gpu: bool,
    dataset_root: str,
    geom_config: str,
    geom_ckpt: str,
    gnn_config: str,
    gnn_ckpt: str,
    *,
    pred_geom_pkl: str,
    pred_gnn_pkl: str,
) -> tuple[dict, dict, object]:
    import pickle

    records = load_pixel_records(dataset_root, stems)
    gnn_raw: dict = {}

    t0 = time.time()
    print("[1/2] geom (%d images, %s)..." % (len(stems), "cuda" if use_gpu else "cpu"))
    geom_args, fwd19, ds19, loader19, dev19 = _setup_run_subset(
        geom_config, geom_ckpt, dataset_root, use_gpu, stems
    )
    run_geom_predictions(fwd19, ds19, loader19, dev19, records)
    del fwd19, ds19, loader19
    if use_gpu:
        torch.cuda.empty_cache()
    print("      geom done in %.1fs" % (time.time() - t0))

    t1 = time.time()
    print("[2/2] GNN (%d images)..." % len(stems))
    _a55, fwd55, ds55, loader55, dev55 = _setup_run_subset(
        gnn_config, gnn_ckpt, dataset_root, use_gpu, stems
    )
    run_gnn_predictions(fwd55, ds55, loader55, dev55, records, gnn_raw)
    del fwd55, _a55, ds55, loader55
    print("      GNN done in %.1fs" % (time.time() - t1))

    geom_pkl = build_geom_pkl(
        records,
        adjacency_threshold=float(geom_args.adjacency_threshold),
        min_segments_for_polyline=int(geom_args.min_segments_for_polyline),
    )
    gnn_pkl = build_gnn_pkl(records, gnn_raw)
    with open(pred_geom_pkl, "wb") as f:
        pickle.dump(geom_pkl, f)
    with open(pred_gnn_pkl, "wb") as f:
        pickle.dump(gnn_pkl, f)
    print("[OK] Cached predictions:", pred_geom_pkl, pred_gnn_pkl)
    return records, gnn_raw, geom_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick geom vs GNN val comparison")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split (default: val)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limit images (0 = all in split)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="ISQ post-process worker processes (0 = auto, min(cpu_count-1, 16))",
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip mask visualization")
    parser.add_argument(
        "--from-pkl",
        action="store_true",
        help="Skip model inference; use cached prediction pickles",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Results JSON path (default: results/quick_compare.json)",
    )
    parser.add_argument(
        "--pkl-tag",
        type=str,
        default="quick",
        help="Pickle suffix: pred_geom_<tag>.pkl (use 'full' for all-val run)",
    )
    parser.add_argument(
        "--geom-config",
        type=str,
        default=EXP19_CONFIG,
        help="Geom experiment config path (default: exp19 config)",
    )
    parser.add_argument(
        "--geom-ckpt",
        type=str,
        default=EXP19_CKPT,
        help="Geom checkpoint path (default: exp19 best checkpoint)",
    )
    parser.add_argument(
        "--gnn-config",
        type=str,
        default=EXP55_CONFIG,
        help="GNN experiment config path (default: exp55 config)",
    )
    parser.add_argument(
        "--gnn-ckpt",
        type=str,
        default=EXP55_CKPT,
        help="GNN checkpoint path (default: exp55 best checkpoint)",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DATASET_ROOT,
        help="TTPLA dataset root for GT + inference (default: smallset, same as eval_pixel_f1)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use CUDA GPU for inference (error if unavailable)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even when CUDA is available",
    )
    args = parser.parse_args()

    global SPLIT
    SPLIT = args.split
    epf.SPLIT = args.split

    workers = args.workers
    if workers <= 0:
        workers = max(1, min((os.cpu_count() or 4) - 1, 16))

    results_json = os.path.abspath(args.output_json or RESULTS_JSON)
    pred_geom_pkl, pred_gnn_pkl = _pkl_paths(args.pkl_tag)
    geom_config = os.path.abspath(args.geom_config)
    geom_ckpt = os.path.abspath(args.geom_ckpt)
    gnn_config = os.path.abspath(args.gnn_config)
    gnn_ckpt = os.path.abspath(args.gnn_ckpt)
    dataset_root = os.path.abspath(args.dataset_root)

    if args.cpu and args.use_gpu:
        raise ValueError("Cannot specify both --cpu and --use-gpu")
    if args.cpu:
        use_gpu = False
    elif args.use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("--use-gpu requested but torch.cuda.is_available() is False")
        use_gpu = True
    else:
        use_gpu = torch.cuda.is_available()
    if use_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        print("[INFO] device: cuda:0 (%s)" % gpu_name)
    else:
        print("[INFO] device: cpu (expect ~1–2 min/image for GNN)")
    for p in (geom_config, geom_ckpt, gnn_config, gnn_ckpt, dataset_root):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    os.makedirs(OUT_DIR, exist_ok=True)
    stems = _val_stems(dataset_root, args.max_images)
    print("[INFO] dataset_root (GT + inference):", dataset_root)
    print("[INFO] %s images:" % SPLIT, len(stems))
    print("[INFO] ISQ workers:", workers)
    print(
        "[INFO] All models use DATASET_TTPLA=%s; yaml dataset_ttpla is ignored when set."
        % dataset_root
    )
    print("[INFO] output:", results_json)
    print("[INFO] pickles:", pred_geom_pkl, pred_gnn_pkl)
    print("[INFO] geom config:", geom_config)
    print("[INFO] geom ckpt:", geom_ckpt)
    print("[INFO] gnn config:", gnn_config)
    print("[INFO] gnn ckpt:", gnn_ckpt)

    import pickle

    if args.from_pkl:
        if not (os.path.isfile(pred_geom_pkl) and os.path.isfile(pred_gnn_pkl)):
            raise FileNotFoundError(
                "Missing %s or %s — run without --from-pkl first." % (pred_geom_pkl, pred_gnn_pkl)
            )
        with open(pred_geom_pkl, "rb") as f:
            pred_geom = pickle.load(f)
        with open(pred_gnn_pkl, "rb") as f:
            pred_gnn = pickle.load(f)
        for stem in stems:
            if stem not in pred_geom or stem not in pred_gnn:
                raise ValueError("Pickles missing stem %s" % stem)
        px_geom = None
        px_gnn = None
        records_px = None
        print("[INFO] Skipped inference (--from-pkl); ISQ only")
    else:
        records_px, _, _geom_args = run_inference(
            stems,
            use_gpu,
            dataset_root,
            geom_config,
            geom_ckpt,
            gnn_config,
            gnn_ckpt,
            pred_geom_pkl=pred_geom_pkl,
            pred_gnn_pkl=pred_gnn_pkl,
        )
        px_geom = evaluate_gt_b_vs_pred(records_px, "geom", THICKNESS)
        px_gnn = evaluate_gt_b_vs_pred(records_px, "gnn", THICKNESS)
        with open(pred_geom_pkl, "rb") as f:
            pred_geom = pickle.load(f)
        with open(pred_gnn_pkl, "rb") as f:
            pred_gnn = pickle.load(f)
    isq_meta = _load_isq_records(dataset_root, stems)
    t_isq = time.time()
    print("[INFO] ISQ geom (%d workers)..." % workers)
    isq_geom = _eval_isq(isq_meta, pred_geom, workers=workers)
    print("[INFO] ISQ GNN (%d workers)..." % workers)
    isq_gnn = _eval_isq(isq_meta, pred_gnn, workers=workers)
    print("[INFO] ISQ done in %.1fs" % (time.time() - t_isq))

    if not args.no_viz and records_px is not None:
        viz_dir = os.path.join(OUT_DIR, "quick_viz")
        epf.VIZ_DIR = viz_dir
        os.makedirs(viz_dir, exist_ok=True)
        save_viz(records_px, stems, thickness=VIZ_THICKNESS)
        print("[INFO] viz ->", viz_dir)

    results = {
        "settings": {
            "dataset_root": dataset_root,
            "split": SPLIT,
            "n_images": len(stems),
            "stems": stems,
            "pixel_line_thickness": THICKNESS,
            "geom_confidence": GEOM_CONF,
            "gnn_edge_thresh": GNN_EDGE_THRESH,
            "gnn_node_conf": GNN_NODE_CONF,
            "geom_config": geom_config,
            "geom_checkpoint": geom_ckpt,
            "gnn_config": gnn_config,
            "gnn_checkpoint": gnn_ckpt,
            "device": ("cuda:0" if use_gpu else "cpu"),
            "use_gpu_flag": bool(args.use_gpu),
            "cpu_flag": bool(args.cpu),
            "from_pkl": bool(args.from_pkl),
            "isq_workers": workers,
            "pkl_tag": args.pkl_tag,
            "pred_geom_pkl": pred_geom_pkl,
            "pred_gnn_pkl": pred_gnn_pkl,
            "results_json": results_json,
        },
        "isq": {
            "Pred-1_geom_exp19": isq_geom,
            "Pred-2_gnn_exp55": isq_gnn,
        },
    }
    if px_geom is not None and px_gnn is not None:
        results["pixel_f1_gtB"] = {
            "Pred-1_geom_exp19": px_geom,
            "Pred-2_gnn_exp55": px_gnn,
        }

    if px_geom is not None:
        print("\n=== Pixel F1 (GT-B, line thickness=%d) ===" % THICKNESS)
        for label, block in (
            ("Pred-1 geom (exp19)", px_geom),
            ("Pred-2 GNN (exp55)", px_gnn),
        ):
            s = block["summary"]
            print(
                "  %-22s  P=%.4f  R=%.4f  F1=%.4f  IoU=%.4f  (n=%d)"
                % (label, s["precision"], s["recall"], s["f1"], s["iou"], s["n_images"])
            )

    if len(stems) <= 20:
        print("\n=== GT vs pred segment counts (per image) ===")
        for stem in stems:
            row_g = next(x for x in isq_geom["per_image"] if x["stem"] == stem)
            row_n = next(x for x in isq_gnn["per_image"] if x["stem"] == stem)
            print(
                "  %s  GT=%d  pred_geom=%d  pred_gnn=%d"
                % (stem, row_g["n_gt_segments"], row_g["n_pred_segments"], row_n["n_pred_segments"])
            )

    print("\n=== ISQ (grid segment matching) ===")
    for label, block in (
        ("Pred-1 geom (exp19)", isq_geom),
        ("Pred-2 GNN (exp55)", isq_gnn),
    ):
        s = block["summary"]
        print(
            "  %-22s  P=%.4f  R=%.4f  F1=%.4f  OS/img=%.2f  UM/img=%.2f  (OS_total=%d)"
            % (
                label,
                s["precision"],
                s["recall"],
                s["f1"],
                s["over_split"],
                s["under_merge"],
                s.get("over_split_count", 0),
            )
        )

    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n[OK] Wrote", results_json)


if __name__ == "__main__":
    main()
