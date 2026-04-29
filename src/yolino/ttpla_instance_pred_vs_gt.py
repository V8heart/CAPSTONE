# TTPLA: load ep11 (or any) best checkpoint, run inference, cluster line-segment embeddings
# (epsilon-graph / DBSCAN-like), and save a side-by-side PNG: prediction | GT (per-instance colors).
#
# Example:
#   source venv/bin/activate
#   export DATASET_TTPLA=/path/to/TTPLA_YOLinO_Dataset
#   export YOLINO_IGNORE_DIRTY=1
#   cd YOLinO && PYTHONPATH=src python -m yolino.ttpla_instance_pred_vs_gt \
#     --dvc ttpla_train_exp --root . --run_name ep11_pred_vs_gt --split test --gpu \
#     --explicit_model log/checkpoints/ep11_embed_v1/best_model.pth \
#     --max_images 5 --dbscan_eps 0.35 --dbscan_min_samples 1 \
#     --viz_confidence 0.75
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from tqdm import tqdm

from yolino.dataset.dataset_factory import DatasetFactory
from yolino.instance_embed_cluster_infer import (
    cluster_embeddings_dbscan_like,
    draw_overlay,
    segment_lines_and_embeddings,
    _color_for_cluster,
)
from yolino.model.activations import get_activations
from yolino.model.model_factory import load_checkpoint
from yolino.utils.enums import Dataset, TaskType
from yolino.utils.general_setup import general_setup


def parse_extra():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--max_images", type=int, default=5)
    p.add_argument("--dbscan_eps", type=float, default=0.35)
    p.add_argument("--dbscan_min_samples", type=int, default=1)
    p.add_argument("--l2_normalize", type=str, default="true")
    p.add_argument("--out_subdir", type=str, default="",
                   help="Optional subfolder under debug/<run_name>/. Empty = use debug/<run_name> only.")
    p.add_argument("--line_width", type=int, default=2)
    p.add_argument(
        "--viz_confidence",
        type=float,
        default=None,
        help="Min confidence to keep a line segment in the prediction panel (overrides params.yaml / --confidence for this script only). Example: 0.75",
    )
    return p


def str_to_bool(s):
    return str(s).lower() in ("1", "true", "yes", "y")


def tensor_to_rgb_uint8(img_t):
    if img_t.shape[0] == 3:
        im = img_t.numpy().transpose(1, 2, 0)
    else:
        im = img_t.numpy()
    if im.max() <= 1.0 + 1e-6:
        im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    else:
        im = np.clip(im, 0, 255).astype(np.uint8)
    if im.ndim == 2:
        im = np.stack([im, im, im], axis=-1)
    return im


def draw_gt_polylines(rgb, polylines, width=2):
    """GT: list/array of polylines; instance index = enumerate order (1..N), same spirit as visualize_ttpla_gt_instances."""
    im = Image.fromarray(rgb)
    dr = ImageDraw.Draw(im)
    for inst_idx, poly in enumerate(polylines, start=1):
        pts = np.asarray(poly, dtype=np.float32)
        if len(pts) < 2:
            continue
        c = _color_for_cluster(inst_idx - 1)
        for k in range(len(pts) - 1):
            x0, y0 = float(pts[k, 0]), float(pts[k, 1])
            x1, y1 = float(pts[k + 1, 0]), float(pts[k + 1, 1])
            dr.line((x0, y0, x1, y1), fill=c, width=width)
        x0, y0 = int(round(pts[0, 0])), int(round(pts[0, 1]))
        try:
            font = ImageFont.load_default()
            dr.text((x0, max(0, y0 - 10)), str(inst_idx), fill=c, font=font)
        except Exception:
            dr.text((x0, max(0, y0 - 10)), str(inst_idx), fill=c)
    return np.asarray(im)


def hstack_panels(left_rgb, right_rgb, titles=("Pred (embed cluster)", "GT instances")):
    h = max(left_rgb.shape[0], right_rgb.shape[0])
    w1, w2 = left_rgb.shape[1], right_rgb.shape[1]
    banner_h = 36
    canvas = np.ones((h + banner_h, w1 + w2, 3), dtype=np.uint8) * 255
    canvas[banner_h : banner_h + h, 0:w1] = left_rgb
    canvas[banner_h : banner_h + h, w1 : w1 + w2] = right_rgb
    im = Image.fromarray(canvas)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    dr.text((8, 8), titles[0], fill=(0, 0, 0), font=font)
    dr.text((w1 + 8, 8), titles[1], fill=(0, 0, 0), font=font)
    return np.asarray(im)


if __name__ == "__main__":
    extra_parser = parse_extra()
    extra_args, remaining = extra_parser.parse_known_args()

    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _default_cfg = os.path.join(_root, "ttpla_train_exp", "default_params.yaml")
    _params_cfg = os.path.join(_root, "ttpla_train_exp", "params.yaml")
    if not os.path.isfile(_params_cfg):
        _params_cfg = os.path.join(_root, "params.yaml")
    if not os.path.isfile(_default_cfg):
        _default_cfg = os.path.join(_root, "res", "default_params.yaml")

    args = general_setup(
        "TTPLA pred vs GT instance viz",
        config_file=_params_cfg,
        default_config=_default_cfg,
        task_type=TaskType.TEST,
        ignore_cmd_args=True,
        alternative_args=remaining,
    )

    if args.dataset != Dataset.TTPLA:
        raise SystemExit("Use dataset ttpla in params.yaml")

    ds_path = os.getenv("DATASET_TTPLA")
    if not ds_path:
        raise SystemExit("Set DATASET_TTPLA to your TTPLA root (e.g. .../TTPLA_YOLinO_Dataset)")

    l2n = str_to_bool(extra_args.l2_normalize)
    out_root = (
        os.path.join(args.paths.debug_folder, extra_args.out_subdir)
        if extra_args.out_subdir
        else args.paths.debug_folder
    )
    os.makedirs(out_root, exist_ok=True)

    dataset, loader = DatasetFactory.get(
        args.dataset, only_available=True, split=args.split, args=args,
        shuffle=False, augment=False, load_full=True,
    )
    coords = dataset.coords

    model, _, _ = load_checkpoint(args, coords, allow_failure=False, load_best=False)
    model.eval()
    activations = get_activations(args.activations, coords=coords, linerep=args.linerep)
    device = args.cuda
    model = model.to(device)

    conf_thr = (
        float(extra_args.viz_confidence)
        if extra_args.viz_confidence is not None
        else float(args.confidence)
    )
    print("Prediction panel confidence threshold: %s (params default was %s)" % (conf_thr, args.confidence))
    processed = 0
    label_dir = os.path.join(ds_path, "labels", args.split)

    summary = []

    for batch_idx, data in enumerate(tqdm(loader, desc="pred_vs_gt")):
        if extra_args.max_images > 0 and processed >= extra_args.max_images:
            break

        images, _, filenames, _, _ = data
        images = images.to(device)

        with torch.no_grad():
            logits = model(images)
            geom_preds, embed_preds = activations(logits)

        bs = images.shape[0]
        for bi in range(bs):
            if extra_args.max_images > 0 and processed >= extra_args.max_images:
                break

            name = filenames[bi]
            gp = geom_preds[bi : bi + 1]
            ep = embed_preds[bi : bi + 1]
            if l2n:
                ep = F.normalize(ep, p=2, dim=-1)

            segs, emb_mat = segment_lines_and_embeddings(gp, ep, args, coords, conf_thr)
            rgb = tensor_to_rgb_uint8(images[bi].detach().cpu())

            label_path = os.path.join(label_dir, name + ".npy")
            if not os.path.isfile(label_path):
                polylines = []
            else:
                polylines = np.load(label_path, allow_pickle=True)

            gt_panel = draw_gt_polylines(rgb.copy(), polylines, width=extra_args.line_width)

            if len(segs) == 0:
                pred_panel = rgb.copy()
                lab = np.array([])
            else:
                lab = cluster_embeddings_dbscan_like(
                    emb_mat, eps=extra_args.dbscan_eps, min_samples=extra_args.dbscan_min_samples
                )
                pred_panel = draw_overlay(rgb.copy(), segs, lab, width=extra_args.line_width)

            combined = hstack_panels(pred_panel, gt_panel)
            out_png = os.path.join(out_root, name + "_pred_vs_gt.png")
            Image.fromarray(combined).save(out_png)

            n_gt = len(polylines) if hasattr(polylines, "__len__") else 0
            n_clu = int(len(set(lab.tolist())) - (1 if -1 in lab else 0)) if len(lab) else 0
            rec = {
                "file": name,
                "num_segments_pred": len(segs),
                "num_gt_instances": n_gt,
                "n_clusters_pred": n_clu,
                "confidence_threshold": conf_thr,
                "png": out_png,
            }
            with open(os.path.join(out_root, name + "_pred_vs_gt.json"), "w") as f:
                json.dump(rec, f, indent=2)
            summary.append(rec)
            processed += 1

    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump({"items": summary}, f, indent=2)

    print("Saved %d PNGs under file://%s" % (len(summary), out_root))
