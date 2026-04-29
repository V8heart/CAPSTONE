# Copyright 2023 Karlsruhe Institute of Technology, Institute for Measurement
# and Control Systems
#
# Inference: line segments + decoupled instance embeddings, then cluster embeddings
# (DBSCAN) so each segment gets a predicted instance id. This is NOT geometric NMS;
# NMS in YOLinO is DBSCAN in UV space — here we group by embedding similarity.
#
# Usage (example):
#   export DATASET_TTPLA=/path/to/TTPLA_YOLinO_Dataset
#   export YOLINO_IGNORE_DIRTY=1   # optional: skip interactive prompt if git working tree is dirty
#   cd YOLinO && PYTHONPATH=src python -m yolino.instance_embed_cluster_infer \
#     -c ttpla_train_exp/params.yaml --dvc ttpla_train_exp --root . \
#     --run_name embed_cluster_demo --split test --gpu \
#     --explicit_model log/checkpoints/ep11_embed_v1/best_model.pth \
#     --max_images 10 --dbscan_eps 0.35 --dbscan_min_samples 1
import argparse
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from tqdm import tqdm

from yolino.dataset.dataset_factory import DatasetFactory
from yolino.grid.predictor import Predictor
from yolino.model.activations import get_activations
from yolino.model.model_factory import load_checkpoint
from yolino.utils.enums import Dataset, LINE, TaskType, Variables
from yolino.utils.general_setup import general_setup


def parse_extra():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--max_images", type=int, default=10, help="Max images to process from the split.")
    p.add_argument("--dbscan_eps", type=float, default=0.35,
                   help="DBSCAN eps on L2-normalized embeddings (euclidean distance).")
    p.add_argument("--dbscan_min_samples", type=int, default=1, help="DBSCAN min_samples.")
    p.add_argument("--l2_normalize", type=str, default="true", help="L2-normalize embedding vectors before clustering.")
    p.add_argument("--out_subdir", type=str, default="instance_embed_cluster",
                   help="Under debug/<run_name>/ this folder receives PNG + JSON.")
    p.add_argument(
        "--viz_confidence",
        type=float,
        default=None,
        help="Override --confidence for filtering segments (viz only).",
    )
    return p


def str_to_bool(s):
    return s.lower() in ("1", "true", "yes", "y")


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def cluster_embeddings_dbscan_like(emb: np.ndarray, eps: float, min_samples: int):
    """
    DBSCAN-style clustering without sklearn: build epsilon-neighborhood graph, transitive closure via union-find.
    emb: (n, d) row vectors (typically L2-normalized). Metric: Euclidean distance.
    Returns labels (int), -1 = noise (component size < min_samples).
    """
    n = emb.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    if n == 1:
        return np.array([0], dtype=np.int64)

    # Pairwise Euclidean distances
    diff = emb[:, None, :] - emb[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))

    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] <= eps:
                uf.union(i, j)

    roots = {}
    next_id = 0
    raw = np.empty(n, dtype=np.int64)
    for i in range(n):
        r = uf.find(i)
        if r not in roots:
            roots[r] = next_id
            next_id += 1
        raw[i] = roots[r]

    sizes = np.bincount(raw, minlength=next_id)
    labels = raw.copy()
    for i in range(n):
        if sizes[labels[i]] < min_samples:
            labels[i] = -1

    # Renumber non-noise labels to 0..K-1
    uniq = sorted(set(labels.tolist()) - {-1})
    remap = {u: k for k, u in enumerate(uniq)}
    out = labels.copy()
    for i in range(n):
        if out[i] == -1:
            continue
        out[i] = remap[out[i]]
    return out


def segment_lines_and_embeddings(geom_preds, embed_preds, args, coords, conf_threshold):
    """Iterate (cell, predictor) like GridFactory; return segment endpoints for drawing.

    YOLinO ``Grid.get_image_lines`` stores geometry as
    ``[v_term, h_term, v_term, h_term]`` where ``v = row * scale`` (vertical / image y)
    and ``h = col * scale`` (horizontal / image x). See ``grid.py`` (training_data branch).

    PIL / numpy image convention is ``(x, y)`` = (column direction, row direction), so we output
    ``(x0, y0, x1, y1) = (h_term_start, v_term_start, h_term_end, v_term_end)``.
    Earlier code mistakenly used ``(v, h)`` as ``(x, y)``, which swaps axes on the image.
    """
    b = 0
    shape = args.grid_shape
    num_cells = shape[0] * shape[1]
    cell_size = [float(args.cell_size[0]), float(args.cell_size[1])]
    use_conf = Variables.CONF in coords.train_vars() and coords[Variables.CONF] > 0
    conf_idx = coords.get_position_within_prediction(Variables.CONF) if use_conf else None

    segs = []
    embs = []
    g = geom_preds[b].detach().cpu()
    e = embed_preds[b].detach().cpu()

    for i in range(num_cells):
        row = int(math.floor(i / shape[1]))
        col = int(i % shape[1])
        for p_idx in range(args.num_predictors):
            predictor_arr = g[i, p_idx].numpy()
            if np.any(np.isnan(predictor_arr)):
                continue
            if use_conf and conf_threshold > 0 and float(predictor_arr[conf_idx]) < conf_threshold:
                continue
            try:
                line = Predictor.from_linesegment(
                    predictor_arr, args.linerep, input_coords=coords,
                    is_prediction=True, is_offset=args.offset, anchor=None,
                )
            except ValueError:
                continue

            v = row * cell_size[0]
            h = col * cell_size[1]
            s0, s1 = cell_size[0], cell_size[1]
            # Match grid.py get_image_lines exactly (then map to PIL x,y).
            v0 = v + line.start[0] * s0
            h0 = h + line.start[1] * s0
            v1 = v + line.end[0] * s0
            h1 = h + line.end[1] * s1
            segs.append((float(h0), float(v0), float(h1), float(v1)))
            embs.append(e[i, p_idx].numpy())

    return segs, np.asarray(embs, dtype=np.float32)


def _color_for_cluster(lab):
    if lab < 0:
        return (140, 140, 140)
    # Deterministic distinct colors without matplotlib
    tab = [
        (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
        (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190), (0, 128, 128),
        (230, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
        (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128), (255, 255, 255),
    ]
    return tab[lab % len(tab)]


def draw_overlay(rgb, segments, labels, width=2):
    """rgb: HxWx3 uint8 RGB. labels: int per segment (-1 = noise). Returns RGB uint8."""
    im = Image.fromarray(rgb)
    dr = ImageDraw.Draw(im)
    for (x0, y0, x1, y1), lab in zip(segments, labels):
        dr.line((x0, y0, x1, y1), fill=_color_for_cluster(int(lab)), width=width)
    return np.asarray(im)


if __name__ == "__main__":
    extra_parser = parse_extra()
    extra_args, remaining = extra_parser.parse_known_args()

    # Resolve config paths so this works regardless of CWD (needs full paths when using ignore_cmd_args).
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _default_cfg = os.path.join(_root, "ttpla_train_exp", "default_params.yaml")
    _params_cfg = os.path.join(_root, "ttpla_train_exp", "params.yaml")
    if not os.path.isfile(_params_cfg):
        _params_cfg = os.path.join(_root, "params.yaml")
    if not os.path.isfile(_default_cfg):
        _default_cfg = os.path.join(_root, "res", "default_params.yaml")

    args = general_setup(
        "Instance embed cluster infer",
        config_file=_params_cfg,
        default_config=_default_cfg,
        task_type=TaskType.TEST,
        ignore_cmd_args=True,
        alternative_args=remaining,
    )

    if args.dataset != Dataset.TTPLA:
        raise SystemExit("This script is tested with dataset ttpla; your config has %s" % args.dataset)

    l2n = str_to_bool(extra_args.l2_normalize)
    out_root = os.path.join(args.paths.debug_folder, extra_args.out_subdir)
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

    processed = 0
    conf_thr = (
        float(extra_args.viz_confidence)
        if extra_args.viz_confidence is not None
        else float(args.confidence)
    )

    for batch_idx, data in enumerate(tqdm(loader, desc="Embed cluster")):
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

            gp = geom_preds[bi : bi + 1]
            ep = embed_preds[bi : bi + 1]
            if l2n:
                ep = F.normalize(ep, p=2, dim=-1)

            segs, emb_mat = segment_lines_and_embeddings(gp, ep, args, coords, conf_thr)
            name = filenames[bi]
            rec = {"file": name, "num_segments": len(segs), "confidence_threshold": conf_thr}

            if len(segs) == 0:
                rec["clusters"] = []
                with open(os.path.join(out_root, name + "_clusters.json"), "w") as f:
                    json.dump(rec, f, indent=2)
                processed += 1
                continue

            lab = cluster_embeddings_dbscan_like(
                emb_mat, eps=extra_args.dbscan_eps, min_samples=extra_args.dbscan_min_samples
            )
            rec["dbscan_eps"] = extra_args.dbscan_eps
            rec["dbscan_min_samples"] = extra_args.dbscan_min_samples
            rec["l2_normalize"] = l2n
            rec["n_clusters_est"] = int(len(set(lab)) - (1 if -1 in lab else 0))
            rec["segments"] = [
                {"xyxy": segs[i], "cluster": int(lab[i])} for i in range(len(segs))
            ]

            # Image: dataset returns tensor; typical CHW float
            img_t = images[bi].detach().cpu()
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

            vis = draw_overlay(im, segs, lab)
            out_png = os.path.join(out_root, name + "_inst_cluster.png")
            Image.fromarray(vis).save(out_png)
            with open(os.path.join(out_root, name + "_clusters.json"), "w") as f:
                json.dump(rec, f, indent=2)

            processed += 1

    print("Wrote outputs under file://%s" % out_root)
