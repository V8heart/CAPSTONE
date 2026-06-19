#!/usr/bin/env python3
"""ISQ (Instance Segmentation Quality) for YOLinO geom (exp19) and GNN (exp55) on TTPLA val."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from isq_core import (
    CELL_SIZE,
    GEOM_CONF,
    GNN_EDGE_THRESH,
    GNN_NODE_CONF,
    GRID_SHAPE,
    compute_isq_image,
    geom_polylines_to_pred_polylines,
    gnn_segments_to_polylines,
    gt_polylines_to_segments,
    pred_polylines_to_segment_tuples,
    segment_tuples_to_pred_polylines,
    summarize_isq,
)

_torch_load = torch.load


def _torch_load_trusted(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_trusted  # noqa: PLW0603

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPSTONE_ROOT = os.path.join(BASE, "yolino", "CAPSTONE")
SRC_ROOT = os.path.join(CAPSTONE_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from yolino.dataset.dataset_factory import DatasetFactory  # noqa: E402
from yolino.grid.grid_factory import GridFactory  # noqa: E402
from yolino.model.model_factory import load_checkpoint  # noqa: E402
from yolino.runner.forward_runner import ForwardRunner  # noqa: E402
from yolino.utils.enums import CoordinateSystem, TaskType  # noqa: E402
from yolino.utils.general_setup import general_setup  # noqa: E402
from yolino.visualize_ttpla_gt_instances import load_ttpla_label_with_ids  # noqa: E402

DATASET_ROOT = os.path.join(BASE, "yolino", "ttpla_yolino_dataset_1024x1024_smallset")
EXP19_CONFIG = os.path.join(
    CAPSTONE_ROOT, "configs", "experiments", "exp19_fpn_bottomup_p4_num_predictors4_1024.yaml"
)
EXP55_CONFIG = os.path.join(
    CAPSTONE_ROOT, "configs", "experiments", "exp55_gnn_soft_geom_rw_topology_1024.yaml"
)
EXP19_CKPT = os.path.join(
    CAPSTONE_ROOT,
    "ttpla_train_exp",
    "log",
    "checkpoints",
    "exp19_fpn_bottomup_p4_num_predictors4_1024",
    "best_model.pth",
)
EXP55_CKPT = os.path.join(
    CAPSTONE_ROOT,
    "ttpla_train_exp",
    "log",
    "checkpoints",
    "exp55_gnn_soft_geom_rw_topology_1024_smallset",
    "best_model.pth",
)
EVAL_ISQ_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_GEOM_PKL = os.path.join(EVAL_ISQ_DIR, "pred_geom.pkl")
PRED_GNN_PKL = os.path.join(EVAL_ISQ_DIR, "pred_gnn.pkl")
OUT_DIR = os.path.join(EVAL_ISQ_DIR, "results")
RESULTS_JSON = os.path.join(OUT_DIR, "isq_results.json")
SPLIT = "val"

Segment = Tuple[Tuple[float, float], Tuple[float, float]]
SegmentTuple = Tuple[float, float, float, float, int]


@dataclass
class ImageRecord:
    stem: str
    height: int
    width: int
    label_polylines: List = field(default_factory=list)
    label_instance_ids: List[int] = field(default_factory=list)
    geom_polylines: Optional[List] = None
    gnn_segments: Optional[List[Segment]] = None


def geom_uv_to_polylines(pred_uv: np.ndarray) -> List[List[List[float]]]:
    if pred_uv.ndim == 3:
        rows = pred_uv[0]
    else:
        rows = pred_uv
    polys = []
    for row in rows:
        if row.shape[0] < 4 or np.any(np.isnan(row[:4])):
            continue
        y1, x1, y2, x2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        polys.append([[x1, y1], [x2, y2]])
    return polys


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def extract_gnn_segments(
    gnn_single: dict,
    edge_thresh: float = GNN_EDGE_THRESH,
    node_conf_thresh: float = GNN_NODE_CONF,
) -> List[Segment]:
    node_mid = _to_numpy(gnn_single["node_mid_px"])
    node_valid = _to_numpy(gnn_single["node_valid"]).astype(bool)
    neighbors = _to_numpy(gnn_single["neighbors"]).astype(np.int64)
    neigh_valid = _to_numpy(gnn_single["neigh_valid"]).astype(bool)
    edge_logits = _to_numpy(gnn_single["edge_logits"]).astype(np.float32)
    node_ea = gnn_single.get("node_end_a_px")
    node_eb = gnn_single.get("node_end_b_px")
    node_conf = gnn_single.get("node_conf")
    if node_ea is not None:
        node_ea = _to_numpy(node_ea)
    if node_eb is not None:
        node_eb = _to_numpy(node_eb)
    if node_conf is not None:
        node_conf = _to_numpy(node_conf).astype(np.float32)
        node_valid = node_valid & (node_conf >= float(node_conf_thresh))

    segments: List[Segment] = []
    if not np.any(node_valid) or node_ea is None or node_eb is None:
        return segments

    n_nodes, k = neighbors.shape
    probs = 1.0 / (1.0 + np.exp(-edge_logits))
    edge_pairs = set()
    for ni in range(n_nodes):
        if not bool(node_valid[ni]):
            continue
        for kj in range(k):
            if not bool(neigh_valid[ni, kj]):
                continue
            if float(probs[ni, kj]) < float(edge_thresh):
                continue
            nj = int(neighbors[ni, kj])
            if nj < 0 or nj >= n_nodes or not bool(node_valid[nj]):
                continue
            a, b = (ni, nj) if ni < nj else (nj, ni)
            edge_pairs.add((a, b))

    in_graph = np.zeros((n_nodes,), dtype=bool)
    for a, b in edge_pairs:
        in_graph[a] = True
        in_graph[b] = True

    for i in range(n_nodes):
        if not bool(node_valid[i]) or not bool(in_graph[i]):
            continue
        ea, eb = node_ea[i], node_eb[i]
        segments.append(((float(ea[0]), float(ea[1])), (float(eb[0]), float(eb[1]))))

    for a, b in edge_pairs:
        ma, mb = node_mid[a], node_mid[b]
        segments.append(((float(ma[0]), float(ma[1])), (float(mb[0]), float(mb[1]))))
    return segments


def setup_run(config_path: str, checkpoint_path: str, dataset_root: str, use_gpu: bool):
    os.environ["DATASET_TTPLA"] = os.path.abspath(dataset_root)
    os.environ.setdefault("YOLINO_IGNORE_DIRTY", "1")
    dvc = os.path.join(CAPSTONE_ROOT, "ttpla_train_exp")
    default_cfg = os.path.join(dvc, "default_params.yaml")
    ckpt = os.path.abspath(checkpoint_path)
    alt = [
        "-c",
        os.path.abspath(config_path),
        "--root",
        CAPSTONE_ROOT,
        "--dvc",
        dvc,
        "--split",
        SPLIT,
        "--explicit_model",
        ckpt,
        "--run_name",
        "eval_isq",
        "--loggers",
        "file",
        "--batch_size",
        "1",
        "--loading_workers",
        "0",
    ]
    if use_gpu:
        alt.extend(["--gpu", "--gpu_id", "0"])
    args = general_setup(
        "eval_isq",
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
    return forward, dataset, loader, device


def load_records(dataset_root: str, file_names: Sequence[str]) -> Dict[str, ImageRecord]:
    img_dir = os.path.join(dataset_root, "images", SPLIT)
    label_dir = os.path.join(dataset_root, "labels", SPLIT)
    records: Dict[str, ImageRecord] = {}
    for stem in file_names:
        img_path = os.path.join(img_dir, stem + ".png")
        label_path = os.path.join(label_dir, stem + ".npy")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)
        h, w = img.shape[:2]
        polylines, instance_ids, _ = load_ttpla_label_with_ids(label_path)
        records[stem] = ImageRecord(
            stem=stem,
            height=h,
            width=w,
            label_polylines=polylines,
            label_instance_ids=list(instance_ids),
        )
    return records


def run_geom_predictions(
    forward: ForwardRunner,
    dataset,
    loader: DataLoader,
    device: torch.device,
    records: Dict[str, ImageRecord],
) -> None:
    n_done = 0
    for images, _grid_tensor, fileinfo, *_ in loader:
        stem = str(fileinfo[0] if isinstance(fileinfo, (list, tuple)) else fileinfo)
        if stem not in records:
            continue
        images = images.to(device)
        geom_preds, _, _, _e2e = forward(images, is_train=False, epoch=None)
        ih = int(images.shape[-2])
        pred_grid, _ = GridFactory.get(
            torch.unsqueeze(geom_preds[0].detach().cpu(), dim=0),
            [],
            CoordinateSystem.CELL_SPLIT,
            args=forward.args,
            input_coords=dataset.coords,
            only_train_vars=True,
            anchors=dataset.anchors,
        )
        pred_uv = pred_grid.get_image_lines(
            coords=dataset.coords,
            image_height=ih,
            is_training_data=True,
            confidence_threshold=float(GEOM_CONF),
        )
        records[stem].geom_polylines = geom_uv_to_polylines(np.asarray(pred_uv))
        n_done += 1
        if n_done % 10 == 0:
            print("[geom] %d/%d" % (n_done, len(records)), flush=True)
        if n_done >= len(records):
            break


def run_gnn_predictions(
    forward: ForwardRunner,
    dataset,
    loader: DataLoader,
    device: torch.device,
    records: Dict[str, ImageRecord],
    gnn_raw: Dict[str, dict],
) -> None:
    n_done = 0
    for images, _grid_tensor, fileinfo, *_ in loader:
        stem = str(fileinfo[0] if isinstance(fileinfo, (list, tuple)) else fileinfo)
        if stem not in records:
            continue
        images = images.to(device)
        _geom_preds, _, _, e2e_out = forward(images, is_train=False, epoch=None)
        if e2e_out is None or "edge_logits" not in e2e_out:
            raise RuntimeError("GNN e2e output missing; check exp55 config (e2e_mode=gnn).")
        gnn_single = {}
        for k, v in e2e_out.items():
            if k in ("soft_nms_debug",):
                continue
            if isinstance(v, torch.Tensor) and v.shape[0] == 1:
                gnn_single[k] = v[0]
            else:
                gnn_single[k] = v
        gnn_raw[stem] = gnn_single
        records[stem].gnn_segments = extract_gnn_segments(gnn_single)
        n_done += 1
        if n_done % 10 == 0:
            print("[gnn] %d/%d" % (n_done, len(records)), flush=True)
        if n_done >= len(records):
            break


def build_pred_pkls(
    records: Dict[str, ImageRecord],
    gnn_raw: Dict[str, dict],
) -> Tuple[Dict[str, List[SegmentTuple]], Dict[str, List[SegmentTuple]]]:
    geom_pkl: Dict[str, List[SegmentTuple]] = {}
    gnn_pkl: Dict[str, List[SegmentTuple]] = {}
    for stem, rec in sorted(records.items()):
        if rec.geom_polylines is None:
            raise ValueError("Missing geom for %s" % stem)
        geom_pkl[stem] = pred_polylines_to_segment_tuples(
            geom_polylines_to_pred_polylines(rec.geom_polylines)
        )
        if rec.gnn_segments is None or stem not in gnn_raw:
            raise ValueError("Missing GNN for %s" % stem)
        gnn_pkl[stem] = pred_polylines_to_segment_tuples(
            gnn_segments_to_polylines(rec.gnn_segments, gnn_raw[stem])
        )
    return geom_pkl, gnn_pkl


def evaluate_from_pkl(
    records: Dict[str, ImageRecord],
    pred_by_stem: Dict[str, List[SegmentTuple]],
) -> Dict[str, object]:
    per_image = []
    for stem, rec in sorted(records.items()):
        image_size = max(rec.height, rec.width)
        gt_segments = gt_polylines_to_segments(
            rec.label_polylines,
            rec.label_instance_ids,
            cell_size=CELL_SIZE,
            image_size=image_size,
        )
        pred_polylines = segment_tuples_to_pred_polylines(pred_by_stem[stem])
        metrics = compute_isq_image(gt_segments, pred_polylines)
        metrics["stem"] = stem
        per_image.append(metrics)
    return {"summary": summarize_isq(per_image), "per_image": per_image}


def run_inference(records: Dict[str, ImageRecord], use_gpu: bool) -> Tuple[Dict[str, List[SegmentTuple]], Dict[str, List[SegmentTuple]]]:
    gnn_raw: Dict[str, dict] = {}
    print("[INFO] Loading exp19 — geom (Pred-1)...")
    fwd19, ds19, loader19, dev19 = setup_run(EXP19_CONFIG, EXP19_CKPT, DATASET_ROOT, use_gpu)
    run_geom_predictions(fwd19, ds19, loader19, dev19, records)
    del fwd19, ds19, loader19
    if use_gpu:
        torch.cuda.empty_cache()

    print("[INFO] Loading exp55 — GNN (Pred-2)...")
    fwd55, ds55, loader55, dev55 = setup_run(EXP55_CONFIG, EXP55_CKPT, DATASET_ROOT, use_gpu)
    run_gnn_predictions(fwd55, ds55, loader55, dev55, records, gnn_raw)
    geom_pkl, gnn_pkl = build_pred_pkls(records, gnn_raw)
    with open(PRED_GEOM_PKL, "wb") as f:
        pickle.dump(geom_pkl, f)
    with open(PRED_GNN_PKL, "wb") as f:
        pickle.dump(gnn_pkl, f)
    print("[OK] Cached predictions:", PRED_GEOM_PKL, PRED_GNN_PKL)
    return geom_pkl, gnn_pkl


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ISQ evaluation on TTPLA val.")
    ap.add_argument(
        "--from-pkl",
        action="store_true",
        help="Skip model inference; load pred_geom.pkl / pred_gnn.pkl (from a prior run).",
    )
    ap.add_argument("--max-images", type=int, default=0, help="Limit val images (0 = all).")
    return ap.parse_args()


def main() -> None:
    args_cli = parse_args()
    use_gpu = torch.cuda.is_available()
    print("[INFO] device:", "cuda" if use_gpu else "cpu")

    for p in (DATASET_ROOT,):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    os.makedirs(OUT_DIR, exist_ok=True)
    val_img_dir = os.path.join(DATASET_ROOT, "images", SPLIT)
    file_names = sorted(
        os.path.splitext(f)[0] for f in os.listdir(val_img_dir) if f.lower().endswith(".png")
    )
    if args_cli.max_images > 0:
        file_names = file_names[: args_cli.max_images]
    print("[INFO] val images:", len(file_names))

    records = load_records(DATASET_ROOT, file_names)

    if args_cli.from_pkl:
        for p in (PRED_GEOM_PKL, PRED_GNN_PKL):
            if not os.path.isfile(p):
                raise FileNotFoundError("%s missing; run without --from-pkl first." % p)
        with open(PRED_GEOM_PKL, "rb") as f:
            geom_pkl = pickle.load(f)
        with open(PRED_GNN_PKL, "rb") as f:
            gnn_pkl = pickle.load(f)
        inference_note = "cached pickles"
    else:
        for p in (EXP19_CONFIG, EXP55_CONFIG, EXP19_CKPT, EXP55_CKPT):
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        geom_pkl, gnn_pkl = run_inference(records, use_gpu)
        inference_note = "exp19 geom + exp55 GNN checkpoints"

    results = {
        "settings": {
            "dataset_root": DATASET_ROOT,
            "split": SPLIT,
            "cell_size": CELL_SIZE,
            "grid_shape": list(GRID_SHAPE),
            "match_max_dist_px": 24,
            "match_max_angle_deg": 15,
            "meaningful_coverage_thresh": 0.3,
            "min_segments_per_pred_polyline": 2,
            "geom_confidence": GEOM_CONF,
            "gnn_edge_thresh": GNN_EDGE_THRESH,
            "gnn_node_conf": GNN_NODE_CONF,
            "exp19_checkpoint": EXP19_CKPT,
            "exp55_checkpoint": EXP55_CKPT,
            "inference": inference_note,
        },
        "Pred-1_geom_exp19": evaluate_from_pkl(records, geom_pkl),
        "Pred-2_gnn_exp55": evaluate_from_pkl(records, gnn_pkl),
    }

    for label, block in (
        ("Pred-1 geom (exp19)", results["Pred-1_geom_exp19"]),
        ("Pred-2 GNN (exp55)", results["Pred-2_gnn_exp55"]),
    ):
        s = block["summary"]
        print(
            "\n[ISQ] %s  P=%.4f  R=%.4f  F1=%.4f  over_split=%.2f  under_merge=%.2f  (n=%d)"
            % (
                label,
                s["precision"],
                s["recall"],
                s["f1"],
                s["over_split"],
                s["under_merge"],
                s["n_images"],
            )
        )

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n[OK] Wrote", RESULTS_JSON)


if __name__ == "__main__":
    main()
