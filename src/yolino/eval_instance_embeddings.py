import argparse
import json
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from yolino.dataset.dataset_factory import DatasetFactory
from yolino.eval.matcher_cell import CellMatcher
from yolino.model.activations import get_activations
from yolino.model.model_factory import load_checkpoint
from yolino.utils.enums import TaskType, Variables
from yolino.utils.general_setup import general_setup


def build_args():
    parser = argparse.ArgumentParser("Instance embedding evaluation")
    parser.add_argument("--max_batches", type=int, default=-1,
                        help="Limit the number of batches to inspect. -1 means all batches.")
    parser.add_argument("--out_name", type=str, default="instance_embedding_eval",
                        help="Output file prefix inside the debug folder.")
    parser.add_argument("--sample_pairs", type=int, default=50000,
                        help="Maximum number of same/different-instance pairs to keep for summary stats.")
    parser.add_argument("--tsne", action="store_true",
                        help="Use t-SNE instead of PCA for the 2D visualization.")
    return parser


def collect_pair_distances(embeddings, instance_ids, same_distances, diff_distances, max_pairs):
    if len(embeddings) < 2:
        return

    pair_indices = list(combinations(range(len(embeddings)), 2))
    if len(pair_indices) > max_pairs:
        rng = np.random.default_rng(0)
        sampled = rng.choice(len(pair_indices), size=max_pairs, replace=False)
        pair_indices = [pair_indices[i] for i in sampled]

    for a_idx, b_idx in pair_indices:
        dist = torch.norm(embeddings[a_idx] - embeddings[b_idx]).item()
        if instance_ids[a_idx] == instance_ids[b_idx]:
            if len(same_distances) < max_pairs:
                same_distances.append(dist)
        else:
            if len(diff_distances) < max_pairs:
                diff_distances.append(dist)


def to_numpy_cpu(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def reduce_to_2d(embeddings, use_tsne=False):
    if len(embeddings) == 0:
        return np.empty((0, 2), dtype=np.float32)

    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.shape[1] == 1:
        return np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=np.float32)], axis=1)

    if use_tsne:
        from sklearn.manifold import TSNE
        perplexity = min(30, max(5, len(arr) // 10))
        return TSNE(n_components=2, init="pca", learning_rate="auto",
                    perplexity=perplexity, random_state=0).fit_transform(arr)

    centered = arr - arr.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2].T
    return centered @ basis


def plot_embedding_projection(points_2d, instance_ids, out_path):
    plt.figure(figsize=(8, 8))
    unique_ids = sorted(set(instance_ids))
    cmap = plt.get_cmap("tab20", max(1, len(unique_ids)))

    for color_idx, inst_id in enumerate(unique_ids):
        mask = np.asarray(instance_ids) == inst_id
        pts = points_2d[mask]
        plt.scatter(pts[:, 0], pts[:, 1], s=16, alpha=0.75, label=f"id={int(inst_id)}",
                    color=cmap(color_idx % max(1, len(unique_ids))))

    plt.title("Instance embedding projection")
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    if len(unique_ids) <= 20:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def summarize_distances(values):
    if len(values) == 0:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}

    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


if __name__ == "__main__":
    extra_parser = build_args()
    extra_args, remaining = extra_parser.parse_known_args()

    args = general_setup(
        "Instance Embedding Evaluation",
        task_type=TaskType.TEST,
        ignore_cmd_args=True,
        alternative_args=remaining,
    )

    dataset, loader = DatasetFactory.get(args.dataset, only_available=True, split=args.split, args=args,
                                         shuffle=False, augment=False, ignore_duplicates=False)
    model, _, _ = load_checkpoint(args, dataset.coords, allow_failure=False, load_best=False)
    model.eval()
    activations = get_activations(args.activations, coords=dataset.coords,
                                  linerep=dataset.coords.line_representation.enum)

    matcher = CellMatcher(dataset.coords.clone(dataset.coords.line_representation.enum), args)
    device = args.cuda
    model = model.to(device)

    same_distances = []
    diff_distances = []
    all_embeddings = []
    all_instance_ids = []
    matched_segment_count = 0
    matched_instance_count = 0

    for batch_idx, data in enumerate(tqdm(loader, desc="Evaluate instance embeddings")):
        if extra_args.max_batches >= 0 and batch_idx >= extra_args.max_batches:
            break

        images, grid_tensor, filenames, duplicate_info, params = data
        images = images.to(device)

        with torch.no_grad():
            geom_logits, embed_preds = model(images)
            geom_preds, embed_preds = activations((geom_logits, embed_preds))

        geom_preds = geom_preds.detach()
        embed_preds = embed_preds.detach()
        grid_tensor = grid_tensor.to(device)

        matched_predictions, _ = matcher.match(
            preds=geom_preds,
            grid_tensor=grid_tensor,
            filenames=filenames,
            confidence_threshold=args.confidence,
        )

        num_batch, num_cells, num_preds, embed_dim = embed_preds.shape
        matched_predictions = matched_predictions.view(num_batch, num_cells, num_preds)
        inst_pos = dataset.coords.get_position_of(Variables.INSTANCE)

        for b_idx in range(num_batch):
            sample_embeddings = []
            sample_instance_ids = []

            for c_idx in range(num_cells):
                for p_idx in range(num_preds):
                    gt_idx = int(matched_predictions[b_idx, c_idx, p_idx].item())
                    if gt_idx < 0:
                        continue

                    gt_row = grid_tensor[b_idx, c_idx, gt_idx]
                    if torch.isnan(gt_row[0]):
                        continue

                    inst_value = gt_row[inst_pos[0]].item()
                    if not np.isfinite(inst_value):
                        continue

                    inst_id = int(inst_value)
                    if inst_id <= 0:
                        continue

                    emb = embed_preds[b_idx, c_idx, p_idx].detach().cpu()
                    sample_embeddings.append(emb)
                    sample_instance_ids.append(inst_id)

            if not sample_embeddings:
                continue

            sample_embeddings = torch.stack(sample_embeddings)
            collect_pair_distances(sample_embeddings, sample_instance_ids, same_distances,
                                   diff_distances, extra_args.sample_pairs)

            all_embeddings.extend(to_numpy_cpu(sample_embeddings))
            all_instance_ids.extend(sample_instance_ids)
            matched_segment_count += len(sample_embeddings)
            matched_instance_count += len(set(sample_instance_ids))

    debug_dir = args.paths.debug_folder
    os.makedirs(debug_dir, exist_ok=True)

    projection_path = os.path.join(debug_dir, f"{extra_args.out_name}_projection.png")
    summary_path = os.path.join(debug_dir, f"{extra_args.out_name}_summary.json")

    points_2d = reduce_to_2d(all_embeddings, use_tsne=extra_args.tsne)
    if len(points_2d) > 0:
        plot_embedding_projection(points_2d, all_instance_ids, projection_path)

    same_stats = summarize_distances(same_distances)
    diff_stats = summarize_distances(diff_distances)

    summary = {
        "split": args.split,
        "checkpoint": str(args.paths.pretrain_model),
        "matched_segment_count": matched_segment_count,
        "matched_instance_count": matched_instance_count,
        "same_instance_distance": same_stats,
        "different_instance_distance": diff_stats,
        "distance_gap": None if same_stats["mean"] is None or diff_stats["mean"] is None
        else float(diff_stats["mean"] - same_stats["mean"]),
        "projection_image": projection_path if len(points_2d) > 0 else None,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
