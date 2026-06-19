# ISQ Results — exp70b (TTPLA smallset val)

## Setup

- **Dataset**: `/home/work/caps_drone/yolino/ttpla_yolino_dataset_1024x1024_smallset` (val **65** images)
- **Geom baseline**: exp19 (`confidence=0.7`)
- **GNN model**: exp70b (`edge_thresh=0.2`, `node_conf=0.7`)
- **Config**: `/home/work/caps_drone/yolino/CAPSTONE/configs/experiments/exp70b_gnn_directional2_ctx_cross_ignore_cross15_1024.yaml`
- **Checkpoint**: `/home/work/caps_drone/yolino/CAPSTONE/ttpla_train_exp/log/checkpoints/exp70b_gnn_directional2_ctx_cross_ignore_cross15_1024_smallset/best_model.pth`
- **ISQ**: grid cell=32, match dist=24 px, angle=15 deg, coverage thresh=0.3

## Summary (macro mean over 65 val images)

| Model | Precision | Recall | F1 | over_split (mean/img) | under_merge (mean/img) |
|-------|-----------|--------|-----|------------------------|-------------------------|
| Pred-1 geom (exp19) | 0.6377 | 0.6855 | 0.6514 | 0.00 | 0.98 |
| **Pred-2 GNN (exp70b)** | **0.1654** | **0.0175** | **0.0300** | **0.11** | **0.00** |
| Pred-2 GNN (exp55, ref) | 0.4937 | 0.5083 | 0.4979 | 0.00 | 0.82 |
| Pred-3 GNN (exp65, ref) | 0.7630 | 0.7537 | 0.7519 | 0.14 | 0.72 |

## Notes

- **over_split**: mean per-image count of GT instances split across multiple pred polylines.
- **under_merge**: mean per-image count of pred polylines merging multiple GT instances.
- exp55/exp65 reference rows are from `results/isq_results_oversplit.json` (same smallset val).

Artifacts:
- `/home/work/caps_drone/eval_isq/pred_geom_exp70b.pkl`
- `/home/work/caps_drone/eval_isq/pred_gnn_exp70b.pkl`
- `/home/work/caps_drone/eval_isq/results/isq_results_exp70b.json`
