# ISQ Results — exp70b ep59 (TTPLA smallset val)

## Setup

- **Dataset**: `/home/work/caps_drone/yolino/ttpla_yolino_dataset_1024x1024_smallset` (val **65** images)
- **Checkpoint**: `/home/work/caps_drone/yolino/CAPSTONE/ttpla_train_exp/log/checkpoints/exp70b_gnn_directional2_ctx_cross_ignore_cross15_1024_smallset/ep0059_model.pth` (epoch 59, **not** stale `best_model.pth`)
- **GNN thresholds**: edge=0.2, node_conf=0.7

## Summary

| Model | Precision | Recall | F1 | over_split | under_merge |
|-------|-----------|--------|-----|------------|-------------|
| Pred-1 geom (exp19) | 0.6377 | 0.6855 | 0.6514 | 0.00 | 0.98 |
| **Pred-2 GNN (exp70b ep59)** | **0.7183** | **0.5932** | **0.6223** | **0.11** | **0.52** |
| Pred-2 GNN (exp70b best_model, ref) | 0.1654 | 0.0175 | 0.0300 | 0.11 | 0.00 |
| Pred-3 GNN (exp65, ref) | 0.7630 | 0.7537 | 0.7519 | 0.14 | 0.72 |

Artifacts:
- `/home/work/caps_drone/eval_isq/pred_geom_exp70b.pkl`
- `/home/work/caps_drone/eval_isq/pred_gnn_exp70b_ep59.pkl`
- `/home/work/caps_drone/eval_isq/results/isq_results_exp70b_ep59.json`
