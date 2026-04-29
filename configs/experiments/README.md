# Experiment Configs

This folder stores one YAML per experiment objective.

## Naming

- `exp01_baseline_1k.yaml`
- `exp02_strict_hard_1k.yaml`
- `exp03_backbone_convnext_nofpn_1k.yaml`
- `exp04_focal_learn_log_1k.yaml`

Use purpose-based names, not person names.

## Run

From `YOLinO/`:

```bash
bash run.sh --config configs/experiments/exp01_baseline_1k.yaml --dataset-root /home/work/caps_drone/yolino/ttpla_yolino_dataset_downsample
```

You can also set the dataset path via env var:

```bash
export DATASET_TTPLA=/home/work/caps_drone/yolino/ttpla_yolino_dataset_downsample
bash run.sh --config configs/experiments/exp02_strict_hard_1k.yaml
```

## Principle

- `run.sh` is the single execution entrypoint.
- Experiment differences live in YAML files.
- Keep common params aligned across YAMLs and change one factor at a time.
