# CAPSTONE: Powerline Detection with YOLinO + GNN

**[Korea Institute of Energy Technology](https://www.kentech.ac.kr/) (KENTECH)**

**Students**

- [**최재영**](https://github.com/V8heart) — agew1597@kentech.ac.kr
- [**최동제**](https://github.com/URIBARI) — cdj0418@kentech.ac.kr
- [**강원용**](https://github.com/wonyong-3927) — wonyong3927@kentech.ac.kr
- [**이현승**](https://github.com/Ark-sty) — pruina@kentech.ac.kr

**Advisor**

- [**Seokju Lee**](https://github.com/SeokjuLee) — slee@kentech.ac.kr

*Capstone project — [viewlab-group/Capstone26s-PowerLineDetection-dev](https://github.com/viewlab-group/Capstone26s-PowerLineDetection-dev)*

## Abstract

CAPSTONE detects aerial powerlines in TTPLA imagery using a **two-stage** pipeline. **Stage 1** runs a YOLinO-style single-shot detector (ConvNeXt-Tiny + FPN) to predict per-cell line geometry and confidence. **Stage 2** freezes that backbone and trains a **Graph Attention Network (GAT)** on predicted segments to assemble individual wires into instance-level polylines. Datasets and checkpoints are published on Hugging Face; this repository provides training, inference, experiment configs, and evaluation code.

## Model architecture

![CAPSTONE two-stage architecture: YOLinO geom head + GNN assembly](docs/images/Model_Architecture.png)

**Pipeline**
- **Stage 1 (`exp80`)** — train YOLinO geometry + confidence (ConvNeXt-Tiny, FPN+PANet, 512×512, scale 16 / P3).
- **Stage 2 (`exp83`)** — freeze backbone/FPN/geom head; train GNN head with strengthened topology loss (`directional2`, BCE + random-walk). **Best checkpoint: epoch 6.**

Large artifacts (datasets & checkpoints) are hosted on [Hugging Face](https://huggingface.co/V8heart); this repo contains code and experiment configs only.

## Results

| Stage | Config | Description |
|-------|--------|-------------|
| **Stage 1** | `exp80` | YOLinO geom + confidence (conf ≥ 0.7) |
| **Stage 2** | `exp83` | GNN assembly on top of exp80 (ep6) |

**Quantitative results (TTPLA test set, 220 images)**

| Metric | exp80 (Geom only) | exp83 (+ GNN) | LSNetv2 (reported) |
|--------|:-----------------:|:-------------:|:------------------:|
| APR (2 px) | 0.610 | 0.617 | 0.714 |
| ARR (2 px) | 0.581 | 0.579 | 0.560 |
| F1  (2 px) | 0.595 | 0.597 | 0.628 |
| F_β (β²=0.3) | 0.603 | 0.608 | 0.671 |
| ISQ Precision | 0.354 | 0.361 | — |
| ISQ Recall    | 0.663 | 0.693 | — |
| ISQ F1        | 0.455 | 0.469 | — |

**Stage 1 example** (`71_4520`, val split):

![Stage 1 example — YOLinO geom prediction (left) vs GT (right)](docs/images/stage1_example.png)

---

## Requirements

- Python 3.10+ (tested with the project venv below)
- CUDA-capable GPU(s) for training (4-GPU DDP by default)
- [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub) CLI (`hf`) for downloading data/weights

### Virtual environment

We use a **Python venv** (not conda) for this project:

```bash
# create once (example)
python3 -m venv /path/to/venv
source /path/to/venv/bin/activate
pip install -e .

# our server default (also used by run.sh)
# /home/work/caps_drone/yolino/venv/bin/python
```

`run.sh` picks up `PYTHON_BIN` automatically; override if your venv lives elsewhere:

```bash
export PYTHON_BIN=/path/to/venv/bin/python
```

---

## Quick start

```bash
git clone https://github.com/V8heart/CAPSTONE.git
cd CAPSTONE
source /path/to/venv/bin/activate   # or set PYTHON_BIN
pip install -e .
pip install -U huggingface_hub

# download TTPLA benchmark (512×512)
hf download V8heart/yolino-ttpla-benchmark \
  --repo-type dataset \
  --local-dir ./YOLinO_benchmark

export DATASET_TTPLA="$(pwd)/YOLinO_benchmark"

# download Stage 2 checkpoint (exp83, epoch 6 — final model)
hf download V8heart/CAPSTONE-gnn-weights \
  exp83_gnn_ttpla_512512_from_exp80/ep0006_model.pth \
  --repo-type model \
  --local-dir ttpla_train_exp/log/checkpoints/exp83_gnn_ttpla_512512_from_exp80
```

Expected dataset layout:

```
YOLinO_benchmark/
├── images/{train,val,test}/*.png
└── labels/{train,val,test}/*.npy
```

### Hugging Face resources

| Resource | Hugging Face | Notes |
|----------|--------------|-------|
| TTPLA benchmark (512×512) | [V8heart/yolino-ttpla-benchmark](https://huggingface.co/datasets/V8heart/yolino-ttpla-benchmark) | Main training / test set |
| TTPLA ablation tiles (1024×1024) | [V8heart/ttpla-yolino-1024](https://huggingface.co/datasets/V8heart/ttpla-yolino-1024) | 1024² ablation & ISQ full-test eval |
| KEPCO cross-domain dataset | [V8heart/Yolino-KEPCO](https://huggingface.co/datasets/V8heart/Yolino-KEPCO) | exp23 / exp71 evaluation |
| Model weights | [V8heart/CAPSTONE-gnn-weights](https://huggingface.co/V8heart/CAPSTONE-gnn-weights) | See checkpoint table below |

**Published checkpoints** (`V8heart/CAPSTONE-gnn-weights`):

| Experiment | File | Role |
|------------|------|------|
| `exp80_ttpla_512512_scale16` | `best_model.pth` | Stage 1 geom (512²) |
| `exp83_gnn_ttpla_512512_from_exp80` | `ep0006_model.pth` | **Stage 2 GNN (512², final)** |
| `exp81_gnn_ttpla_512512_from_exp80` | `ep0058_model.pth` | Previous Stage 2 (reference) |
| `exp19_fpn_bottomup_p4_num_predictors4_1024` | `ep0042_model.pth` | 1024² ablation geom |
| `exp76_gnn_ttpla_full` | `ep0005_model.pth` | 1024² ablation GNN |
| `exp23_finetune_kepco` | `best_model.pth` | KEPCO geom fine-tune |
| `exp71_gnn_kepco` | `best_model.pth` | KEPCO GNN |

Download any checkpoint:

```bash
hf download V8heart/CAPSTONE-gnn-weights \
  <run_name>/<filename>.pth \
  --repo-type model \
  --local-dir ttpla_train_exp/log/checkpoints/<run_name>
```

---

## Training

All training goes through `run.sh`, which sets `DATASET_TTPLA`, `PYTHONPATH`, and launches `torch.distributed.run`.

Default: **4 GPUs** (`--nproc 4`, `CUDA_VISIBLE_DEVICES=0,1,2,3`). Adjust to your machine.

### Stage 1 — YOLinO geometry baseline (`exp80`)

Config: `configs/experiments/exp80_ttpla_512512_scale16.yaml`

Trains geom + confidence only (no GNN). Output checkpoint:

```
ttpla_train_exp/log/checkpoints/exp80_ttpla_512512_scale16/best_model.pth
```

```bash
export DATASET_TTPLA="$(pwd)/YOLinO_benchmark"

bash run.sh \
  --config configs/experiments/exp80_ttpla_512512_scale16.yaml \
  --dataset-root "$DATASET_TTPLA" \
  --nproc 4
```

### Stage 2 — GNN head (`exp83`)

Config: `configs/experiments/exp83_gnn_ttpla_512512_from_exp80.yaml`

Warm-starts from Stage 1 `best_model.pth` (`explicit_model` in the YAML). Backbone/FPN/geom are **frozen** (`lr=0.0`); only the GNN head (`e2e_mode: gnn`) is trained.

Key hyperparameters vs. baseline GNN (exp82):
- `gnn_pos_weight: 4.0` (↑ from 2.0) — stronger positive edge signal
- `gnn_rw_topology_weight: 0.35` (↑ from 0.1) — stronger random-walk regularisation
- `gnn_rw_steps: 8` (↑ from 6) — longer chain connectivity
- `gnn_cc_edge_thresh: 0.2` (↓ from 0.3) — more segments connected at inference

```bash
# requires Stage 1 checkpoint at:
# ttpla_train_exp/log/checkpoints/exp80_ttpla_512512_scale16/best_model.pth

bash run.sh \
  --config configs/experiments/exp83_gnn_ttpla_512512_from_exp80.yaml \
  --dataset-root "$DATASET_TTPLA" \
  --nproc 3
```

> Previous Stage 2 experiments: `exp81` (directional2_ctx), `exp82` (directional2 baseline).

---

## Inference & evaluation

Run from the project root with the venv active.

### Visual prediction (overlay images)

```bash
cd ttpla_train_exp
export DATASET_TTPLA="../YOLinO_benchmark"
export PYTHONPATH="../src"

../venv/bin/python ../src/yolino/predict.py \
  -c ../configs/experiments/exp83_gnn_ttpla_512512_from_exp80.yaml \
  --root .. \
  --dvc . \
  --log_dir ttpla_experiments \
  --split val \
  --gpu \
  --explicit_model log/checkpoints/exp83_gnn_ttpla_512512_from_exp80/ep0006_model.pth
```

Debug images are written under `ttpla_train_exp/debug/prediction/`.

### Built-in metric evaluation

```bash
cd ttpla_train_exp
export DATASET_TTPLA="../YOLinO_benchmark"
export PYTHONPATH="../src"

../venv/bin/python ../src/yolino/eval.py \
  -c ../configs/experiments/exp83_gnn_ttpla_512512_from_exp80.yaml \
  --root .. \
  --dvc . \
  --log_dir ttpla_experiments \
  --split val \
  --gpu \
  --explicit_model log/checkpoints/exp83_gnn_ttpla_512512_from_exp80/ep0006_model.pth
```

Replace `../venv/bin/python` with your `PYTHON_BIN` if the venv path differs.

> **Note:** Experiment YAMLs may contain machine-local `dataset_ttpla` paths. Always set `DATASET_TTPLA` or pass `--dataset-root` via `run.sh` when running on a new machine.

---

## Evaluation Metrics

This repository includes two standalone evaluation modules under `eval_isq/` and `eval_pixel_f1/`.

### ISQ — Instance Segmentation Quality (`eval_isq/`)

ISQ is a **proposed segment-level metric** that explicitly accounts for instance identity (polyline ID), over-splitting, and under-merging. Unlike pixel-level metrics, it operates at the level of predicted *polyline instances* and measures how faithfully the model recovers individual wire identities.

**Key files:**

| File | Description |
|------|-------------|
| `isq_core.py` | Core ISQ computation: GT segment generation, pred-to-GT matching, TP/FP/FN counting, OS/UM rate |
| `eval_isq.py` | Evaluation runner: runs inference and ISQ over a dataset split |

**Algorithm (3-step):**

1. **GT segment generation** — each GT polyline is intersected with a 32×32 px grid; at most one segment per (cell, polyline) pair.
2. **Matching** — each predicted segment is matched to the closest GT segment within distance < 24 px **and** angle difference < 15°. The dominant GT polyline ID is the majority-vote among matched GT IDs.
3. **TP/FP/FN** — a GT segment counts as TP at most once; any re-match is FP. FN = unmatched GT segments.

**OS / UM rates:**
- **Over-split (OS):** ≥ 2 predicted polylines each cover ≥ 30% of the same GT polyline.
- **Under-merge (UM):** a single predicted polyline meaningfully covers ≥ 2 distinct GT polylines (precision or recall ≥ 30% against each).

**Usage (512² main results):**

```bash
cd eval_isq
export PYTHONPATH="../src"
export DATASET_TTPLA="/path/to/YOLinO_benchmark"

python eval_isq.py \
  --geom-config ../configs/experiments/exp80_ttpla_512512_scale16.yaml \
  --gnn-config  ../configs/experiments/exp83_gnn_ttpla_512512_from_exp80.yaml \
  --geom-ckpt   ../ttpla_train_exp/log/checkpoints/exp80_ttpla_512512_scale16/best_model.pth \
  --gnn-ckpt    ../ttpla_train_exp/log/checkpoints/exp83_gnn_ttpla_512512_from_exp80/ep0006_model.pth \
  --dataset-root "$DATASET_TTPLA" \
  --split test --gpu
```

**Usage (1024² ablation):**

```bash
python eval_isq.py \
  --geom-config ../configs/experiments/exp19_fpn_bottomup_p4_num_predictors4_1024.yaml \
  --gnn-config  ../configs/experiments/exp76_gnn_ttpla_full.yaml \
  --geom-ckpt   ../ttpla_train_exp/log/checkpoints/exp19_fpn_bottomup_p4_num_predictors4_1024/ep0042_model.pth \
  --gnn-ckpt    ../ttpla_train_exp/log/checkpoints/exp76_gnn_ttpla_full/ep0005_model.pth \
  --dataset-root /path/to/ttpla_yolino_dataset_1024x1024 \
  --split test --gpu
```

---

### Pixel F1 — LSNetv2-style Raster Metrics (`eval_pixel_f1/`)

Pixel-level evaluation following the **LSNetv2** protocol: both GT polylines and predictions are rasterised to binary masks at a fixed line width, and macro-averaged precision / recall / F1 / F_β (β²=0.3) are reported over all images.

**Key files:**

| File | Description |
|------|-------------|
| `eval_pixel_f1.py` | Main evaluation: rasterisation, per-image PR, macro APR/ARR/F1/F_β |

**Protocol:**

- **GT type:** GT-B (polyline raster, thin wire centre lines)
- **Line widths evaluated:** 2 px and 4 px (LSNetv2 standard)
- **Metric:** macro APR / ARR / F1 / F_β (β²=0.3), averaged over images

```
F_β = (1 + β²) · APR · ARR / (β² · APR + ARR),   β² = 0.3
```

**Usage:**

```bash
cd eval_pixel_f1
export PYTHONPATH="../src"
export DATASET_TTPLA="/path/to/YOLinO_benchmark"

python eval_pixel_f1.py \
  --geom-config ../configs/experiments/exp80_ttpla_512512_scale16.yaml \
  --gnn-config  ../configs/experiments/exp83_gnn_ttpla_512512_from_exp80.yaml \
  --geom-ckpt   ../ttpla_train_exp/log/checkpoints/exp80_ttpla_512512_scale16/best_model.pth \
  --gnn-ckpt    ../ttpla_train_exp/log/checkpoints/exp83_gnn_ttpla_512512_from_exp80/ep0006_model.pth \
  --dataset-root "$DATASET_TTPLA" \
  --split test --gpu
```

---

## Dataset preprocessing scripts

1024×1024 TTPLA tiles and the KEPCO dataset are built with helper scripts under `scripts/`:

| Script | Purpose |
|--------|---------|
| `scripts/build_downsampled_yolino_dataset.py` | Core resize / label scaling utilities used by the tiling pipeline |
| `scripts/build_ttpla_style_dataset.py` | CVAT XML or LabelMe → YOLinO `{images,labels}/{train,val,test}` layout |
| `scripts/prepare_kepco_yolino.py` | KEPCO CVAT → dual-crop 1024² YOLinO dataset |

Example (1024² TTPLA tiles from CVAT):

```bash
python scripts/build_ttpla_style_dataset.py cvat-xml \
  --cvat-xml /path/to/annotations.xml \
  --image-dir /path/to/images \
  --guide-dir data_guide \
  --output-root ttpla_yolino_dataset_1024x1024 \
  --target-width 1024
```

---

## Project layout

```
CAPSTONE/
├── run.sh                              # main training launcher (DDP)
├── configs/experiments/
│   ├── exp80_ttpla_512512_scale16.yaml           # Stage 1 (geom)
│   ├── exp83_gnn_ttpla_512512_from_exp80.yaml    # Stage 2 (GNN, final — ep6)
│   ├── exp82_gnn_ttpla_512512_from_exp80.yaml    # Stage 2 (prev.)
│   └── exp19_fpn_bottomup_p4_num_predictors4_1024.yaml  # 1024² ablation
├── scripts/
│   ├── build_downsampled_yolino_dataset.py       # tiling / resize utilities
│   ├── build_ttpla_style_dataset.py              # TTPLA / CVAT dataset builder
│   └── prepare_kepco_yolino.py                   # KEPCO dataset builder
├── src/yolino/
│   ├── train.py                        # training entry
│   ├── predict.py                      # inference + visualisation
│   ├── eval.py                         # built-in evaluation
│   ├── dataset/ttpla.py                # TTPLA dataset loader
│   ├── model/yolino_gnn_head.py        # GAT-based GNN head
│   └── tools/benchmark_yolino_fps.py   # FPS benchmark
├── eval_isq/
│   ├── isq_core.py                     # ISQ metric
│   └── eval_isq.py                     # ISQ evaluation runner
├── eval_pixel_f1/
│   └── eval_pixel_f1.py                # Pixel F1 / LSNetv2-style metrics
└── ttpla_train_exp/
    └── log/checkpoints/                # saved weights (gitignored)
```

---

## References

If you use this code, please cite the original YOLinO paper and the GAT architecture used in our graph head:

```bibtex
@inproceedings{meyer2021yolino,
  title={YOLinO: Generic Single Shot Polyline Detection in Real Time},
  author={Meyer, Annika and Skudlik, Philipp and Pauls, Jan-Hendrik and Stiller, Christoph},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},
  pages={2916--2925},
  year={2021}
}

@inproceedings{velickovic2018graph,
  title={Graph Attention Networks},
  author={Veli{\v{c}}kovi{\'c}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`o}, Pietro and Bengio, Yoshua},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
```

**Links**
- YOLinO (upstream): https://github.com/KIT-MRT/YOLinO
- Graph Attention Networks: https://arxiv.org/abs/1710.10903
- TTPLA dataset (original): cite the TTPLA source paper when using the benchmark tiles

---

## Acknowledgments

This project extends the open-source [YOLinO](https://github.com/KIT-MRT/YOLinO) framework (Karlsruhe Institute of Technology). CAPSTONE-specific changes focus on TTPLA powerline detection with a GNN-based instance assembly stage, developed at the [Korea Institute of Energy Technology](https://www.kentech.ac.kr/) under the supervision of [Seokju Lee](https://github.com/SeokjuLee).
