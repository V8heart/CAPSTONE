#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/work/caps_drone/yolino/venv/bin/python}"
CONFIG_PATH=""
# GPU/process count: pass explicitly via --nproc (or set NPROC_PER_NODE before invoking run.sh).
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DVC_DIR="${DVC_DIR:-$ROOT_DIR/ttpla_train_exp}"
LOG_DIR="${LOG_DIR:-ttpla_experiments}"
DATASET_ROOT="${DATASET_ROOT:-${DATASET_TTPLA:-}}"
EXTRA_ARGS=()

usage() {
  echo "Usage: bash run.sh --config <config.yaml> [--dataset-root <path>] [--nproc <gpus>] [--log-dir <name>] [-- extra train args]"
  echo "  --nproc     torchrun processes per node (typically GPU count). Default: 4 or \$NPROC_PER_NODE."
  echo "Example: bash run.sh --config configs/experiments/exp05_convnext_baseline_1024.yaml --dataset-root ... --nproc 4 -- ..."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --nproc)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --dvc-dir)
      DVC_DIR="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$CONFIG_PATH" ]]; then
  usage
  exit 1
fi

if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$ROOT_DIR/$CONFIG_PATH"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] Config not found: $CONFIG_PATH"
  exit 1
fi

if [[ -z "$DATASET_ROOT" ]]; then
  echo "[ERROR] DATASET_TTPLA or --dataset-root must be set."
  exit 1
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] Dataset root not found: $DATASET_ROOT"
  exit 1
fi

export DATASET_TTPLA="$DATASET_ROOT"
export PYTHONPATH="$ROOT_DIR/src"
export YOLINO_IGNORE_DIRTY=true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

mkdir -p "$DVC_DIR"
REL_CONFIG_PATH="$CONFIG_PATH"
if command -v realpath >/dev/null 2>&1; then
  REL_CONFIG_PATH="$(realpath --relative-to="$DVC_DIR" "$CONFIG_PATH")"
fi

echo "[INFO] ROOT_DIR=$ROOT_DIR"
echo "[INFO] CONFIG=$CONFIG_PATH"
echo "[INFO] CONFIG_REL_TO_DVC=$REL_CONFIG_PATH"
echo "[INFO] DATASET_TTPLA=$DATASET_TTPLA"
echo "[INFO] NPROC_PER_NODE=$NPROC_PER_NODE"
echo "[INFO] LOG_DIR=$LOG_DIR"

cd "$DVC_DIR"

"$PYTHON_BIN" -m torch.distributed.run \
  --rdzv_backend=c10d \
  --nproc_per_node="$NPROC_PER_NODE" \
  "$ROOT_DIR/src/yolino/train.py" -c "$REL_CONFIG_PATH" \
  --root "$ROOT_DIR" \
  --dvc "$DVC_DIR" \
  --log_dir "$LOG_DIR" \
  --gpu \
  "${EXTRA_ARGS[@]}"
