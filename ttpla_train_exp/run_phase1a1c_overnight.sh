#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Phase 1A/1C overnight runner (TTPLA / YOLinO)
#
# Goal for tonight:
# - Keep Phase 1B (FPN on/off) excluded.
# - Start AFTER current manual 0429darknet run finishes.
# - Reproduce ConvNeXt scale-up baseline settings (same as 0428scale32 core setup):
#   * backbone=convnext (default), scale=32, head_level=P4
#   * 4-GPU DDP, amp=false, loading_workers=4
#   * same dataset subset path and environment
# - Apply Phase 1C toggles independently:
#   1) confidence loss: mse_mean vs focal_mean
#   2) hard bipartite loss matching: on vs off
# - Run each variant for 50 epochs with unique run names to keep TensorBoard separation.
#
# IMPORTANT:
# - This script does NOT touch the currently running process.
# - It waits for any active run_name=0429darknet training to finish first.
# ---------------------------------------------------------------------------

REPO_ROOT="/home/work/caps_drone/yolino/YOLinO"
EXP_DIR="${REPO_ROOT}/ttpla_train_exp"
PYBIN="/home/work/caps_drone/yolino/venv/bin/python"
TRAIN_PY="${REPO_ROOT}/src/yolino/train.py"

DATASET_TTPLA_DEFAULT="/home/work/caps_drone/yolino/TTPLA_YOLinO_Dataset_30p"
export DATASET_TTPLA="${DATASET_TTPLA:-${DATASET_TTPLA_DEFAULT}}"
export PYTHONPATH="${REPO_ROOT}/src"
export YOLINO_IGNORE_DIRTY=true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
EPOCHS="${EPOCHS:-50}"
BASE_LOG_DIR="${BASE_LOG_DIR:-ttpla_convnext_fpn_p3}"

cd "${EXP_DIR}"

echo "[INFO] Waiting for currently running 0429darknet to finish..."
while pgrep -f "yolino/train.py.*--run_name 0429darknet( |$)" >/dev/null; do
  sleep 30
done
echo "[INFO] No active 0429darknet process found. Starting Phase 1A/1C runs."

# ---------------------------------------------------------------------------
# Shared command parts (0428scale32-equivalent core settings)
# ---------------------------------------------------------------------------
COMMON_ARGS=(
  -c params.yaml
  --root "${REPO_ROOT}"
  --dvc "${EXP_DIR}"
  --log_dir "${BASE_LOG_DIR}"
  --gpu
  --scale 32
  --head_level P4
  --amp false
  --loading_workers 4
  --epoch "${EPOCHS}"
  --retrain
)

# ---------------------------------------------------------------------------
# Phase 1C matrix on ConvNeXt baseline:
#   A) mse + hard matching ON   (reference)
#   B) focal + hard matching ON
#   C) mse + hard matching OFF
#   D) focal + hard matching OFF
# ---------------------------------------------------------------------------

declare -a RUN_NAMES=(
  "0501_convnext_s32_p1c_mse_hardon_e${EPOCHS}"
  "0501_convnext_s32_p1c_focal_hardon_e${EPOCHS}"
  "0501_convnext_s32_p1c_mse_hardoff_e${EPOCHS}"
  "0501_convnext_s32_p1c_focal_hardoff_e${EPOCHS}"
)

declare -a LOSS_ARGS=(
  "norm_mean,mse_mean"
  "norm_mean,focal_mean"
  "norm_mean,mse_mean"
  "norm_mean,focal_mean"
)

declare -a HARD_MATCH_ARGS=(
  "true"
  "true"
  "false"
  "false"
)

for i in "${!RUN_NAMES[@]}"; do
  RUN_NAME="${RUN_NAMES[$i]}"
  LOSS_SPEC="${LOSS_ARGS[$i]}"
  HARD_MATCH="${HARD_MATCH_ARGS[$i]}"
  RDZV_PORT=$((29610 + i))
  RDZV_ID="phase1a1c_${RUN_NAME}"

  echo "------------------------------------------------------------------"
  echo "[RUN] ${RUN_NAME}"
  echo "  loss=${LOSS_SPEC}"
  echo "  loss_hard_matching=${HARD_MATCH}"
  echo "  conf_match_weight=5,1"
  echo "  confidence=0.3"
  echo "  nproc=${NPROC_PER_NODE}, epoch=${EPOCHS}, scale=32, head=P4"
  echo "------------------------------------------------------------------"

  "${PYBIN}" -m torch.distributed.run \
    --rdzv_backend=c10d \
    --rdzv_endpoint="127.0.0.1:${RDZV_PORT}" \
    --rdzv_id="${RDZV_ID}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    "${TRAIN_PY}" \
    "${COMMON_ARGS[@]}" \
    --run_name "${RUN_NAME}" \
    --loss "${LOSS_SPEC}" \
    --loss_hard_matching "${HARD_MATCH}" \
    --conf_match_weight 5,1 \
    --confidence 0.3
done

echo "[DONE] Phase 1A/1C overnight matrix finished."
