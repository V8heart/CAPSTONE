#!/usr/bin/env bash
# ---------------------------------------------------------------------------- #
# YOLinO + ConvNeXt-Tiny + FPN 학습 실행 스크립트 (TTPLA)
#
# 두 가지 모드를 지원합니다.
#   1) two_stage  : Stage1(geom only) → Stage2(geom + embedding joint)  [기본]
#   2) one_stage  : 처음부터 geom + embedding 동시 학습 (warmup 적용)
#
# 사용:
#   bash run_train.sh                # two_stage (권장)
#   bash run_train.sh two_stage
#   bash run_train.sh one_stage
#   NPROC_PER_NODE=4 bash run_train.sh two_stage   # DDP (4GPU)
# ---------------------------------------------------------------------------- #
set -euo pipefail

MODE="${1:-two_stage}"

REPO_ROOT="/home/work/caps_drone/yolino/YOLinO"
EXP_DIR="${REPO_ROOT}/ttpla_train_exp"
# Default to the prepared 30% subset; override with DATA_DIR=... if needed.
DATA_DIR="${DATA_DIR:-/home/work/caps_drone/yolino/TTPLA_YOLinO_Dataset_30p}"
PYBIN="/home/work/caps_drone/yolino/venv/bin/python"

RUN_NAME="${RUN_NAME:-ttpla_convnext_fpn_p3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

export DATASET_TTPLA="${DATA_DIR}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export YOLINO_IGNORE_DIRTY=true

COMMON_ARGS=(
  --root "${REPO_ROOT}"
  --dvc  "${EXP_DIR}"
  --log_dir "ttpla_convnext_fpn_p3"
  --gpu
  --run_name "${RUN_NAME}"
)

run_stage() {
  local cfg="$1"
  local cfg_base
  cfg_base="$(basename "${cfg}")"
  echo "============================================================"
  echo " Run: ${cfg}"
  echo "============================================================"
  pushd "${EXP_DIR}" >/dev/null
  if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    "${PYBIN}" -m torch.distributed.run \
      --standalone \
      --nproc_per_node="${NPROC_PER_NODE}" \
      "${REPO_ROOT}/src/yolino/train.py" \
      -c "${cfg_base}" \
      "${COMMON_ARGS[@]}"
  else
    "${PYBIN}" "${REPO_ROOT}/src/yolino/train.py" \
      -c "${cfg_base}" \
      "${COMMON_ARGS[@]}"
  fi
  popd >/dev/null
}

case "${MODE}" in
  two_stage)
    run_stage "${EXP_DIR}/params.yaml"            # Stage 1: geometry-first
    run_stage "${EXP_DIR}/params_stage2.yaml"     # Stage 2: + embedding (warmup)
    ;;
  one_stage)
    run_stage "${EXP_DIR}/params_stage2.yaml"     # joint w/ warmup from epoch 0
    ;;
  *)
    echo "Unknown MODE='${MODE}'. Use 'two_stage' or 'one_stage'."
    exit 2
    ;;
esac
