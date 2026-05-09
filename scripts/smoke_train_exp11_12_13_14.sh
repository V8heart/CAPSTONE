#!/usr/bin/env bash
# exp11·12·13: 전체 TTPLA + DDP 멀티 GPU (--nproc NPROC_DDP).
# exp14: Sample_YOLinO + 단일 GPU (--nproc 1).
#
# 환경변수 예:
#   NPROC_DDP=4           # exp11/12/13 torchrun 프로세스 수 (= 보통 가시 GPU 개수)
#   CUDA_MULTI=0,1,2,3    # 미설정 시 0..NPROC_DDP-1 자동
#   CUDA_SINGLE=0         # exp14 전용 GPU
#   TTPLA_FULL, SAMPLE_DS, PER_RUN_TIMEOUT_SEC
set -u
set +e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || exit 1

# 기본: make_square_crops 1024² 듀얼 크롭 패키지. 더 많은 타일 쓰려면 TTPLA_FULL=/home/work/caps_drone/yolino/TTPLA_YOLinO_Dataset 등으로 덮어쓰기.
TTPLA_FULL="${TTPLA_FULL:-/home/work/caps_drone/yolino/ttpla_yolino_dataset_1024_downsample}"
SAMPLE_DS="${SAMPLE_DS:-/home/work/caps_drone/yolino/Sample_YOLinO}"
PER_RUN_TIMEOUT_SEC="${PER_RUN_TIMEOUT_SEC:-14400}"

NPROC_DDP="${NPROC_DDP:-4}"
if [[ -z "${CUDA_MULTI:-}" ]]; then
  CUDA_MULTI=""
  for ((i = 0; i < NPROC_DDP; i++)); do
    [[ -n "${CUDA_MULTI}" ]] && CUDA_MULTI+=","
    CUDA_MULTI+="${i}"
  done
fi
CUDA_SINGLE="${CUDA_SINGLE:-0}"

run_one() {
  local tag="$1"
  shift
  echo ""
  echo "========== ${tag}  $(date -Is) =========="
  local ec=0
  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after=300 "${PER_RUN_TIMEOUT_SEC}" "$@"
    ec=$?
  else
    "$@"
    ec=$?
  fi
  if [[ "${ec}" -eq 124 ]]; then
    echo "[RESULT] ${tag}  TIMEOUT (exit ${ec})"
  elif [[ "${ec}" -eq 137 ]]; then
    echo "[RESULT] ${tag}  KILLED (SIGKILL / OOM 가능, exit ${ec})"
  elif [[ "${ec}" -ne 0 ]]; then
    echo "[RESULT] ${tag}  FAIL (exit ${ec})"
  else
    echo "[RESULT] ${tag}  OK"
  fi
  echo "==========================================="
  return 0
}

echo "[INFO] ROOT=${ROOT}"
echo "[INFO] exp11/12/13  CUDA_VISIBLE_DEVICES=${CUDA_MULTI}  --nproc ${NPROC_DDP}"
echo "[INFO] exp14        CUDA_VISIBLE_DEVICES=${CUDA_SINGLE}  --nproc 1"

# exp11–13: 전체 TTPLA + 분산 GPU
run_one "smoke_exp11_e1" \
  env CUDA_VISIBLE_DEVICES="${CUDA_MULTI}" \
  bash run.sh --config configs/experiments/exp11_fpn_bottomup_p4_focal_1024.yaml \
    --dataset-root "${TTPLA_FULL}" --nproc "${NPROC_DDP}" -- \
    --epoch 1 --run_name smoke_exp11_e1

run_one "smoke_exp12_e1" \
  env CUDA_VISIBLE_DEVICES="${CUDA_MULTI}" \
  bash run.sh --config configs/experiments/exp12_fpn_bottomup_p4_strict_match_1024.yaml \
    --dataset-root "${TTPLA_FULL}" --nproc "${NPROC_DDP}" -- \
    --epoch 1 --run_name smoke_exp12_e1

run_one "smoke_exp13_e1" \
  env CUDA_VISIBLE_DEVICES="${CUDA_MULTI}" \
  bash run.sh --config configs/experiments/exp13_fpn_bottomup_p4_equal_direction_1024.yaml \
    --dataset-root "${TTPLA_FULL}" --nproc "${NPROC_DDP}" -- \
    --epoch 1 --run_name smoke_exp13_e1

# exp14만 Sample_YOLinO + 단일 GPU
# --epoch 는 학습 루프 상한(미포함): 체크포인트 epoch >= 1 이면 --epoch 1 은 루프 0회 → 최소 --epoch (ckpt+1)
run_one "smoke_exp14_e1" \
  env CUDA_VISIBLE_DEVICES="${CUDA_SINGLE}" \
  bash run.sh --config configs/experiments/exp14_finetune_sample_yolino_from_exp10_1024.yaml \
    --dataset-root "${SAMPLE_DS}" --nproc 1 -- \
    --epoch 2 --run_name smoke_exp14_e1

echo "All queued smokes finished  $(date -Is)"
