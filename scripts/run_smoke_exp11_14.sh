#!/usr/bin/env bash
# exp11–13: ttpla_yolino_dataset_1024_downsample + 분할 GPU (--nproc 4)
# exp14: Sample_YOLinO + 단일 GPU
# 한 단계 실패해도 다음 단계 진행하려면: set +e (아래 기본값).
set +e
set -u

cd /home/work/caps_drone/yolino/CAPSTONE || exit 1

TTPLA_FULL="/home/work/caps_drone/yolino/ttpla_yolino_dataset_1024_downsample"
SAMPLE_DS="/home/work/caps_drone/yolino/Sample_YOLinO"

# exp11–13: 분할 GPU (예: 4GPU → CUDA 0–3, --nproc 4)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh \
  --config configs/experiments/exp11_fpn_bottomup_p4_focal_1024.yaml \
  --dataset-root "${TTPLA_FULL}" --nproc 4 -- \
  --epoch 1 --run_name smoke_exp11_e1

CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh \
  --config configs/experiments/exp12_fpn_bottomup_p4_strict_match_1024.yaml \
  --dataset-root "${TTPLA_FULL}" --nproc 4 -- \
  --epoch 1 --run_name smoke_exp12_e1

CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh \
  --config configs/experiments/exp13_fpn_bottomup_p4_equal_direction_1024.yaml \
  --dataset-root "${TTPLA_FULL}" --nproc 4 -- \
  --epoch 1 --run_name smoke_exp13_e1

# exp14: 단일 GPU (체크포인트 epoch>=1 이면 --epoch 1 은 학습 0회 → --epoch 2 로 1스텝 스모크)
CUDA_VISIBLE_DEVICES=0 bash run.sh \
  --config configs/experiments/exp14_finetune_sample_yolino_from_exp10_1024.yaml \
  --dataset-root "${SAMPLE_DS}" --nproc 1 -- \
  --epoch 2 --run_name smoke_exp14_e1

echo "done $(date -Is)"
