#!/bin/bash

# ================== GPU & ENV ==================
export CUDA_VISIBLE_DEVICES=2
export CUDA_LAUNCH_BLOCKING=1

# ================== Experiment Config ==================
MODEL="qwen18"
DATASET="hh"
N_EPOCHS=6
DATE=$(date +%m%d)

# 训练时的保存目录（与你 train.py 一致）
SAVE_ROOT="outputs"   # 如果你 config.save_path 不是 outputs，这里改掉

# entropy 分析脚本
ANALYSIS_SCRIPT="analyze_entropy.py"

# ================== Run ==================
echo "=============================================="
echo " Running entropy analysis"
echo " Model   : ${MODEL}"
echo " Dataset : ${DATASET}"
echo " Epochs  : 1 ~ ${N_EPOCHS}"
echo " GPU     : ${CUDA_VISIBLE_DEVICES}"
echo "=============================================="

for ((EP=1; EP<=N_EPOCHS; EP++)); do
  CKPT_PATH="${SAVE_ROOT}/ep${EP}/policy.pt"

  if [ ! -f "${CKPT_PATH}" ]; then
    echo "[WARNING] Checkpoint not found: ${CKPT_PATH}, skip epoch ${EP}"
    continue
  fi

  echo "----------------------------------------------"
  echo " [Epoch ${EP}] Analyzing entropy"
  echo " Checkpoint: ${CKPT_PATH}"
  echo "----------------------------------------------"

  python -u ${ANALYSIS_SCRIPT} \
    epoch=${EP} \
    model=${MODEL} \
    datasets=${DATASET}

done

echo "=============================================="
echo " Entropy analysis finished"
echo " Results saved to:"
echo "   ${SAVE_ROOT}/entropy_analysis/"
echo "=============================================="
