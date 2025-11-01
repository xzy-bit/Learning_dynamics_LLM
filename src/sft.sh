export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=3
MODEL="qwen18"
EXP_NAME="base_sft_qwen18_ep8"
TRAINER="BasicTrainer"
TRAIN_SPLIT="train_dpo"
N_EPOCHS=8
N_EXAMPLES=40000
SAVE_CKP="true"
EVAL_EVERY=1000

python -u train.py \
  model="qwen18" \
  exp_name="base_ultrfb_sft_qwen18_ep8" \
  datasets="ultrafb" \
  trainer="BasicTrainer" \
  train_split="train_dpo" \
  n_epochs="8" \
  n_examples="40000" \
  save_ckp="true" \
  eval_every="1000"
