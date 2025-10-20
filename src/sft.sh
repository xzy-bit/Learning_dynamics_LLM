export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

MODEL="qwen18"
EXP_NAME="base_sft_qwen18_ep4"
TRAINER="BasicTrainer"
TRAIN_SPLIT="train_dpo"
N_EPOCHS=4
N_EXAMPLES=20000
SAVE_CKP="true"
EVAL_EVERY=1000

python -u train.py \
  model="$MODEL" \
  exp_name="$EXP_NAME" \
  trainer="$TRAINER" \
  train_split="$TRAIN_SPLIT" \
  n_epochs="$N_EPOCHS" \
  n_examples="$N_EXAMPLES" \
  save_ckp="$SAVE_CKP" \
  eval_every="$EVAL_EVERY"
