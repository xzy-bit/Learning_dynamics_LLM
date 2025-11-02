export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1
# MODEL="qwen18"
MODEL="pythia410m"
DATASETS="hh"
TRAIN_SPLIT="train_sft_extend"
N_EPOCHS=4
N_EXAMPLES=40000
EVAL_EVERY=1000

python -u train.py \
  model=$MODEL \
  datasets=$DATASETS \
  exp_name="extend_${DATASETS}_sft_${MODEL}_ep${N_EPOCHS}" \
  trainer=BasicTrainer \
  train_split=train_sft_extend \
  n_epochs=$N_EPOCHS \
  n_examples=$N_EXAMPLES \
  save_ckp="true" \
  eval_every=1000
