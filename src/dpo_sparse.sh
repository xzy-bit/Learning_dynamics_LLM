export CUDA_VISIBLE_DEVICES=2
export CUDA_LAUNCH_BLOCKING=1
# MODEL="qwen18"
MODEL="pythia410m"
N_EPOCHS=6
# DATASET="ultrafb"
DATASET="hh"

python -u train.py \
    loss=sp_dpo \
    loss.beta=0.1 \
    model=$MODEL \
    exp_name="sparse_dpo_${DATASET}_${MODEL}_ep6" \
    trainer=BasicTrainer \
    n_epochs=6 \
    n_examples=30000 \
    model.archive="base_${DATASET}_sft_${MODEL}_ep8" \
    save_ckp=true \
    eval_every=1000
