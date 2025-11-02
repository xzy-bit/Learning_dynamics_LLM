export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
MODEL="pythia410m"
# MODEL= "qwen18"
# DATASET="ultrafb"
DATASET="hh"
N_EPOCHS=6

python -u train.py \
    loss=dpo \
    loss.beta=0.1 \
    model=$MODEL \
    exp_name="base_dpo_${DATASET}_${MODEL}_ep${N_EPOCHS}"\
    trainer=BasicTrainer \
    n_epochs=$N_EPOCHS \
    n_examples=30000 \
    model.archive="base_${DATASET}_sft_${MODEL}_ep8" \
    save_ckp=true \
    eval_every=1000
