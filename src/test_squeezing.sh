export CUDA_VISIBLE_DEVICES=1
MODEL="qwen18"
#MODEL="pythia410m"
#DATASET="hh"
DATASET="ultrafb"
TRAINER="BasicTrainer"
TRAIN_SPLIT="train_dpo"
N_EPOCHS=8
N_EXAMPLES=40000
EVAL_EVERY=1000
DATE="0109"
#python -u train.py \
#  model=$MODEL \
#  exp_name="sft_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}" \
#  datasets=$DATASET \
#  trainer="BasicTrainer" \
#  train_split="train_dpo" \
#  n_epochs=$N_EPOCHS \
#  n_examples=$N_EXAMPLES \
#  save_ckp="true" \
#  eval_every=$EVAL_EVERY


#export CUDA_VISIBLE_DEVICES=1
export CUDA_LAUNCH_BLOCKING=1
#MODEL="pythia410m"
MODEL="qwen18"
DATASET="ultrafb"
#DATASET="hh"
SFT_EPOCHS=8
N_EPOCHS=6
N_EXAMPLES=30000
EVAL_EVERY=500
python -u train.py \
    loss=dpo \
    loss.beta=0.1 \
    datasets=$DATASET \
    model=$MODEL \
    exp_name="dpo_base_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
    trainer=BasicTrainer \
    n_epochs=$N_EPOCHS \
    n_examples=$N_EXAMPLES \
    model.archive="sft_${MODEL}_${DATASET}_ep${SFT_EPOCHS}_${DATE}"\
    save_ckp=true \
    eval_every=$EVAL_EVERY
