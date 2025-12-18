export CUDA_VISIBLE_DEVICES=2
export CUDA_LAUNCH_BLOCKING=1
#MODEL="pythia410m"
MODEL="qwen18"
#DATASET="ultrafb"
DATASET="hh"
SFT_EPOCHS=2
N_EPOCHS=4
N_EXAMPLES=20000
EVAL_EVERY=100000
DATE=$(date +%m%d)
# sparsemax, ratio, topk
MASK_TYPE="sparsemax"
#MASK_TYPE="ratio"
#MASK_TYPE="topk"
MASK_RATIO=0.1
MASK_TOP_K=3
MASK_STRENGTH=0.0

python -u train.py \
    loss=masked_dpo \
    loss.beta=0.1 \
    loss.mask_type=$MASK_TYPE\
    loss.mask_ratio=$MASK_RATIO\
    loss.mask_top_k=$MASK_TOP_K\
    loss.mask_strength=$MASK_STRENGTH\
    datasets=$DATASET \
    model=$MODEL \
    exp_name="dpo_mask_${MASK_TYPE}_r_${MASK_RATIO}_k_${MASK_TOP_K}_s_${MASK_STRENGTH}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
    trainer=BasicTrainer \
    n_epochs=$N_EPOCHS \
    n_examples=$N_EXAMPLES \
    model.archive="sft_${MODEL}_${DATASET}_ep${SFT_EPOCHS}_1202" \
    save_ckp=true \
    eval_every=$EVAL_EVERY
