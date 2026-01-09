export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
#MODEL="pythia410m"
MODEL="qwen18"
DATASET="ultrafb"
#DATASET="hh"
N_EPOCHS=14
N_EXAMPLES=70000
EVAL_EVERY=500
DATE="0109"
python -u train.py \
    loss=dpo \
    loss.beta=0.1 \
    datasets=$DATASET \
    model=$MODEL \
    exp_name="dpo_base_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
    trainer=BasicTrainer \
    n_epochs=$N_EPOCHS \
    n_examples=$N_EXAMPLES \
    save_ckp=true \
    eval_every=$EVAL_EVERY

#export CUDA_VISIBLE_DEVICES=3
#export CUDA_LAUNCH_BLOCKING=1
#MODEL="pythia410m"
#MODEL="qwen18"
#DATASET="ultrafb"
#DATASET="hh"
#ALPHA=1.5
#BETA=1.0
#USING_NS=true

#python -u train.py \
#       loss=ent_dpo \
#       loss.beta=0.1 \
#       loss.alpha=$ALPHA \
#       loss.ent_beta=$BETA\
#       loss.using_ns=$USING_NS\
#       datasets=$DATASET \
#       model=$MODEL \
#       exp_name="dpo_entmax_${ALPHA}_beta_${BETA}_usingNs_${USING_NS}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
#       trainer=BasicTrainer \
#       using_extra_ce=false\
#       n_epochs=$N_EPOCHS \
#       n_examples=$N_EXAMPLES \
#       save_ckp=true \
#       eval_every=$EVAL_EVERY

