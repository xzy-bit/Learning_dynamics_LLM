export CUDA_VISIBLE_DEVICES=3
export CUDA_LAUNCH_BLOCKING=1
MODEL="llama3_1"
#MODEL="pythia410m"
#MODEL="qwen18"
#DATASET="ultrafb"
DATASET="hh"
SFT_EPOCHS=1
N_EPOCHS=10
N_EXAMPLES=50000
EVAL_EVERY=100000
DATE="0113"

#LAMBDAS=(0.01)
#ALPHA=1.5
#BETA=0.25
#TEMP=1.0
#USING_NS=true


#for LAMBDA in "${LAMBDAS[@]}"; do
#python -u train.py \
#    loss=dpo \
#    loss.beta=0.1 \
    #loss.alpha=$ALPHA \
    #loss.ent_beta=$BETA\
    #loss.using_ns=$USING_NS\
    #loss.temperature=$TEMP\
#    using_extra_ce=false \
    #ce_lambda=$LAMBDA \
#    model=$MODEL \
#    datasets=$DATASET \
#    exp_name="dpo_extend_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
#    trainer=BasicTrainer \
#    n_epochs=$N_EPOCHS \
#    n_examples=$N_EXAMPLES \
#    model.archive="extend_sft_llama3_1_hh_ep1_0109" \
#    save_ckp=true \
#    eval_every=$EVAL_EVERY
#done

#python -u gen_multipt.py \
#   model=${MODEL} \
#   model.archive="dpo_extend_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}" \
#   exp_name="eval_dpo_extend_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"

python -u train.py \
    loss=dpo \
    loss.beta=0.1 \
    using_extra_ce=false \
    model=$MODEL\
    datasets=$DATASET \
    exp_name="dpo_extend_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
    trainer=BasicTrainer \
    n_epochs=$N_EPOCHS \
    n_examples=$N_EXAMPLES \
    model.archive="extend_sft_llama3_1_hh_ep1_0109"\
    save_ckp=true \
    eval_every=$EVAL_EVERY

python -u gen_multipt.py \
   model=${MODEL} \
   model.archive="dpo_extend_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}" \
   exp_name="eval_dpo_extend_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"
