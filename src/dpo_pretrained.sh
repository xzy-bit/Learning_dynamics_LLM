export CUDA_VISIBLE_DEVICES=3
export CUDA_LAUNCH_BLOCKING=1
#MODEL="pythia410m"
MODEL="qwen18"
#DATASET="ultrafb"
DATASET="hh"
N_EPOCHS=6
N_EXAMPLES=30000
EVAL_EVERY=80000
#DATE=$(date +%m%d)
DATE="0108"
ALPHA=1.5
ENT_BETA=1.0
USING_NS=true
LAMBDAS=(0.0)
for LAMBDA in "${LAMBDAS[@]}"; do
    python -u train.py \
        loss=ent_dpo \
        loss.beta=0.1 \
        loss.alpha=$ALPHA \
        loss.ent_beta=$ENT_BETA\
        loss.using_ns=$USING_NS\
    	datasets=$DATASET \
        model=$MODEL \
    	exp_name="dpo_mixed_entmax_l_${LAMBDA}_a_${ALPHA}_b_${}_usingNs_${USING_NS}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}" \
        trainer=BasicTrainer \
        n_epochs=$N_EPOCHS \
        n_examples=$N_EXAMPLES \
        using_extra_ce=true \
        ce_lambda=$LAMBDA \
        save_ckp=true \
        eval_every=$EVAL_EVERY
        
done

for LAMBDA in "${LAMBDAS[@]}"; do
    python -u gen_multipt.py \
        model=${MODEL} \
        model.archive="dpo_mixed_entmax_l_${LAMBDA}_a_${ALPHA}_usingNs_${USING_NS}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
        exp_name="eval_dpo_mixed_entmax_l_${LAMBDA}_a_${ALPHA}_usingNs_${USING_NS}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"
done

#bash score.sh


