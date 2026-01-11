export CUDA_VISIBLE_DEVICES=2
export CUDA_LAUNCH_BLOCKING=1
MODEL="llama3_1"
#MODEL="pythia410m"
#MODEL="qwen18"
#DATASET="ultrafb"
DATASET="hh"
N_EPOCHS=6
N_EXAMPLES=30000
EVAL_EVERY=80000
#DATE=$(date +%m%d)
DATE="0110"
ALPHA=1.5
ENT_BETA=1.0
USING_NS=true
LAMBDAS=(0.1 0.5)
for LAMBDA in "${LAMBDAS[@]}"; do
    python -u train.py \
        loss=dpo \
	loss.beta=0.1\
    	datasets=$DATASET \
        model=$MODEL \
    	exp_name="dpo_mixed_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}" \
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
        model.archive="dpo_mixed_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
        exp_name="eval_dpo_mixed_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"
done

#bash score.sh


