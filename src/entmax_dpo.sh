export CUDA_VISIBLE_DEVICES=1
export CUDA_LAUNCH_BLOCKING=1
gamma=1.0
tau=1.0
python -u train.py \
    loss=entmax_dpo \
    loss.beta=0.1 \
    +loss.gamma=1.0\
    +loss.tau=1.0\
    model=qwen18 \
    exp_name=entmax_dpo_gamma_${gamma}_tau_${tau}_qwen18_ep6 \
    trainer=BasicTrainer \
    n_epochs=6 \
    n_examples=30000 \
    model.archive=base_sft_qwen18_ep8 \
    save_ckp=true \
    eval_every=100

