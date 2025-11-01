export CUDA_VISIBLE_DEVICES=1
export CUDA_LAUNCH_BLOCKING=1
python -u train.py \
    ldoss=sp_dpo \
    loss.beta=0.1 \
    model=qwen18 \
    exp_name=sparse_dpo_qwen18_ep6 \
    trainer=BasicTrainer \
    n_epochs=6 \
    n_examples=30000 \
    model.archive=base_sft_qwen18_ep8 \
    save_ckp=true \
    eval_every=1000

