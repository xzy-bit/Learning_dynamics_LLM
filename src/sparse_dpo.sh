export CUDA_VISIBLE_DEVICES=2
export CUDA_LAUNCH_BLOCKING=1
python -u train.py \
    loss=sp_dpo \
    loss.beta=0.1 \
    model=qwen18 \
    exp_name=sparse_dpo_ultrafb_qwen18_ep6 \
    trainer=BasicTrainer \
    n_epochs=6 \
    n_examples=30000 \
    model.archive=base_ultrfb_sft_qwen18_ep8 \
    save_ckp=true \
    eval_every=1000

