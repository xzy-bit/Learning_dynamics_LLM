export CUDA_VISIBLE_DEVICES=0
python -u train.py \
  loss=dpo \
  loss.beta=0.1 \
  model=qwen18 \
  exp_name=extend_dpo_qwen18_ep6 \
  trainer=BasicTrainer \
  n_epochs=6 \
  n_examples=30000 \
  model.archive=extend_qwen18_ep4 \
  save_ckp=true \
  eval_every=1000

