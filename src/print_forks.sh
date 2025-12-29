export CUDA_VISIBLE_DEVICES=0

python -u print_forking_token.py \
  --base_model Qwen/Qwen1.5-1.8B \
  --archive sft_qwen18_hh_ep2_1202\
  --ckpt policy.pt\
  --json_path forking_tokens.json \
  --tau 0.8
