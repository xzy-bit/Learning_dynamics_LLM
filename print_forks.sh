export CUDA_VISIBLE_DEVICES=0
python -u print_forking_token.py \
  --model qwen18 \
  --json_path forking_tokens.json \
  --tau 0.8
