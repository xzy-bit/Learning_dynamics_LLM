export CUDA_VISIBLE_DEVICES=0
python -u gen_multipt.py \
	model=qwen18\
	model.archive=dpo_gated_qwen18_hh_ep16_1214\
	exp_name=eval_gated_qwen18_hh_ep16_1214
