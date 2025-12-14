export CUDA_VISIBLE_DEVICES=2
python -u gen_multipt.py \
	model=qwen18\
	model.archive=dpo_no_normal_masked_qwen18_hh_ep16_1213-2\
	exp_name=dpo_no_normal_masked_qwen18_hh_ep16_1213-2
