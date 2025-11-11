export CUDA_VISIBLE_DEVICES=1
python -u gen_multipt.py \
	model=qwen18\
	model.archive=masked_dpo_hh_qwen18_ep6\
	exp_name=eval_masked_dpo_hh_qwen_ep6_11_11
