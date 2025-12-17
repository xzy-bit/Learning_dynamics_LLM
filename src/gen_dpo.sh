export CUDA_VISIBLE_DEVICES=2
python -u gen_multipt.py \
	model=qwen18\
	model.archive=dpo_entmax_1.5_qwen18_hh_ep4_1217\
	exp_name=eval_dpo_entmax_1.5_qwen18_hh_ep4_1217
