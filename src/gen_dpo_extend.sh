export CUDA_VISIBLE_DEVICES=2
python -u gen_multipt.py \
	model=qwen18\
	model.archive=dpo_extend_qwen18_hh_ep6_1215\
	exp_name=eval_dpo_extend_qwen18_hh_ep6_1215
