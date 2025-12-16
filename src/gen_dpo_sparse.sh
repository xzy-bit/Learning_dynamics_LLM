export CUDA_VISIBLE_DEVICES=2
python -u gen_multipt.py \
	model=qwen18\
	model.archive=dpo_sparse_qwen18_hh_ep6_1215\
	exp_name=eval_dpo_sparse_qwen18_hh_ep6_1215
