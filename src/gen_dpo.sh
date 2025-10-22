export CUDA_VISIBLE_DEVICES=2
python -u gen_multipt.py \
	model=qwen18\
	model.archive=sparse_dpo_qwen18_ep6\
	exp_name=eval_sparse_dpo_qwen18
