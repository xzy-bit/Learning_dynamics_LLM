export CUDA_VISIBLE_DEVICES=0
python -u gen_inference.py \
	model=qwen18\
	model.archive=dpo_sparse_qwen18_hh_ep6_1221\
	exp_name=gen_dpo_sparse_qwen18_hh_ep6_1221
