export CUDA_VISIBLE_DEVICES=0
python -u gen_multipt.py \
	model=llama3_1\
	model.archive=llama32_1b_dpo_extend_hh_ep6\
	exp_name=eval_llama32_1b_dpo_extend_hh_ep6
python -u gen_multipt.py \
        model=llama3_1\
        model.archive=llama32_1b_dpo_base_hh_ep6\
        exp_name=eval_llama32_1b_dpo_base_hh_ep6
bash score.sh
