export CUDA_VISIBLE_DEVICES=3

INPUT_FILE="train_dpo.jsonl"
OUTPUT_FILE="train_dpo_reward.jsonl"
RW_MODEL="Skywork/Skywork-Reward-V2-Llama-3.1-8B"

python score_response.py \
	--input_file $INPUT_FILE \
	--output_file $OUTPUT_FILE \
	--rw_model $RW_MODEL \

