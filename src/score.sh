export CUDA_VISIBLE_DEVICES=3

BASE_DIR="./exp_results"
INPUT_NAME="prob_test_gen_response.jsonl"
OUTPUT_NAME="prmpt_resp_reward_new.jsonl"
RW_MODEL="Skywork/Skywork-Reward-V2-Llama-3.1-8B"

find "$BASE_DIR" -type f -name "$INPUT_NAME" | while read -r INPUT_FILE; do
    EXP_DIR=$(dirname "$INPUT_FILE")
    OUTPUT_FILE="${EXP_DIR}/${OUTPUT_NAME}"

    if [ -f "$OUTPUT_FILE" ]; then
        echo "[SKIP] Already scored: $OUTPUT_FILE"
        continue
    fi

    echo "[RUN ] Scoring $INPUT_FILE"

    python score_response.py \
	--input_file $INPUT_FILE \
	--output_file $OUTPUT_FILE \
	--rw_model $RW_MODEL \
	--max_length 1024

done
