export CUDA_VISIBLE_DEVICES=0

BASE_DIR="./exp_results"
INPUT_NAME="prob_train_gen_response.jsonl"
OUTPUT_NAME="prmpt_resp_reward_train.jsonl"
RW_MODEL="Skywork/Skywork-Reward-V2-Llama-3.1-8B"

find "$BASE_DIR" -type f -name "$INPUT_NAME" | while read -r INPUT_FILE; do
    EXP_DIR=$(dirname "$INPUT_FILE")
    OUTPUT_FILE="${EXP_DIR}/${OUTPUT_NAME}"

    INPUT_LINES=$(wc -l < "$INPUT_FILE")

    if [ -f "$OUTPUT_FILE" ]; then
        OUTPUT_LINES=$(wc -l < "$OUTPUT_FILE")

        if [ "$INPUT_LINES" -eq "$OUTPUT_LINES" ]; then
            echo "[SKIP] Already scored (lines match): $OUTPUT_FILE"
            continue
        else
            echo "[RERUN] Line mismatch (input: $INPUT_LINES, output: $OUTPUT_LINES), re-scoring"
        fi
    else
        echo "[RUN ] Output not found, scoring $INPUT_FILE"
    fi

    python score_response.py \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_FILE" \
        --rw_model "$RW_MODEL" \
        --max_length 1024

done
BASE_DIR="./exp_results"
INPUT_NAME="prob_test_gen_response.jsonl"
OUTPUT_NAME="prmpt_resp_reward_test.jsonl"
RW_MODEL="Skywork/Skywork-Reward-V2-Llama-3.1-8B"

find "$BASE_DIR" -type f -name "$INPUT_NAME" | while read -r INPUT_FILE; do
    EXP_DIR=$(dirname "$INPUT_FILE")
    OUTPUT_FILE="${EXP_DIR}/${OUTPUT_NAME}"

    INPUT_LINES=$(wc -l < "$INPUT_FILE")

    if [ -f "$OUTPUT_FILE" ]; then
        OUTPUT_LINES=$(wc -l < "$OUTPUT_FILE")

        if [ "$INPUT_LINES" -eq "$OUTPUT_LINES" ]; then
            echo "[SKIP] Already scored (lines match): $OUTPUT_FILE"
            continue
        else
            echo "[RERUN] Line mismatch (input: $INPUT_LINES, output: $OUTPUT_LINES), re-scoring"
        fi
    else
        echo "[RUN ] Output not found, scoring $INPUT_FILE"
    fi

    python score_response.py \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_FILE" \
        --rw_model "$RW_MODEL" \
        --max_length 1024

done
