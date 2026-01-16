import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================================================
# Prompt / Response handling
# =========================================================

def split_prompt_response(raw_prompt: str, raw_response: str):
    """
    正确拆分多轮对话的 prompt / response

    约定：
    - raw_prompt 必须以 '\n\nAssistant:' 结尾
    - raw_response 可能包含 prompt 的 echo，也可能只包含回答

    返回：
    - prompt: 原样保留（包含完整多轮对话 + Assistant:）
    - response: 仅最后一轮 assistant 的纯回答文本
    """

    prompt = raw_prompt.strip()

    if not prompt.endswith("Assistant:"):
        raise ValueError(
            "Invalid prompt format: prompt must end with 'Assistant:'\n"
            f"Got prompt tail: {prompt[-50:]}"
        )

    response = raw_response

    # 情况 1：模型把 prompt 原样 echo 了
    if response.startswith(prompt):
        response = response[len(prompt):]

    response = response.strip()

    return prompt, response


# =========================================================
# Main
# =========================================================

def main(args):
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    # ---------- load reward model ----------
    tokenizer = AutoTokenizer.from_pretrained(args.rw_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.rw_model,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    device = model.device
    print(f"[INFO] Loaded reward model on {device}")

    # ---------- scoring ----------
    with input_path.open(encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line_idx, line in tqdm(enumerate(fin), desc="Scoring"):
            item = json.loads(line)

            raw_prompt = item["prompt"]
            raw_response_chosen = item["chosen"]
            raw_response_rejected = item["rejected"]
            try:
                prompt, response_chosen = split_prompt_response(
                    raw_prompt, raw_response_chosen
                )
                prompt, response_rejected = split_prompt_response(
                    raw_prompt, raw_response_rejected
                )
            except Exception as e:
                print(f"[WARN] Skip line {line_idx}: {e}")
                continue

            # Reward model input = 完整对话 + 最后一轮 assistant
            rm_input_chosen = prompt + response_chosen

            inputs_chosen = tokenizer(
                rm_input_chosen,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length
            ).to(device)

            with torch.no_grad():
                reward_chosen = model(**inputs_chosen).logits.squeeze().item()
            
            rm_input_rejected = prompt + response_rejected
            
            inputs_rejected = tokenizer(
                rm_input_rejected,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length
            ).to(device)
            
            with torch.no_grad():
                reward_rejected = model(**inputs_rejected).logits.squeeze().item()

            fout.write(json.dumps({
                "prompt": prompt,
                "chosen": response_chosen,
                "rejected": response_rejected,
                "chosen_reward": reward_chosen,
                "rejected_reward": reward_rejected
            }, ensure_ascii=False) + "\n")

    print(f"[DONE] Saved scored results to {output_path}")


# =========================================================
# Argparse
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score multi-turn prompt-response pairs with a reward model"
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input jsonl (must contain prompt & response)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output jsonl (prompt, response, reward)"
    )
    parser.add_argument(
        "--rw_model",
        type=str,
        required=True,
        help="HuggingFace reward model name or path"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Max token length for reward model input"
    )
    args = parser.parse_args()
    main(args)
