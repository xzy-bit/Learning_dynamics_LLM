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
    raw_prompt 必须以 'Assistant:' 结尾
    raw_response 可能包含 prompt echo
    """
    prompt = raw_prompt.strip()

    if not prompt.endswith("Assistant:"):
        raise ValueError(
            "Invalid prompt format: prompt must end with 'Assistant:'\n"
            f"Got prompt tail: {prompt[-50:]}"
        )

    response = raw_response
    if response.startswith(prompt):
        response = response[len(prompt):]

    response = response.strip()
    return prompt, response


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch score prompt-response pairs with a reward model"
    )
    parser.add_argument(
        "--input_list",
        type=str,
        required=True,
        help="Text file, one input jsonl path per line"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="Output jsonl filename (saved next to input file)"
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
        default=1024
    )

    args = parser.parse_args()

    # ---------- load reward model ONCE ----------
    tokenizer = AutoTokenizer.from_pretrained(args.rw_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.rw_model,
        torch_dtype=torch.float16,
        device_map="cuda"
    ).eval()

    device = model.device
    print(f"[INFO] Reward model loaded on {device}")

    # ---------- process all files ----------
    with open(args.input_list) as f:
        for line in f:
            input_file = Path(line.strip())
            output_file = input_file.parent / args.output_name

            if output_file.exists():
                print(f"[SKIP] {output_file}")
                continue

            print(f"[RUN ] {input_file}")

            with input_file.open(encoding="utf-8") as fin, \
                 output_file.open("w", encoding="utf-8") as fout:

                for line_idx, line in tqdm(
                    enumerate(fin),
                    desc=f"Scoring {input_file.name}"
                ):
                    item = json.loads(line)

                    try:
                        prompt, response = split_prompt_response(
                            item["prompt"],
                            item["response"]
                        )
                    except Exception as e:
                        print(f"[WARN] Skip {input_file}:{line_idx} ({e})")
                        continue

                    rm_input = prompt + response

                    inputs = tokenizer(
                        rm_input,
                        return_tensors="pt",
                        truncation=True,
                        max_length=args.max_length
                    ).to(device)

                    with torch.no_grad():
                        reward = model(**inputs).logits.squeeze().item()

                    fout.write(json.dumps({
                        "prompt": prompt,
                        "response": response,
                        "reward": reward
                    }, ensure_ascii=False) + "\n")

    print("[DONE] All files processed")


if __name__ == "__main__":
    main()

