import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------- clean text ----------

def extract_last_answer(text: str) -> str:
    idx = text.rfind("Assistant:")
    if idx == -1:
        return text.strip()
    return text[idx + len("Assistant:"):].strip()


def clean_answer(prompt: str, response: str) -> str:
    # 去掉 prompt continuation
    if response.startswith(prompt):
        response = response[len(prompt):]
    return extract_last_answer(response)


def clean_prompt(prompt: str) -> str:
    return prompt.replace("\n\nAssistant:", "").strip()

# ---------- main process ----------

def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output)

    # load reward model
    tokenizer = AutoTokenizer.from_pretrained(args.rm_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.rm_name,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        device_map="auto"
    ).eval()

    device = model.device
    print(f"[INFO] Loaded reward model on {device}")

    # score the response
    with input_path.open(encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line_idx, line in enumerate(fin):
            item = json.loads(line)

            raw_prompt = item["prompt"]
            raw_response = item["response"]

            prompt = clean_prompt(raw_prompt)
            response = clean_answer(raw_prompt, raw_response)

            rm_input = f"Human: {prompt}\nAssistant: {response}"

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

            if (line_idx + 1) % args.log_every == 0:
                print(f"[INFO] Processed {line_idx + 1} samples")

    print(f"[DONE] Saved scored results to {output_path}")


# ---------- argparse ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score prompt-response pairs with a reward model"
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input jsonl (with prompt & response)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output jsonl (with prompt, response, reward)"
    )
    parser.add_argument(
        "--rw_model",
        type=str,
        default="OpenAssistant/oasst-rm-2-pythia-6.9b",
        help="HuggingFace reward model name"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Max token length for reward model input"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 for reward model"
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log progress every N samples"
    )

    args = parser.parse_args()
    main(args)
