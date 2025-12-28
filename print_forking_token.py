# print_forking_token.py
import json
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


@torch.no_grad()
def token_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: [1, T, V]
    return: entropy [T]
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


@torch.no_grad()
def analyze_text(
    model,
    tokenizer,
    prompt: str,
    response: str,
    tau: float = 0.8,
    device: str = "cuda",
    tag: str = ""
):
    """
    打印 prompt+response 中的 forking token（entropy > quantile）
    """
    text = prompt + response
    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False
    ).to(device)

    input_ids = enc.input_ids              # [1, T]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    outputs = model(input_ids)
    logits = outputs.logits[:, :-1, :]     # [1, T-1, V]
    tokens = tokens[1:]                    # 对齐 next-token
    entropy = token_entropy_from_logits(logits)[0]  # [T-1]

    thr = torch.quantile(entropy, tau)
    forking = entropy > thr

    print("\n" + "=" * 90)
    print(f"{tag} | tau={tau:.2f} | entropy threshold={thr.item():.4f}")
    print("=" * 90)

    # 逐 token 打印（只打印 forking）
    for i, (tok, H, f) in enumerate(zip(tokens, entropy, forking)):
        if f:
            print(f"[FORK] idx={i:03d}  token={tok:<12s}  entropy={H:.4f}")

    print("\n--- Context around forking tokens ---\n")
    for i, f in enumerate(forking):
        if f:
            l = max(0, i - 6)
            r = min(len(tokens), i + 7)
            ctx_tokens = tokens[l:r]
            ctx = tokenizer.convert_tokens_to_string(ctx_tokens)
            print(f"[idx {i:03d}] ... {ctx} ...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="HF model name or local checkpoint path")
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to a JSON with fields: prompt, chosen, rejected")
    parser.add_argument("--tau", type=float, default=0.8,
                        help="Percentile for entropy threshold (e.g., 0.8)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    ).eval()

    with open(args.json_path, "r", encoding="utf-8") as f:
        sample = json.load(f)

    prompt = sample["prompt"]

    # 同时看 chosen / rejected
    analyze_text(
        model, tokenizer,
        prompt, sample["chosen"],
        tau=args.tau, device=device,
        tag="CHOSEN"
    )
    analyze_text(
        model, tokenizer,
        prompt, sample["rejected"],
        tau=args.tau, device=device,
        tag="REJECTED"
    )


if __name__ == "__main__":
    main()
