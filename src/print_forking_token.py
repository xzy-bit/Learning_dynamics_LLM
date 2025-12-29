import json
import argparse
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def log_print(s):
    print(s)

def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy

@torch.no_grad()
def analyze_text(
    model,
    tokenizer,
    prompt: str,
    response: str,
    tau: float,
    tag: str,
    device: str,
):
    """
    Print full sequence with <FORK>...</FORK> markers.
    """

    # ===== tokenize =====
    text = prompt + response
    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False
    ).to(device)

    input_ids = enc.input_ids                     # [1, T]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # ===== forward =====
    outputs = model(input_ids)
    logits = outputs.logits[:, :-1, :].float()    # [1, T-1, V]
    tokens = tokens[1:]                           # shift to align

    # ===== entropy (logsumexp form, fp32, stable) =====
    entropy = entropy_from_logits(logits)[0]      # [T-1]

    # ===== per-sample quantile (no flatten!) =====
    # since B=1 here, this is still "per-response"
    thr = torch.quantile(entropy, tau)
    forking = entropy > thr                       # [T-1]

    # ===== mark tokens =====
    marked_tokens = []
    for tok, is_fork in zip(tokens, forking.tolist()):
        if is_fork:
            marked_tokens.append(f"<font color=red>{tok}</font>")
        else:
            marked_tokens.append(tok)

    marked_text = tokenizer.convert_tokens_to_string(marked_tokens)

    # ===== output =====
    log_print("\n" + "=" * 30 + f" {tag} " + "=" * 30)
    log_print(f"tau = {tau:.2f} | entropy threshold = {thr.item():.4f}")
    log_print(marked_text)
    log_print("=" * 70)



def main():
    parser = argparse.ArgumentParser()

    # ===== match your launcher script =====
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--archive", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="policy.pt")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # ===== load tokenizer =====
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True
    )

    # ===== load base model =====
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    # ===== load checkpoint =====
    ckpt_path = os.path.join(
        "exp_results",
        args.archive,
        args.ckpt
    )

    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model.eval()

    # ===== load json =====
    with open(args.json_path, "r", encoding="utf-8") as f:
        sample = json.load(f)

    prompt = sample["prompt"]

    analyze_text(
        model,
        tokenizer,
        prompt,
        sample["chosen"],
        tau=args.tau,
        tag="CHOSEN",
        device=device,
    )

    analyze_text(
        model,
        tokenizer,
        prompt,
        sample["rejected"],
        tau=args.tau,
        tag="REJECTED",
        device=device,
    )


if __name__ == "__main__":
    main()

