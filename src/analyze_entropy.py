import os
import json
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from preference_datasets import get_batch_iterator
from utils import get_local_dir

def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    k = 0
    if k == 0:
        pd = torch.nn.functional.softmax(logits, dim=-1)
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    else:
        pd = torch.softmax(logits, dim=-1)          # [..., V]
        topk_pd, _ = torch.topk(pd, k=k, dim=-1)    # [..., k]
        eps = 1e-12
        entropy = -(topk_pd * torch.log(topk_pd + eps)).sum(dim=-1)

    return entropy

@torch.no_grad()
def entropy_binning_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    bin_edges: torch.Tensor,
):
    """
    logits: [B, T, V]
    labels: [B, T]
    return: histogram counts [num_bins]
    """
    # shift
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    entropy = entropy_from_logits(logits)  # [B, T]
    mask = labels != -100
    entropy = entropy[mask]

    # clamp to [min, max)
    entropy = entropy.clamp(
        min=bin_edges[0].item(),
        max=bin_edges[-1].item() - 1e-6
    )

    hist = torch.histc(
        entropy,
        bins=len(bin_edges) - 1,
        min=bin_edges[0].item(),
        max=bin_edges[-1].item(),
    )
    return hist.cpu()

def get_ckpt_path(save_path, epoch):
    p = os.path.join(save_path, f"policy_{epoch}.pt")
    print(p)
    if os.path.exists(p):
        return p
    p = os.path.join(save_path, "policy.pt")
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"Checkpoint not found for epoch {epoch}")

@torch.no_grad()
def update_token_entropy_stats(
    logits: torch.Tensor,    # [B, T+1, V]
    labels: torch.Tensor,    # [B, T+1]
    token_count: dict,
    token_entropy_sum: dict,
):
    # shift like NLL
    logits = logits[:, :-1, :]   # [B, T, V]
    labels = labels[:, 1:]       # [B, T]

    entropy = entropy_from_logits(logits)  # [B, T]
    mask = labels != -100

    B, T = labels.shape
    for b in range(B):
        for t in range(T):
            if not mask[b, t]:
                continue
            tok = labels[b, t].item()
            token_count[tok] += 1
            token_entropy_sum[tok] += entropy[b, t].item()

def decode_token_stats(token_count, token_entropy_sum, tokenizer):
    return {
        tokenizer.decode([tok]): {
            "token_id": tok,
            "count": token_count[tok],
            "avg_entropy": token_entropy_sum[tok] / token_count[tok],
        }
        for tok in token_count
    }

@torch.no_grad()
def analyze_entropy(
    ckpt_path: str,
    config,
    device: str = "cuda",
):
    # ===== tokenizer =====
    tokenizer_name = config.model.tokenizer_name_or_path or config.model.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=get_local_dir(config.local_dirs),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["state"] if isinstance(ckpt, dict) and "state" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
    else:
        print("[INFO] Using pretrained model (no checkpoint loaded)")

    model.eval()

    entropy_min = 0.0
    entropy_max = 10.0
    num_bins = 100
    entropy_bins = torch.linspace(entropy_min, entropy_max, num_bins + 1).to(device)

    entropy_hist = {
        "chosen": torch.zeros(num_bins),
        "rejected": torch.zeros(num_bins),
    }

    # ===== token-level stats =====
    token_count = {
        "chosen": defaultdict(int),
        "rejected": defaultdict(int),
    }
    token_entropy_sum = {
        "chosen": defaultdict(float),
        "rejected": defaultdict(float),
    }

    data_iterator_kwargs = dict(
        names=config.datasets,
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
    )

    train_iterator = get_batch_iterator(
        **data_iterator_kwargs,
        split=config.train_split,
        shuffle=False,
        n_epochs=1,
        batch_size=config.eval_batch_size,
        n_examples=config.n_examples,
        silent=False,
    )

    for batch in tqdm(train_iterator):
        for key in ["chosen", "rejected"]:
            logits = model(
                batch[f"{key}_input_ids"].to(device),
                attention_mask=batch[f"{key}_attention_mask"].to(device),
            ).logits.float()

            # --- entropy histogram ---
            hist = entropy_binning_from_logits(
                logits,
                batch[f"{key}_labels"].to(device),
                entropy_bins,
            )
            entropy_hist[key] += hist.cpu()
            # --- token-level avg entropy ---
            update_token_entropy_stats(
                logits,
                batch[f"{key}_labels"].to(device),
                token_count[key],
                token_entropy_sum[key],
            )

    return entropy_bins, entropy_hist, token_count, token_entropy_sum,tokenizer

@torch.no_grad()
def analyze_entropy_gen(
    ckpt_path: str,
    config,
    device: str = "cuda",
):
    tokenizer_name = config.model.tokenizer_name_or_path or config.model.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=get_local_dir(config.local_dirs),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["state"] if isinstance(ckpt, dict) and "state" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
    else:
        print("[INFO] Using pretrained model (no checkpoint loaded)")

    model.eval()

    entropy_min = 0.0
    entropy_max = 10.0
    num_bins = 100
    entropy_bins = torch.linspace(entropy_min, entropy_max, num_bins + 1).to(device)
    entropy_hist = torch.zeros(num_bins)

    data_iterator = get_batch_iterator(
        names=config.datasets,
        tokenizer=tokenizer,
        split=config.test_split,   # 一般用 test
        batch_size=config.eval_batch_size,
        shuffle=False,
    )

    token_count = defaultdict(int)
    token_entropy_sum = defaultdict(float)

    for batch in tqdm(data_iterator):
        prompt_ids = batch["prompt_input_ids"].to(device)
        prompt_mask = batch["prompt_attention_mask"].to(device)
        prompt_len = prompt_ids.shape[1]

        outputs = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=config.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # outputs.scores: List[T] of [B, V]
        gen_ids = outputs.sequences[:, prompt_len:]  # [B, T]

        for t, step_logits in enumerate(outputs.scores):
            entropy = entropy_from_logits(step_logits)  # [B]

            ent = entropy.clamp(
                min=entropy_bins[0].item(),
                max=entropy_bins[-1].item() - 1e-6,
            )
            hist = torch.histc(
                ent,
                bins=num_bins,
                min=entropy_bins[0].item(),
                max=entropy_bins[-1].item(),
            )
            entropy_hist += hist.cpu()

            tok_ids = gen_ids[:, t]

            for b in range(tok_ids.size(0)):
                tok = tok_ids[b].item()
                token_count[tok] += 1
                token_entropy_sum[tok] += entropy[b].item()

    return token_count, token_entropy_sum, tokenizer

def save_results(
    config,
    epoch,
    entropy_bins,
    entropy_hist,
    token_count,
    token_entropy_sum,
    tokenizer
):
    # ===== entropy histogram =====
    entropy_out = {
        "epoch": epoch,
        "bins": entropy_bins.tolist(),
        "chosen": entropy_hist["chosen"].tolist(),
        "rejected": entropy_hist["rejected"].tolist(),
    }

    # with open(
    #     os.path.join(
    #         config.save_path,
    #         f"entropy_hist_epoch_{epoch}.json"
    #     ),
    #     "w"
    # ) as f:
    #     json.dump(entropy_out, f, indent=2)

    token_entropy_out = {
        "epoch": epoch,
        "chosen": decode_token_stats(
            token_count["chosen"],token_entropy_sum["chosen"],tokenizer),
        "rejected": decode_token_stats(
            token_count["rejected"],token_entropy_sum["rejected"],tokenizer),
    }

    with open(
        os.path.join(
            config.save_path,
            f"max_entropy_top3_words_epoch_{epoch}.json"
        ),
        "w"
    ) as f:
        json.dump(token_entropy_out, f, indent=2)

def save_results_gen(
    config,
    epoch,
    entropy_bins,
    entropy_hist,
    token_count,
    token_entropy_sum,
    tokenizer
):
    entropy_out = {
        "epoch": epoch,
        "bins": entropy_bins.tolist(),
        "generation": entropy_hist.tolist(),
    }

    with open(
        os.path.join(config.save_path, f"entropy_hist_gen_epoch_{epoch}.json"),
        "w"
    ) as f:
        json.dump(entropy_out, f, indent=2)

    token_entropy_out = {
        "epoch": epoch,
        "generation": decode_token_stats(
            token_count, token_entropy_sum, tokenizer
        ),
    }

    with open(
        os.path.join(config.save_path, f"token_entropy_gen_epoch_{epoch}.json"),
        "w"
    ) as f:
        json.dump(token_entropy_out, f, indent=2)

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config: DictConfig):
    if config.analysis.mode == "tf":
        if config.analysis.use_pretrained:
            bins, hist, token_count, token_entropy_sum,tokenizer  = analyze_entropy(
                ckpt_path=None,
                config=config,
            )

            save_results(
                config=config,
                epoch="pretrained",
                entropy_bins=bins,
                entropy_hist=hist,
                token_count=token_count,
                token_entropy_sum=token_entropy_sum,
                tokenizer=tokenizer
            )
            return

        for epoch in range(1, config.n_epochs + 1):
            ckpt_path = get_ckpt_path(config.save_path, epoch)

            bins, hist, token_count, token_entropy_sum,tokenizer = analyze_entropy(
                ckpt_path=ckpt_path,
                config=config,
            )

            save_results(
                config=config,
                epoch=epoch,
                entropy_bins=bins,
                entropy_hist=hist,
                token_count=token_count,
                token_entropy_sum=token_entropy_sum,
                tokenizer=tokenizer,
            )
    else:
        if config.analysis.use_pretrained:
            bins, hist, token_count, token_entropy_sum, tokenizer = analyze_entropy_gen(
                ckpt_path=None,
                config=config,
            )

            save_results_gen(
                config=config,
                epoch="pretrained",
                entropy_bins=bins,
                entropy_hist=hist,
                token_count=token_count,
                token_entropy_sum=token_entropy_sum,
                tokenizer=tokenizer
            )
            return

        for epoch in range(1, config.n_epochs + 1):
            ckpt_path = get_ckpt_path(config.save_path, epoch)

            bins, hist, token_count, token_entropy_sum, tokenizer = analyze_entropy_gen(
                ckpt_path=ckpt_path,
                config=config,
            )

            save_results_gen(
                config=config,
                epoch=epoch,
                entropy_bins=bins,
                entropy_hist=hist,
                token_count=token_count,
                token_entropy_sum=token_entropy_sum,
                tokenizer=tokenizer,
            )

if __name__ == "__main__":
    main()

