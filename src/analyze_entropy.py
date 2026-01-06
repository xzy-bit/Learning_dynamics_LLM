import os
import json
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from preference_datasets import get_batch_iterator
from utils import get_local_dir

# ===== 直接复用你 trainer 里的 entropy 定义 =====
from trainers import (
    entropy_binning_from_logits,
    entropy_from_logits,
)

def get_ckpt_path(save_path, epoch):
    p = os.path.join(save_path, f"policy_{epoch}.pt")
    print(p)
    if os.path.exists(p):
        return p
    # 兼容最后一个 epoch 只保存 policy.pt 的情况
    p = os.path.join(save_path, "policy.pt")
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"Checkpoint not found for epoch {epoch}")

# =====================================================
# token-level entropy online 统计（不存 list）
# =====================================================
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


# =====================================================
# 主分析函数：只 forward，不训练
# =====================================================
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

    # ===== model =====
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

    # ===== entropy bins（与你 trainer 里一致）=====
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

    # ===== train_data（与你训练时完全一致）=====
    data_iterator_kwargs = dict(
        names=config.datasets,
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
    )

    train_iterator = get_batch_iterator(
        **data_iterator_kwargs,
        split=config.train_split,   # train data
        shuffle=False,              # 固定顺序，保证可复现
        n_epochs=1,
        batch_size=config.eval_batch_size,
        n_examples=config.n_examples,
        silent=False,
    )

    # ===== 统计 =====
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

    return entropy_bins, entropy_hist, token_count, token_entropy_sum


# =====================================================
# 保存结果
# =====================================================
def save_results(
    config,
    epoch,
    entropy_bins,
    entropy_hist,
    token_count,
    token_entropy_sum,
):
    # ===== entropy histogram =====
    entropy_out = {
        "epoch": epoch,
        "bins": entropy_bins.tolist(),
        "chosen": entropy_hist["chosen"].tolist(),
        "rejected": entropy_hist["rejected"].tolist(),
    }

    with open(
        os.path.join(
            config.save_path,
            f"entropy_hist_epoch_{epoch}.json"
        ),
        "w"
    ) as f:
        json.dump(entropy_out, f, indent=2)

    # ===== token-level entropy stats（复用原文件名）=====
    token_entropy_out = {
        "epoch": epoch,
        "chosen": {
            str(tok): {
                "count": token_count["chosen"][tok],
                "avg_entropy":
                    token_entropy_sum["chosen"][tok]
                    / token_count["chosen"][tok],
            }
            for tok in token_count["chosen"]
        },
        "rejected": {
            str(tok): {
                "count": token_count["rejected"][tok],
                "avg_entropy":
                    token_entropy_sum["rejected"][tok]
                    / token_count["rejected"][tok],
            }
            for tok in token_count["rejected"]
        },
    }

    with open(
        os.path.join(
            config.save_path,
            f"max_entropy_top3_words_epoch_{epoch}.json"
        ),
        "w"
    ) as f:
        json.dump(token_entropy_out, f, indent=2)

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config: DictConfig):

    if config.analysis.use_pretrained:
        bins, hist, token_count, token_entropy_sum = analyze_entropy(
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
        )
        return

    for epoch in range(1, config.n_epochs + 1):
        ckpt_path = get_ckpt_path(config.save_path, epoch)

        bins, hist, token_count, token_entropy_sum = analyze_entropy(
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
        )

if __name__ == "__main__":
    main()

