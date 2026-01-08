import torch

torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig
from entmax import sparsemax_loss, sparsemax, entmax_bisect_loss,entmax15
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig
from torch.distributed.fsdp.api import FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib
import pdb
from preference_datasets import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
import numpy as np
import wandb
import tqdm
import random
import os
from collections import defaultdict,Counter
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple

def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False):
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1 / (2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    sigma_m = torch.sigmoid(beta * logits).detach()
    return losses, chosen_rewards, rejected_rewards, sigma_m

def asymmetric_preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False):
    p_r = policy_rejected_logps.exp()
    
    #tau = 0.05
    #gate = ((1.0 - p_r) / tau).clamp(max=1.0)
    #gate = gate.detach()
    gate=1.0

    # ========== rejected gradient source ==========
    neglog1mp = -torch.log1p(-p_r.clamp(max=1 - 1e-6))

    # ========== gradient replacement ==========
    policy_rejected_score = (
            policy_rejected_logps.detach()
            + (gate * neglog1mp - (gate * neglog1mp).detach())
    )

    pi_logratios = policy_chosen_logps - policy_rejected_score
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1 / (2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    sigma_m = torch.sigmoid(beta * logits).detach()
    print("==================Using Asymmetric Preference Loss==================")

    return losses, chosen_rewards, rejected_rewards, sigma_m

def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor,
                     average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size,
                sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens
                with a value of -100 are ignored.
                Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per 
                          (non-masked) token. Otherwise, return the sum of the 
                          log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log
        probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0
    logprob_logits = logits.log_softmax(-1)
    V = logprob_logits.shape[-1]
    per_token_logps = torch.gather(logprob_logits, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    # --------- Observe the argmax for each token
    labels_argmax = torch.argmax(logits, dim=-1)  # [B, M], argmax p(y*|chi_u^+/-)
    per_token_logps_argmax = torch.gather(logprob_logits, dim=2, index=labels_argmax.unsqueeze(2)).squeeze(2)
    # breakpoint()
    # ------ 2024-11-15 get all other metrics, e.g., expect_argmax, |A_o|_F, |p-e|_2
    prob_logits = logits.softmax(-1)  # prob version of logits, [B, M, V], easy to get underflow, take care!!!
    # --------- expect_argmax, should be [B, M]
    per_token_prob_argmax = torch.exp(
        per_token_logps_argmax)  # torch.gather(prob_logits, dim=2, index=labels_argmax.unsqueeze(2)).squeeze(2) #[B, M]

    # per_token_prob_exceptargmax =  torch.ones_like(per_token_prob_argmax)* loss_mask - per_token_prob_argmax* loss_mask #[B, M]
    # per_token_logp_exceptargmax = torch.log(per_token_prob_exceptargmax + 1e-100)

    # solove nan
    p_argmax_clamped = per_token_prob_argmax.clamp(max=1 - 1e-9)
    per_token_logp_exceptargmax = torch.log1p(-p_argmax_clamped) * loss_mask

    # print("except_argmax_logp mean:", per_token_logp_exceptargmax.mean().item())
    # print("any NaN:", torch.isnan(per_token_logp_exceptargmax).any().item())

    # --------- |A_o|_F, should be [B, 1]
    # prob_norm = torch.norm(prob_logits, dim=-1) # [B, M, V] -> [B, M]
    prob_norm = torch.linalg.vector_norm(prob_logits, ord=2,
                                         dim=-1)  # [B, M, V] -> [B, M], doing the same thing with previous line
    prob_norm = prob_norm * loss_mask  # [B, M], all other dims are zeros
    prob_norm2_mean = torch.square(prob_norm.sum(-1) / loss_mask.sum(-1))  # [B, M] -> [B, 1]
    A_norm = torch.sqrt(V * prob_norm2_mean + (V - 2) * torch.ones_like(
        prob_norm2_mean))  # [B, 1], align with the shape of all other metrics
    # ---------- |pi-e|_2, or
    # breakpoint()
    e_oht = torch.nn.functional.one_hot(labels, num_classes=V)  # [B, M, V]
    prob_gap_norm = torch.linalg.vector_norm(prob_logits - e_oht, ord=2, dim=-1)  # [B, M, V] -> [B, M]
    prob_gap_norm = prob_gap_norm * loss_mask
    prob_gap2_mean = prob_gap_norm.sum(-1) / loss_mask.sum(-1)
    # --------- (p_label - 1), only the pull-up energy
    prob_label = torch.gather(prob_logits, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    prob_label_gap = torch.ones_like(prob_label) - prob_label  # [B,M]
    prob_energy = (prob_label_gap * loss_mask).sum(-1) / loss_mask.sum(-1)
    # breakpoint()

    out_token = (per_token_logps * loss_mask).sum(-1)  # [B, 1]
    out_argmax = (per_token_logps_argmax * loss_mask).sum(-1)
    out_except_argmax = (per_token_logp_exceptargmax * loss_mask).sum(-1)

    if average_log_prob:
        return out_token / loss_mask.sum(-1), (out_argmax / loss_mask.sum(-1), out_except_argmax / loss_mask.sum(-1),
                                               A_norm, prob_gap2_mean, prob_energy, labels_argmax)
    else:
        return out_token, (out_argmax, out_except_argmax, A_norm, prob_gap2_mean, prob_energy, labels_argmax)

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


@torch.no_grad()
def max_entropy_token_topk(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    k: int = 3,
):
    """
    For each response:
      - find token with maximum entropy
      - return its top-k candidate tokens (strings)
    """
    logits = logits[:, :-1, :]      # [B, T, V]
    labels = labels[:, 1:]          # [B, T]
    mask = labels != -100

    entropy = entropy_from_logits(logits)   # [B, T]
    entropy = entropy.masked_fill(~mask, -1e9)

    max_pos = entropy.argmax(dim=1)          # [B]

    # gather logits at that position
    B = logits.size(0)
    sel_logits = logits[torch.arange(B), max_pos]  # [B, V]

    # top-k candidates
    topk_vals, topk_idx = sel_logits.topk(k, dim=-1)  # [B, k]

    # convert to tokens
    tokens = []
    for i in range(B):
        toks = tokenizer.convert_ids_to_tokens(
            topk_idx[i].tolist()
        )
        tokens.append(toks)

    return tokens   # List[List[str]] length B

@torch.no_grad()
def max_entropy_token_topk_words(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    k: int = 3,
):
    """
    return: List[List[str]]  # per response
    """
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    mask = labels != -100

    entropy = entropy_from_logits(logits)
    entropy = entropy.masked_fill(~mask, -1e9)

    max_pos = entropy.argmax(dim=1)  # [B]

    B = logits.size(0)
    sel_logits = logits[torch.arange(B), max_pos]  # [B, V]

    _, topk_idx = sel_logits.topk(k, dim=-1)

    out = []
    for i in range(B):
        toks = tokenizer.convert_ids_to_tokens(topk_idx[i].tolist())
        out.append(toks)
    return out


def row_quantile_masked(x: torch.Tensor, mask: torch.Tensor, q: float, eps=1e-8):

    B, T = x.shape
    qs = []
    for b in range(B):
        xb = x[b][mask[b]]
        if xb.numel() == 0:
            xb = x[b]
        qs.append(torch.quantile(xb, q))
    return torch.stack(qs, dim=0)  # [B]

def _get_batch_fy_score(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False
):
    """
    Compute sequence-level Fenchel–Young (sparsemax) scores for each example.
    Returns: scores of shape (B,1)
    """
    B, M, V = logits.shape

    # shift like NLL
    shift_logits = logits[:, :-1, :].contiguous()  # [B, M-1, V]
    shift_labels = labels[:, 1:].contiguous()  # [B, M-1]
    mask = (shift_labels != -100)
    shift_labels = shift_labels.masked_fill(~mask, 0)

    # compute sparsemax loss per token
    flat_logits = shift_logits.view(-1, V)
    flat_labels = shift_labels.view(-1)
    flat_loss = sparsemax_loss(flat_logits, flat_labels)  # [B*(M-1)]

    # reshape back to [B, M-1]
    token_loss = flat_loss.view(B, M - 1)

    # apply mask
    token_loss = token_loss * mask

    # sum over valid tokens
    # out_token  = (per_token_logps * loss_mask).sum(-1)  #[B, 1]
    scores = -token_loss.sum(-1)
    return scores


def _get_batch_ent_score(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        alpha: float = 1.5,
        beta:float = 0.5,
        ispos: bool = False,
        using_ns: bool = False,
):
    """
    Compute sequence-level Fenchel–Young (sparsemax) scores for each example.
    Returns: scores of shape (B,1)
    """
    B, M, V = logits.shape

    # shift like NLL
    shift_logits = logits[:, :-1, :].contiguous()  # [B, M-1, V]
    shift_labels = labels[:, 1:].contiguous()  # [B, M-1]
    mask = (shift_labels != -100)
    shift_labels = shift_labels.masked_fill(~mask, 0)

    # compute sparsemax loss per token
    flat_logits = shift_logits.view(-1, V)
    flat_labels = shift_labels.view(-1)
    flat_loss = entmax_bisect_loss(flat_logits, flat_labels, alpha, n_iter=50)  # [B*(M-1)]

    # reshape back to [B, M-1]
    token_loss = flat_loss.view(B, M - 1)

    if ispos and using_ns:
        entmax_probs = entmax15(flat_logits, dim=-1)
        softmax_probs = F.softmax(flat_logits, dim=-1)

        # one-hot label mask
        one_hot = F.one_hot(flat_labels, num_classes=softmax_probs.size(-1)).bool()

        # tail support excluding label
        tail_mask = (entmax_probs==0.0)&(~one_hot)

        suppressed_mass = (softmax_probs * tail_mask.float()).sum(dim=-1)
        suppressed_mass = torch.clamp(suppressed_mass, max = 0.99)
        
        ns_loss = -torch.log(1.0 - suppressed_mass)
        ns_loss = ns_loss.view(B, M-1)
        token_loss = token_loss + beta * ns_loss - beta*ns_loss.detach()

    token_loss = token_loss * mask

    # sum over valid tokens
    # out_token  = (per_token_logps * loss_mask).sum(-1)  #[B, 1]
    scores = -token_loss.sum(-1)
    return scores


def _get_batch_logps_masked(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        masked: bool = False,
        mask_type: str = "sparsemax",   # ["sparsemax", "ratio", "topk"]
        mask_ratio: float = 0.5,        # for quantile: e.g. 0.5 / 0.9
        topk: int = 50,                 # for topk
        mask_strength: float = 0.0,     # λ in gradient scaling
        threshold_prob: float = None,
):
    """
    Returns:
        out_token: [B] sequence-level log-prob
        stats: dict with zero_ratio
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()           # [B, M]
    logits = logits[:, :-1, :]               # [B, M, V]
    loss_mask = (labels != -100)             # [B, M]

    labels[labels == -100] = 0

    logprob_logits = logits.log_softmax(-1)  # [B, M, V]
    per_token_logps = torch.gather(logprob_logits, dim=2, index=labels.unsqueeze(2)).squeeze(2) # [B, M]

    zero_ratio = None
    sparsemax_rank_mean = None
    sparsemax_rank_ratio_mean = None
    if masked:
        with torch.no_grad():
            prob_logits = logits.softmax(-1)     # [B, M, V]
            label_prob = torch.gather(prob_logits, dim=2, index=labels.unsqueeze(2)).squeeze(2) # [B, M]
            if mask_type == "sparsemax":
                sparsemax_logits = sparsemax(logits, dim=-1)
                per_token_sparsemax = torch.gather(sparsemax_logits, dim=2, index=labels.unsqueeze(2)).squeeze(2)
                tail = (per_token_sparsemax == 0)    # bool
                # ===== NEW: rank in original softmax =====
                # rank = 1 + number of tokens with prob > p(label)
                rank = (prob_logits > label_prob.unsqueeze(-1)).sum(dim=-1) + 1  # [B, M]
                V = prob_logits.shape[-1]
                rank_ratio = rank.float() / V       # ∈ (0,1]
                
                sparsemax_rank_mean = rank[tail].float().mean()
                sparsemax_rank_ratio_mean = rank_ratio[tail].mean()

            elif mask_type == "ratio":
                print("=============Using Ratio============")
                # rank = number of probs >= p(label)
                V = prob_logits.shape[-1]
                rank = (prob_logits >= label_prob.unsqueeze(-1)).sum(dim=-1)
                rank_ratio = rank.float() / V
                print(
                    "rank_ratio stats:",
                    rank_ratio.min().item(),
                    rank_ratio.mean().item(),
                    rank_ratio.max().item()
                )
                tail = rank > int(mask_ratio * V)

            elif mask_type == "topk":
                topk_vals, _ = prob_logits.topk(topk, dim=-1)  # [B, M, K]
                kth_val = topk_vals[..., -1]                   # [B, M]
                tail = label_prob < kth_val
            elif mask_type == "hard_threshold":
                logits_fp32 = logits.float()
                entropy = entropy_from_logits(logits_fp32)

                with torch.no_grad():
                    thr = row_quantile_masked(entropy, loss_mask, q=0.8)  # [B]

                forking = entropy > thr[:, None]  # [B, T]

                tail = label_prob < threshold_prob
                tail = tail & forking

            elif mask_type == "entropy_neg_top1":
                logits_fp32 = logits.float()
                entropy = entropy_from_logits(logits_fp32)

                with torch.no_grad():
                    thr = row_quantile_masked(entropy, loss_mask, q=0.8)  # [B]

                forking = entropy > thr[:, None]  # [B, T]
                top1 = logits.argmax(dim=-1)
                is_top1 = labels == top1
                tail = forking & is_top1

            else:
                raise ValueError(f"Unknown mask_type: {mask_type}")

        if mask_type == "hard_threshold":
            eps = 1e-6
            prob_logits_g = logits.softmax(-1)  # requires_grad=True
            p_g = torch.gather(prob_logits_g, dim=2, index=labels.unsqueeze(2)).squeeze(2)
            p_g = p_g.clamp(min=eps, max=1 - eps)
            log1mp = torch.log1p(-p_g)

            print("log1mp requires_grad:", log1mp.requires_grad)
            per_token_logps = torch.where(
                tail,
                per_token_logps.detach() + (log1mp.detach()- log1mp)*mask_strength,
                per_token_logps
                )
            #print("==========Using -log(1-p)=============")
            #print(log1mp.sum(-1).detach()*mask_strength)
        # else:
        #     w = torch.where(tail,torch.full_like(per_token_logps, mask_strength),torch.ones_like(per_token_logps))
        #
        #     per_token_logps = per_token_logps * w + per_token_logps.detach() * (1.0 - w)
        elif mask_type == "entropy_neg_top1":
            per_token_logps = torch.where(
                tail,
                per_token_logps * mask_strength - per_token_logps.detach() * mask_strength + per_token_logps.detach(),
                per_token_logps
            )

        valid_mask = loss_mask.bool()
        zero_ratio = (tail & valid_mask).sum().float() / (valid_mask.sum().float() + 1e-8)
    
    per_token_logps = per_token_logps * loss_mask
    out_token = per_token_logps.sum(-1)      # [B]

    return out_token, {
        "zero_ratio": zero_ratio.detach() if zero_ratio is not None else None,
        "rank":
            sparsemax_rank_mean.detach() if sparsemax_rank_mean is not None else None,
        "rank_ratio":
            sparsemax_rank_ratio_mean.detach() if sparsemax_rank_ratio_mean is not None else None}

@torch.no_grad()
def _aggregate_token_metrics(logits: torch.FloatTensor,
                             labels: torch.LongTensor,
                             eps: float = 1e-8):
    """
    Compute token-level softmax/sparsemax stats but return sample-level aggregates
    (same spirit as _get_batch_logps: per-token -> masked mean over sequence).

    Returns: dict of shape [B] tensors (sample-level)
      - softmax_label_mean:     平均 softmax(z_chosen)
      - softmax_max_mean:       平均 max softmax(z)
      - chosen_is_argmax_ratio:  label 是否为 argmax 的比例
    """
    assert logits.shape[:-1] == labels.shape, "shapes must match on (B, M)"
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)  # [B, M]

    labels[labels == -100] = 0

    prob_soft = torch.softmax(logits, dim=-1)  # [B, M, V]

    # p(label)
    p_label_soft = torch.gather(prob_soft, 2, labels.unsqueeze(2)).squeeze(2)  # [B, M]
    soft_max_vals, soft_argmax_idx = prob_soft.max(dim=-1)  # [B, M]

    # 是否 label==argmax
    logits_argmax_idx = logits.argmax(dim=-1)  # [B, M]
    chosen_is_argmax = (labels == logits_argmax_idx) & loss_mask

    # pos 是否等于该步的最大概率
    label_pos_equal_max = (torch.abs(p_label_soft - soft_max_vals) <= eps) & loss_mask

    # masked mean 辅助
    denom = loss_mask.sum(-1).clamp_min(1)  # [B]

    def mm(x):  # masked mean over time
        return (x * loss_mask).sum(-1) / denom

    softmax_label = mm(p_label_soft)
    softmax_argmax = mm(soft_max_vals)
    label_eq_argmax_ratio = mm(label_pos_equal_max.float())
    return softmax_label,softmax_argmax,label_eq_argmax_ratio


@torch.no_grad()
def _record_eval_probs(metrics, train_test, prob_set, k, logits, labels):
    """
    Compute and record SOFTMAX-based statistics for eval stage.
    包含：
      - chosen/rejected: softmax(y*)
      - max softmax(z)
      - label是否是argmax的比例
    （去掉 sparsemax 相关指标）
    """
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)
    labels[labels == -100] = 0
    denom = loss_mask.sum(-1).clamp_min(1)

    # ---- softmax 概率分布 ----
    prob_soft = torch.softmax(logits, dim=-1)

    # p(label)
    soft_label_prob = torch.gather(prob_soft, 2, labels.unsqueeze(2)).squeeze(2)  # [B, M]

    # p(argmax)
    soft_argmax_vals, soft_argmax_idx = prob_soft.max(dim=-1)  # [B, M]

    # label 是否等于 argmax
    label_eq_argmax = (labels == soft_argmax_idx) & loss_mask
    label_eq_argmax_ratio = (label_eq_argmax.float().sum(-1) / denom)

    # masked mean helper
    def mm(x):
        return (x * loss_mask).sum(-1) / denom

    # ---- 聚合统计 ----
    soft_label_mean = mm(soft_label_prob)
    soft_argmax_mean = mm(soft_argmax_vals)

    # ---- 写入 metrics ----
    metrics[f'softmax_mean_{train_test}_{prob_set}/{k}'] = soft_label_mean.cpu().numpy().tolist()
    metrics[f'softmax_argmax)_mean_{train_test}_{prob_set}/{k}'] = soft_argmax_mean.cpu().numpy().tolist()
    metrics[f'label_eq_argmax_ratio_mean_{train_test}_{prob_set}/{k}'] = label_eq_argmax_ratio.cpu().numpy().tolist()

def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch


class BasicTrainer(object):
    def __init__(
            self, policy: nn.Module, config: DictConfig, seed: int,
            run_dir: str, reference_model: Optional[nn.Module] = None,
            rank: int = 0, world_size: int = 1
    ) -> None:
        """A trainer for a language model, supporting either SFT training.

        If multiple GPUs are present, naively splits the model across them, 
        effectively offering N times available memory, but without any parallel 
        computation.
        """
        # entropy bins
        self.entropy_min = 0.0
        self.entropy_max = 10.0
        self.num_bins = 100
        self.entropy_bins = torch.linspace(
            self.entropy_min, self.entropy_max, self.num_bins + 1
        )

        # histogram accumulator
        self.entropy_hist = {
            "chosen": torch.zeros(self.num_bins),
            "rejected": torch.zeros(self.num_bins),
        }

        self.maxent_word_counter = {
            "chosen": Counter(),
            "rejected": Counter(),
        }

        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        self.prob_dicts = ['chosen', 'chosen_initial', 'chosen_gptsemantic', 'chosen_gptformat',  # 'chosen_selfr'
                           'rejected', 'reject_gptsemantic', 'reject_gptformat',
                           'irr_train', 'irr_test', 'irr_hum',
                           'random_permute', 'random_nonhum']

        tokenizer_name_or_path = \
            config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs)
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
        )

        self.policy = policy
        self.reference_model = reference_model

        self.probtrain_iterator = get_batch_iterator(
            **data_iterator_kwargs,
            split='formal_prob_train',
            n_examples=500,
            shuffle=False,
            batch_size=config.eval_batch_size,
            silent=rank != 0,
        )
        self.probtrain_batches = list(self.probtrain_iterator)
        rank0_print(f'===========Loaded {len(self.probtrain_batches)} prob_train batches ')

        self.probtest_iterator = get_batch_iterator(
            **data_iterator_kwargs,
            split='formal_prob_test',
            n_examples=500,
            shuffle=False,
            batch_size=config.eval_batch_size,
            silent=rank != 0,
        )
        self.probtest_batches = list(self.probtest_iterator)
        rank0_print(f'========Loaded {len(self.probtest_batches)} prob_test batches ')

        if config.train_using_prob:
            self.train_iterator = get_batch_iterator(
                **data_iterator_kwargs,
                split="formal_prob_train",
                shuffle=True,
                n_epochs=config.n_epochs,
                n_examples=config.n_examples,
                batch_size=config.batch_size,
                silent=rank != 0,
            )
        else:
            self.train_iterator = get_batch_iterator(
                **data_iterator_kwargs,
                split=config.train_split,  # "train_dpo",
                shuffle=True,
                n_epochs=config.n_epochs,
                n_examples=config.n_examples,
                batch_size=config.batch_size,
                silent=rank != 0,
            )
        # self.train_batches = list(self.train_iterator)
        # rank0_print(f'===========Loaded {len(self.train_batches)} train batches ')

    def get_batch_samples(
            self, batch: Dict[str, torch.LongTensor],
            sample_flag=True
    ) -> Tuple[str, str]:
        """Generate samples from the policy for the given batch of inputs."""

        # FSDP generation according to
        # https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (
            FSDP.summon_full_params(
                self.policy, writeback=False, recurse=False
            ) if 'FSDP' in self.config.trainer else contextlib.nullcontext()
        )
        with ctx():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'],
                attention_mask=batch['prompt_attention_mask'],
                max_length=self.config.max_length,
                do_sample=sample_flag,
                pad_token_id=self.tokenizer.pad_token_id
            )

        if self.config.loss.name in {'dpo', 'ipo', 'sp_dpo', 'masked_dpo','ent_dpo','asym_dpo'}:
            ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False,
                                                   recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'],
                    max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(
            policy_output, self.config.max_length, self.tokenizer.pad_token_id
        )
        policy_output = all_gather_if_needed(
            policy_output, self.rank, self.world_size
        )
        policy_output_decoded = self.tokenizer.batch_decode(
            policy_output, skip_special_tokens=True
        )

        if self.config.loss.name in {'dpo', 'ipo', 'sp_dpo', 'masked_dpo','ent_dpo','asym_dpo'}:
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_input_ids'],
                           attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        all_logps, _ = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps

    def concatenated_forward_masked(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]],mask_type:str,mask_ratio:float,mask_top_k:int,mask_strength:float,mask_threshold_prob:float):
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_input_ids'],
                           attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        chosen_logis = all_logits[:batch['chosen_input_ids'].shape[0]]
        chosen_labels = concatenated_batch['concatenated_labels'][:batch['chosen_input_ids'].shape[0]]

        rejected_logits = all_logits[batch['chosen_input_ids'].shape[0]:]
        rejected_labels = concatenated_batch['concatenated_labels'][batch['chosen_input_ids'].shape[0]:]

        chosen_logps,_ = _get_batch_logps_masked(chosen_logis, chosen_labels, masked=False)
        rejected_logps,zero_ratio = _get_batch_logps_masked(rejected_logits, rejected_labels, masked=True,mask_type=mask_type,mask_ratio=mask_ratio,topk=mask_top_k,mask_strength=mask_strength,threshold_prob=mask_threshold_prob)

        return chosen_logps, rejected_logps,zero_ratio
    
    def concatenated_forward_sparse(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
           But change the logps into FY-loss
        """
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_input_ids'],
                        attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
    
        #all_logps, _ = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
        all_scores = _get_batch_fy_score(all_logits, concatenated_batch['concatenated_labels'])
    
        # record the logprob
        with torch.no_grad():
            all_logps, _ = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'],average_log_prob=False)
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
    
        chosen_score = all_scores[:batch['chosen_input_ids'].shape[0]]
        rejected_score = all_scores[batch['chosen_input_ids'].shape[0]:]
    
        #all_soft_probs, metrics = _aggregate_token_metrics(all_logits, concatenated_batch['concatenated_labels'])
        #chosen_soft_probs = all_soft_probs[:batch['chosen_input_ids'].shape[0]]
        #rejected_soft_probs = all_soft_probs[batch['chosen_input_ids'].shape[0]:]
    
        #sparsemax_label = metrics['sparsemax_label']
        #sparsemax_argmax = metrics['sparsemax_argmax']
        #sparsemax_argmax_eq1_ratio = metrics['sparsemax_argmax_eq1_ratio']
        #label_eq_argmax_ratio = metrics['label_eq_argmax_ratio']
    
        #chosen_sparse_probs = sparsemax_label[:batch['chosen_input_ids'].shape[0]]
        #rejected_sparse_probs = sparsemax_label[batch['chosen_input_ids'].shape[0]:]
    
        #chosen_sparse_argmax = sparsemax_argmax[:batch['chosen_input_ids'].shape[0]]
        #chosen_sparse_eq1_ratio = sparsemax_argmax_eq1_ratio[:batch['chosen_input_ids'].shape[0]]
    
        #chosen_label_eq_argmax_ratio = label_eq_argmax_ratio[:batch['chosen_input_ids'].shape[0]]
    
        #return chosen_score, rejected_score, chosen_logps, rejected_logps, chosen_soft_probs, rejected_soft_probs, chosen_sparse_probs, rejected_sparse_probs, chosen_sparse_argmax, chosen_sparse_eq1_ratio, chosen_label_eq_argmax_ratio
        return chosen_score, rejected_score, chosen_logps, rejected_logps

    def concatenated_forward_ent(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]],alpha,beta,using_ns):
         """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
            But change the logps into FY-loss
         """
         concatenated_batch = concatenated_inputs(batch)
         all_logits = model(concatenated_batch['concatenated_input_ids'],
                            attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)

         chosen_logis = all_logits[:batch['chosen_input_ids'].shape[0]]
         chosen_labels = concatenated_batch['concatenated_labels'][:batch['chosen_input_ids'].shape[0]]

         rejected_logits = all_logits[batch['chosen_input_ids'].shape[0]:]
         rejected_labels = concatenated_batch['concatenated_labels'][batch['chosen_input_ids'].shape[0]:]

         chosen_score = _get_batch_ent_score(chosen_logis, chosen_labels, alpha=alpha,beta=beta,ispos=True,using_ns=using_ns)
         rejected_score = _get_batch_ent_score(rejected_logits, rejected_labels, alpha=alpha,beta=beta,ispos=False,using_ns=using_ns)

         # all_scores = _get_batch_ent_score(all_logits, concatenated_batch['concatenated_labels'],alpha=alpha)
         # chosen_score = all_scores[:batch['chosen_input_ids'].shape[0]]
         # rejected_score = all_scores[batch['chosen_input_ids'].shape[0]:]

          
         all_logps, _ = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'],
                                             average_log_prob=False)
         chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
         rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]


         return chosen_score, rejected_score, chosen_logps, rejected_logps

    def get_batch_metrics(
            self,
            batch: Dict[str, Union[List, torch.LongTensor]],
            loss_config: DictConfig,
            train=True,
            prob_set=None,
            force_sft=False
    ) -> Tuple[torch.FloatTensor, Dict[str, List]]:
        """Compute the SFT loss and other metrics for the given batch of inputs.
        """

        metrics = {}
        train_test = 'train' if train else 'eval'

        # if self.config.train_supervise=='rejected':
        #     chosen = 'rejected'
        #     print('@@@@@@@@@@@ Here we will use rejected sample as y+ @@@@@@@@@@@@@@@@@@@@@')
        # else:
        #     chosen = 'chosen'  

        if self.config.train_supervise is None:
            chosen = 'chosen'
        else:
            chosen = self.config.train_supervise
        argmax_token = np.array([0])  # dummy variable for the stupid bug, ugly but useful!!!
        if train:
            if loss_config.name in {'dpo', 'ipo', 'masked_dpo','sp_dpo','ent_dpo','asym_dpo'} and not force_sft:
                if loss_config.name == 'masked_dpo' or loss_config.name == 'dpo' or loss_config.name == 'ipo' or loss_config.name=='asym_dpo':
                    if loss_config.name == 'masked_dpo':
                        policy_chosen_score, policy_rejected_score,zero_ratio = self.concatenated_forward_masked(
                            self.policy, batch,mask_type=loss_config.mask_type,mask_ratio=loss_config.mask_ratio,
                            mask_top_k=loss_config.mask_top_k,mask_strength=loss_config.mask_strength,mask_threshold_prob=loss_config.mask_threshold_prob
                        )
                        metrics["masked_dpo/tail_ratio"] = [zero_ratio["zero_ratio"].item()]
                        if loss_config.mask_type=="sparsemax":
                            metrics["masked_dpo/rank_ratio"] = [zero_ratio["rank_ratio"]]
                            metrics["masked_dpo/rank"] = [zero_ratio["rank"]]
                    elif loss_config.name == 'dpo' or loss_config.name == 'ipo' or loss_config.name == 'asym_dpo':
                        policy_chosen_score, policy_rejected_score = self.concatenated_forward(
                            self.policy, batch)
                    with torch.no_grad():
                        reference_chosen_score, reference_rejected_score = self.concatenated_forward(
                            self.reference_model, batch)
                    policy_chosen_logps, policy_rejected_logps = policy_chosen_score, policy_rejected_score

                elif loss_config.name in {'sp_dpo', 'ent_dpo'}:
                    if loss_config.name == 'sp_dpo':
                        policy_chosen_score, policy_rejected_score, policy_chosen_logps, policy_rejected_logps = self.concatenated_forward_sparse(self.policy, batch)
                        with torch.no_grad():
                            reference_chosen_score, reference_rejected_score,_,_ = self.concatenated_forward_sparse(
                                self.reference_model, batch)
                    else:
                        alpha = loss_config.alpha
                        beta = loss_config.ent_beta
                        using_ns = loss_config.using_ns
                        policy_chosen_score, policy_rejected_score, policy_chosen_logps, policy_rejected_logps = self.concatenated_forward_ent(self.policy, batch,alpha,beta,using_ns)
                        with torch.no_grad():
                            reference_chosen_score, reference_rejected_score,_,_ = self.concatenated_forward_ent(
                                self.reference_model, batch,alpha,beta,using_ns)

                if loss_config.name in {'dpo', 'masked_dpo','sp_dpo','ent_dpo','asym_dpo'}:
                    loss_kwargs = {'beta': loss_config.beta, 'reference_free': loss_config.reference_free,
                                   'label_smoothing': loss_config.label_smoothing, 'ipo': False}
                elif loss_config.name == 'ipo':
                    loss_kwargs = {'beta': loss_config.beta, 'ipo': True}
                else:
                    raise ValueError(f'unknown loss {loss_config.name}')

                if loss_config.name=='asym_dpo':
                    losses, chosen_rewards, rejected_rewards, sigma_m = asymmetric_preference_loss(
                        policy_chosen_score, policy_rejected_score, reference_chosen_score, reference_rejected_score,**loss_kwargs)
                else:
                    losses, chosen_rewards, rejected_rewards,sigma_m = preference_loss(
                        policy_chosen_score, policy_rejected_score, reference_chosen_score, reference_rejected_score,
                        **loss_kwargs)

                if self.config.using_extra_ce==True:
                    print("===============================================")
                    ce_losses = -policy_chosen_logps - policy_rejected_logps
                    losses = losses + self.config.ce_lambda * ce_losses

                reward_accuracies = (chosen_rewards > rejected_rewards).float()

                chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
                rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
                reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

                metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
                metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
                metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
                metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
                metrics[f'rewards_{train_test}/sigma_m'] = sigma_m.cpu().numpy().tolist()

                policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
                metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()
                argmax_token = np.array([-1])


            # elif loss_config.name == "sp_dpo":
            #     (policy_chosen_score, policy_rejected_score,
            #      policy_chosen_logps, policy_rejected_logps,
            #      policy_chosen_soft_probs, policy_rejected_soft_probs,
            #      policy_chosen_sparse_probs, policy_rejected_sparse_probs,
            #      policy_chosen_sparse_argmax, policy_chosen_sparse_eq1_ratio, policy_chosen_label_eq_argmax_ratio
            #      ) = self.concatenated_forward_sparse(self.policy, batch)
            #
            #      policy_chosen_score, policy_rejected_score,policy_chosen_logps, policy_rejected_logps = self.concatenated_forward_sparse(self.policy, batch)
            #      with torch.no_grad():
            #         (reference_chosen_score, reference_rejected_score,
            #          reference_chosen_logps, reference_rejected_logps,
            #          reference_chosen_soft_probs, reference_rejected_soft_probs,
            #          reference_chosen_sparse_probs, reference_rejected_sparse_probs,
            #          reference_chosen_sparse_argmax, reference_chosen_sparse_eq1_ratio,
            #          reference_chosen_label_eq_argmax_ratio
            #          ) = self.concatenated_forward_sparse(self.reference_model, batch)
            #         reference_chosen_score, reference_rejected_score,reference_chosen_logps, reference_rejected_logps = self.concatenated_forward_sparse(self.reference_model, batch)
            #      loss_kwargs = {'beta': loss_config.beta, 'reference_free': loss_config.reference_free,
            #                     'label_smoothing': loss_config.label_smoothing, 'ipo': False}
            #
            #      losses, chosen_rewards, rejected_rewards = preference_loss_sparse(
            #          policy_chosen_score, policy_rejected_score, reference_chosen_score, reference_rejected_score,
            #          **loss_kwargs)
            #
            #     reward_accuracies = (chosen_rewards > rejected_rewards).float()
            #
            #     chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            #     rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            #     reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)
            #
            #     metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            #     metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            #     metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            #     metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            #
            #     # === policy softmax/sparsemax 指标 ===
            #     metrics[f'softmax_{train_test}/chosen'] = policy_chosen_soft_probs.cpu().numpy().tolist()
            #     metrics[f'softmax_{train_test}/rejected'] = policy_rejected_soft_probs.cpu().numpy().tolist()
            #     metrics[f'sparsemax_{train_test}/chosen'] = policy_chosen_sparse_probs.cpu().numpy().tolist()
            #     metrics[f'sparsemax_{train_test}/rejected'] = policy_rejected_sparse_probs.cpu().numpy().tolist()
            #     metrics[f'sparsemax_argmax_{train_test}/chosen'] = policy_chosen_sparse_argmax.cpu().numpy().tolist()
            #     metrics[
            #         f'sparsemax_eq1_ratio_{train_test}/chosen'] = policy_chosen_sparse_eq1_ratio.cpu().numpy().tolist()
            #     metrics[
            #         f'label_eq_argmax_ratio_{train_test}/chosen'] = policy_chosen_label_eq_argmax_ratio.cpu().numpy().tolist()
            #
            else:  # if loss_config.name == 'sft':
                policy_chosen_logits = self.policy(batch[f'{chosen}_input_ids'],
                                                   attention_mask=batch[f'{chosen}_attention_mask']).logits.to(
                    torch.float32)
                policy_chosen_logps, _ = _get_batch_logps(policy_chosen_logits, batch[f'{chosen}_labels'],
                                                          average_log_prob=False)
                losses = -policy_chosen_logps

            policy_chosen_logps = all_gather_if_needed(
                policy_chosen_logps.detach(), self.rank, self.world_size
            )

            metrics[f'logps_{train_test}/chosen'] = \
                policy_chosen_logps.cpu().numpy().tolist()

            all_devices_losses = all_gather_if_needed(
                losses.detach(), self.rank, self.world_size
            )
            metrics[f'loss/{train_test}'] = \
                all_devices_losses.cpu().numpy().tolist()
            loss_mean = losses.mean()
        else:
            if prob_set is not None:
                with torch.no_grad():
                    for k in self.prob_dicts:
                        policy_predict_logtis = self.policy(
                            input_ids=batch[f'{k}_input_ids'],
                            attention_mask=batch[f'{k}_attention_mask']
                        ).logits.detach().to(torch.float32)
                        _record_eval_probs(metrics, train_test, prob_set, k, policy_predict_logtis,
                                           batch[f'{k}_labels'])
                        policy_predict_logps, policy_argmax_logps = _get_batch_logps(policy_predict_logtis,
                                                                                     batch[f'{k}_labels'],
                                                                                     average_log_prob=False)
                        del policy_predict_logtis
                        metrics[f'{k}_A_o'] = policy_argmax_logps[2].cpu().numpy().tolist()
                        metrics[f'logps_{train_test}_{prob_set}/{k}'] = \
                            policy_predict_logps.cpu().numpy().tolist()
                        if k == 'chosen':
                             metrics[f'argmax_prob_logits'] = policy_argmax_logps[0].cpu().numpy().tolist()
                             metrics[f'except_argmax_prob_logits'] = policy_argmax_logps[1].cpu().numpy().tolist()
                             metrics[f'p_e'] = policy_argmax_logps[3].cpu().numpy().tolist()
                             metrics[f'energy'] = policy_argmax_logps[4].cpu().numpy().tolist()
                             argmax_token = policy_argmax_logps[5].squeeze().cpu().numpy()

            loss_mean = 0
        return loss_mean, metrics, argmax_token

    def evaluation(self, prob_set='prob_train'):
        if prob_set.lower() == 'prob_train':
            data_batches = self.probtrain_batches
        elif prob_set.lower() == 'prob_test':
            data_batches = self.probtest_batches
        else:
            raise ('only have prob_set naming prob_train or prob_test')

        self.policy.eval()
        all_eval_metrics = defaultdict(list)
        all_argmax_token = []
        for eval_batch in (tqdm.tqdm(data_batches, desc='Computing eval metrics') if self.rank == 0 else data_batches):
            local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
            with torch.no_grad():
                # ----- detail_eval_matrics is token-wise, for k, v... then each v[i] contains logp of M tokens
                _, eval_metrics, argmax_token = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False,
                                                                       prob_set=prob_set)
            all_argmax_token.append(
                argmax_token)  # argmax_token is [B, M], all_argmax_token is a list storing those argtokens with different length
            for k, v in eval_metrics.items():
                all_eval_metrics[k].extend(v)

        # -------- Save the corresponding results
        logp_npy = np.zeros((1, 1))
        mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
        if self.config.wandb.enabled and self.rank == 0:
            wandb.log(mean_eval_metrics, step=self.example_counter)
        if self.rank == 0:
            output_dir = os.path.join(self.config.save_path, f'{prob_set}_metrics.json')
            self.save_metrics(output_dir, mean_eval_metrics)
            print(mean_eval_metrics)
            if self.config.fine_evaluation:
                # --------- Save logp; first N are for different y, last four are argmax, expect_argmax, |A_o|, |p-e| for each example
                tmp = []
                for k, v in all_eval_metrics.items():
                    tmp.append(v)
                logp_npy = np.array(tmp)
        return logp_npy, all_argmax_token

    def evaluation_get_response(self, prob_set='prob_train_gen'):
        data_iterator_kwargs = dict(
            names=self.config.datasets,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
        )
        probtest_iterator = get_batch_iterator(
            **data_iterator_kwargs,
            split=prob_set,
            n_examples=500,
            shuffle=False,
            batch_size=self.config.eval_batch_size,
            silent=True,
        )
        data_batches = list(probtest_iterator)
        rank0_print(f'========Loaded {len(data_batches)} prob_test batches ')

        self.policy.eval()
        all_policy_samples = []
        for eval_batch in (tqdm.tqdm(data_batches, desc='Computing eval metrics') if self.rank == 0 else data_batches):
            local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
            with torch.no_grad():
                policy_samples, _ = self.get_batch_samples(local_eval_batch,
                                                           sample_flag=True)  # For greedy decoding, convert this to False
                for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                    all_policy_samples.append({'prompt': prompt, 'response': sample})
        output_dir = os.path.join(self.config.save_path, f'{prob_set}_response.jsonl')
        with open(output_dir, 'a', newline='\n') as f:
            for i in range(len(all_policy_samples)):
                f.write(json.dumps(all_policy_samples[i]))
                f.write('\n')

    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(
            self.policy.parameters(), lr=self.config.lr
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(
                1.0, (step + 1) / (self.config.warmup_steps + 1)
            )
        )

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {'dpo', 'ipo', 'sp_dpo', 'masked_dpo','ent_dpo','asym_dpo'}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None
        logp_npy_all = []
        argmax_npy_all = []
        saving_epoch = 1
        # reload_ref_required = True
        for batch in self.train_iterator:
            if self.config.loss.name in {"dpo", "masked_dpo"}:
                local_batch = slice_and_move_batch_for_device(
                        batch,
                        self.rank,
                        self.world_size,
                        self.rank
                )

                with torch.no_grad():
                    # ===== chosen =====
                    chosen_logits = self.policy(
                        local_batch["chosen_input_ids"],
                        attention_mask=local_batch["chosen_attention_mask"]
                    ).logits.float()

                    chosen_hist = entropy_binning_from_logits(
                        chosen_logits,
                        local_batch["chosen_labels"],
                        self.entropy_bins
                    )
                    self.entropy_hist["chosen"] += chosen_hist

                    # ===== rejected =====
                    rejected_logits = self.policy(
                        local_batch["rejected_input_ids"],
                        attention_mask=local_batch["rejected_attention_mask"]
                    ).logits.float()

                    rejected_hist = entropy_binning_from_logits(
                        rejected_logits,
                        local_batch["rejected_labels"],
                        self.entropy_bins
                    )
                    self.entropy_hist["rejected"] += rejected_hist

                    chosen_words = max_entropy_token_topk_words(
                        chosen_logits,
                        local_batch["chosen_labels"],
                        self.tokenizer,
                        k=3,
                    )
                    for ws in chosen_words:
                        self.maxent_word_counter["chosen"].update(ws)

                    # rejected
                    rejected_words = max_entropy_token_topk_words(
                        rejected_logits,
                        local_batch["rejected_labels"],
                        self.tokenizer,
                        k=3,
                    )
                    for ws in rejected_words:
                        self.maxent_word_counter["rejected"].update(ws)

            #### BEGIN EVALUATION ####
            if self.example_counter % self.config.eval_every == 0 and (
                    self.example_counter > 0 or self.config.do_first_eval):
                rank0_print(f'Running evaluation after {self.example_counter} ' + 'train examples')
                _, _ = self.evaluation(prob_set='prob_train')  # [B,1] and [B, M]
                # if self.rank==0:
                #    logp_npy_all.append(logp_npy)
                #    argmax_npy_all.extend(argmax_npy)
                # self.evaluation(prob_set='prob_test')
            #### END EVALUATION ####
            epoch = self.example_counter // 5000
            if epoch == saving_epoch and epoch!=6:
                output_dir = os.path.join(self.config.save_path)
                self.save_pt(epoch,output_dir)
                saving_epoch+=1

                # word_out = {
                #     "epoch": epoch,
                #     "chosen": dict(self.maxent_word_counter["chosen"]),
                #     "rejected": dict(self.maxent_word_counter["rejected"]),
                # }
                #
                # with open(os.path.join(self.config.save_path,f"max_entropy_top3_words_epoch_{epoch}.json"), "w") as f:
                #     json.dump(word_out, f, indent=2)
                #
                # self.maxent_word_counter["chosen"].clear()
                # self.maxent_word_counter["rejected"].clear()
                #
                # entropy_out = {
                #     "epoch": epoch,
                #     "bins": self.entropy_bins.tolist(),
                #     "chosen": self.entropy_hist["chosen"].tolist(),
                #     "rejected": self.entropy_hist["rejected"].tolist(),
                # }
                #
                # with open(os.path.join(self.config.save_path,f"entropy_hist_epoch_{epoch}.json"), "w") as f:
                #     json.dump(entropy_out, f, indent=2)
                #
                # # reset
                # self.entropy_hist["chosen"].zero_()
                # self.entropy_hist["rejected"].zero_()

            #### BEGIN TRAINING ####
            self.policy.train()
            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(
                    self.config.gradient_accumulation_steps
            ):
                global_microbatch = slice_and_move_batch_for_device(
                    batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank
                )
                local_microbatch = slice_and_move_batch_for_device(
                    global_microbatch, self.rank, self.world_size, self.rank
                )

                # if self.batch_counter < self.config.pre_sft_steps:
                #     loss, metrics, _ = self.get_batch_metrics(local_microbatch, self.config.loss, train=True, force_sft=True)
                # else:
                #     if reload_ref_required:
                #         self.reference_model = self.policy
                #         self.reference_model.eval()
                #         reload_ref_required = False
                loss, metrics, _ = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
                (loss / self.config.gradient_accumulation_steps).backward()
            #if self.example_counter % 50 == 0 and self.rank == 0:
            #    for name, p in self.policy.named_parameters():
            #        if p.grad is not None and "embed" in name:
            #            print(
            #                    f"[STEP {self.batch_counter}] "
            #                    f"{name} grad |mean|={p.grad.abs().mean().item():.3e} "
            #                    f"|norm|={p.grad.norm().item():.3e}"
            #                    )
            #            break

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)
            # ===== clip 前 =====
            #if self.example_counter % 50 == 0 and self.rank == 0:
            #    pre_clip_norm = torch.norm(
            #            torch.stack([
            #                p.grad.norm()
            #                for p in self.policy.parameters()
            #                if p.grad is not None
            #                ])
            #            ).item()
            #    print(f"[STEP {self.batch_counter}] grad_norm_pre_clip = {pre_clip_norm:.3e}")

            grad_norm = self.clip_gradient()

            # ===== clip 后 =====
            #if self.example_counter % 50 == 0 and self.rank == 0:
            #    print(f"[STEP {self.batch_counter}] grad_norm_post_clip = {grad_norm:.3e}")

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if last_log is None or time.time() - last_log > \
                    self.config.minimum_log_interval_secs:
                mean_train_metrics = {
                    k: sum(v) / len(v) for k, v in batch_metrics.items()
                }
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(
                    f'train stats after {self.example_counter} examples: ' + \
                    f'{formatted_dict(mean_train_metrics)}'
                )

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            # else:
            #     rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')
            #### END TRAINING ####
        # --- Train numpy results
        # if self.config.fine_evaluation:
        # output_dir = os.path.join(self.config.save_path, f'logp_npy_all_{self.config.train_supervise}.npy')
        # output_dir_argmax = os.path.join(self.config.save_path, f'argmax_token_all_{self.config.train_supervise}.npy')
        # breakpoint()
        # np.save(output_dir, np.array(logp_npy_all))
        # print("=====================DEBUG================")
        # print(type(argmax_npy_all))
        # print(type(argmax_npy_all[0]))
        # print("=====================DEBUG================")
        # np.save(output_dir_argmax, np.array(argmax_npy_all, dtype=object), allow_pickle=True)

        if self.config.save_ckp:
            output_dir = os.path.join(self.config.save_path)
            self.save(output_dir)
            word_out = {
                "epoch": 6,
                "chosen": dict(self.maxent_word_counter["chosen"]),
                "rejected": dict(self.maxent_word_counter["rejected"]),
            }

            # with open(os.path.join(self.config.save_path,f"max_entropy_top3_words_epoch_6.json"), "w") as f:
            #     json.dump(word_out, f, indent=2)
            #
            # self.maxent_word_counter["chosen"].clear()
            # self.maxent_word_counter["rejected"].clear()
            #
            # entropy_out = {
            #     "epoch": 6,
            #     "bins": self.entropy_bins.tolist(),
            #     "chosen": self.entropy_hist["chosen"].tolist(),
            #     "rejected": self.entropy_hist["rejected"].tolist(),
            # }
            #
            # with open(os.path.join(self.config.save_path, f"entropy_hist_epoch_6.json"), "w") as f:
            #     json.dump(entropy_out, f, indent=2)
            #
            # # reset
            # self.entropy_hist["chosen"].zero_()
            # self.entropy_hist["rejected"].zero_()


    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        ).item()

    def write_state_dict(
            self, step: int, state: Dict[str, torch.Tensor],
            metrics: Dict, filename: str, dir_name: Optional[str] = None
    ) -> None:
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)

    def save_metrics(self, output_name=None, metrics=None):
        with open(output_name, 'a', newline='\n') as f:
            json.dump(metrics, f)
            f.write('\n')

    def save_pt(self, epoch: int, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(
            self.example_counter,
            policy_state_dict,
            metrics,
            f'policy_{epoch}.pt',
            output_dir
        )
        del policy_state_dict

    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""

        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(
            self.example_counter,
            policy_state_dict,
            metrics,
            'policy.pt',
            output_dir
        )
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(
            self.example_counter,
            optimizer_state_dict,
            metrics,
            'optimizer.pt',
            output_dir
        )
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(
            self.example_counter,
            scheduler_state_dict,
            metrics,
            'scheduler.pt',
            output_dir
        )


class FSDPTrainer(BasicTrainer):
    def __init__(
            self,
            policy: nn.Module,
            config: DictConfig,
            seed: int,
            run_dir: str,
            reference_model: Optional[nn.Module] = None,
            rank: int = 0,
            world_size: int = 1
    ) -> None:
        """A trainer subclass that uses PyTorch FSDP to shard the model across
        multiple GPUs.
        
        This trainer will shard both the policy and reference model across all 
        available GPUs. Models are sharded at the block level, where the block 
        class name is provided in the config.
        """

        super().__init__(
            policy, config, seed, run_dir, reference_model, rank, world_size
        )
        assert config.model.block_name is not None, \
            'must specify model.block_name ' + \
            '(e.g., GPT2Block or GPTNeoXLayer) for FSDP'

        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        model_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={wrap_class},
        )

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False
        )

        rank0_print('Sharding policy...')
        mp_dtype = getattr(
            torch, config.model.fsdp_policy_mp
        ) if config.model.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(
            param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype
        )
        self.policy = FSDP(
            policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy
        )

        if config.activation_checkpointing:
            rank0_print('Attempting to enable activation checkpointing...')
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/
                # scaling-multimodal-foundation-models-in-torchmultimodal/
                # -with-pytorch-distributed/
                # first, verify we have FSDP activation support ready by 
                # importing:
                from \
                    torch.distributed.algorithms._checkpoint.checkpoint_wrapper \
                    import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )
                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print('FSDP activation checkpointing not available:', e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print(
                    'Applying activation checkpointing wrapper to policy...'
                )
                apply_activation_checkpointing(
                    self.policy,
                    checkpoint_wrapper_fn=non_reentrant_wrapper,
                    check_fn=check_fn
                )
                rank0_print('FSDP activation checkpointing enabled!')

        if config.loss.name in {'dpo', 'ipo', 'sp_dpo', 'masked_dpo','ent_dpo','asym_dpo'}:
            rank0_print('Sharding reference model...')
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)

        print('Loaded model on rank', rank)
        dist.barrier()

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy,
           gathering the gradients across all GPUs.
        """
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()

    def save(self, output_dir=None, metrics=None):
        """Save policy, optimizer, and scheduler state to disk, gathering from all processes and saving only on the rank 0 process."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
                self.policy, StateDictType.FULL_STATE_DICT,
                state_dict_config=save_policy):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self.write_state_dict(
                self.example_counter,
                policy_state_dict,
                metrics,
                'policy.pt',
                output_dir
            )
        del policy_state_dict
        dist.barrier()

        save_policy = FullOptimStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )
        with FSDP.state_dict_type(
                self.policy,
                StateDictType.FULL_STATE_DICT,
                optim_state_dict_config=save_policy):
            optimizer_state_dict = FSDP.optim_state_dict(
                self.policy, self.optimizer
            )

        if self.rank == 0:
            self.write_state_dict(
                self.example_counter,
                optimizer_state_dict,
                metrics,
                'optimizer.pt',
                output_dir
            )
        del optimizer_state_dict
        dist.barrier()

        if self.rank == 0:
            scheduler_state_dict = self.scheduler.state_dict()
            self.write_state_dict(
                self.example_counter,
                scheduler_state_dict,
                metrics,
                'scheduler.pt',
                output_dir
            )
        dist.barrier()


class TensorParallelTrainer(BasicTrainer):
    def __init__(self, policy, config, seed, run_dir, reference_model=None, rank=0, world_size=1):
        """A trainer subclass that uses TensorParallel to shard the model 
        across multiple GPUs.

        Based on https://github.com/BlackSamorez/tensor_parallel. Note sampling 
        is extremely slow, see 
        https://github.com/BlackSamorez/tensor_parallel/issues/66.
        """
        super().__init__(
            policy, config, seed, run_dir, reference_model, rank, world_size
        )

        rank0_print('Sharding policy...')
        self.policy = tp.tensor_parallel(policy, sharded=True)
        if config.loss.name in {'dpo', 'ipo', 'sp_dpo', 'masked_dpo','ent_dpo','asym_dpo'}:
            rank0_print('Sharding reference model...')
            self.reference_model = tp.tensor_parallel(
                reference_model, sharded=False
            )

    def save(self, output_dir=None, metrics=None):
        """Save (unsharded) policy state to disk."""
        with tp.save_tensor_parallel(self.policy):
            policy_state_dict = self.policy.state_dict()

        self.write_state_dict(
            self.example_counter,
            policy_state_dict,
            metrics,
            'policy.pt',
            output_dir
        )
        del policy_state_dict
