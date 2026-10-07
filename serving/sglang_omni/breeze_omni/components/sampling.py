"""Batched sampler ported from models/cudagraph/sampling.py:14-86."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_repetition_penalty(
    logits: torch.Tensor, history_mask: torch.Tensor, penalty: torch.Tensor
) -> torch.Tensor:
    """Apply HF-style repetition penalty in place to tokens marked in history_mask."""
    if logits.ndim != 2 or history_mask.shape != logits.shape:
        raise ValueError("logits and history_mask must both have shape [B,V]")
    penalties = torch.as_tensor(penalty, device=logits.device, dtype=logits.dtype).reshape(-1)
    if penalties.numel() == 1:
        penalties = penalties.expand(logits.shape[0])
    if penalties.numel() != logits.shape[0]:
        raise ValueError("penalty must be scalar or have shape [B]")
    rows, cols = torch.where(history_mask)
    selected = logits[rows, cols]
    row_penalties = penalties[rows]
    logits[rows, cols] = torch.where(
        selected > 0, selected / row_penalties, selected * row_penalties
    )
    return logits


def sample(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.LongTensor:
    """Sample independently per row; rows with temperature <= 0 are greedy.

    Filtering order matches the HF-compatible implementation in
    ``models/cudagraph/sampling.py``: temperature, top-k, top-p, softmax, multinomial.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [B,V]")
    batch, vocab = logits.shape
    device = logits.device
    temperature = torch.as_tensor(temperature, device=device, dtype=torch.float32).reshape(-1)
    top_k = torch.as_tensor(top_k, device=device, dtype=torch.long).reshape(-1)
    top_p = torch.as_tensor(top_p, device=device, dtype=torch.float32).reshape(-1)
    def rows(x):
        if x.numel() == 1:
            return x.expand(batch)
        if x.numel() != batch:
            raise ValueError("temperature, top_k, and top_p must be scalar or shape [B]")
        return x
    temperature, top_k, top_p = rows(temperature), rows(top_k), rows(top_p)
    work = logits.float().clone()
    greedy = temperature <= 0
    work = torch.where(greedy[:, None], work, work / temperature.clamp_min(torch.finfo(torch.float32).tiny)[:, None])
    ordered, _ = torch.sort(work, descending=True, dim=-1)
    kth = (top_k.clamp(min=1, max=vocab) - 1).unsqueeze(1)
    threshold = ordered.gather(1, kth)
    work = work.masked_fill((top_k[:, None] > 0) & (work < threshold), float("-inf"))
    sorted_logits, sorted_indices = torch.sort(work, descending=True, dim=-1)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative > top_p[:, None]
    remove = torch.cat((torch.zeros_like(remove[:, :1]), remove[:, :-1]), dim=1)
    remove &= top_p[:, None] < 1
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    work = torch.full_like(work, float("-inf")).scatter_(-1, sorted_indices, sorted_logits)
    probs = F.softmax(work, dim=-1)
    sampled = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    return torch.where(greedy, logits.argmax(dim=-1), sampled).to(torch.long)
