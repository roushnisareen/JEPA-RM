"""Shared machinery for the reward models that score in embedding space (JEPA, its ablations, and
the frozen-feature control).

They all embed prompts and responses with the same frozen encoder and train only a small head on
top, so they're compared at equal capacity: any difference between them comes from the training
objective, not from the features.
"""
from __future__ import annotations

import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

from jepa_rm import tracking
from jepa_rm.config import SEED, JepaConfig
from jepa_rm.data import PreferencePair, format_text
from jepa_rm.runtime import default_device, model_dtype

log = logging.getLogger(__name__)
POOLING_MODES = ("mean", "last")


def context_text(prompt: str) -> str:
    return format_text(prompt, "")


def target_text(response: str) -> str:
    return response if response.strip() else "(empty)"


def mlp(n_in: int, hidden: int, n_out: int) -> nn.Module:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                         nn.Linear(hidden, n_out))


def pool(hidden: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    """(batch, seq, dim) hidden states -> (batch, dim), ignoring padding (right-padded inputs).

    "mean" averages all real tokens, so saying everything twice barely moves the embedding.
    "last" takes the final real token, which in a causal LM has attended to the whole sequence.
    """
    if mode == "mean":
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1)
    if mode == "last":
        last = mask.sum(1).long() - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    raise ValueError(f"unknown pooling {mode!r}, expected one of {POOLING_MODES}")


@dataclass
class Standardizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, x: torch.Tensor) -> Standardizer:
        return cls(x.mean(0), x.std(0) + 1e-6)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def to(self, device: str) -> Standardizer:
        return Standardizer(self.mean.to(device), self.std.to(device))

    def state(self, prefix: str) -> dict[str, torch.Tensor]:
        return {f"{prefix}_mean": self.mean.cpu(), f"{prefix}_std": self.std.cpu()}

    @classmethod
    def from_state(cls, state: dict, prefix: str, device: str) -> Standardizer:
        return cls(state[f"{prefix}_mean"], state[f"{prefix}_std"]).to(device)


class Encoder:
    """Frozen LLM; a text's embedding is its pooled hidden state at one layer."""

    def __init__(self, name: str, layer: int, max_len: int, pooling: str = "mean", device: str | None = None):
        if pooling not in POOLING_MODES:
            raise ValueError(f"unknown pooling {pooling!r}, expected one of {POOLING_MODES}")
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.tokenizer.padding_side = "right"
        self.device = device or default_device()
        self.model = AutoModel.from_pretrained(name, torch_dtype=model_dtype()).to(self.device).eval()
        self.layer = layer
        self.max_len = max_len
        self.pooling = pooling

    @classmethod
    def from_config(cls, cfg: JepaConfig, device: str | None = None) -> Encoder:
        return cls(cfg.encoder, cfg.embed_layer, cfg.max_len, cfg.pooling, device)

    @torch.no_grad()
    def embed(self, texts: Sequence[str], batch_size: int = 32) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                list(texts[i : i + batch_size]), return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_len,
            ).to(self.device)
            hidden = self.model(**batch, output_hidden_states=True).hidden_states[self.layer].float()
            out.append(pool(hidden, batch["attention_mask"], self.pooling))
        return torch.cat(out)


def split_by_prompt(pairs: Sequence[PreferencePair], holdout: float, seed: int = SEED) -> tuple[list[int], list[int]]:
    """(fit, holdout) pair indices with each prompt's pairs all on one side, so the best epoch is
    picked on prompts the head never trained on."""
    prompts = sorted({p.prompt for p in pairs})
    if len(prompts) < 2:
        raise ValueError("need pairs from at least 2 prompts to hold some out")
    random.Random(seed).shuffle(prompts)
    held = set(prompts[: max(1, round(len(prompts) * holdout))])
    fit = [i for i, p in enumerate(pairs) if p.prompt not in held]
    hold = [i for i, p in enumerate(pairs) if p.prompt in held]
    return fit, hold


@dataclass
class PairEmbeddings:
    """Every training pair, embedded once. ctx[P[i]] is pair i's prompt; tgt[C[i]] and tgt[R[i]]
    its chosen and rejected responses (standardized, then unit length)."""

    ctx: torch.Tensor
    tgt: torch.Tensor
    P: torch.Tensor
    C: torch.Tensor
    R: torch.Tensor
    fit: torch.Tensor
    hold: torch.Tensor
    ctx_norm: Standardizer
    tgt_norm: Standardizer

    @property
    def dim(self) -> int:
        return self.ctx.shape[1]


def embed_pairs(encoder: Encoder, pairs: Sequence[PreferencePair], holdout: float) -> PairEmbeddings:
    prompts = sorted({p.prompt for p in pairs})
    responses = sorted({r for p in pairs for r in (p.chosen, p.rejected)})
    log.info("embedding %d prompts and %d responses", len(prompts), len(responses))
    ctx_raw = encoder.embed([context_text(p) for p in prompts])
    tgt_raw = encoder.embed([target_text(r) for r in responses])

    device = encoder.device
    p_idx = {p: i for i, p in enumerate(prompts)}
    r_idx = {r: i for i, r in enumerate(responses)}
    fit, hold = split_by_prompt(pairs, holdout)
    # normalization statistics come from the fit side only, so the holdout that picks the best epoch
    # has no influence on the features the head trains on
    fit_ctx = sorted({p_idx[pairs[i].prompt] for i in fit})
    fit_tgt = sorted({r_idx[r] for i in fit for r in (pairs[i].chosen, pairs[i].rejected)})
    ctx_norm, tgt_norm = Standardizer.fit(ctx_raw[fit_ctx]), Standardizer.fit(tgt_raw[fit_tgt])
    return PairEmbeddings(
        ctx=ctx_norm(ctx_raw),
        tgt=F.normalize(tgt_norm(tgt_raw), dim=-1),
        P=torch.tensor([p_idx[p.prompt] for p in pairs], device=device),
        C=torch.tensor([r_idx[p.chosen] for p in pairs], device=device),
        R=torch.tensor([r_idx[p.rejected] for p in pairs], device=device),
        fit=torch.tensor(fit, device=device),
        hold=torch.tensor(hold, device=device),
        ctx_norm=ctx_norm,
        tgt_norm=tgt_norm,
    )


def fit_head(
    head: nn.Module,
    emb: PairEmbeddings,
    cfg: JepaConfig,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    scores_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    tag: str,
) -> float:
    """Train `head` on the fit pairs, keep the epoch with the best holdout accuracy, return it.

    loss_fn(idx) is the training loss on pairs idx; scores_fn(idx) gives (chosen, rejected) scores.
    """
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_acc, best_state = -1.0, None
    for epoch in range(cfg.epochs):
        head.train()
        perm = emb.fit[torch.randperm(len(emb.fit), device=emb.fit.device)]
        losses = []
        for i in range(0, len(perm), cfg.batch_size):
            loss = loss_fn(perm[i : i + cfg.batch_size])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        epoch_loss = sum(losses) / len(losses)

        head.eval()
        with torch.no_grad():
            chosen, rejected = scores_fn(emb.hold)
            acc = (chosen > rejected).float().mean().item()
        log.info("[%s] epoch %d/%d  loss %.3f  holdout acc %.3f", tag, epoch + 1, cfg.epochs, epoch_loss, acc)
        tracking.log({f"{tag}/epoch": epoch + 1, f"{tag}/loss": epoch_loss, f"{tag}/holdout_acc": acc})
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    head.eval()
    return best_acc
