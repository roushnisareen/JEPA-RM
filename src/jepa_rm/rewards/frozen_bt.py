"""The capacity-matched control for the JEPA reward model.

Same frozen encoder, same embeddings, and a head of the same size as the JEPA predictor, but
trained as an ordinary Bradley-Terry reward model: an MLP reads the prompt and response embeddings
and outputs a scalar score. It answers the question a reviewer would ask first: does the JEPA
objective help, or does any reward model on frozen features behave the same way?
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from jepa_rm.config import SEED, JepaConfig
from jepa_rm.data import PreferencePair
from jepa_rm.rewards.bradley_terry import bt_loss
from jepa_rm.rewards.embeddings import (
    Encoder,
    PairEmbeddings,
    Standardizer,
    context_text,
    embed_pairs,
    fit_head,
    mlp,
    target_text,
)
from jepa_rm.runtime import default_device

CHECKPOINT_FILE = "frozen_bt.pt"


def build_head(dim: int, hidden: int) -> nn.Module:
    # input is [prompt; response]. With the JEPA predictor's hidden size h, both heads have
    # 2·dim·h + h² weights (plus h for this one's scalar output), so capacity is matched
    return mlp(2 * dim, hidden, 1)


class FrozenBTRM:
    def __init__(self, cfg: JepaConfig, encoder: Encoder, head: nn.Module,
                 ctx_norm: Standardizer, tgt_norm: Standardizer):
        self.cfg = cfg
        self.encoder = encoder
        self.head = head
        self.ctx_norm = ctx_norm
        self.tgt_norm = tgt_norm

    def head_scores(self, ctx: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([ctx, tgt], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def score(self, prompts: Sequence[str], responses: Sequence[str]) -> list[float]:
        ctx = self.ctx_norm(self.encoder.embed([context_text(p) for p in prompts]))
        tgt = nn.functional.normalize(self.tgt_norm(self.encoder.embed([target_text(r) for r in responses])), dim=-1)
        return self.head_scores(ctx, tgt).tolist()

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.cfg),
                "dim": self.head[0].in_features // 2,
                "head": {k: v.cpu() for k, v in self.head.state_dict().items()},
                **self.ctx_norm.state("ctx"),
                **self.tgt_norm.state("tgt"),
            },
            out_dir / CHECKPOINT_FILE,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> FrozenBTRM:
        device = device or default_device()
        ckpt = torch.load(Path(path) / CHECKPOINT_FILE, map_location="cpu")
        cfg = JepaConfig(**ckpt["config"])
        head = build_head(ckpt["dim"], cfg.predictor_hidden)
        head.load_state_dict(ckpt["head"])
        return cls(
            cfg,
            Encoder.from_config(cfg, device),
            head.to(device).eval(),
            Standardizer.from_state(ckpt, "ctx", device),
            Standardizer.from_state(ckpt, "tgt", device),
        )


def train_frozen_bt(
    pairs: Sequence[PreferencePair],
    out_dir: str | Path,
    cfg: JepaConfig,
    device: str | None = None,
    encoder: Encoder | None = None,
    embeddings: PairEmbeddings | None = None,
) -> FrozenBTRM:
    encoder = encoder or Encoder.from_config(cfg, device)
    emb = embeddings or embed_pairs(encoder, pairs, cfg.holdout)
    torch.manual_seed(SEED)  # same as the JEPA variants
    head = build_head(emb.dim, cfg.predictor_hidden).to(encoder.device)
    rm = FrozenBTRM(cfg, encoder, head, emb.ctx_norm, emb.tgt_norm)

    def scores_fn(idx):
        ctx = emb.ctx[emb.P[idx]]
        return rm.head_scores(ctx, emb.tgt[emb.C[idx]]), rm.head_scores(ctx, emb.tgt[emb.R[idx]])

    def loss_fn(idx):
        return bt_loss(*scores_fn(idx))

    fit_head(head, emb, cfg, loss_fn, scores_fn, "frozen_bt")
    rm.save(out_dir)
    return rm
