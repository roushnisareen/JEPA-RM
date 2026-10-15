"""JEPA reward model.

A frozen encoder embeds the prompt (context) and the response on its own (target). A predictor
learns to map the context embedding to the target embedding of a preferred response. The reward for
a new response is the cosine similarity between the predicted and the actual target embedding.

Two weights control the training loss, which gives the ablations: the JEPA term alone ("jepa_pred",
no negatives at all), the ranking term alone ("jepa_rank"), and both (the default "jepa").
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from jepa_rm.config import SEED, JepaConfig
from jepa_rm.data import PreferencePair
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

CHECKPOINT_FILE = "jepa.pt"


def build_predictor(dim: int, hidden: int) -> nn.Module:
    return mlp(dim, hidden, dim)


def jepa_loss(
    pred: torch.Tensor, chosen: torch.Tensor, rejected: torch.Tensor, tau: float,
    jepa_weight: float = 1.0, rank_weight: float = 1.0,
) -> torch.Tensor:
    """Inputs are L2-normalised, shape (batch, dim).

    The JEPA term pulls the prediction toward the chosen response; the ranking term keeps it
    closer to the chosen response than to the rejected one.
    """
    sim_chosen = (pred * chosen).sum(-1)
    sim_rejected = (pred * rejected).sum(-1)
    return (jepa_weight * (1 - sim_chosen).mean()
            + rank_weight * F.softplus(-(sim_chosen - sim_rejected) / tau).mean())


class JepaRM:
    def __init__(self, cfg: JepaConfig, encoder: Encoder, predictor: nn.Module,
                 ctx_norm: Standardizer, tgt_norm: Standardizer):
        self.cfg = cfg
        self.encoder = encoder
        self.predictor = predictor
        self.ctx_norm = ctx_norm
        self.tgt_norm = tgt_norm

    def contexts(self, prompts: Sequence[str]) -> torch.Tensor:
        return self.ctx_norm(self.encoder.embed([context_text(p) for p in prompts]))

    def targets(self, responses: Sequence[str]) -> torch.Tensor:
        return F.normalize(self.tgt_norm(self.encoder.embed([target_text(r) for r in responses])), dim=-1)

    def predict(self, contexts: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.predictor(contexts), dim=-1)

    @torch.no_grad()
    def score(self, prompts: Sequence[str], responses: Sequence[str]) -> list[float]:
        return (self.predict(self.contexts(prompts)) * self.targets(responses)).sum(-1).tolist()

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.cfg),
                "dim": self.predictor[0].in_features,
                "predictor": {k: v.cpu() for k, v in self.predictor.state_dict().items()},
                **self.ctx_norm.state("ctx"),
                **self.tgt_norm.state("tgt"),
            },
            out_dir / CHECKPOINT_FILE,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> JepaRM:
        device = device or default_device()
        ckpt = torch.load(Path(path) / CHECKPOINT_FILE, map_location="cpu")
        cfg = JepaConfig(**ckpt["config"])
        predictor = build_predictor(ckpt["dim"], cfg.predictor_hidden)
        predictor.load_state_dict(ckpt["predictor"])
        return cls(
            cfg,
            Encoder.from_config(cfg, device),
            predictor.to(device).eval(),
            Standardizer.from_state(ckpt, "ctx", device),
            Standardizer.from_state(ckpt, "tgt", device),
        )


def train_jepa(
    pairs: Sequence[PreferencePair],
    out_dir: str | Path,
    cfg: JepaConfig,
    device: str | None = None,
    encoder: Encoder | None = None,
    embeddings: PairEmbeddings | None = None,
    tag: str = "jepa",
) -> JepaRM:
    """Pass `encoder` and `embeddings` to share one embedding pass across several reward models."""
    encoder = encoder or Encoder.from_config(cfg, device)
    emb = embeddings or embed_pairs(encoder, pairs, cfg.holdout)
    torch.manual_seed(SEED)  # same initialization and batch order for every variant
    predictor = build_predictor(emb.dim, cfg.predictor_hidden).to(encoder.device)
    rm = JepaRM(cfg, encoder, predictor, emb.ctx_norm, emb.tgt_norm)

    def terms(idx: torch.Tensor):
        return rm.predict(emb.ctx[emb.P[idx]]), emb.tgt[emb.C[idx]], emb.tgt[emb.R[idx]]

    def loss_fn(idx):
        return jepa_loss(*terms(idx), cfg.tau, cfg.jepa_weight, cfg.rank_weight)

    def scores_fn(idx):
        pred, chosen, rejected = terms(idx)
        return (pred * chosen).sum(-1), (pred * rejected).sum(-1)

    fit_head(predictor, emb, cfg, loss_fn, scores_fn, tag)
    rm.save(out_dir)
    return rm
