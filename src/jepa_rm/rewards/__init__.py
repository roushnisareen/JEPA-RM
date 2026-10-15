"""Reward models. Every kind has the same interface, so training and evaluation never special-case them.

    bt          the standard reward model: LLM + LoRA + scalar head, Bradley-Terry loss
    bt_lenpen   bt with its linear length trend removed (the cheap, known fix, as a baseline)
    frozen_bt   capacity-matched control: frozen encoder + MLP head, Bradley-Terry loss
    jepa        frozen encoder + predictor, JEPA + ranking loss: the idea under test
    jepa_pred   ablation: JEPA loss only, no negatives
    jepa_rank   ablation: ranking loss only

frozen_bt and the three jepa kinds share one frozen encoder, one set of embeddings and the same
head size, so differences between them come from the training objective alone.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

REWARD_KINDS = ("bt", "bt_lenpen", "frozen_bt", "jepa", "jepa_pred", "jepa_rank")
JEPA_VARIANTS = {"jepa": {}, "jepa_pred": {"rank_weight": 0.0}, "jepa_rank": {"jepa_weight": 0.0}}

# BT reads the text itself and never pools; the embedding-space kinds all depend on the pooling
BT_KINDS = ("bt", "bt_lenpen")
EMBEDDING_KINDS = ("frozen_bt", "jepa", "jepa_pred", "jepa_rank")

# what JEPA is compared against, head to head on the same prompts
COMPARISONS = (("jepa", "bt"), ("jepa", "bt_lenpen"), ("jepa", "frozen_bt"), ("jepa", "jepa_pred"),
               ("jepa", "jepa_rank"))

# the two groups each figure compares
MAIN_COMPARISON = ("bt", "bt_lenpen", "frozen_bt", "jepa")
JEPA_ABLATION = ("frozen_bt", "jepa", "jepa_pred", "jepa_rank")


class RewardModel(Protocol):
    def score(self, prompts: Sequence[str], responses: Sequence[str]) -> list[float]: ...


def check_kind(kind: str) -> str:
    if kind not in REWARD_KINDS:
        raise ValueError(f"unknown reward model {kind!r}, expected one of {REWARD_KINDS}")
    return kind


def load_reward_model(kind: str, path: str | Path) -> RewardModel:
    check_kind(kind)
    if kind == "bt":
        from jepa_rm.rewards.bradley_terry import BradleyTerryRM

        return BradleyTerryRM.load(path)
    if kind == "bt_lenpen":
        from jepa_rm.rewards.bradley_terry import LengthPenalizedRM

        return LengthPenalizedRM.load(path)
    if kind == "frozen_bt":
        from jepa_rm.rewards.frozen_bt import FrozenBTRM

        return FrozenBTRM.load(path)
    from jepa_rm.rewards.jepa import JepaRM

    return JepaRM.load(path)
