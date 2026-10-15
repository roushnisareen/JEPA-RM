"""Blind human spot check of the gold judge.

The whole evaluation trusts one judge model. This samples answer pairs (base model vs. a trained
checkpoint, same prompt), hides which is which, and lets a person pick the better one. Agreement
between the person and the judge says how far the judge can be trusted; a person preferring the
base model where the judge doesn't is direct evidence that the judge got hacked too.
"""
from __future__ import annotations

import csv
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

from jepa_rm.config import SEED

CHOICES = ("A", "B", "tie")
FIELDS = ("id", "prompt", "answer_a", "answer_b", "better")


def make_blind_pairs(
    generations: Sequence[Mapping], run: str, step: int, n: int, run_seed: int = 0, seed: int = SEED
) -> tuple[list[dict], dict[str, dict]]:
    """(rows to rate, hidden key). Each row shows the base and the trained answer in random order.

    `run_seed` picks which training seed's checkpoint to compare; `seed` drives the sampling.
    """
    base = {g["prompt"]: g for g in generations if g["run"] == "base"}
    trained = [
        g for g in generations
        if g["run"] == run and g["step"] == step and g.get("seed", 0) == run_seed and g["prompt"] in base
    ]
    if not trained:
        raise ValueError(f"no generations for {run} seed {run_seed} at step {step}")
    rng = random.Random(seed)
    rows, key = [], {}
    for i, t in enumerate(rng.sample(trained, min(n, len(trained)))):
        b = base[t["prompt"]]
        trained_is_a = rng.random() < 0.5
        a, other = (t, b) if trained_is_a else (b, t)
        rows.append({"id": str(i), "prompt": t["prompt"], "answer_a": a["response"], "answer_b": other["response"],
                     "better": ""})
        key[str(i)] = {"trained": "A" if trained_is_a else "B", "gold_a": a["gold"], "gold_b": other["gold"]}
    return rows, key


def agreement(ratings: Sequence[Mapping], key: Mapping[str, Mapping]) -> dict:
    """How often the person agrees with the judge, and how often they prefer the trained model."""
    rated = [r for r in ratings if r["better"].strip() in CHOICES]
    decided = [r for r in rated if r["better"].strip() != "tie"]
    if not decided:
        raise ValueError("no ratings yet: fill the 'better' column with A, B or tie")
    agree = sum(
        (r["better"].strip() == "A") == (key[r["id"]]["gold_a"] > key[r["id"]]["gold_b"]) for r in decided
    )
    prefers_trained = sum(r["better"].strip() == key[r["id"]]["trained"] for r in decided)
    return {
        "n_rated": len(rated),
        "n_ties": len(rated) - len(decided),
        "human_judge_agreement": agree / len(decided),
        "human_prefers_trained": prefers_trained / len(decided),
    }


def write_csv(path: str | Path, rows: Sequence[Mapping]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
