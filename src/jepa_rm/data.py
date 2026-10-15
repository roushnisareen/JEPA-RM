"""HelpSteer2 loading and the text formats shared by every stage.

Everything comes from HelpSteer2's preference subset: direct human comparisons of two answers to
the same prompt, which is exactly what a Bradley-Terry reward model models. Its train and val
splits share no prompts, and both the reward models' data and the RL/eval prompts come from it,
so no prompt a reward model was trained on can leak into evaluation.
"""
from __future__ import annotations

import random
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from datasets import load_dataset

from jepa_rm.config import DATASET, SEED

NO_THINK = " /no_think"  # switches off Qwen3's thinking mode
_MULTI_TURN_MARKER = "<extra_id_1>"
_SPLITS = {"train": "train", "validation": "val"}  # our split names -> the subset's "split" column
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    strength: int = 1  # how strongly the annotator preferred `chosen`: 1 slightly, 2 better, 3 much better


def format_text(prompt: str, response: str) -> str:
    return f"User: {prompt}\n\nAssistant: {response}"


def user_message(prompt: str, policy: str) -> str:
    return prompt + NO_THINK if policy == "qwen" else prompt


def strip_think(text: str) -> str:
    return _THINK_BLOCK.sub("", text).strip()


def make_pairs(rows: Iterable[Mapping], max_chars: int = 6000) -> list[PreferencePair]:
    """One pair per human comparison on a single-turn prompt.

    `preference_strength` runs from -3 (response 1 much better) to +3 (response 2 much better), as
    the annotators' own statements confirm. Ties (0) carry no preference and are dropped. With the
    default `max_chars`, the longest resulting text is 2,896 tokens, inside the reward models' 3,072.
    """
    pairs = []
    for row in rows:
        prompt, strength = row["prompt"], row["preference_strength"]
        first, second = row["response_1"], row["response_2"]
        if strength == 0 or _MULTI_TURN_MARKER in prompt:
            continue
        if len(prompt) + max(len(first), len(second)) > max_chars:
            continue
        chosen, rejected = (first, second) if strength < 0 else (second, first)
        pairs.append(PreferencePair(prompt, chosen, rejected, abs(strength)))
    return pairs


def select_prompts(rows: Iterable[Mapping], n: int, max_chars: int = 1000, seed: int = SEED) -> list[str]:
    """Unique single-turn prompts, shuffled deterministically."""
    prompts = list(dict.fromkeys(
        row["prompt"] for row in rows
        if _MULTI_TURN_MARKER not in row["prompt"] and len(row["prompt"]) <= max_chars
    ))
    random.Random(seed).shuffle(prompts)
    return prompts[:n]


def _preference_rows(split: str):
    rows = load_dataset(DATASET, data_dir="preference", split="train")  # one file; real split is a column
    return rows.filter(lambda row: row["split"] == _SPLITS[split])


def load_pairs(split: str) -> list[PreferencePair]:
    return make_pairs(_preference_rows(split))


def load_prompts(split: str, n: int) -> list[str]:
    return select_prompts(_preference_rows(split), n)
