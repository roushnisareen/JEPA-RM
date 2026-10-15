"""Held-out comparison of reward models: how often each one picks the preferred answer, and how
much it just prefers the longer one.

With a few hundred held-out pairs an accuracy is only known to within several points, so every
accuracy comes with a 95% CI, and every model's gap to the standard reward model with a paired one
(same pairs, so the comparison is much tighter than two separate CIs would suggest).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from scipy.stats import spearmanr

from jepa_rm.data import PreferencePair
from jepa_rm.evaluation.stats import bootstrap_ci
from jepa_rm.rewards import RewardModel

REFERENCE = "bt"


def rm_stats(pairs: Sequence[PreferencePair], chosen: Sequence[float], rejected: Sequence[float]) -> dict:
    n = len(pairs)
    correct = [float(c > r) for c, r in zip(chosen, rejected, strict=True)]
    chosen_longer = [len(p.chosen) > len(p.rejected) for p in pairs]
    lengths = [len(p.chosen) for p in pairs] + [len(p.rejected) for p in pairs]
    return {
        "accuracy": sum(correct) / n,
        "accuracy_ci": list(bootstrap_ci(correct)),
        "prefers_longer": sum(bool(c) == cl for c, cl in zip(correct, chosen_longer, strict=True)) / n,
        "corr_with_length": float(spearmanr(list(chosen) + list(rejected), lengths).correlation),
        "correct": correct,  # per pair, for the paired comparison; dropped from the saved summary
    }


def compare_reward_models(pairs: Sequence[PreferencePair], rms: Mapping[str, RewardModel]) -> dict:
    prompts = [p.prompt for p in pairs]
    chosen = [p.chosen for p in pairs]
    rejected = [p.rejected for p in pairs]
    models = {name: rm_stats(pairs, rm.score(prompts, chosen), rm.score(prompts, rejected)) for name, rm in rms.items()}
    if REFERENCE in models:
        ref = models[REFERENCE]["correct"]
        for name, m in models.items():
            if name != REFERENCE:
                diffs = [a - b for a, b in zip(m["correct"], ref, strict=True)]
                m[f"accuracy_vs_{REFERENCE}"] = sum(diffs) / len(diffs)
                m[f"accuracy_vs_{REFERENCE}_ci"] = list(bootstrap_ci(diffs))
    for m in models.values():
        del m["correct"]
    return {
        "n_pairs": len(pairs),
        "humans_prefer_longer": sum(len(c) > len(r) for c, r in zip(chosen, rejected, strict=True)) / len(pairs),
        "models": models,
    }


def format_table(results: dict) -> str:
    lines = [
        f"humans prefer the longer answer {results['humans_prefer_longer']:.1%} of the time "
        f"({results['n_pairs']} held-out pairs)",
        f"{'RM':<11}{'accuracy (95% CI)':>22}{'vs bt (95% CI)':>24}{'prefers longer':>16}{'corr w/ length':>16}",
    ]
    for name, m in results["models"].items():
        lo, hi = m["accuracy_ci"]
        acc = f"{m['accuracy']:.1%} [{lo:.0%}, {hi:.0%}]"
        if f"accuracy_vs_{REFERENCE}" in m:
            dlo, dhi = m[f"accuracy_vs_{REFERENCE}_ci"]
            gap = f"{m[f'accuracy_vs_{REFERENCE}']:+.1%} [{dlo:+.0%}, {dhi:+.0%}]"
        else:
            gap = "-"
        lines.append(f"{name:<11}{acc:>22}{gap:>24}{m['prefers_longer']:>16.1%}{m['corr_with_length']:>16.2f}")
    return "\n".join(lines)
