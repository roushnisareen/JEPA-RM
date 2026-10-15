from __future__ import annotations

import random
from collections.abc import Sequence

from jepa_rm.config import SEED


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 2000, level: float = 0.95, seed: int = SEED
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values`."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    tail = (1 - level) / 2
    return means[int(tail * n_boot)], means[min(n_boot - 1, int((1 - tail) * n_boot))]


def paired_gap(a: dict[str, float], b: dict[str, float]) -> dict:
    """Mean of a - b over the keys (prompts) both have, with a bootstrap 95% CI.

    Pairing by prompt removes prompt difficulty from the comparison, so it's much tighter than
    comparing two separate CIs.
    """
    shared = sorted(a.keys() & b.keys())
    diffs = [a[k] - b[k] for k in shared]
    if len(diffs) < 2:
        return {"gap": None, "ci": None, "n_prompts": len(diffs)}
    return {"gap": sum(diffs) / len(diffs), "ci": list(bootstrap_ci(diffs)), "n_prompts": len(diffs)}


def format_comparisons(comparisons: list[dict]) -> str:
    lines = [f"{'comparison':<24}{'gold gap':>10}{'95% CI':>22}{'prompts':>9}"]
    for c in comparisons:
        where = f" @{c['n']}" if "n" in c else ""
        name = f"{c['a']} - {c['b']}{where}"
        if c["gap"] is None:
            lines.append(f"{name:<24}{'-':>10}{'-':>22}{c['n_prompts']:>9}")
            continue
        lo, hi = c["ci"]
        lines.append(f"{name:<24}{c['gap']:>+10.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>22}{c['n_prompts']:>9}")
    return "\n".join(lines)
