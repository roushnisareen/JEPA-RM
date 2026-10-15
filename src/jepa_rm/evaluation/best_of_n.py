"""Best-of-n: reward hacking without any training.

Sample answers from the base model, keep the one a reward model likes best, and score that one
with the gold judge. Picking the best of more samples (larger n) is more optimization pressure on
the reward model. If the gold score of the picked answer stops improving or falls as n grows, the
reward model is being over-optimized.

This is the second over-optimization method in Gao et al. (2022), alongside RL. It needs no
training, so it's a cheap, independent check on the GRPO result.
"""
from __future__ import annotations

import logging
import math
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict

from jepa_rm import tracking
from jepa_rm.artifacts import Store
from jepa_rm.config import SEED, BonConfig, EvalConfig
from jepa_rm.data import load_prompts
from jepa_rm.evaluation.plots import plot_reward_hacking
from jepa_rm.evaluation.policies import format_table, generate, score_all, score_keys
from jepa_rm.evaluation.stats import bootstrap_ci, format_comparisons, paired_gap
from jepa_rm.rewards import COMPARISONS, JEPA_ABLATION, MAIN_COMPARISON, REWARD_KINDS

log = logging.getLogger(__name__)


def bon_kl(n: int) -> float:
    """KL between best-of-n sampling and the base model (Gao et al., 2022)."""
    return math.log(n) - (n - 1) / n


def expected_best_of_n(proxy: Sequence[float], values: Sequence[float], n: int) -> float:
    """Expected `values` of the sample with the highest `proxy`, averaged over every n-subset of
    the samples. That's the unbiased estimator from Gao et al., computed without enumerating
    subsets: the sample ranked r-th by proxy (r = 0 is best) wins a random n-subset with
    probability C(N-1-r, n-1) / C(N, n)."""
    total = len(proxy)
    if not 1 <= n <= total:
        raise ValueError(f"n must be between 1 and {total}, got {n}")
    order = sorted(range(total), key=lambda i: proxy[i], reverse=True)
    subsets = math.comb(total, n)
    return sum(math.comb(total - 1 - rank, n - 1) / subsets * values[i] for rank, i in enumerate(order))


def best_of_n_curves(
    samples: Sequence[dict], ns: Sequence[int], selectors: Sequence[str] = REWARD_KINDS
) -> tuple[list[dict], dict[str, float]]:
    """Rows in the same shape as the policy evaluation, so the same figure can draw them.

    "base" is n = 1 (a plain sample). Each selector row is what best-of-n picks when that reward
    model does the picking: every score of the picked answer, averaged over prompts, plus a 95% CI
    on its gold-score change from n = 1.
    """
    scores = score_keys(samples[0])
    averaged = (*scores, "words", "truncated")
    selectors = [s for s in selectors if s in scores]
    by_prompt: dict[str, list[dict]] = {}
    for s in samples:
        by_prompt.setdefault(s["prompt"], []).append(s)
    groups = list(by_prompt.values())

    def base_of(group, key):
        return statistics.mean(s[key] for s in group)

    rows = [{"run": "base", "step": 1, "bon_kl": 0.0, **{k: statistics.mean(s[k] for s in samples) for k in averaged}}]
    for selector in selectors:
        for n in ns:
            picked = [
                {k: expected_best_of_n([s[selector] for s in g], [float(s[k]) for s in g], n) for k in averaged}
                for g in groups
            ]
            gold_gain = [p["gold"] - base_of(g, "gold") for p, g in zip(picked, groups, strict=True)]
            rows.append({
                "run": selector, "step": n, "bon_kl": bon_kl(n),
                **{k: statistics.mean(p[k] for p in picked) for k in averaged},
                "gold_ci": list(bootstrap_ci(gold_gain)) if len(gold_gain) > 1 else None,
            })
    base_std = {k: statistics.stdev(s[k] for s in samples) for k in scores}
    return rows, base_std


def picked_gold(samples: Sequence[dict], selector: str, n: int) -> dict[str, float]:
    """Per prompt: the expected gold score of the answer `selector` picks out of n."""
    by_prompt: dict[str, list[dict]] = {}
    for s in samples:
        by_prompt.setdefault(s["prompt"], []).append(s)
    return {p: expected_best_of_n([s[selector] for s in g], [s["gold"] for s in g], n) for p, g in by_prompt.items()}


def compare_selectors(samples: Sequence[dict], n: int, pairs=COMPARISONS) -> list[dict]:
    """Head-to-head gold gaps at one n: how much better (or worse) a's picks are than b's on the
    same prompts, with a paired 95% CI."""
    present = score_keys(samples[0])
    return [{"a": a, "b": b, "n": n, **paired_gap(picked_gold(samples, a, n), picked_gold(samples, b, n))}
            for a, b in pairs if a in present and b in present]


def sample_answers(
    store: Store, cfg: BonConfig, prompts: Sequence[str], first: int, count: int
) -> list[dict]:
    """`count` new scored answers per prompt, numbered from `first`.

    The sampling seed depends on `first`, so extending a run draws fresh answers instead of
    repeating the ones it already has.
    """
    gen_cfg = EvalConfig(policy=cfg.policy, batch_size=cfg.batch_size, max_new_tokens=cfg.max_new_tokens,
                         temperature=cfg.temperature, top_p=1.0)
    log.info("sampling %d answers for each of %d prompts", count, len(prompts))
    outputs = generate(list(prompts) * count, None, gen_cfg, seed=SEED + first)
    samples = [
        {"prompt": prompts[k % len(prompts)], "sample": first + k // len(prompts), "response": gen.response,
         "words": len(gen.response.split()), "truncated": gen.truncated}
        for k, gen in enumerate(outputs)
    ]
    score_all(store, samples, cfg.gold_rm)
    return samples


def run_best_of_n(store: Store, cfg: BonConfig, extend: bool = False) -> list[dict]:
    """With `extend`, keep the samples of the previous run (same prompts, already scored) and only
    sample the extra answers needed to reach `cfg.n_samples` per prompt. The previous result is
    saved under a separate name first, so nothing is lost."""
    if max(cfg.ns) > cfg.n_samples:
        raise ValueError(f"largest n ({max(cfg.ns)}) needs at least that many samples, got {cfg.n_samples}")
    name = f"best_of_n_{cfg.policy}"
    previous = store.load_result(f"{name}.json") if extend else None
    if extend and previous is None:
        raise ValueError(f"nothing to extend: results/{name}.json not found")

    if previous:
        old = previous["samples"]
        prompts = list(dict.fromkeys(s["prompt"] for s in old))
        have = min(Counter(s["prompt"] for s in old).values())
        old = [s for s in old if s["sample"] < have]  # an equal number per prompt
        store.save_result(f"{name}_before_extend.json", previous)
        log.info("extending %d prompts from %d to %d samples each", len(prompts), have, cfg.n_samples)
    else:
        old, prompts, have = [], load_prompts("validation", cfg.n_prompts), 0
    new = sample_answers(store, cfg, prompts, have, cfg.n_samples - have) if cfg.n_samples > have else []
    samples = old + new

    rows, base_std = best_of_n_curves(samples, cfg.ns)
    comparisons = compare_selectors(samples, max(cfg.ns))
    store.save_result(f"{name}.json", {
        "config": {**asdict(cfg), "n_prompts": len(prompts), "extended_from": have if previous else None},
        "summary": rows, "base_std": base_std, "comparisons": comparisons, "samples": samples,
    })
    log.info("\n%s\n\n%s", format_table(rows), format_comparisons(comparisons))
    tracking.log_table("best_of_n/comparisons", comparisons)
    for name, png in best_of_n_figures(rows, base_std).items():
        store.save_result(f"{name}_{cfg.policy}.png", png)
        tracking.log_image(f"best_of_n/{name}", png)
    tracking.log_table("best_of_n/summary", rows)
    return rows


def best_of_n_figures(rows: list[dict], base_std: dict[str, float], fmt: str = "png") -> dict[str, bytes]:
    """The main comparison (BT, its length-penalized fix, the frozen control and JEPA) and the
    JEPA ablation, as separate figures so neither gets crowded."""
    own = "Selecting reward model  (what best-of-n picks on)"
    return {
        "best_of_n": plot_reward_hacking(rows, base_std, x="bon_kl", fmt=fmt, own_title=own, runs=MAIN_COMPARISON),
        "best_of_n_ablation": plot_reward_hacking(rows, base_std, x="bon_kl", fmt=fmt, own_title=own,
                                                  runs=JEPA_ABLATION),
    }
