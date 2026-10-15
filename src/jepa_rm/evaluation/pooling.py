"""Mean vs. last-token pooling for the embedding-space reward models, side by side.

Each arm is one HF repo trained with one pooling setting. The last-pooling arm reuses the mean arm's
BT models and rescores the mean arm's best-of-n answers, so the two arms differ only in how the
frozen encoder pools. That makes the best-of-n comparison paired: on every prompt, both arms pick
from the same 64 answers, each with the same gold-judge score.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from jepa_rm.evaluation.best_of_n import picked_gold
from jepa_rm.evaluation.stats import paired_gap
from jepa_rm.rewards import EMBEDDING_KINDS

ARMS = ("mean", "last")


def accuracy_rows(arms: Mapping[str, Mapping], kinds: Sequence[str]) -> list[dict]:
    rows = []
    for kind in kinds:
        for pooling, results in arms.items():
            m = results.get("reward_models", {}).get("models", {}).get(kind)
            if m:
                rows.append({"kind": kind, "pooling": pooling, "accuracy": m["accuracy"], "ci": m["accuracy_ci"],
                             "prefers_longer": m["prefers_longer"]})
    return rows


def loophole_rows(arms: Mapping[str, Mapping], kinds: Sequence[str]) -> list[dict]:
    """Every "shouldn't help" edit's shift for each kind and pooling, with its CI."""
    rows = []
    for pooling, results in arms.items():
        for p in results.get("stress", {}).get("perturbations", []):
            if p["expected"] != "flat":
                continue
            for kind in kinds:
                st = p["scores"].get(kind)
                if st:
                    rows.append({"kind": kind, "pooling": pooling, "edit": p["name"],
                                 "delta_sigma": st["delta_sigma"], "ci": st.get("ci_sigma")})
    return rows


def best_of_n_rows(arms: Mapping[str, Mapping], kinds: Sequence[str]) -> list[dict]:
    """Per kind and pooling: where its gold curve peaks, and its own and the gold judge's change at
    the largest n, in σ of the base answers."""
    rows = []
    for pooling, results in arms.items():
        bon = results.get("bon")
        if not bon:
            continue
        summary, std = bon["summary"], bon["base_std"]
        base = next(r for r in summary if r["run"] == "base")
        for kind in kinds:
            curve = sorted((r for r in summary if r["run"] == kind), key=lambda r: r["step"])
            if not curve:
                continue
            last = curve[-1]
            rows.append({
                "kind": kind, "pooling": pooling, "max_n": last["step"],
                "peak_n": max(curve, key=lambda r: r["gold"])["step"],
                "own_rise_sigma": (last[kind] - base[kind]) / (std[kind] or 1.0),
                "gold_change_sigma": (last["gold"] - base["gold"]) / (std["gold"] or 1.0),
            })
    return rows


def paired_rows(arms: Mapping[str, Mapping], kinds: Sequence[str], n: int | None = None) -> list[dict]:
    """last − mean, per kind: the gold score of what each pooling picks out of n, paired by prompt.

    Only valid when both arms scored the same answers (the last arm rescored the mean arm's samples),
    which is checked rather than assumed.
    """
    mean, last = arms.get("mean", {}).get("bon"), arms.get("last", {}).get("bon")
    if not mean or not last:
        return []
    key = lambda s: (s["prompt"], s["sample"])  # noqa: E731
    mean_gold = {key(s): (s["response"], s["gold"]) for s in mean["samples"]}
    last_gold = {key(s): (s["response"], s["gold"]) for s in last["samples"]}
    if mean_gold != last_gold:
        raise ValueError("the two arms didn't score the same answers, so they can't be compared pair by pair")
    n = n or max(r["step"] for r in last["summary"] if r["run"] != "base")
    std = mean["base_std"]["gold"] or 1.0
    rows = []
    for kind in kinds:
        if kind not in mean["samples"][0] or kind not in last["samples"][0]:
            continue
        gap = paired_gap(picked_gold(last["samples"], kind, n), picked_gold(mean["samples"], kind, n))
        rows.append({"kind": kind, "n": n, "gap_sigma": gap["gap"] / std if gap["gap"] is not None else None,
                     "ci_sigma": [c / std for c in gap["ci"]] if gap["ci"] else None, "n_prompts": gap["n_prompts"]})
    return rows


def compare_pooling(arms: Mapping[str, Mapping], kinds: Sequence[str] = EMBEDDING_KINDS) -> dict:
    """`arms` maps "mean" and "last" to their results: {"reward_models", "stress", "bon"}, any of
    which may be missing."""
    return {
        "accuracy": accuracy_rows(arms, kinds),
        "loopholes": loophole_rows(arms, kinds),
        "best_of_n": best_of_n_rows(arms, kinds),
        "paired_best_of_n": paired_rows(arms, kinds),
    }


def format_pooling(result: Mapping) -> str:
    def ci(c):
        return f"[{c[0]:+.2f}, {c[1]:+.2f}]" if c else ""

    lines = ["held-out accuracy", f"{'kind':<11}{'pooling':>8}{'accuracy (95% CI)':>24}{'prefers longer':>16}"]
    for r in result["accuracy"]:
        lo, hi = r["ci"]
        acc = f"{r['accuracy']:.1%} [{lo:.0%}, {hi:.0%}]"
        lines.append(f"{r['kind']:<11}{r['pooling']:>8}{acc:>24}{r['prefers_longer']:>16.1%}")
    lines += ["", "\"shouldn't help\" edits, shift in σ",
              f"{'kind':<11}{'pooling':>8}{'edit':>12}{'shift (95% CI)':>26}"]
    for r in result["loopholes"]:
        shift = f"{r['delta_sigma']:+.2f} {ci(r['ci'])}"
        lines.append(f"{r['kind']:<11}{r['pooling']:>8}{r['edit']:>12}{shift:>26}")
    lines += ["", "best-of-n", f"{'kind':<11}{'pooling':>8}{'peak n':>8}{'own rise':>10}{'gold change':>13}"]
    for r in result["best_of_n"]:
        lines.append(f"{r['kind']:<11}{r['pooling']:>8}{r['peak_n']:>8}{r['own_rise_sigma']:>+10.2f}"
                     f"{r['gold_change_sigma']:>+13.2f}")
    lines += ["", "paired: gold of last-pooling picks − mean-pooling picks, same answers (σ)"]
    for r in result["paired_best_of_n"]:
        lines.append(f"{r['kind']:<11} @{r['n']:<4}{r['gap_sigma']:+.3f} {ci(r['ci_sigma'])}"
                     f"  ({r['n_prompts']} prompts)")
    return "\n".join(lines)
