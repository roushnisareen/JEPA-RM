"""Paper-style figures.

Every figure shares one look: a fixed colour per method, labels at the end of each line instead of
a legend, and faint raw curves under bold smoothed ones. All functions return the encoded image
(PNG by default, or PDF for print), so they work the same on Modal and locally.
"""
from __future__ import annotations

import io
import statistics
from collections.abc import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from jepa_rm.rewards import REWARD_KINDS  # noqa: E402

# warm colours for the standard reward model and its fixes, blues for the JEPA family
COLORS = {
    "bt": "#D1603D", "bt_lenpen": "#E3A33B", "frozen_bt": "#8E6BB0",
    "jepa": "#3B6EA8", "jepa_pred": "#6FA8DC", "jepa_rank": "#2E9C9C",
    "dpo": "#5E9C6E", "gold": "#3A3A3A", "base": "#8C8C8C",
}
LABELS = {
    "bt": "Bradley-Terry RM", "bt_lenpen": "BT + length penalty", "frozen_bt": "frozen-feature BT (control)",
    "jepa": "JEPA RM", "jepa_pred": "JEPA loss only", "jepa_rank": "ranking loss only",
    "dpo": "DPO", "gold": "gold judge", "base": "base",
}
RUN_ORDER = ("bt", "bt_lenpen", "frozen_bt", "jepa", "jepa_pred", "jepa_rank", "dpo", "gold")
FALLBACK_COLOR = "#9A7FB0"
LENGTH_KEYS = ("completion_length", "completions/mean_length")  # renamed across TRL versions

STYLE = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "axes.labelcolor": "#333333",
    "axes.edgecolor": "#999999",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E6E6E6",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


def color(run: str) -> str:
    return COLORS.get(run, FALLBACK_COLOR)


def label(run: str) -> str:
    return LABELS.get(run, run)


def ordered(runs: Iterable[str]) -> list[str]:
    runs = set(runs)
    return [r for r in RUN_ORDER if r in runs] + sorted(runs - set(RUN_ORDER))


def ema(values: Sequence[float], alpha: float = 0.2) -> list[float]:
    """Exponential moving average, the same smoothing W&B uses."""
    out: list[float] = []
    for v in values:
        out.append(v if not out else alpha * v + (1 - alpha) * out[-1])
    return out


def rescale(values: Sequence[float]) -> list[float]:
    """Min-max scale to [0, 1], so rewards on different scales can share an axis."""
    lo, hi = min(values), max(values)
    return [0.0] * len(values) if hi == lo else [(v - lo) / (hi - lo) for v in values]


def series(history: Sequence[Mapping], keys: Sequence[str]) -> tuple[list[int], list[float]]:
    """(steps, values) for the first of `keys` found in each trainer log entry."""
    steps, values = [], []
    for entry in history:
        key = next((k for k in keys if k in entry), None)
        if key is not None and "step" in entry:
            steps.append(entry["step"])
            values.append(entry[key])
    return steps, values


def spread_labels(ys: Sequence[float], gap: float, lo: float, hi: float) -> list[float]:
    """Label heights for line ends at `ys`: at least `gap` apart, as close to their lines as
    possible, and inside [lo, hi] when they fit. Returned in the order of `ys`."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    placed: list[float] = []
    for i in order:
        y = ys[i]
        placed.append(y if not placed or y - placed[-1] >= gap else placed[-1] + gap)
    # a pile-up near the top pushes labels out of the axes: cap the top one, then push down only
    # the labels it now collides with, so labels with room stay next to their lines
    if placed and placed[-1] > hi:
        placed[-1] = hi
        for k in range(len(placed) - 2, -1, -1):
            placed[k] = min(placed[k], placed[k + 1] - gap)
    out = [0.0] * len(ys)
    for i, y in zip(order, placed, strict=True):
        out[i] = y
    return out


def _label_ends(ax, ends: list[tuple[str, float, float, str]]) -> None:
    """Write each run's label just past the end of its line, nudged apart so they don't overlap."""
    if not ends:
        return
    xmax = max(x for _, x, _, _ in ends)
    xmin = ax.get_xlim()[0]
    # a few round ticks over the data only: none under the labels, and none crowded together
    ax.set_xticks([t for t in MaxNLocator(nbins=4).tick_values(max(xmin, 0.0), xmax) if xmin <= t <= xmax])
    ax.set_xlim(left=xmin, right=xmax + 0.32 * (xmax - xmin))

    lo, hi = ax.get_ylim()
    gap = 0.08 * (hi - lo)
    ys = spread_labels([y for _, _, y, _ in ends], gap, lo + gap / 2, hi - gap / 2)
    if min(ys) < lo:  # still too many labels to fit: make room rather than draw them off the axes
        ax.set_ylim(bottom=min(ys) - gap / 2)
    for (run, x, _, text), y in zip(ends, ys, strict=True):
        ax.annotate(
            text, xy=(x, y), xytext=(7, 0), textcoords="offset points", va="center",
            color=color(run), fontsize=9, fontweight="bold", annotation_clip=False,
        )


def _reference_line(ax, y: float, text: str, x: float) -> None:
    """Dashed line for the base model, labelled at `x` (the end of the data) so no curve covers it."""
    ax.axhline(y, color=COLORS["base"], lw=1, ls=(0, (4, 3)), zorder=0)
    lo, hi = ax.get_ylim()
    if y - lo < 0.1 * (hi - lo):  # leave room for the label under the line
        ax.set_ylim(bottom=y - 0.1 * (hi - lo))
    ax.annotate(text, xy=(x, y), xytext=(0, -4), textcoords="offset points", ha="right", va="top",
                color=COLORS["base"], fontsize=8)


def _encode(fig, fmt: str) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt)
    plt.close(fig)
    return buf.getvalue()


X_AXES = {
    "step": "training step",
    "kl": "KL to base model (per token)",
    "bon_kl": "best-of-n KL  (log n − (n−1)/n)",
}


def plot_reward_hacking(
    rows: Sequence[Mapping],
    base_std: Mapping[str, float],
    x: str = "step",
    fmt: str = "png",
    own_title: str = "Own reward model  (what RL optimizes)",
    runs: Sequence[str] | None = None,
) -> bytes:
    """Three panels: each run's own RM score, the gold judge's score, and answer length.

    Scores are the change from the base model in units of the base model's std (σ), so both
    reward models and the judge share a scale. Reward hacking is the left panel rising while
    the middle one doesn't. With several seeds, lines are the mean and the shaded band runs from the
    lowest to the highest seed.

    x="kl" plots against how far each policy has moved from the base model instead of against
    steps, as in Gao et al. (2022). That's the fairer comparison: a reward model that's simply a
    weaker signal moves its policy less, and would look "unhacked" at any given step. Rows without
    a KL value are left out of that version.
    """
    base = next(r for r in rows if r["run"] == "base")
    rows = [r for r in rows if r.get(x) is not None and (runs is None or r["run"] in runs or r["run"] == "base")]

    # a scorer with no spread on the base model's answers (e.g. a constant one) is drawn in raw units
    # rather than crashing the figure
    std = {k: v or 1.0 for k, v in base_std.items()}

    def delta(row, key):
        return (row[key] - base[key]) / std[key]

    def spread(row, key, end):  # the seed with the lowest or highest score, as a change from base
        return (row.get(f"{key}_{end}", row[key]) - base[key]) / std[key]

    with plt.rc_context(STYLE):
        fig, (own_ax, gold_ax, len_ax) = plt.subplots(1, 3, figsize=(13.5, 3.9))
        ends: dict = {own_ax: [], gold_ax: [], len_ax: []}

        for run in ordered(r["run"] for r in rows if r["run"] != "base"):
            points = [base] + sorted((r for r in rows if r["run"] == run), key=lambda r: r["step"])
            steps = [0] + [r[x] for r in points[1:]]
            style = {"color": color(run), "lw": 2.2, "marker": "o", "ms": 5, "mec": "white", "mew": 1.2}

            seeds = max(r.get("n_seeds", 1) for r in points)
            band = {"color": color(run), "alpha": 0.12, "lw": 0}  # min-max across seeds

            if run in REWARD_KINDS:  # DPO has no reward model of its own
                own = [delta(r, run) for r in points]
                own_ax.plot(steps, own, **style)
                if seeds > 1:
                    own_ax.fill_between(steps, [spread(r, run, "min") for r in points],
                                        [spread(r, run, "max") for r in points], **band)
                ends[own_ax].append((run, steps[-1], own[-1], f"{label(run)}  {own[-1]:+.1f}σ"))

            gold = [delta(r, "gold") for r in points]
            gold_ax.plot(steps, gold, **style)
            if seeds > 1:
                gold_ax.fill_between(steps, [spread(r, "gold", "min") for r in points],
                                     [spread(r, "gold", "max") for r in points], **band)
            cis = [r.get("gold_ci") for r in points[1:]]
            if all(cis):  # 95% CI on the change from base, in the same σ units
                # clamped at 0: a percentile CI usually contains the mean, but with small or skewed
                # samples it needn't, and matplotlib refuses negative error bars
                lo = [max(0.0, g - ci[0] / std["gold"]) for g, ci in zip(gold[1:], cis, strict=True)]
                hi = [max(0.0, ci[1] / std["gold"] - g) for g, ci in zip(gold[1:], cis, strict=True)]
                gold_ax.errorbar(steps[1:], gold[1:], yerr=[lo, hi], fmt="none", ecolor=color(run),
                                 elinewidth=1.2, capsize=3, alpha=0.6)
            ends[gold_ax].append((run, steps[-1], gold[-1], f"{label(run)}  {gold[-1]:+.1f}σ"))

            words = [r["words"] for r in points]
            len_ax.plot(steps, words, **style)
            if seeds > 1:
                len_ax.fill_between(steps, [r.get("words_min", r["words"]) for r in points],
                                    [r.get("words_max", r["words"]) for r in points], **band)
            ends[len_ax].append((run, steps[-1], words[-1], f"{label(run)}  {words[-1]:.0f}"))

        last_step = max(r[x] for r in rows)
        _reference_line(own_ax, 0, "base model", last_step)
        _reference_line(gold_ax, 0, "base model", last_step)
        _reference_line(len_ax, base["words"], "base model", last_step)
        own_ax.set(title=own_title, ylabel="change vs. base (σ)")
        gold_ax.set(title="Gold judge  (true quality, 95% CI)", ylabel="change vs. base (σ)")
        len_ax.set(title="Answer length", ylabel="mean words")
        for ax in (own_ax, gold_ax, len_ax):
            ax.set_xlabel(X_AXES[x])
            _label_ends(ax, ends[ax])

        fig.tight_layout(w_pad=2.5)
        return _encode(fig, fmt)


def plot_training_curves(logs: Mapping[str, Sequence[Sequence[Mapping]]], fmt: str = "png") -> bytes:
    """GRPO training reward and answer length per run, from the trainer's log histories.

    `logs` maps each run to one history per seed; seeds share their run's colour. The two reward
    models score on different scales, so each run's rewards (all seeds together) are rescaled to [0, 1].
    """
    with plt.rc_context(STYLE):
        fig, (reward_ax, len_ax) = plt.subplots(1, 2, figsize=(10, 3.9))
        ends: dict = {reward_ax: [], len_ax: []}

        for run in ordered(logs):
            for ax, keys, rescaled in ((reward_ax, ("reward",), True), (len_ax, LENGTH_KEYS, False)):
                curves = [c for c in (series(history, keys) for history in logs[run]) if c[0]]
                if not curves:
                    continue
                if rescaled:  # one scale for all seeds of a run, so they stay comparable
                    flat = rescale([v for _, values in curves for v in values])
                    sizes = [len(values) for _, values in curves]
                    starts = [sum(sizes[:i]) for i in range(len(sizes))]
                    curves = [(steps, flat[a : a + n]) for (steps, _), a, n in zip(curves, starts, sizes, strict=True)]
                finals = []
                for steps, values in curves:
                    smooth = ema(values)
                    ax.plot(steps, values, color=color(run), lw=1, alpha=0.2)
                    ax.plot(steps, smooth, color=color(run), lw=2.2 if len(curves) == 1 else 1.6)
                    finals.append(smooth[-1])
                text = label(run) if len(curves) == 1 else f"{label(run)}  ({len(curves)} seeds)"
                ends[ax].append((run, max(steps[-1] for steps, _ in curves), statistics.mean(finals), text))

        reward_ax.set(title="Training reward", ylabel="reward (rescaled per run)")
        len_ax.set(title="Answer length during training", ylabel="mean tokens")
        for ax in (reward_ax, len_ax):
            ax.set_xlabel("training step")
            _label_ends(ax, ends[ax])

        fig.tight_layout(w_pad=2.5)
        return _encode(fig, fmt)


def _stress_label(perturbation: Mapping) -> str:
    """Row label, noting when an edit only applied to some answers (scores average over all)."""
    changed = perturbation.get("changed", 1.0)
    return perturbation["label"] if changed >= 0.995 else f"{perturbation['label']}  ({changed:.0%} edited)"


def plot_stress_test(results: Mapping, fmt: str = "png") -> bytes:
    """One row per edit, one dot per scorer: the mean score change (σ) from editing a good answer.

    Top section: edits that shouldn't help, so a dot right of 0 means that scorer can be gamed
    that way. Bottom section: edits that should hurt, a sanity check that each scorer still
    notices worse answers.
    """
    flat = [p for p in results["perturbations"] if p["expected"] == "flat"]
    down = [p for p in results["perturbations"] if p["expected"] == "down"]
    scorers = ordered(results["perturbations"][0]["scores"])

    # rows top to bottom, with a gap between the two sections
    rows = [(p, i) for i, p in enumerate(flat)] + [(p, len(flat) + 1 + i) for i, p in enumerate(down)]
    n_rows = len(flat) + 1 + len(down)
    step = min(0.22, 0.8 / len(scorers))  # each row's dots stay inside its own band
    dodge = {s: (i - (len(scorers) - 1) / 2) * step for i, s in enumerate(scorers)}
    ms = 7 if len(scorers) <= 3 else 5
    xs = [p["scores"][s]["delta_sigma"] for p, _ in rows for s in scorers]
    span = max(1.0, max(abs(x) for x in xs)) * 1.15

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.5, (0.3 + 0.07 * len(scorers)) * n_rows + 1.8))
        ax.axvspan(0, span, ymin=1 - (len(flat) + 0.5) / (n_rows + 0.5), ymax=1, color=COLORS["bt"], alpha=0.06, lw=0)
        ax.axvline(0, color=COLORS["base"], lw=1, ls=(0, (4, 3)))

        for scorer in scorers:
            for p, y in rows:
                x = p["scores"][scorer]["delta_sigma"]
                yy = y + dodge[scorer]
                ax.plot([0, x], [yy, yy], color=color(scorer), lw=1.4, alpha=0.35)
                ax.plot(x, yy, "o", color=color(scorer), ms=ms, mec="white", mew=1.0)
            ax.plot([], [], "o", color=color(scorer), ms=7, label=label(scorer))

        ax.set_yticks([y for _, y in rows], [_stress_label(p) for p, _ in rows])
        ax.set_ylim(n_rows - 0.4, -0.9)
        ax.set_xlim(-span, span)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("score change vs. the original answer (σ of each scorer)")
        ax.annotate("shouldn't raise the score  (right of 0 = gameable)", xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 6), textcoords="offset points", fontsize=9, fontweight="bold", color="#555555")
        ax.annotate("should lower the score  (sanity check)", xy=(-span, len(flat)), xytext=(0, 0),
                    textcoords="offset points", fontsize=9, fontweight="bold", color="#555555", va="center")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=min(len(scorers), 4), frameon=False,
                  handletextpad=0.3)
        ax.set_title(f"Can each scorer be gamed?  (n = {results['n_examples']} answers)", pad=24)

        fig.tight_layout()
        return _encode(fig, fmt)


def plot_pooling(
    bon_by_pooling: Mapping[str, Mapping], kinds: Sequence[str] = ("frozen_bt", "jepa"), fmt: str = "png"
) -> bytes:
    """Gold-judge change of each kind's best-of-n picks as n grows: solid = mean pooling, dashed =
    last-token pooling. Both arms pick from the same answers, so any gap is the pooling's doing."""
    styles = {"mean": "-", "last": (0, (5, 3))}
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9.5, 4.4))
        ends = []
        for pooling, bon in bon_by_pooling.items():
            summary, std = bon["summary"], bon["base_std"]["gold"] or 1.0
            base = next(r for r in summary if r["run"] == "base")
            for kind in kinds:
                curve = sorted((r for r in summary if r["run"] == kind), key=lambda r: r["step"])
                if not curve:
                    continue
                ns = [1] + [r["step"] for r in curve]
                gold = [0.0] + [(r["gold"] - base["gold"]) / std for r in curve]
                ax.plot(ns, gold, color=color(kind), ls=styles.get(pooling, "-"), lw=2.2, marker="o", ms=4,
                        mec="white", mew=1.0)
                ends.append((kind, ns[-1], gold[-1], f"{label(kind)}, {pooling} pooling  {gold[-1]:+.2f}σ"))
        ax.set_xscale("log", base=2)
        all_ns = sorted({1} | {r["step"] for bon in bon_by_pooling.values()
                               for r in bon["summary"] if r["run"] != "base"})
        ax.set_xticks(all_ns, [str(n) for n in all_ns])
        ax.axhline(0, color=COLORS["base"], lw=1, ls=(0, (4, 3)), zorder=0)
        ax.set(title="Best-of-n with mean vs. last-token pooling  (same answers)",
               xlabel="n, answers sampled per question (log scale)", ylabel="gold judge, change vs. base (σ)")
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(right=xmax * 6)  # room for the line labels on a log axis
        lo, hi = ax.get_ylim()
        gap = 0.08 * (hi - lo)
        ys = spread_labels([y for _, _, y, _ in ends], gap, lo + gap / 2, hi - gap / 2)
        for (kind, x, _, text), y in zip(ends, ys, strict=True):
            ax.annotate(text, xy=(x, y), xytext=(7, 0), textcoords="offset points", va="center",
                        color=color(kind), fontsize=9, fontweight="bold", annotation_clip=False)
        fig.tight_layout()
        return _encode(fig, fmt)
