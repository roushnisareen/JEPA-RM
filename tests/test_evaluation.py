import math

import pytest
import torch

from jepa_rm.data import PreferencePair
from jepa_rm.evaluation.plots import ema, ordered, plot_reward_hacking, plot_training_curves, rescale, series
from jepa_rm.evaluation.policies import (
    aggregate_seeds,
    bootstrap_ci,
    completion_mask,
    hit_length_limit,
    kl_per_token,
    summarize_generations,
)
from jepa_rm.evaluation.reward_models import compare_reward_models, rm_stats


class LengthRM:
    """A reward model that only looks at length, i.e. the worst case we're testing for."""

    def score(self, prompts, responses):
        return [float(len(r)) for r in responses]


PAIRS = [
    PreferencePair("p1", "long good answer", "bad"),
    PreferencePair("p2", "ok", "long bad answer"),
]


def test_rm_stats_detects_a_length_only_reward_model():
    rm = LengthRM()
    stats = rm_stats(PAIRS, rm.score([], [p.chosen for p in PAIRS]), rm.score([], [p.rejected for p in PAIRS]))
    assert stats["accuracy"] == 0.5
    assert stats["prefers_longer"] == 1.0
    assert stats["corr_with_length"] == pytest.approx(1.0)


def test_compare_reward_models_reports_every_model():
    results = compare_reward_models(PAIRS, {"length": LengthRM()})
    assert results["n_pairs"] == 2
    assert results["humans_prefer_longer"] == 0.5
    assert set(results["models"]) == {"length"}


def generation(run, step, gold, bt, jepa, words, truncated=False, prompt="p", seed=0):
    return {"run": run, "seed": seed, "step": step, "prompt": prompt, "gold": gold, "bt": bt, "jepa": jepa,
            "words": words, "truncated": truncated}


def test_summarize_averages_per_checkpoint():
    gens = [
        generation("base", 0, 1.0, 0.0, 0.1, 10, prompt="a"),
        generation("base", 0, 3.0, 2.0, 0.3, 20, prompt="b"),
        generation("bt", 50, 2.0, 5.0, 0.2, 40, truncated=True, prompt="a"),
        generation("bt", 50, 2.0, 7.0, 0.2, 60, prompt="b"),
    ]
    rows, base_std = summarize_generations(gens, [("base", 0, 0), ("bt", 0, 50)])
    assert rows[1] == {"run": "bt", "seed": 0, "step": 50, "gold": 2.0, "bt": 6.0, "jepa": 0.2, "words": 50,
                       "truncated": 0.5, "kl": None}  # these generations carry no KL
    assert base_std["gold"] == pytest.approx(2 ** 0.5)
    ci = aggregate_seeds(rows, gens)[1]["gold_ci"]
    assert ci[0] <= 0.0 <= ci[1]  # per-prompt changes are +1 and -1, so no clear effect


def test_aggregate_seeds_averages_and_keeps_the_spread():
    gens = [generation("base", 0, 0.0, 0, 0, 10, prompt=p) for p in "ab"]
    gens += [generation("bt", 50, gold, 0, 0, 10, prompt=p, seed=s)
             for s, gold in ((0, 1.0), (1, 3.0)) for p in "ab"]
    seed_rows, _ = summarize_generations(gens, [("base", 0, 0), ("bt", 0, 50), ("bt", 1, 50)])
    seed_rows[1]["kl"], seed_rows[2]["kl"] = 0.1, None  # one seed's log is missing
    row = next(r for r in aggregate_seeds(seed_rows, gens) if r["run"] == "bt")
    assert (row["n_seeds"], row["gold"], row["gold_min"], row["gold_max"]) == (2, 2.0, 1.0, 3.0)
    assert row["kl"] == 0.1  # mean of the seeds that have one
    assert row["gold_ci"] == [2.0, 2.0]  # each prompt's change, averaged over seeds, is exactly +2


ROWS = [
    {"run": "base", "step": 0, "gold": 0.0, "bt": 0.0, "jepa": 0.0, "words": 100},
    {"run": "bt", "step": 50, "gold": -1.0, "bt": 3.0, "jepa": 0.1, "words": 180},
    {"run": "jepa", "step": 50, "gold": 0.4, "bt": 0.5, "jepa": 0.3, "words": 105},
    {"run": "dpo", "step": 90, "gold": 0.5, "bt": 0.5, "jepa": 0.2, "words": 110},
]
STD = {"gold": 1.0, "bt": 1.0, "jepa": 0.1}


def grpo_log(n, reward_key="reward", length_key="completion_length"):
    return [{"step": i, reward_key: i / n, length_key: 100 + i} for i in range(1, n + 1)] + [{"train_runtime": 1.0}]


def test_bootstrap_ci_brackets_the_mean_and_tightens_with_more_data():
    small = bootstrap_ci([0.0, 1.0, 2.0, 3.0])
    large = bootstrap_ci([0.0, 1.0, 2.0, 3.0] * 50)
    assert small[0] <= 1.5 <= small[1]
    assert large[1] - large[0] < small[1] - small[0]
    assert bootstrap_ci([1.0, 1.0, 1.0]) == (1.0, 1.0)


def test_cut_off_answers_are_the_ones_without_an_end_token():
    assert hit_length_limit([5, 6, 7], eos_ids={2})
    assert not hit_length_limit([5, 2, 0, 0], eos_ids={2})


def test_completion_mask_stops_after_the_first_end_token():
    ids = torch.tensor([[5, 6, 2, 0, 0], [5, 6, 7, 8, 9]])
    assert completion_mask(ids, eos_ids={2}).tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]


def test_kl_is_zero_for_identical_models_and_positive_otherwise():
    logps = torch.log(torch.tensor([[0.5, 0.25, 0.1]]))
    mask = torch.tensor([[1, 1, 0]])
    assert kl_per_token(logps, logps, mask).item() == pytest.approx(0.0)
    moved = kl_per_token(logps, logps - 0.5, mask).item()
    assert moved > 0
    r = -0.5  # k3 = e^r - r - 1, the same on every unmasked token
    assert moved == pytest.approx(math.exp(r) - r - 1)

def test_reward_hacking_plot_can_use_kl_and_skips_runs_without_it():
    rows = [dict(r, kl=None if r["run"] == "dpo" else 0.02 * r["step"] / 50) for r in ROWS]
    rows[1]["gold_ci"] = [-1.5, -0.5]  # error bars where a CI exists
    assert plot_reward_hacking(rows, STD, x="kl").startswith(b"\x89PNG")


def test_reward_hacking_plot_renders_png_and_pdf():
    assert plot_reward_hacking(ROWS, STD).startswith(b"\x89PNG")
    assert plot_reward_hacking(ROWS, STD, fmt="pdf").startswith(b"%PDF")


def test_training_curves_render_with_either_trl_length_key():
    logs = {"bt": [grpo_log(20)], "jepa": [grpo_log(20, length_key="completions/mean_length")]}
    assert plot_training_curves(logs).startswith(b"\x89PNG")


def test_training_curves_draw_several_seeds_per_run():
    assert plot_training_curves({"bt": [grpo_log(20), grpo_log(10)]}).startswith(b"\x89PNG")


def test_reward_hacking_plot_draws_a_band_across_seeds():
    rows = [dict(r, n_seeds=3, gold_min=r["gold"] - 0.5, gold_max=r["gold"] + 0.5) for r in ROWS]
    assert plot_reward_hacking(rows, STD).startswith(b"\x89PNG")


def test_series_reads_steps_and_skips_entries_without_the_key():
    steps, values = series(grpo_log(3), ("reward",))
    assert steps == [1, 2, 3]
    assert values == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_ema_smooths_toward_new_values():
    assert ema([0.0, 1.0, 1.0], alpha=0.5) == [0.0, 0.5, 0.75]
    assert ema([]) == []


def test_rescale_maps_to_unit_range_and_handles_constants():
    assert rescale([2.0, 4.0, 3.0]) == [0.0, 1.0, 0.5]
    assert rescale([5.0, 5.0]) == [0.0, 0.0]


def test_runs_are_drawn_in_a_fixed_order():
    assert ordered({"dpo", "zzz", "jepa", "bt"}) == ["bt", "jepa", "dpo", "zzz"]


def test_rm_comparison_reports_cis_and_a_paired_gap_to_bt():
    class Perfect:
        def score(self, prompts, responses):
            return [1.0 if "good" in r else 0.0 for r in responses]

    class Backwards:
        def score(self, prompts, responses):
            return [0.0 if "good" in r else 1.0 for r in responses]

    pairs = [PreferencePair(f"p{i}", f"good {i}", f"bad {i}") for i in range(20)]
    results = compare_reward_models(pairs, {"bt": Perfect(), "jepa": Backwards()})
    bt, jepa = results["models"]["bt"], results["models"]["jepa"]
    assert (bt["accuracy"], bt["accuracy_ci"]) == (1.0, [1.0, 1.0])
    assert jepa["accuracy_vs_bt"] == -1.0 and jepa["accuracy_vs_bt_ci"] == [-1.0, -1.0]
    assert "accuracy_vs_bt" not in bt and "correct" not in bt  # internals never reach the saved results


def test_a_ci_that_misses_its_point_estimate_does_not_break_the_figure():
    rows = [dict(r) for r in ROWS]
    rows[1]["gold_ci"] = [0.5, 0.9]  # the point estimate (-1.0) lies outside it
    assert plot_reward_hacking(rows, STD).startswith(b"\x89PNG")


def test_labels_keep_their_gap_and_stay_inside_the_axes():
    from jepa_rm.evaluation.plots import spread_labels

    ys = spread_labels([1.0, 1.0, 1.0, 0.0], gap=0.1, lo=0.0, hi=1.0)
    stacked = sorted(ys)
    assert all(b - a >= 0.1 - 1e-9 for a, b in zip(stacked, stacked[1:], strict=False))
    assert max(ys) <= 1.0  # the pile-up at the top was slid back down
    assert ys[3] == 0.0  # a label with room stays next to its line


def test_runs_are_compared_head_to_head_at_their_final_checkpoints():
    from jepa_rm.evaluation.policies import compare_runs

    gens = [generation("base", 0, 0.0, 0, 0, 10, prompt=p) for p in "abc"]
    gens += [generation("bt", 50, 9.0, 0, 0, 10, prompt=p) for p in "abc"]  # an earlier checkpoint, ignored
    gens += [generation("bt", 100, 1.0, 0, 0, 10, prompt=p) for p in "abc"]
    gens += [generation("jepa", 100, 1.5, 0, 0, 10, prompt=p) for p in "abc"]
    [c] = compare_runs(gens)
    assert (c["a"], c["b"], c["step_a"], c["step_b"]) == ("jepa", "bt", 100, 100)
    assert c["gap"] == pytest.approx(0.5) and c["ci"] == pytest.approx([0.5, 0.5]) and c["n_prompts"] == 3
