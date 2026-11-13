import itertools
import random
import statistics

import pytest

from jepa_rm.config import BonConfig
from jepa_rm.evaluation.best_of_n import (
    best_of_n_curves,
    bon_kl,
    check_same_sampling,
    compare_ns,
    expected_best_of_n,
    format_verdicts,
    verdicts,
    within_prompt_std,
)
from jepa_rm.evaluation.plots import plot_reward_hacking


def brute_force(proxy, values, n):
    """Average over every n-subset of the value of its highest-proxy sample."""
    picks = [values[max(subset, key=lambda i: proxy[i])] for subset in itertools.combinations(range(len(proxy)), n)]
    return statistics.mean(picks)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_closed_form_matches_enumerating_every_subset(n):
    rng = random.Random(n)
    proxy = [rng.random() for _ in range(8)]
    values = [rng.gauss(0, 1) for _ in range(8)]
    assert expected_best_of_n(proxy, values, n) == pytest.approx(brute_force(proxy, values, n))


def test_n_of_one_is_the_plain_average_and_n_of_all_is_the_top_pick():
    proxy, values = [0.1, 0.9, 0.5], [10.0, 30.0, 20.0]
    assert expected_best_of_n(proxy, values, 1) == pytest.approx(20.0)
    assert expected_best_of_n(proxy, values, 3) == 30.0


def test_n_outside_the_sample_count_is_rejected():
    with pytest.raises(ValueError):
        expected_best_of_n([0.1, 0.2], [1.0, 2.0], 3)


def test_bon_kl_starts_at_zero_and_grows():
    kls = [bon_kl(n) for n in (1, 2, 4, 8, 16)]
    assert kls[0] == 0.0
    assert kls == sorted(kls)


def test_curves_catch_a_reward_model_that_rewards_length():
    # longer answers are worse here: "bt" is fooled by length, "jepa" happens to agree with the judge
    rng = random.Random(0)
    samples = []
    for prompt in range(30):
        for _ in range(16):
            words = rng.randint(20, 400)
            gold = -words / 100 + rng.gauss(0, 0.3)
            samples.append({"prompt": f"p{prompt}", "words": words, "truncated": False,
                            "gold": gold, "bt": float(words), "jepa": gold})
    rows, base_std = best_of_n_curves(samples, ns=(2, 4, 8, 16))
    gold = {run: [r["gold"] for r in rows if r["run"] == run] for run in ("bt", "jepa")}
    assert gold["bt"] == sorted(gold["bt"], reverse=True)  # more selection pressure, worse answers
    assert gold["jepa"] == sorted(gold["jepa"])
    assert all(r["gold_ci"][1] < 0 for r in rows if r["run"] == "bt")  # clearly worse than n = 1
    assert plot_reward_hacking(rows, base_std, x="bon_kl").startswith(b"\x89PNG")


def test_selector_comparisons_keep_n_and_the_prompt_count_apart():
    from jepa_rm.evaluation.best_of_n import compare_selectors

    rng = random.Random(1)
    samples = [{"prompt": f"p{i}", "gold": g, "bt": -g, "jepa": g, "words": 10, "truncated": False}
               for i in range(5) for g in (rng.gauss(0, 1) for _ in range(8))]
    [c] = compare_selectors(samples, n=4)
    assert (c["a"], c["b"], c["n"], c["n_prompts"]) == ("jepa", "bt", 4, 5)
    assert c["ci"][0] > 0  # picking by gold beats picking by its negation on every prompt


def test_extending_reuses_old_samples_and_draws_fresh_ones(tmp_path, monkeypatch):
    from jepa_rm import config
    from jepa_rm.artifacts import LocalStore
    from jepa_rm.config import BonConfig
    from jepa_rm.evaluation import best_of_n
    from jepa_rm.evaluation.policies import Generated

    seeds = []

    def fake_generate(prompts, adapter, cfg, seed=config.SEED):
        seeds.append(seed)
        return [Generated(f"answer {seed}-{i}", False, 0.0) for i in range(len(prompts))]

    def fake_score_all(store, samples, gold_rm):
        for s in samples:
            s.update({"gold": float(len(s["response"])), **{k: float(s["sample"]) for k in ("bt", "jepa")}})

    monkeypatch.setattr(best_of_n, "generate", fake_generate)
    monkeypatch.setattr(best_of_n, "score_all", fake_score_all)
    monkeypatch.setattr(best_of_n, "load_prompts", lambda split, n: ["p0", "p1", "p2"][:n])
    store = LocalStore(tmp_path)

    best_of_n.run_best_of_n(store, BonConfig(n_prompts=3, n_samples=2, ns=(2,)))
    first = store.load_result("best_of_n_qwen.json")["samples"]
    best_of_n.run_best_of_n(store, BonConfig(n_samples=5, ns=(2, 4)), extend=True)
    extended = store.load_result("best_of_n_qwen.json")

    assert seeds == [config.SEED, config.SEED + 2]  # fresh randomness for the new samples
    samples = extended["samples"]
    assert samples[:6] == first  # old samples kept as they were, scores included
    assert len(samples) == 15 and sorted({s["sample"] for s in samples}) == [0, 1, 2, 3, 4]
    assert len({s["response"] for s in samples}) == 15  # nothing repeated
    assert extended["config"]["n_prompts"] == 3 and extended["config"]["extended_from"] == 2
    assert store.load_result("best_of_n_qwen_before_extend_2.json")["samples"] == first  # named by sample count


def test_extending_needs_something_to_extend(tmp_path):
    from jepa_rm.artifacts import LocalStore
    from jepa_rm.config import BonConfig
    from jepa_rm.evaluation.best_of_n import run_best_of_n

    with pytest.raises(ValueError, match="nothing to extend"):
        run_best_of_n(LocalStore(tmp_path), BonConfig(), extend=True)


def test_tied_proxy_scores_share_the_pick_instead_of_the_first_sample_taking_it():
    values = [10.0, 20.0, 30.0, 40.0]
    assert expected_best_of_n([1.0] * 4, values, 4) == pytest.approx(25.0)  # a blind selector: the plain average
    assert expected_best_of_n([1.0] * 4, values, 2) == pytest.approx(25.0)
    assert expected_best_of_n([2.0, 2.0, 1.0, 1.0], values, 4) == pytest.approx(15.0)  # the tied top pair, averaged


def long_answers_win(seed=0):
    """3 prompts x 8 answers; gold likes long answers except the longest, which is bad. 'bt' sees
    that and keeps improving with n; 'jepa' loves the bad one most, so its picks peak early and fall."""
    rng = random.Random(seed)
    out = []
    for p in range(3):
        for i in range(8):
            words = 50 + 10 * i + rng.random()
            gold = 5 * p + words / 20 if i < 7 else 5 * p - 2.0  # the longest answer is bad
            out.append({"prompt": f"p{p}", "sample": i, "response": "x" * i, "words": words, "truncated": False,
                        "gold": gold, "bt": words if i < 7 else 0.0, "jepa": words})
    return out


def test_verdicts_apply_the_pre_registered_rules():
    samples = long_answers_win()
    rows, std = best_of_n_curves(samples, ns=(2, 4, 8), selectors=("bt", "jepa"))
    by = {v["selector"]: v for v in verdicts(rows, samples, std)}
    assert by["bt"]["optimizable"] and by["bt"]["peak_n"] == 8 and not by["bt"]["hacked"]
    assert by["bt"]["peak_to_max"] is None  # peak is the largest n: nothing to drop
    assert by["jepa"]["optimizable"] and by["jepa"]["peak_n"] < 8 and by["jepa"]["hacked"]
    drop = by["jepa"]["peak_to_max"]
    assert drop["gap"] < 0 and drop["ci"][1] < 0 and drop["n_prompts"] == 3
    assert compare_ns(samples, "jepa", by["jepa"]["peak_n"], 8) == drop
    assert "hacked" in format_verdicts(list(by.values()), std)


def test_within_prompt_spread_ignores_differences_between_prompts():
    samples = long_answers_win()
    total = best_of_n_curves(samples, ns=(2,), selectors=("bt",))[1]
    within = within_prompt_std(samples)
    assert within["gold"] < total["gold"]  # prompts sit 5 points apart; answers to one prompt don't
    assert within["bt"] == pytest.approx(total["bt"], rel=0.05)  # 'bt' has no between-prompt component


def test_extending_or_rescoring_refuses_samples_drawn_with_other_settings():
    cfg = BonConfig(n_samples=8, ns=(2,))
    check_same_sampling({"policy": "qwen", "temperature": 1.0, "n_samples": 16}, cfg)  # n_samples may differ
    with pytest.raises(ValueError, match="temperature"):
        check_same_sampling({"temperature": 0.7}, cfg)
    with pytest.raises(ValueError, match="policy"):
        check_same_sampling({"policy": "phi"}, cfg)
