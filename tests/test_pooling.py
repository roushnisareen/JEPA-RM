import random

import pytest

from jepa_rm.artifacts import LocalStore
from jepa_rm.config import BonConfig
from jepa_rm.evaluation import best_of_n
from jepa_rm.evaluation.best_of_n import best_of_n_curves
from jepa_rm.evaluation.plots import plot_pooling
from jepa_rm.evaluation.pooling import compare_pooling, format_pooling, paired_rows

KINDS = ("bt", "frozen_bt", "jepa")


def samples(jepa_tracks_gold: bool, seed: int = 0) -> list[dict]:
    """6 prompts x 8 answers. The gold score and the answers are the same for every call with the
    same seed; only the JEPA score depends on `jepa_tracks_gold`."""
    rng = random.Random(seed)
    out = []
    for p in range(6):
        for i in range(8):
            gold = rng.gauss(0, 1)
            out.append({"prompt": f"p{p}", "sample": i, "response": f"answer {p}-{i}", "gold": gold,
                        "words": 100 + i, "truncated": False, "bt": gold + rng.gauss(0, 0.5),
                        "frozen_bt": rng.gauss(0, 1), "jepa": gold if jepa_tracks_gold else -gold})
    return out


def arm(jepa_tracks_gold: bool) -> dict:
    s = samples(jepa_tracks_gold)
    summary, base_std = best_of_n_curves(s, ns=(2, 4, 8), selectors=KINDS)
    return {
        "reward_models": {"models": {"jepa": {"accuracy": 0.66, "accuracy_ci": [0.6, 0.72], "prefers_longer": 0.6}}},
        "stress": {"perturbations": [
            {"name": "restate", "expected": "flat", "scores": {"jepa": {"delta_sigma": 0.2, "ci_sigma": [0.15, 0.25]}}},
            {"name": "truncate", "expected": "down",
             "scores": {"jepa": {"delta_sigma": -0.4, "ci_sigma": [-0.5, -0.3]}}},
        ]},
        "bon": {"summary": summary, "base_std": base_std, "samples": s},
    }


def test_last_pooling_that_tracks_the_judge_beats_one_that_does_not_pair_by_pair():
    result = compare_pooling({"mean": arm(jepa_tracks_gold=False), "last": arm(jepa_tracks_gold=True)},
                             kinds=("frozen_bt", "jepa"))
    [jepa] = [r for r in result["paired_best_of_n"] if r["kind"] == "jepa"]
    assert jepa["n"] == 8 and jepa["n_prompts"] == 6
    assert jepa["ci_sigma"][0] > 0  # picking by gold beats picking by its negation on every prompt
    [frozen] = [r for r in result["paired_best_of_n"] if r["kind"] == "frozen_bt"]
    assert frozen["gap_sigma"] == pytest.approx(0.0)  # same scores in both arms, same picks
    assert [r["edit"] for r in result["loopholes"]] == ["restate", "restate"]  # "shouldn't help" edits only
    assert {r["pooling"] for r in result["accuracy"]} == {"mean", "last"}
    assert "paired" in format_pooling(result)


def test_arms_that_scored_different_answers_are_not_compared_pair_by_pair():
    other = arm(jepa_tracks_gold=True)
    other["bon"]["samples"] = samples(jepa_tracks_gold=True, seed=1)  # different answers and gold scores
    with pytest.raises(ValueError, match="same answers"):
        paired_rows({"mean": arm(jepa_tracks_gold=False), "last": other}, kinds=("jepa",))


def test_missing_results_are_skipped_not_fatal():
    result = compare_pooling({"mean": {"reward_models": {}, "stress": {}, "bon": None},
                              "last": {"reward_models": {}, "stress": {}, "bon": None}})
    assert result == {"accuracy": [], "loopholes": [], "best_of_n": [], "paired_best_of_n": []}


def test_pooling_figure_renders():
    bons = {"mean": arm(False)["bon"], "last": arm(True)["bon"]}
    assert plot_pooling(bons, kinds=("frozen_bt", "jepa")).startswith(b"\x89PNG")


def test_rescoring_keeps_answers_gold_and_bt_and_replaces_only_the_embedding_kinds(tmp_path, monkeypatch):
    source, target = LocalStore(tmp_path / "mean"), LocalStore(tmp_path / "last")
    original = samples(jepa_tracks_gold=False)
    source.save_result("best_of_n_qwen.json", {"samples": original})
    rescored_kinds = []

    def fake_score_all(store, gens, gold_rm, kinds, gold):
        assert gold is False and store is target  # the judge is never rerun; this store's models score
        rescored_kinds.extend(kinds)
        for g in gens:
            g.update({k: 42.0 for k in kinds})

    monkeypatch.setattr(best_of_n, "score_all", fake_score_all)
    best_of_n.run_best_of_n(target, BonConfig(n_samples=8, ns=(2, 4, 8)), rescore_from=source)
    result = target.load_result("best_of_n_qwen.json")

    assert rescored_kinds == ["frozen_bt", "jepa", "jepa_pred", "jepa_rank"]
    for old, new in zip(original, result["samples"], strict=True):
        assert (new["response"], new["gold"], new["bt"]) == (old["response"], old["gold"], old["bt"])
        assert new["jepa"] == 42.0 and new["frozen_bt"] == 42.0
    assert original[0]["jepa"] != 42.0  # the source's samples weren't modified in place


def test_rescoring_needs_enough_samples_and_cannot_be_combined_with_extend(tmp_path):
    source, target = LocalStore(tmp_path / "mean"), LocalStore(tmp_path / "last")
    source.save_result("best_of_n_qwen.json", {"samples": samples(False)})
    with pytest.raises(ValueError, match="can't be combined"):
        best_of_n.run_best_of_n(target, BonConfig(n_samples=8, ns=(2,)), extend=True, rescore_from=source)
    with pytest.raises(ValueError, match="nothing to rescore"):
        best_of_n.run_best_of_n(target, BonConfig(n_samples=8, ns=(2,)), rescore_from=LocalStore(tmp_path / "empty"))
