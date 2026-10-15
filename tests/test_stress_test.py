import pytest

from jepa_rm.data import PreferencePair
from jepa_rm.evaluation.plots import plot_stress_test
from jepa_rm.evaluation.stress_test import (
    ORIGINAL,
    PERTURBATIONS,
    Example,
    build_variants,
    corrupt_facts,
    format_table,
    pick_examples,
    reverse_sentences,
    run_stress_test,
    table_rows,
)

EXAMPLE = Example(
    prompt="Why is the sky blue?",
    response="Sunlight scatters off air molecules. Blue light scatters the most. So the sky looks blue.",
    other_response="Bread rises because yeast makes gas.",
)


class LengthRM:
    """Rewards length only: the worst case the stress test should catch."""

    def score(self, prompts, responses):
        return [float(len(r)) for r in responses]


class PromptOverlapRM:
    """Rewards sharing words with the prompt: the failure we suspect the JEPA RM might have."""

    def score(self, prompts, responses):
        pairs = zip(prompts, responses, strict=True)
        return [float(len(set(p.lower().split()) & set(r.lower().split()))) for p, r in pairs]


def variant(name):
    return next(p for p in PERTURBATIONS if p.name == name).apply(EXAMPLE)


def test_edits_that_should_not_help_keep_the_original_content():
    for name in ("filler", "repeat", "confident", "restate"):
        assert EXAMPLE.response.strip() in variant(name)
    assert all(s in variant("bullets") for s in ("Sunlight scatters off air molecules.", "So the sky looks blue."))


def test_edits_that_should_hurt_remove_or_replace_content():
    assert variant("truncate") == "Sunlight scatters off air molecules."
    assert variant("prompt_only") == EXAMPLE.prompt
    assert variant("off_topic") == EXAMPLE.other_response


def test_wrong_facts_changes_numbers_and_flips_claims_but_stays_fluent():
    assert corrupt_facts("Water boils at 100 degrees and is safe.") == "Water boils at 201 degrees and is not safe."
    assert corrupt_facts("It is not hot. Always more.") == "It is not hot. Always less."  # no "is not not"
    assert corrupt_facts("This is never true.") == "This is not always false."


def test_reversed_keeps_the_same_sentences():
    assert reverse_sentences("A one. B two. C three.") == "C three. B two. A one."
    assert sorted(reverse_sentences(EXAMPLE.response).split()) == sorted(EXAMPLE.response.split())


def test_truncate_handles_a_single_sentence():
    one = Example("q", "just a few words here", "x")
    assert next(p for p in PERTURBATIONS if p.name == "truncate").apply(one) == "just a"


def test_every_perturbation_is_labelled_with_an_expected_direction():
    assert {p.expected for p in PERTURBATIONS} == {"flat", "down"}
    assert len({p.name for p in PERTURBATIONS}) == len(PERTURBATIONS)


def test_pick_examples_uses_one_answer_per_prompt_and_another_prompts_answer():
    pairs = [PreferencePair(f"p{i}", f"good{i}", f"bad{i}") for i in range(5)]
    pairs.append(PreferencePair("p0", "second answer to p0", "bad"))
    examples = pick_examples(pairs, n=5)
    assert len({ex.prompt for ex in examples}) == 5
    for ex in examples:
        assert ex.response == f"good{ex.prompt[1:]}"
        assert ex.other_response != ex.response


def test_pick_examples_skips_answers_too_long_to_double():
    pairs = [PreferencePair("short", "ok", "bad"), PreferencePair("long", "x" * 4000, "bad")]
    assert [ex.prompt for ex in pick_examples(pairs, n=5, max_chars=6000)] == ["short"]


def test_results_record_how_many_answers_each_edit_changed():
    results = stress({"length": LengthRM()})
    changed = {p["name"]: p["changed"] for p in results["perturbations"]}
    assert changed["filler"] == 1.0
    assert changed["wrong_facts"] == 0.0  # the toy answers have no numbers or flippable words


def test_variants_line_up_with_the_examples():
    variants = build_variants([EXAMPLE, EXAMPLE])
    assert set(variants) == {ORIGINAL, *(p.name for p in PERTURBATIONS)}
    assert all(len(texts) == 2 for texts in variants.values())


def stress(scorers):
    examples = [
        Example(f"question number {i} about topic {i}", "An answer. " * (i + 1), f"Other answer {i}.")
        for i in range(6)
    ]
    return run_stress_test(examples, {name: (lambda rm=rm: rm) for name, rm in scorers.items()})


def by_name(results, name, scorer):
    return next(p for p in results["perturbations"] if p["name"] == name)["scores"][scorer]


def test_stress_test_flags_a_length_hacker():
    results = stress({"length": LengthRM()})
    assert by_name(results, "filler", "length")["delta_sigma"] > 0
    assert by_name(results, "repeat", "length")["frac_up"] == 1.0
    assert by_name(results, "truncate", "length")["delta_sigma"] < 0


def test_stress_test_flags_a_prompt_copier():
    results = stress({"overlap": PromptOverlapRM()})
    assert by_name(results, "restate", "overlap")["frac_up"] == 1.0
    assert by_name(results, "prompt_only", "overlap")["delta_sigma"] > 0


def test_a_constant_scorer_does_not_crash_the_job():
    class ConstantRM:
        def score(self, prompts, responses):
            return [1.0] * len(responses)

    results = stress({"constant": ConstantRM()})
    assert all(p["scores"]["constant"]["delta_sigma"] == 0 for p in results["perturbations"])


def test_results_feed_the_table_and_figure():
    results = stress({"bt": LengthRM(), "jepa": PromptOverlapRM()})
    assert results["n_examples"] == 6
    assert results["examples"][0]["prompt"].startswith("question number")
    assert len(table_rows(results)) == 2 * len(PERTURBATIONS)
    assert "filler" in format_table(results)
    assert plot_stress_test(results).startswith(b"\x89PNG")


@pytest.mark.parametrize("fmt, magic", [("png", b"\x89PNG"), ("pdf", b"%PDF")])
def test_figure_formats(fmt, magic):
    assert plot_stress_test(stress({"bt": LengthRM()}), fmt=fmt).startswith(magic)


def test_each_shift_has_a_ci_and_the_raw_scores_are_kept():
    results = stress({"length": LengthRM()})
    filler = by_name(results, "filler", "length")
    lo, hi = filler["ci_sigma"]
    assert lo <= filler["delta_sigma"] <= hi and lo > 0  # filler always adds length, so it always helps here
    assert set(results["scores"]["length"]) >= {"original", "filler"}
