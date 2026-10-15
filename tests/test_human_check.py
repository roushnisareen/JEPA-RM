import pytest

from jepa_rm.evaluation.human_check import agreement, make_blind_pairs, read_csv, write_csv


def gen(run, step, prompt, response, gold):
    return {"run": run, "step": step, "prompt": prompt, "response": response, "gold": gold}


GENERATIONS = [gen("base", 0, f"p{i}", f"base answer {i}", 1.0) for i in range(10)] + [
    gen("bt", 200, f"p{i}", f"trained answer {i}", 2.0 if i % 2 else 0.0) for i in range(10)
]


def test_blind_pairs_hide_which_answer_is_which():
    rows, key = make_blind_pairs(GENERATIONS, "bt", 200, n=10)
    assert len(rows) == 10
    assert {k["trained"] for k in key.values()} == {"A", "B"}  # both orders occur
    for row in rows:
        trained = row["answer_a"] if key[row["id"]]["trained"] == "A" else row["answer_b"]
        assert trained.startswith("trained") and row["better"] == ""


def test_blind_pairs_are_reproducible_and_reject_missing_checkpoints():
    assert make_blind_pairs(GENERATIONS, "bt", 200, n=5) == make_blind_pairs(GENERATIONS, "bt", 200, n=5)
    with pytest.raises(ValueError):
        make_blind_pairs(GENERATIONS, "jepa", 200, n=5)


def test_agreement_with_a_rater_who_always_picks_the_judges_choice():
    rows, key = make_blind_pairs(GENERATIONS, "bt", 200, n=10)
    for row in rows:
        row["better"] = "A" if key[row["id"]]["gold_a"] > key[row["id"]]["gold_b"] else "B"
    result = agreement(rows, key)
    assert result["human_judge_agreement"] == 1.0
    assert result["human_prefers_trained"] == 0.5  # the judge prefers the trained answer on half the prompts


def test_ties_are_counted_but_left_out_of_rates():
    rows, key = make_blind_pairs(GENERATIONS, "bt", 200, n=2)
    rows[0]["better"], rows[1]["better"] = "tie", key[rows[1]["id"]]["trained"]
    result = agreement(rows, key)
    assert (result["n_rated"], result["n_ties"], result["human_prefers_trained"]) == (2, 1, 1.0)


def test_unrated_sheet_is_rejected():
    rows, key = make_blind_pairs(GENERATIONS, "bt", 200, n=3)
    with pytest.raises(ValueError):
        agreement(rows, key)


def test_csv_round_trip(tmp_path):
    rows, _ = make_blind_pairs(GENERATIONS, "bt", 200, n=3)
    write_csv(tmp_path / "sheet.csv", rows)
    assert read_csv(tmp_path / "sheet.csv") == rows
