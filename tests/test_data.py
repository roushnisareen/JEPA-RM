from jepa_rm.data import NO_THINK, PreferencePair, make_pairs, select_prompts, strip_think, user_message


def comparison(prompt, first, second, strength):
    return {"prompt": prompt, "response_1": first, "response_2": second, "preference_strength": strength}


def row(prompt):
    return {"prompt": prompt}


def test_negative_strength_means_the_first_response_is_better():
    # the convention the annotators' statements use: -3 = "@Response 1 is much better"
    assert make_pairs([comparison("p", "good", "bad", -3)]) == [PreferencePair("p", "good", "bad", 3)]


def test_positive_strength_means_the_second_response_is_better():
    assert make_pairs([comparison("p", "bad", "good", 1)]) == [PreferencePair("p", "good", "bad", 1)]


def test_pairs_skip_ties():
    assert make_pairs([comparison("p", "a", "b", 0)]) == []


def test_pairs_skip_multi_turn_prompts():
    prompt = "hi<extra_id_1>Assistant\nhello<extra_id_1>User\nmore"
    assert make_pairs([comparison(prompt, "a", "b", -2)]) == []


def test_pairs_skip_overlong_examples():
    assert make_pairs([comparison("p", "a" * 50, "b", -1)], max_chars=10) == []


def test_select_prompts_dedupes_filters_and_is_deterministic():
    rows = [row("a"), row("a"), row("b"), row("x" * 20), row("c<extra_id_1>")]
    first = select_prompts(rows, n=10, max_chars=10, seed=1)
    assert sorted(first) == ["a", "b"]
    assert select_prompts(rows, n=10, max_chars=10, seed=1) == first


def test_select_prompts_respects_n():
    rows = [row(str(i)) for i in range(10)]
    assert len(select_prompts(rows, n=3)) == 3


def test_strip_think_removes_thinking_blocks():
    assert strip_think("<think>\nhmm\n</think>\n\nanswer") == "answer"
    assert strip_think("no thinking here") == "no thinking here"


def test_user_message_only_disables_thinking_for_qwen():
    assert user_message("hi", "qwen") == "hi" + NO_THINK
    assert user_message("hi", "phi") == "hi"
