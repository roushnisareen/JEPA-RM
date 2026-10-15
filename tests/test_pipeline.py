"""Every stage, end to end, on tiny models on CPU.

This is the test that matters most before spending GPU money: it exercises the real TRL, PEFT and
transformers code paths (trainer arguments, reward-function signatures, checkpoint saving, adapter
loading, generation) that the unit tests can't. The tiny models are random, so the numbers mean
nothing; the point is that nothing crashes and every artifact lands where the next stage expects it.

Needs network once to download the tiny models (~20 MB each). Skip with `pytest -m "not integration"`.
"""
from dataclasses import replace

import pytest
import torch
from safetensors.torch import load_file

from jepa_rm.artifacts import LocalStore, MissingArtifactError, checkpoint_dir, rm_dir
from jepa_rm.config import POLICIES, BonConfig, BTConfig, DpoRunConfig, EvalConfig, GrpoRunConfig, JepaConfig
from jepa_rm.data import PreferencePair
from jepa_rm.evaluation import best_of_n, policies
from jepa_rm.evaluation.plots import plot_stress_test
from jepa_rm.evaluation.policies import GoldJudge, evaluate_policies
from jepa_rm.evaluation.reward_models import compare_reward_models
from jepa_rm.evaluation.stress_test import pick_examples, run_stress_test
from jepa_rm.rewards import REWARD_KINDS, load_reward_model
from jepa_rm.rewards.bradley_terry import train_bradley_terry
from jepa_rm.rewards.suite import train_reward_models
from jepa_rm.training import dpo, grpo

pytestmark = pytest.mark.integration

TINY_LM = "trl-internal-testing/tiny-Qwen3ForCausalLM"
TINY_JUDGE = "trl-internal-testing/tiny-Qwen3ForSequenceClassification"
TINY_PHI = "trl-internal-testing/tiny-Phi3ForCausalLM"  # Phi-4-mini uses the Phi-3 architecture

PAIRS = [
    PreferencePair(
        f"Question {i}: what is {i} plus {i}?",
        f"The answer is {2 * i}. Adding {i} to itself gives {2 * i}.",
        f"Maybe {i}.",
    )
    for i in range(24)
]
PROMPTS = [p.prompt for p in PAIRS]


@pytest.fixture
def tiny(monkeypatch):
    """Point every stage at the tiny models and the toy data instead of HelpSteer2."""
    monkeypatch.setitem(POLICIES, "qwen", TINY_LM)
    monkeypatch.setitem(POLICIES, "phi", TINY_PHI)
    monkeypatch.setattr(grpo, "load_prompts", lambda split, n: PROMPTS[:n])
    monkeypatch.setattr(dpo, "load_pairs", lambda split: list(PAIRS))
    monkeypatch.setattr(policies, "load_prompts", lambda split, n: PROMPTS[:n])
    monkeypatch.setattr(best_of_n, "load_prompts", lambda split, n: PROMPTS[:n])


GRPO_CFG = GrpoRunConfig(max_steps=2, n_prompts=8, batch_size=4, grad_accum=1, num_generations=4,
                         max_prompt_len=64, max_completion_len=16, n_checkpoints=1,
                         mask_truncated=False)  # a random tiny model never stops, so nothing would be left


def test_running_stages_out_of_order_says_what_to_run_first(tiny, tmp_path):
    with pytest.raises(MissingArtifactError, match="reward_models"):
        grpo.train_grpo(LocalStore(tmp_path), "bt", GRPO_CFG, work_dir=tmp_path)


def test_full_pipeline_runs_on_cpu(tiny, tmp_path):
    store = LocalStore(tmp_path / "store")
    work = tmp_path / "work"

    # 1. every reward model kind, via the same function the GPU job uses, saved to the store and
    # reloaded from it the way later jobs do
    jepa_cfg = JepaConfig(encoder=TINY_LM, embed_layer=-1, max_len=128, predictor_hidden=32, epochs=2,
                          batch_size=8, holdout=0.25)
    models = train_reward_models(PAIRS, work / "rms", BTConfig(base_model=TINY_LM, batch_size=4, max_len=128),
                                 jepa_cfg, device="cpu")
    assert set(models) == set(REWARD_KINDS)
    for kind in REWARD_KINDS:
        store.upload_dir(work / "rms" / rm_dir(kind), rm_dir(kind))
        reloaded = load_reward_model(kind, store.download_dir(rm_dir(kind)))
        assert len(reloaded.score(PROMPTS[:2], ["a short answer", "another one"])) == 2

    comparison = compare_reward_models(PAIRS[:6], models)
    assert set(comparison["models"]) == set(REWARD_KINDS)
    assert "accuracy_vs_bt_ci" in comparison["models"]["jepa"]

    # 2. stress test, including the gold judge
    scorers = {kind: (lambda rm=rm: rm) for kind, rm in models.items()}
    stress = run_stress_test(pick_examples(PAIRS, 4), {**scorers, "gold": lambda: GoldJudge(TINY_JUDGE, device="cpu")})
    assert set(stress["perturbations"][0]["scores"]) == {*REWARD_KINDS, "gold"}
    assert plot_stress_test(stress).startswith(b"\x89PNG")

    # best-of-n: sample from the base model, let each reward model pick, score with the judge
    bon_rows = best_of_n.run_best_of_n(store, BonConfig(n_prompts=2, n_samples=4, ns=(2, 4), batch_size=8,
                                                       max_new_tokens=8, gold_rm=TINY_JUDGE))
    assert {(r["run"], r["step"]) for r in bon_rows} == {("base", 1)} | {(k, n) for k in REWARD_KINDS for n in (2, 4)}
    for figure in ("best_of_n_qwen.png", "best_of_n_ablation_qwen.png"):
        assert (store.root / "results" / figure).exists(), f"missing {figure}"

    # 3. GRPO against each reward model. A stale checkpoint from an "earlier smoke test" is planted
    # first; it must be gone afterwards, or evaluation would mix it into this run's results.
    stale = work / "stale"
    stale.mkdir(parents=True)
    (stale / "adapter_config.json").write_text("{}")
    store.upload_dir(stale, checkpoint_dir("qwen", "bt", 999))
    for rm in ("bt", "jepa"):
        history = grpo.train_grpo(store, rm, GRPO_CFG, work_dir=work)
        assert any("reward" in entry for entry in history)
        assert any("kl" in entry for entry in history), "kl_beta > 0 should make TRL log KL"
        assert any("completions/clipped_ratio" in entry for entry in history)
        assert store.checkpoints("qwen", rm) == [(2, f"policy_qwen_{rm}/checkpoint-2")]
    # a second seed sits next to the first instead of overwriting it
    grpo.train_grpo(store, "bt", replace(GRPO_CFG, seed=1), work_dir=work)
    assert store.checkpoints("qwen", "bt", seed=1) == [(2, "policy_qwen_bt_s1/checkpoint-2")]
    assert store.checkpoints("qwen", "bt") == [(2, "policy_qwen_bt/checkpoint-2")]

    # 4. DPO baseline
    dpo.train_dpo(store, DpoRunConfig(max_pairs=8, batch_size=2, grad_accum=1, max_len=128, max_prompt_len=64),
                  work_dir=work)
    assert store.checkpoints("qwen", "dpo")

    # 5. evaluation of every checkpoint, with KL, CIs and every figure
    eval_cfg = EvalConfig(runs=("bt", "jepa", "dpo"), seeds=(0, 1), n_prompts=3, batch_size=3, max_new_tokens=8,
                          gold_rm=TINY_JUDGE)
    rows = evaluate_policies(store, eval_cfg)
    assert {r["run"] for r in rows} == {"base", "bt", "jepa", "dpo"}
    seeds = {r["run"]: r["n_seeds"] for r in rows}
    assert (seeds["bt"], seeds["jepa"]) == (2, 1)  # jepa has no seed 1, which is skipped, not an error
    # KL is measured during evaluation, so every run has it, DPO included; base is 0 by definition
    assert all(r["kl"] is not None and r["kl"] >= 0 for r in rows)
    assert next(r for r in rows if r["run"] == "base")["kl"] == 0.0
    assert all(r["gold_ci"] for r in rows if r["run"] != "base")
    for name in ("eval_qwen.json", "grpo_qwen_bt.json", "grpo_qwen_jepa.json", "dpo_qwen.json"):
        assert store.load_result(name), f"missing {name}"
    assert store.load_result("grpo_qwen_bt.json")["complete"] is True
    for figure in ("reward_hacking_qwen.png", "reward_hacking_kl_qwen.png", "training_qwen.png"):
        assert (store.root / "results" / figure).exists(), f"missing {figure}"


def test_phi_policy_trains_and_generates(tiny, tmp_path):
    """Phi-4-mini has its own chat template and layer names; check both work end to end."""
    store = LocalStore(tmp_path / "store")
    train_bradley_terry(PAIRS, tmp_path / "rm_bt", BTConfig(base_model=TINY_LM, batch_size=4, max_len=128), "cpu")
    store.upload_dir(tmp_path / "rm_bt", rm_dir("bt"))
    # LoRA's B matrices start at exactly zero: non-zero after training means gradients got through
    # gradient checkpointing (a known way for LoRA training to run but silently learn nothing)
    weights = load_file(tmp_path / "rm_bt" / "adapter_model.safetensors")
    lora_b = [w for name, w in weights.items() if "lora_B" in name]
    assert lora_b and any(w.abs().sum() > 0 for w in lora_b)

    grpo.train_grpo(store, "bt", replace(GRPO_CFG, policy="phi"), work_dir=tmp_path / "work")
    [(step, folder)] = store.checkpoints("phi", "bt")
    answers = policies.generate(PROMPTS[:2], store.download_dir(folder),
                                EvalConfig(policy="phi", batch_size=2, max_new_tokens=8), device="cpu")
    assert len(answers) == 2 and all(isinstance(a.response, str) and a.kl >= 0 for a in answers)


def test_pooling_ablation_reuses_bt_rescores_the_same_answers_and_compares(tiny, tmp_path):
    from jepa_rm.evaluation.plots import plot_pooling
    from jepa_rm.evaluation.pooling import compare_pooling
    from jepa_rm.rewards import EMBEDDING_KINDS

    work = tmp_path / "work"
    stores = {"mean": LocalStore(tmp_path / "mean"), "last": LocalStore(tmp_path / "last")}
    bt_cfg = BTConfig(base_model=TINY_LM, batch_size=4, max_len=128)
    jepa_cfg = JepaConfig(encoder=TINY_LM, embed_layer=-1, max_len=128, predictor_hidden=32, epochs=2,
                          batch_size=8, holdout=0.25)

    # mean arm: everything trained, then best-of-n
    mean_models = train_reward_models(PAIRS, work / "mean", bt_cfg, jepa_cfg, device="cpu")
    # last arm: BT copied from the mean arm, only the embedding-space kinds retrained
    last_models = train_reward_models(PAIRS, work / "last", bt_cfg, replace(jepa_cfg, pooling="last"), device="cpu",
                                      bt_from=work / "mean")
    for pooling, models in (("mean", mean_models), ("last", last_models)):
        for kind in REWARD_KINDS:
            stores[pooling].upload_dir(work / pooling / rm_dir(kind), rm_dir(kind))
        stores[pooling].save_result("reward_models.json", compare_reward_models(PAIRS[:6], models))
    bt_weights = "rm_bt/adapter_model.safetensors"
    assert (stores["mean"].root / bt_weights).read_bytes() == (stores["last"].root / bt_weights).read_bytes()
    assert torch.load(stores["last"].root / "rm_jepa/jepa.pt")["config"]["pooling"] == "last"

    bon_cfg = BonConfig(n_prompts=2, n_samples=4, ns=(2, 4), batch_size=8, max_new_tokens=8, gold_rm=TINY_JUDGE)
    best_of_n.run_best_of_n(stores["mean"], bon_cfg)
    best_of_n.run_best_of_n(stores["last"], bon_cfg, rescore_from=stores["mean"])

    mean_bon, last_bon = (stores[p].load_result("best_of_n_qwen.json") for p in ("mean", "last"))
    assert [s["response"] for s in mean_bon["samples"]] == [s["response"] for s in last_bon["samples"]]
    assert [s["gold"] for s in mean_bon["samples"]] == [s["gold"] for s in last_bon["samples"]]

    arms = {p: {"reward_models": stores[p].load_result("reward_models.json"), "stress": {},
                "bon": stores[p].load_result("best_of_n_qwen.json")} for p in stores}
    result = compare_pooling(arms)
    assert {r["kind"] for r in result["paired_best_of_n"]} == set(EMBEDDING_KINDS)
    assert {r["pooling"] for r in result["accuracy"]} == {"mean", "last"}
    assert plot_pooling({p: a["bon"] for p, a in arms.items()}).startswith(b"\x89PNG")
