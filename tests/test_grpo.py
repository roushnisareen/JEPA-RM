from types import SimpleNamespace

import torch

from jepa_rm.artifacts import LocalStore
from jepa_rm.config import GrpoRunConfig
from jepa_rm.data import NO_THINK
from jepa_rm.training.grpo import UploadCheckpoints, make_reward_fn


class EchoRM:
    """Records what it was asked to score."""

    def __init__(self):
        self.seen = []

    def score(self, prompts, responses):
        self.seen.append((list(prompts), list(responses)))
        return [float(len(r)) for r in responses]


def test_reward_fn_gives_the_rm_clean_prompts_and_answers():
    rm = EchoRM()
    reward = make_reward_fn(rm, "bt")
    prompts = [[{"role": "user", "content": "Why is the sky blue?" + NO_THINK}]] * 2
    completions = [[{"role": "assistant", "content": "<think>\n\n</think>\n\nRayleigh scattering."}],
                   [{"role": "assistant", "content": "Because."}]]
    assert reward(prompts=prompts, completions=completions, completion_ids=None) == [20.0, 8.0]
    assert rm.seen == [(["Why is the sky blue?"] * 2, ["Rayleigh scattering.", "Because."])]
    assert reward.__name__ == "reward_bt"  # TRL logs each reward function under its name


def test_each_checkpoint_upload_also_saves_the_log_so_far(tmp_path):
    checkpoint = tmp_path / "out" / "checkpoint-50"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}")
    (checkpoint / "adapter_model.safetensors").write_text("weights")
    (checkpoint / "optimizer.pt").write_text("not needed")
    store = LocalStore(tmp_path / "store")

    UploadCheckpoints(store, GrpoRunConfig(), "bt").on_save(
        SimpleNamespace(output_dir=str(tmp_path / "out")),
        SimpleNamespace(global_step=50, log_history=[{"step": 50, "kl": 0.1, "reward": 0.3}]),
        control=None,
    )

    assert store.checkpoints("qwen", "bt") == [(50, "policy_qwen_bt/checkpoint-50")]
    assert not (store.root / "policy_qwen_bt/checkpoint-50/optimizer.pt").exists()  # adapters only
    saved = store.load_result("grpo_qwen_bt.json")
    assert saved["complete"] is False
    assert saved["log"] == [{"step": 50, "kl": 0.1, "reward": 0.3}]


class FlakyStore(LocalStore):
    """Its first `failures` uploads fail, like a network outage during training."""

    def __init__(self, root, failures):
        super().__init__(root)
        self.failures = failures

    def upload_dir(self, *args, **kwargs):
        if self.failures:
            self.failures -= 1
            raise ConnectionError("network down")
        super().upload_dir(*args, **kwargs)


def test_a_failed_checkpoint_upload_does_not_stop_training_and_is_retried(tmp_path):
    checkpoint = tmp_path / "out" / "checkpoint-50"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}")
    store = FlakyStore(tmp_path / "store", failures=1)
    uploader = UploadCheckpoints(store, GrpoRunConfig(), "bt")

    uploader.on_save(SimpleNamespace(output_dir=str(tmp_path / "out")),
                     SimpleNamespace(global_step=50, log_history=[]), control=None)  # must not raise
    assert store.checkpoints("qwen", "bt") == []

    uploader.retry_failed()
    assert store.checkpoints("qwen", "bt") == [(50, "policy_qwen_bt/checkpoint-50")]


def test_sampling_runs_in_eval_mode_and_training_mode_is_restored(monkeypatch):
    """Eval mode is what keeps the KV cache on under gradient checkpointing (see the trainer's docstring)."""
    from trl import GRPOTrainer

    from jepa_rm.training.grpo import EvalModeSamplingGRPOTrainer

    seen = {}

    def fake_generate(self, inputs):
        seen["training_during_sampling"] = self.model.training
        return "batch"

    monkeypatch.setattr(GRPOTrainer, "_generate_and_score_completions", fake_generate)
    trainer = object.__new__(EvalModeSamplingGRPOTrainer)  # no model downloads: only the override is under test
    trainer.model = torch.nn.Linear(2, 2)
    trainer.model.train()

    assert trainer._generate_and_score_completions({}) == "batch"
    assert seen["training_during_sampling"] is False
    assert trainer.model.training is True


def test_a_resumed_run_gets_exactly_the_prompts_trl_had_not_reached():
    from dataclasses import replace

    from trl.trainer.grpo_trainer import RepeatSampler

    from jepa_rm.training.grpo import unseen_prompts

    cfg = replace(GrpoRunConfig(), resume_step=10)  # 4 prompts a step, as in the real runs
    prompts = [f"p{i}" for i in range(100)]
    per_step = cfg.batch_size * cfg.grad_accum // cfg.num_generations
    # the sampler TRL 0.17 builds in _get_train_sampler, one index per answer
    sampler = RepeatSampler(prompts, mini_repeat_count=cfg.num_generations, batch_size=per_step,
                            repeat_count=cfg.grad_accum, seed=cfg.seed)
    drawn = list(dict.fromkeys(sampler))[: cfg.resume_step * per_step]
    left = unseen_prompts(prompts, cfg, [10])
    assert len(left) == 100 - 40 and not {prompts[i] for i in drawn} & set(left)

    # the continuation was cut off too, after 5 more steps; it had shuffled its own 60 prompts
    sampler = RepeatSampler(left, mini_repeat_count=cfg.num_generations, batch_size=per_step,
                            repeat_count=cfg.grad_accum, seed=cfg.seed)
    drawn_again = {left[i] for i in list(dict.fromkeys(sampler))[: 5 * per_step]}
    left_again = unseen_prompts(prompts, cfg, [10, 15])
    assert len(left_again) == 100 - 60 and set(left_again) == set(left) - drawn_again


def test_a_resumed_run_numbers_its_checkpoints_and_log_from_where_it_stopped(tmp_path):
    from dataclasses import replace

    store = LocalStore(tmp_path / "store")
    cfg = replace(GrpoRunConfig(), resume_step=100)
    earlier = [{"step": 50, "reward": 0.1}, {"step": 100, "reward": 0.2}]
    cb = UploadCheckpoints(store, cfg, "jepa", earlier, [100])
    local = tmp_path / "out" / "checkpoint-50"
    local.mkdir(parents=True)
    (local / "adapter_config.json").write_text("{}")
    cb.on_save(SimpleNamespace(output_dir=str(tmp_path / "out")),
               SimpleNamespace(global_step=50, log_history=[{"step": 50, "reward": 0.3}]), None)
    assert store.checkpoints("qwen", "jepa") == [(150, "policy_qwen_jepa/checkpoint-150")]
    assert [e["step"] for e in store.load_result("grpo_qwen_jepa.json")["log"]] == [50, 100, 150]


def test_saves_between_checkpoints_go_to_resume_points_that_evaluation_never_sees(tmp_path):
    from dataclasses import replace

    store = LocalStore(tmp_path / "store")
    cfg = replace(GrpoRunConfig(), resume_step=100)  # 200 steps, a checkpoint every 50
    cb = UploadCheckpoints(store, cfg, "jepa", [{"step": 100, "reward": 0.2}], [100])
    local = tmp_path / "out" / "checkpoint-10"
    local.mkdir(parents=True)
    (local / "adapter_config.json").write_text("{}")
    cb.on_save(SimpleNamespace(output_dir=str(tmp_path / "out")),
               SimpleNamespace(global_step=10, log_history=[{"step": 10, "reward": 0.3}]), None)
    assert store.checkpoints("qwen", "jepa") == []
    assert store.checkpoints("qwen", "jepa_resume") == [(110, "policy_qwen_jepa_resume/checkpoint-110")]
    saved = store.load_result("grpo_qwen_jepa.json")
    assert saved["resumed_from"] == [100] and [e["step"] for e in saved["log"]] == [100, 110]
