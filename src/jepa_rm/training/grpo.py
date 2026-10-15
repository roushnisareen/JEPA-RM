"""RLHF with GRPO against one reward model.

Checkpoints are uploaded as soon as they're saved, so evaluation can follow how the policy's own
reward and the gold judge's score drift apart over training.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from jepa_rm import tracking
from jepa_rm.artifacts import Store, checkpoint_dir, grpo_log_name, policy_dir, rm_dir
from jepa_rm.config import LORA, POLICIES, GrpoRunConfig
from jepa_rm.data import NO_THINK, load_prompts, strip_think, user_message
from jepa_rm.rewards import RewardModel, check_kind, load_reward_model
from jepa_rm.runtime import model_dtype, on_gpu

log = logging.getLogger(__name__)
ADAPTER_FILES = ["adapter_*"]


class EvalModeSamplingGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that samples answers with the model in eval mode.

    With gradient checkpointing on, transformers switches the KV cache off whenever the model is in
    training mode, so every new token re-reads the whole answer; a GPU check measured ~130 s per
    step that way. In eval mode the cache stays on, and LoRA dropout is off while sampling, which is
    what sampling wants anyway. Training passes still run in training mode, with checkpointing.
    """

    def _generate_and_score_completions(self, inputs):
        was_training = self.model.training
        self.model.eval()
        try:
            return super()._generate_and_score_completions(inputs)
        finally:
            if was_training:
                self.model.train()


def make_reward_fn(rm: RewardModel, name: str) -> Callable[..., list[float]]:
    """Adapt a reward model to TRL's reward function signature (conversational format)."""

    def reward(prompts, completions, **kwargs) -> list[float]:
        users = [p[-1]["content"].removesuffix(NO_THINK) for p in prompts]
        answers = [strip_think(c[0]["content"]) for c in completions]
        return rm.score(users, answers)

    reward.__name__ = f"reward_{name}"
    return reward


def save_log(store: Store, cfg: GrpoRunConfig, run: str, history: list[dict], complete: bool) -> None:
    store.save_result(grpo_log_name(cfg.policy, run, cfg.seed),
                      {"config": asdict(cfg), "rm": run, "complete": complete, "log": history})


class UploadCheckpoints(TrainerCallback):
    """Uploads each checkpoint with the training log so far, so a run cut off by the job's
    timeout still has its KL and reward curves for every checkpoint it saved.

    An upload that still fails after the store's own retries is logged and remembered rather than
    raised: crashing would throw away hours of training over one network error. `retry_failed`
    tries those again once training is over (checkpoints stay on disk until the job ends).
    """

    def __init__(self, store: Store, cfg: GrpoRunConfig, run: str):
        self.store, self.cfg, self.run = store, cfg, run
        self.failed: list[tuple[Path, str]] = []

    def on_save(self, args, state, control, **kwargs):
        local = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        remote = checkpoint_dir(self.cfg.policy, self.run, state.global_step, self.cfg.seed)
        try:
            self.store.upload_dir(local, remote, allow_patterns=ADAPTER_FILES)
            save_log(self.store, self.cfg, self.run, state.log_history, complete=False)
            log.info("uploaded %s", remote)
        except Exception:
            log.exception("could not upload %s; training continues and it will be retried at the end", remote)
            self.failed.append((local, remote))

    def retry_failed(self) -> None:
        """Upload anything that failed during training; raise if it still can't be uploaded."""
        for local, remote in self.failed:
            self.store.upload_dir(local, remote, allow_patterns=ADAPTER_FILES)
            log.info("uploaded %s on retry", remote)
        self.failed.clear()


def train_grpo(store: Store, rm_kind: str, cfg: GrpoRunConfig, work_dir: str | Path = "/tmp") -> list[dict]:
    rm = load_reward_model(check_kind(rm_kind), store.download_dir(rm_dir(rm_kind)))
    out = policy_dir(cfg.policy, rm_kind, cfg.seed)
    # checkpoints left by an earlier run (e.g. a quick smoke test) would otherwise be evaluated
    # alongside this run's
    if store.delete_dir(out):
        log.warning("deleted old checkpoints in %s before starting", out)
    prompts = load_prompts("train", cfg.n_prompts)
    dataset = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": user_message(p, cfg.policy)}]} for p in prompts]
    )

    args = GRPOConfig(
        output_dir=str(Path(work_dir) / out),
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_len,
        max_completion_length=cfg.max_completion_len,
        mask_truncated_completions=cfg.mask_truncated,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # the variant that works with LoRA
        temperature=cfg.temperature,
        beta=cfg.kl_beta,
        max_steps=cfg.max_steps,
        save_steps=max(1, cfg.max_steps // cfg.n_checkpoints),
        logging_steps=5,
        bf16=on_gpu(),
        use_cpu=not on_gpu(),  # lets the CPU integration test run on any laptop
        report_to="wandb" if tracking.active() else "none",
        log_completions=tracking.active(),  # sample answers in W&B, handy for spotting hacking
        seed=cfg.seed,
        model_init_kwargs={"torch_dtype": str(model_dtype()).removeprefix("torch.")},
    )
    trainer = EvalModeSamplingGRPOTrainer(
        model=POLICIES[cfg.policy],
        reward_funcs=make_reward_fn(rm, rm_kind),
        args=args,
        train_dataset=dataset,
        peft_config=LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear", **LORA),
        callbacks=[uploader := UploadCheckpoints(store, cfg, rm_kind)],
    )
    trainer.train()
    uploader.retry_failed()

    step = trainer.state.global_step
    if step % args.save_steps:  # the trainer only saves on multiples of save_steps
        final = Path(args.output_dir) / "final"
        trainer.save_model(str(final))
        store.upload_dir(final, checkpoint_dir(cfg.policy, rm_kind, step, cfg.seed), allow_patterns=ADAPTER_FILES)

    history = trainer.state.log_history
    save_log(store, cfg, rm_kind, history, complete=True)
    return history
