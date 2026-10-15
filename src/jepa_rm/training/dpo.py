"""DPO baseline.

DPO learns straight from the preference pairs with no reward model, so there's nothing to hack.
It shows how far a policy drifts from the same data without a reward model in the loop.
"""
from __future__ import annotations

import logging
import random
from dataclasses import asdict
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from jepa_rm import tracking
from jepa_rm.artifacts import Store, checkpoint_dir, dpo_log_name, policy_dir
from jepa_rm.config import LORA, POLICIES, DpoRunConfig
from jepa_rm.data import load_pairs, user_message
from jepa_rm.runtime import model_dtype, on_gpu

log = logging.getLogger(__name__)
RUN_NAME = "dpo"


def train_dpo(store: Store, cfg: DpoRunConfig, work_dir: str | Path = "/tmp") -> list[dict]:
    out = policy_dir(cfg.policy, RUN_NAME, cfg.seed)
    if store.delete_dir(out):  # see train_grpo
        log.warning("deleted old checkpoints in %s before starting", out)
    pairs = load_pairs("train")
    random.Random(cfg.seed).shuffle(pairs)
    dataset = Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": user_message(p.prompt, cfg.policy)}],
            "chosen": [{"role": "assistant", "content": p.chosen}],
            "rejected": [{"role": "assistant", "content": p.rejected}],
        }
        for p in pairs[: cfg.max_pairs]
    ])

    model_name = POLICIES[cfg.policy]
    args = DPOConfig(
        output_dir=str(Path(work_dir) / out),
        learning_rate=cfg.lr,
        beta=cfg.beta,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=1,
        max_length=cfg.max_len,
        max_prompt_length=cfg.max_prompt_len,
        logging_steps=10,
        gradient_checkpointing=True,  # long pairs with a 151k-token vocabulary (DPO doesn't generate)
        gradient_checkpointing_kwargs={"use_reentrant": False},  # the variant that works with LoRA
        save_strategy="no",
        bf16=on_gpu(),
        use_cpu=not on_gpu(),  # lets the CPU integration test run on any laptop
        report_to="wandb" if tracking.active() else "none",
        seed=cfg.seed,
    )
    trainer = DPOTrainer(
        model=AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype()),
        args=args,
        train_dataset=dataset,
        processing_class=AutoTokenizer.from_pretrained(model_name),
        peft_config=LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear", **LORA),
    )
    trainer.train()

    final = Path(args.output_dir) / "final"
    trainer.save_model(str(final))
    remote = checkpoint_dir(cfg.policy, RUN_NAME, trainer.state.global_step, cfg.seed)
    store.upload_dir(final, remote, allow_patterns=["adapter_*"])

    history = trainer.state.log_history
    store.save_result(dpo_log_name(cfg.policy, cfg.seed), {"config": asdict(cfg), "log": history})
    return history
