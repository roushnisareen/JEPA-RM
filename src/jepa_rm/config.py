"""Model names and hyperparameters for every stage.

Each run saves its config next to its results, so a number in the results can always be traced
back to the settings that produced it.
"""
from __future__ import annotations

from dataclasses import dataclass

DATASET = "nvidia/HelpSteer2"
SEED = 0
POLICIES = {
    "qwen": "Qwen/Qwen3-1.7B",
    "phi": "microsoft/Phi-4-mini-instruct",
}
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}
GOLD_RM = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"  # judge for eval only, never trained against


@dataclass(frozen=True)
class BTConfig:
    """Bradley-Terry reward model."""

    base_model: str = "Qwen/Qwen3-0.6B"
    # long enough that nothing is ever cut off: the longest training text is 2,896 tokens and the
    # longest stress-test edit 1,570 (measured). Batches pad to their longest text, so the few long
    # examples are the only ones that pay for it.
    max_len: int = 3072
    batch_size: int = 8
    gradient_checkpointing: bool = True  # see train_bradley_terry
    lr: float = 1e-4
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class JepaConfig:
    """JEPA reward model."""

    encoder: str = "Qwen/Qwen3-0.6B"
    embed_layer: int = -8  # mid-to-late layers carry more meaning than the last, next-token one
    # "mean" averages every token, which makes the reward blind to repetition by construction;
    # "last" uses the final token, which sees the whole sequence. Run both as an ablation.
    pooling: str = "mean"
    max_len: int = 3072  # see BTConfig.max_len
    predictor_hidden: int = 2048
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.01
    tau: float = 0.05  # temperature of the ranking loss
    # weights of the two loss terms (see rewards/jepa.py); zeroing one gives the ablations
    jepa_weight: float = 1.0
    rank_weight: float = 1.0
    holdout: float = 0.05  # share of training pairs used to pick the best epoch


# Sampling, for every place answers are generated: temperature 1.0, top-p 1.0, top-k 20.
# Temperature 1.0 so GRPO, best-of-n and the KL estimate all work with the policy's own
# distribution. Top-k 20 keeps 1,536-token answers from wandering into junk tokens; it's also
# Qwen3's own default, and what the 6,400 best-of-n answers were in fact sampled with.
# These have to reach the model explicitly (runtime.set_sampling): transformers >= 4.50 swaps any
# setting a caller leaves at its global default (temperature 1.0, top-p 1.0) for the model's own,
# which for Qwen3 is temperature 0.6 and top-p 0.95.


@dataclass(frozen=True)
class GrpoRunConfig:
    policy: str = "qwen"
    max_steps: int = 200
    n_prompts: int = 2000
    # the first pilot at 1e-5 left the policy where it started (KL ~0.001 after 35 steps); 1e-4 is the
    # one planned increase from the pre-registration
    lr: float = 1e-4
    # TRL 0.17 generates all batch_size * grad_accum answers at once, then trains on them in
    # grad_accum slices, so each update sees 4 prompts x 8 answers. Slices of 8 keep a training pass
    # at ~9 GB of logits even with 1,536-token answers.
    batch_size: int = 8
    grad_accum: int = 4
    num_generations: int = 8
    max_prompt_len: int = 384
    # Qwen3-1.7B writes long, formatted answers: at 768 tokens a GPU check found 31% of them cut off,
    # which would cap the very length hacking being measured. Cut-off answers are still scored by the
    # reward model (TRL scores every answer before masking), but with mask_truncated they contribute
    # no gradient, so the policy isn't pushed to ramble past the limit. The share that get cut is
    # logged as clipped_ratio.
    max_completion_len: int = 1536
    mask_truncated: bool = True
    # on: without it, backprop through 8 answers of up to 1,536 tokens (and their 151k-way logits)
    # ran out of memory on an 80 GB A100 in a GPU check. The KV cache that checkpointing would
    # otherwise switch off during sampling is kept by sampling in eval mode (training/grpo.py).
    gradient_checkpointing: bool = True
    # the same sampling everywhere (GRPO, best-of-n, evaluation); see SAMPLING below
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 20
    # far too small to hold the policy back, but nonzero so TRL measures KL to the base model,
    # which is the x-axis for the over-optimization plot (Gao et al., 2022)
    kl_beta: float = 1e-3
    n_checkpoints: int = 4
    seed: int = 0  # each seed saves to its own folder, so runs with different seeds sit side by side
    # continue a run that stopped after saving this checkpoint (see training/grpo.py); 0 starts fresh.
    # If a resumed run was itself cut off (Modal restarts a preempted job with the same arguments),
    # it carries on from the latest resume point instead.
    resume_step: int = 0
    # extra adapters saved this often, only to restart from; kept apart from the checkpoints, so
    # evaluation never sees them. A preemption then costs at most this many steps.
    resume_every: int = 10


@dataclass(frozen=True)
class DpoRunConfig:
    policy: str = "qwen"
    max_pairs: int = 3000
    lr: float = 2e-5
    beta: float = 0.1
    batch_size: int = 4
    grad_accum: int = 4
    max_len: int = 2048  # all but a handful of pairs fit (see data.make_pairs)
    max_prompt_len: int = 1024
    seed: int = 0


@dataclass(frozen=True)
class EvalConfig:
    policy: str = "qwen"
    runs: tuple[str, ...] = ("bt", "jepa")
    seeds: tuple[int, ...] = (0,)  # every seed of every run is evaluated; missing ones are skipped
    n_prompts: int = 200
    batch_size: int = 32
    max_new_tokens: int = 1536  # same as GrpoRunConfig.max_completion_len
    # the same sampling as GRPO: KL to the base model is estimated from these answers, which is only
    # right when they come from the distribution being trained
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 20
    gold_rm: str = GOLD_RM


@dataclass(frozen=True)
class BonConfig:
    """Best-of-n: sample n_samples answers per prompt from the base model, then for each n in `ns`
    keep the one each reward model likes best."""

    policy: str = "qwen"
    n_prompts: int = 64
    # up to n = 64 (KL ≈ 3.2 nats): n = 16 only reaches ≈ 1.8, short of where over-optimization
    # typically sets in for best-of-n (Gao et al., 2022)
    n_samples: int = 64
    ns: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
    # same sampling as GRPO, so both methods search the same distribution
    temperature: float = 1.0
    top_k: int = 20
    max_new_tokens: int = 1536  # same as GrpoRunConfig.max_completion_len
    batch_size: int = 64
    gold_rm: str = GOLD_RM


@dataclass(frozen=True)
class StressConfig:
    n_examples: int = 200
    with_gold: bool = True  # also stress-test the judge, since the whole eval relies on it
    gold_rm: str = GOLD_RM
