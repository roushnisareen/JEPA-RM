"""Device selection and memory cleanup, so the same code runs on a Modal GPU and on a laptop CPU."""
from __future__ import annotations

import gc

import torch


def default_device() -> str:
    # MPS is skipped on purpose: the HF Trainer would also pick it, and bf16 support there is patchy
    return "cuda" if torch.cuda.is_available() else "cpu"


def on_gpu() -> bool:
    return torch.cuda.is_available()


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_dtype() -> torch.dtype:
    # bf16 halves memory on GPU; on CPU, float32 is faster and avoids precision surprises in tests
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def set_sampling(model, temperature: float, top_p: float, top_k: int | None) -> None:
    """Make `model.generate` sample with exactly these settings, whoever calls it.

    transformers >= 4.50 replaces every setting a caller leaves at its global default (temperature
    1.0, top-p 1.0, top-k 50) with the model's own default. TRL's GRPO asks for temperature 1.0 and
    so got Qwen3's 0.6 instead. Writing the settings into the model's own generation config makes
    the model's default and the request agree, so nothing gets swapped.
    """
    config = model.generation_config
    config.temperature, config.top_p, config.top_k = temperature, top_p, top_k
