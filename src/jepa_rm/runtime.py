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
