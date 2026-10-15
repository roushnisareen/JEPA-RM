"""Weights & Biases logging.

Each Modal job is one W&B run in the same project. Every function here is a no-op when no run
is active, so the library code can log freely while tests and local use need no W&B account.
"""
from __future__ import annotations

import io
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import wandb

PROJECT = os.environ.get("WANDB_PROJECT", "jepa-rm")


def active() -> bool:
    return wandb.run is not None


@contextmanager
def run(name: str, job_type: str, config: Mapping[str, Any]) -> Iterator[None]:
    wandb.init(project=PROJECT, name=name, job_type=job_type, config=dict(config))
    try:
        yield
    except BaseException:
        wandb.finish(exit_code=1)
        raise
    wandb.finish()


def define_axis(prefix: str, step_key: str) -> None:
    """Plot every `prefix/*` metric against `step_key`, so stages in one run don't share a step axis."""
    if active():
        wandb.define_metric(step_key)
        wandb.define_metric(f"{prefix}/*", step_metric=step_key)


def log(metrics: Mapping[str, Any]) -> None:
    if active():
        wandb.log(dict(metrics))


def summary(metrics: Mapping[str, Any]) -> None:
    if active():
        wandb.run.summary.update(dict(metrics))


def _cell(value: Any) -> Any:
    # W&B tables need one type per column; lists and dicts (e.g. confidence intervals) go in as text
    return value if value is None or isinstance(value, (bool, int, float, str)) else json.dumps(value)


def log_table(key: str, rows: Sequence[Mapping[str, Any]]) -> None:
    if active() and rows:
        columns = list(dict.fromkeys(c for r in rows for c in r))
        wandb.log({key: wandb.Table(columns=columns, data=[[_cell(r.get(c)) for c in columns] for r in rows])})


def log_image(key: str, png: bytes) -> None:
    if active():
        from PIL import Image

        wandb.log({key: wandb.Image(Image.open(io.BytesIO(png)))})
