"""The Modal jobs' command-line defaults override the config dataclasses, so they must match.

A best-of-n run once used 100 x 16 instead of the configured 64 x 64 because the two had drifted.
This reads modal_app.py's source, so it needs neither Modal credentials nor a GPU.
"""
import ast
from dataclasses import fields
from pathlib import Path

import pytest

from jepa_rm.config import BonConfig, DpoRunConfig, EvalConfig, GrpoRunConfig, StressConfig

APP = Path(__file__).resolve().parents[1] / "modal_app.py"

# (job, its parameter, config class, config field)
LINKED = [
    ("grpo", "max_steps", GrpoRunConfig, "max_steps"),
    ("grpo", "lr", GrpoRunConfig, "lr"),
    ("grpo", "seed", GrpoRunConfig, "seed"),
    ("grpo", "policy", GrpoRunConfig, "policy"),
    ("dpo", "max_pairs", DpoRunConfig, "max_pairs"),
    ("dpo", "seed", DpoRunConfig, "seed"),
    ("evaluate", "n_prompts", EvalConfig, "n_prompts"),
    ("stress_test", "n_examples", StressConfig, "n_examples"),
    ("stress_test", "with_gold", StressConfig, "with_gold"),
    ("best_of_n", "n_prompts", BonConfig, "n_prompts"),
    ("best_of_n", "n_samples", BonConfig, "n_samples"),
]


def job_defaults() -> dict[str, dict[str, object]]:
    tree = ast.parse(APP.read_text())
    defaults = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            args = node.args.args[-len(node.args.defaults):] if node.args.defaults else []
            defaults[node.name] = {a.arg: ast.literal_eval(d) for a, d in zip(args, node.args.defaults, strict=True)}
    return defaults


@pytest.mark.parametrize("job, param, config, field", LINKED, ids=[f"{j}.{p}" for j, p, _, _ in LINKED])
def test_job_default_matches_config(job, param, config, field):
    config_default = {f.name: f.default for f in fields(config)}[field]
    assert job_defaults()[job][param] == config_default
