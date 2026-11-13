"""Sampling settings reach the model as configured.

transformers >= 4.50 swaps any generation setting a caller leaves at its global default for the
model's own default. A GRPO pilot asked for temperature 1.0 and trained at Qwen3's 0.6 because of
it. These tests build a tiny Qwen3 locally (no download) with Qwen3's real sampling defaults.
"""
import pytest
from peft import LoraConfig, get_peft_model
from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM

from jepa_rm.config import BonConfig, EvalConfig, GrpoRunConfig
from jepa_rm.runtime import set_sampling


def tiny_qwen3():
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                         num_attention_heads=2, num_key_value_heads=1, head_dim=8))
    # what Qwen/Qwen3-1.7B ships with
    model.generation_config = GenerationConfig(do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                                               transformers_version="4.51.3")
    return model


def sampling_used(model, requested: GenerationConfig) -> tuple:
    """What generate() would actually sample with, given the config a caller (here: TRL) passes."""
    config, _ = model._prepare_generation_config(requested)
    return config.temperature, config.top_p, config.top_k


def trl_request(cfg: GrpoRunConfig) -> GenerationConfig:
    # how TRL 0.17's GRPOTrainer builds its generation config
    return GenerationConfig(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k)


def test_without_the_fix_grpo_would_sample_at_the_models_own_temperature():
    cfg = GrpoRunConfig()
    assert sampling_used(tiny_qwen3(), trl_request(cfg)) == (0.6, 0.95, 20)  # the trap this guards against


def test_grpo_samples_with_the_configured_settings_through_the_lora_wrapper():
    cfg = GrpoRunConfig()
    model = tiny_qwen3()
    policy = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear"))
    set_sampling(policy, cfg.temperature, cfg.top_p, cfg.top_k)  # as train_grpo does on trainer.model
    assert sampling_used(model, trl_request(cfg)) == (cfg.temperature, cfg.top_p, cfg.top_k)


@pytest.mark.parametrize("cfg", [EvalConfig(), BonConfig()], ids=["evaluate", "best_of_n"])
def test_every_place_that_samples_uses_the_same_settings_as_grpo(cfg):
    grpo = GrpoRunConfig()
    assert (cfg.temperature, cfg.top_k) == (grpo.temperature, grpo.top_k)
    assert getattr(cfg, "top_p", 1.0) == grpo.top_p
