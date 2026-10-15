"""Standard reward model: an LLM with a LoRA adapter and a scalar head, trained with the
Bradley-Terry loss on preference pairs."""
from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from jepa_rm import tracking
from jepa_rm.config import LORA, BTConfig
from jepa_rm.data import PreferencePair, format_text
from jepa_rm.runtime import default_device, free_memory, model_dtype

log = logging.getLogger(__name__)
CONFIG_FILE = "bt_config.json"


def bt_loss(chosen: torch.Tensor, rejected: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(chosen - rejected).mean()


def _tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "right"
    return tok


class BradleyTerryRM:
    def __init__(self, tokenizer, model, max_len: int):
        self.tokenizer = tokenizer
        self.model = model
        self.max_len = max_len

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> BradleyTerryRM:
        path = Path(path)
        cfg = BTConfig(**json.loads((path / CONFIG_FILE).read_text()))
        tok = _tokenizer(cfg.base_model)
        base = AutoModelForSequenceClassification.from_pretrained(
            cfg.base_model, num_labels=1, torch_dtype=model_dtype()
        )
        base.config.pad_token_id = tok.pad_token_id
        model = PeftModel.from_pretrained(base, str(path)).merge_and_unload()
        return cls(tok, model.to(device or default_device()).eval(), cfg.max_len)

    @torch.no_grad()
    def score(self, prompts: Sequence[str], responses: Sequence[str], batch_size: int = 16) -> list[float]:
        all_texts = [format_text(p, r) for p, r in zip(prompts, responses, strict=True)]
        scores: list[float] = []
        for i in range(0, len(all_texts), batch_size):
            texts = all_texts[i : i + batch_size]
            enc = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len
            ).to(self.model.device)
            scores += self.model(**enc).logits[:, 0].float().tolist()
        return scores


def train_bradley_terry(
    pairs: Sequence[PreferencePair], out_dir: str | Path, cfg: BTConfig, device: str | None = None
) -> BradleyTerryRM:
    out_dir = Path(out_dir)
    device = device or default_device()
    steps = len(pairs) // cfg.batch_size
    if steps == 0:
        raise ValueError(f"need at least {cfg.batch_size} pairs, got {len(pairs)}")

    tok = _tokenizer(cfg.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.base_model, num_labels=1, torch_dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", target_modules=list(cfg.lora_targets), **LORA))
    if cfg.gradient_checkpointing:
        # a batch is 2 x batch_size texts of up to max_len tokens; without this, a batch of long pairs
        # ran out of memory on an 80 GB A100 in the first full run
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device).train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)

    bs = cfg.batch_size
    for step in range(steps):
        batch = pairs[step * bs : (step + 1) * bs]
        texts = [format_text(p.prompt, p.chosen) for p in batch] + [format_text(p.prompt, p.rejected) for p in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_len).to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            rewards = model(**enc).logits[:, 0].float()
        loss = bt_loss(rewards[:bs], rewards[bs:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        if (step + 1) % 10 == 0:
            tracking.log({"bt/step": step + 1, "bt/loss": loss.item(), "bt/lr": sched.get_last_lr()[0]})
        if (step + 1) % 50 == 0:
            log.info("step %d/%d  loss %.3f", step + 1, steps, loss.item())

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / CONFIG_FILE).write_text(json.dumps(asdict(cfg), indent=2))
    del model
    free_memory()
    return BradleyTerryRM.load(out_dir, device)


LENGTH_PENALTY_FILE = "length_penalty.json"


def length_slope(scores: Sequence[float], lengths: Sequence[float]) -> float:
    """Least-squares slope of score on length: how much the reward model pays per extra word."""
    n = len(scores)
    mean_s, mean_l = sum(scores) / n, sum(lengths) / n
    var = sum((x - mean_l) ** 2 for x in lengths)
    if var == 0:
        return 0.0
    return sum((x - mean_l) * (y - mean_s) for x, y in zip(lengths, scores, strict=True)) / var


def words(text: str) -> int:
    return len(text.split())


class LengthPenalizedRM:
    """The standard reward model with its linear length trend removed: score - slope * words.

    This is the cheap, well-known fix for length hacking. It's here as a baseline: if the JEPA
    reward model can't beat a one-line length penalty, that's worth knowing.
    """

    def __init__(self, bt: BradleyTerryRM, slope: float):
        self.bt = bt
        self.slope = slope

    def score(self, prompts: Sequence[str], responses: Sequence[str]) -> list[float]:
        raw = self.bt.score(prompts, responses)
        return [s - self.slope * words(r) for s, r in zip(raw, responses, strict=True)]

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> LengthPenalizedRM:
        slope = json.loads((Path(path) / LENGTH_PENALTY_FILE).read_text())["slope_per_word"]
        return cls(BradleyTerryRM.load(path, device), slope)


def make_length_penalized(
    bt: BradleyTerryRM, bt_dir: str | Path, out_dir: str | Path, pairs: Sequence[PreferencePair]
) -> LengthPenalizedRM:
    """Fit the slope on training responses (never the held-out ones it's evaluated on) and save a
    self-contained copy: the BT adapter plus the slope."""
    prompts = [p.prompt for p in pairs for _ in range(2)]
    responses = [r for p in pairs for r in (p.chosen, p.rejected)]
    slope = length_slope(bt.score(prompts, responses), [words(r) for r in responses])
    log.info("BT reward model pays %.4f per word; subtracting it", slope)
    shutil.copytree(bt_dir, out_dir, dirs_exist_ok=True)
    (Path(out_dir) / LENGTH_PENALTY_FILE).write_text(
        json.dumps({"slope_per_word": slope, "fit_on_responses": len(responses)}, indent=2)
    )
    return LengthPenalizedRM(bt, slope)
