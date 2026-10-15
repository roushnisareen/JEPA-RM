"""Policy evaluation: generate from the base model and every checkpoint, then score each answer
with the gold judge and with both reward models.

A run is reward hacking when its own reward model's score goes up and the gold judge's doesn't.
"""
from __future__ import annotations

import logging
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from jepa_rm import tracking
from jepa_rm.artifacts import Store, grpo_log_name, rm_dir
from jepa_rm.config import POLICIES, SEED, EvalConfig
from jepa_rm.data import load_prompts, strip_think, user_message
from jepa_rm.evaluation.plots import plot_reward_hacking, plot_training_curves
from jepa_rm.evaluation.stats import bootstrap_ci, format_comparisons, paired_gap
from jepa_rm.rewards import COMPARISONS, REWARD_KINDS, load_reward_model
from jepa_rm.runtime import default_device, free_memory, model_dtype

log = logging.getLogger(__name__)
SCORES = ("gold", *REWARD_KINDS)
EXTRAS = ("words", "truncated")


def score_keys(record: Mapping) -> tuple[str, ...]:
    """The scores a record actually has: results from older runs, or with fewer reward model kinds,
    keep working."""
    return tuple(k for k in SCORES if k in record)


class GoldJudge:
    def __init__(self, name: str, device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name, torch_dtype=model_dtype(), num_labels=1
        ).to(device or default_device()).eval()

    @torch.no_grad()
    def score(self, prompts: Sequence[str], responses: Sequence[str]) -> list[float]:
        scores = []
        for prompt, response in zip(prompts, responses, strict=True):
            conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
            ids = self.tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(self.model.device)
            scores.append(self.model(ids).logits[0][0].item())
        return scores


def hit_length_limit(token_ids: Sequence[int], eos_ids: set[int]) -> bool:
    """True if generation stopped at max_new_tokens instead of ending on its own."""
    return not any(t in eos_ids for t in token_ids)


@dataclass(frozen=True)
class Generated:
    response: str
    truncated: bool
    kl: float  # per-token KL from this answer's policy to the base model (0 for the base model itself)


def completion_mask(token_ids: torch.Tensor, eos_ids: set[int]) -> torch.Tensor:
    """1 for every generated token up to and including the first end-of-sequence token, 0 after it
    (padding). An answer with no end token was cut off, and all of it counts."""
    is_eos = torch.zeros_like(token_ids, dtype=torch.bool)
    for e in eos_ids:
        is_eos |= token_ids == e
    length = token_ids.shape[1]
    first = torch.where(is_eos.any(1), is_eos.int().argmax(1), torch.full_like(is_eos[:, 0], length, dtype=torch.long))
    return (torch.arange(length, device=token_ids.device) <= first.unsqueeze(1)).long()


def kl_per_token(policy_logps: torch.Tensor, base_logps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-answer mean of the k3 estimator of KL(policy || base) over its tokens: unbiased for
    answers sampled from the policy, never negative, and lower-variance than the plain log ratio."""
    r = base_logps - policy_logps
    k3 = torch.exp(r) - r - 1
    return (k3 * mask).sum(1) / mask.sum(1).clamp(min=1)


@torch.no_grad()
def answer_logps(
    model, sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], chunk: int = 4
) -> list[torch.Tensor]:
    """Log-probability of every answer token, for (prompt_ids, prompt_mask, answer_ids) batches as
    generated.

    Only the answer positions' logits are computed (logits_to_keep), in chunks, so a 151k-token
    vocabulary over 1,536-token answers stays within a few GB.
    """
    out = []
    for prompt_ids, prompt_mask, answer_ids in sequences:
        batch = []
        for i in range(0, len(prompt_ids), chunk):
            p, m, a = prompt_ids[i : i + chunk], prompt_mask[i : i + chunk], answer_ids[i : i + chunk]
            ids = torch.cat([p, a], dim=1)
            mask = torch.cat([m, torch.ones_like(a)], dim=1)
            logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=a.shape[1] + 1).logits[:, :-1].float()
            batch.append(torch.gather(logits.log_softmax(-1), 2, a.unsqueeze(-1)).squeeze(-1))
        out.append(torch.cat(batch))
    return out


def _load_policy(cfg: EvalConfig, device: str, adapter: Path | None):
    model = AutoModelForCausalLM.from_pretrained(POLICIES[cfg.policy], torch_dtype=model_dtype()).to(device)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    return model.eval()


def generate(
    prompts: Sequence[str], adapter: Path | None, cfg: EvalConfig, device: str | None = None, seed: int = SEED
) -> list[Generated]:
    """Sample one answer per prompt from the base model plus `adapter`, and measure how far each
    answer's policy has moved from the base model (KL, the x-axis of the over-optimization plots).

    KL is measured here rather than read from training logs: it works for every run (DPO included),
    and the base model's log-probabilities come from a fresh copy, so they're exact.
    """
    device = device or default_device()
    tok = AutoTokenizer.from_pretrained(POLICIES[cfg.policy])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = _load_policy(cfg, device, adapter)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) | {tok.eos_token_id}

    torch.manual_seed(seed)  # the same seed gives every checkpoint the same sampling noise
    sequences, texts, cut = [], [], []
    for i in range(0, len(prompts), cfg.batch_size):
        chats = [
            tok.apply_chat_template(
                [{"role": "user", "content": user_message(p, cfg.policy)}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts[i : i + cfg.batch_size]
        ]
        enc = tok(chats, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=cfg.max_new_tokens, do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p
            )
        answer_ids = out[:, enc["input_ids"].shape[1] :]
        sequences.append((enc["input_ids"], enc["attention_mask"], answer_ids))
        texts += tok.batch_decode(answer_ids, skip_special_tokens=True)
        cut += [hit_length_limit(ids.tolist(), eos_ids) for ids in answer_ids]

    if adapter is None:
        kls = [0.0] * len(texts)
    else:
        policy_logps = answer_logps(model, sequences)
        del model
        free_memory()
        model = _load_policy(cfg, device, None)  # a fresh base model: exact, unlike un-merging the adapter
        base_logps = answer_logps(model, sequences)
        kls = []
        for (_, _, answer_ids), pol, base in zip(sequences, policy_logps, base_logps, strict=True):
            kls += kl_per_token(pol, base, completion_mask(answer_ids, eos_ids)).tolist()

    del model
    free_memory()
    return [Generated(strip_think(t), c, k) for t, c, k in zip(texts, cut, kls, strict=True)]


def score_all(
    store: Store, generations: list[dict], gold_rm: str, kinds: Sequence[str] = REWARD_KINDS, gold: bool = True
) -> None:
    """Add the gold judge's and each reward model kind's score to every generation, loading one
    scorer at a time. `kinds` and `gold` pick a subset, e.g. to rescore only the kinds that changed."""
    P = [g["prompt"] for g in generations]
    R = [g["response"] for g in generations]
    scores = {}
    if gold:
        scores["gold"] = GoldJudge(gold_rm).score(P, R)
        free_memory()
    for kind in kinds:
        scores[kind] = load_reward_model(kind, store.download_dir(rm_dir(kind))).score(P, R)
        free_memory()
    for i, g in enumerate(generations):
        g.update({k: v[i] for k, v in scores.items()})


Key = tuple[str, int, int]  # (run, seed, step)


def summarize_generations(generations: list[dict], keys: list[Key]) -> tuple[list[dict], dict[str, float]]:
    """Mean of every score for each (run, seed, step), plus the base model's std for each score."""
    scores = score_keys(generations[0])
    rows = []
    for run, seed, step in keys:
        group = [g for g in generations if (g["run"], g.get("seed", 0), g["step"]) == (run, seed, step)]
        rows.append({"run": run, "seed": seed, "step": step,
                     **{k: statistics.mean(g[k] for g in group) for k in (*scores, *EXTRAS)},
                     "kl": statistics.mean(g["kl"] for g in group) if all("kl" in g for g in group) else None})
    base = [g for g in generations if g["run"] == "base"]
    base_std = {k: statistics.stdev(g[k] for g in base) for k in scores}
    return rows, base_std


def aggregate_seeds(seed_rows: list[dict], generations: list[dict]) -> list[dict]:
    """One row per (run, step): the mean over seeds, the min and max across seeds, and a 95% CI on
    the gold-score change from the base model.

    The CI is paired: every checkpoint answers the same prompts as the base model, so it's computed
    on per-prompt differences (averaged over seeds first). It covers which prompts happened to be
    sampled; the min-max band across seeds is what shows run-to-run variance.
    """
    base_gold = {g["prompt"]: g["gold"] for g in generations if g["run"] == "base"}
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in seed_rows:
        groups.setdefault((row["run"], row["step"]), []).append(row)

    rows = []
    for (run, step), group in groups.items():
        row = {"run": run, "step": step, "n_seeds": len(group)}
        for k in (*score_keys(group[0]), *EXTRAS):
            values = [r[k] for r in group]
            row[k] = statistics.mean(values)
            row[f"{k}_min"], row[f"{k}_max"] = min(values), max(values)
        kls = [r["kl"] for r in group if r.get("kl") is not None]
        row["kl"] = statistics.mean(kls) if kls else None
        if run != "base":
            seeds = {r["seed"] for r in group}
            per_prompt: dict[str, list[float]] = {}
            for g in generations:
                if g["run"] == run and g["step"] == step and g.get("seed", 0) in seeds and g["prompt"] in base_gold:
                    per_prompt.setdefault(g["prompt"], []).append(g["gold"] - base_gold[g["prompt"]])
            diffs = [statistics.mean(d) for d in per_prompt.values()]
            row["gold_ci"] = list(bootstrap_ci(diffs)) if len(diffs) > 1 else None
        rows.append(row)
    return rows


def compare_runs(generations: list[dict], pairs=COMPARISONS) -> list[dict]:
    """Head-to-head gold gaps between runs at each one's final checkpoint, on the same prompts
    (averaged over seeds), with a paired 95% CI."""
    last: dict[str, int] = {}
    for g in generations:
        if g["run"] != "base":
            last[g["run"]] = max(last.get(g["run"], 0), g["step"])

    def gold_by_prompt(run: str) -> dict[str, float]:
        by_prompt: dict[str, list[float]] = {}
        for g in generations:
            if g["run"] == run and g["step"] == last[run]:
                by_prompt.setdefault(g["prompt"], []).append(g["gold"])
        return {p: statistics.mean(v) for p, v in by_prompt.items()}

    return [{"a": a, "b": b, "step_a": last[a], "step_b": last[b], **paired_gap(gold_by_prompt(a), gold_by_prompt(b))}
            for a, b in pairs if a in last and b in last]


def format_table(rows: list[dict]) -> str:
    kinds = [k for k in REWARD_KINDS if k in rows[0]]
    lines = [f"{'run':<11}{'step':>6}{'seeds':>6}{'KL':>8}{'gold':>9}" + "".join(f"{k:>11}" for k in kinds)
             + f"{'words':>8}{'cut off':>9}"]
    for r in rows:
        kl = f"{r['kl']:>8.3f}" if r.get("kl") is not None else f"{'-':>8}"
        lines.append(
            f"{r['run']:<11}{r['step']:>6}{r.get('n_seeds', 1):>6}{kl}{r['gold']:>9.2f}"
            + "".join(f"{r[k]:>11.3f}" for k in kinds) + f"{r['words']:>8.0f}{r['truncated']:>9.0%}"
        )
    return "\n".join(lines)


def evaluate_policies(store: Store, cfg: EvalConfig) -> list[dict]:
    prompts = load_prompts("validation", cfg.n_prompts)

    todo: list[tuple[str, int, int, Path | None]] = [("base", 0, 0, None)]
    for run in cfg.runs:
        for seed in cfg.seeds:
            checkpoints = store.checkpoints(cfg.policy, run, seed)
            if not checkpoints:
                log.warning("no checkpoints for %s/%s seed %d, skipping", cfg.policy, run, seed)
            todo += [(run, seed, step, store.download_dir(folder)) for step, folder in checkpoints]

    generations = []
    for run, seed, step, adapter in todo:
        log.info("generating: %s seed %d step %d", run, seed, step)
        for prompt, gen in zip(prompts, generate(prompts, adapter, cfg), strict=True):
            generations.append({"run": run, "seed": seed, "step": step, "prompt": prompt, "response": gen.response,
                                "words": len(gen.response.split()), "truncated": gen.truncated, "kl": gen.kl})

    score_all(store, generations, cfg.gold_rm)

    # the GRPO jobs' logs (possibly from other Modal accounts) give the training curves
    grpo_logs: dict[tuple[str, int], list[dict]] = {}
    for run in (r for r in cfg.runs if r in REWARD_KINDS):
        for seed in cfg.seeds:
            if result := store.load_result(grpo_log_name(cfg.policy, run, seed)):
                grpo_logs[(run, seed)] = result["log"]

    seed_rows, base_std = summarize_generations(generations, [(run, seed, step) for run, seed, step, _ in todo])
    rows = aggregate_seeds(seed_rows, generations)

    comparisons = compare_runs(generations)
    store.save_result(f"eval_{cfg.policy}.json", {
        "config": asdict(cfg), "summary": rows, "seed_summary": seed_rows, "base_std": base_std,
        "comparisons": comparisons, "generations": generations,
    })
    log.info("\n%s\n\n%s", format_table(rows), format_comparisons(comparisons))
    tracking.log_table("eval/comparisons", comparisons)

    figures = {"reward_hacking": plot_reward_hacking(rows, base_std)}
    # gold vs. proxy against KL, as in Gao et al. (2022). KL can be exactly 0 early on, so test for presence
    if any(r["kl"] is not None for r in rows if r["run"] != "base"):
        figures["reward_hacking_kl"] = plot_reward_hacking(rows, base_std, x="kl")
    if grpo_logs:
        by_run: dict[str, list[list[dict]]] = {}
        for (run, _), history in sorted(grpo_logs.items()):
            by_run.setdefault(run, []).append(history)
        figures["training"] = plot_training_curves(by_run)
    for name, png in figures.items():
        store.save_result(f"{name}_{cfg.policy}.png", png)
        tracking.log_image(f"eval/{name}", png)

    tracking.log_table("eval/summary", rows)
    tracking.log_table("eval/generations", generations)
    tracking.summary({f"{r['run']}/{k}": r[k] for r in rows for k in (*score_keys(r), *EXTRAS)})
    return rows
