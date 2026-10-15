"""GPU jobs on Modal. Each job is a thin wrapper around the `jepa_rm` package.

Jobs read from and write to one private HF repo, so they can run one at a time and on different
Modal accounts:

    modal run modal_app.py::reward_models --hf-repo USER/jepa-rlhf
    modal run modal_app.py::stress_test --hf-repo USER/jepa-rlhf
    modal run modal_app.py::best_of_n --hf-repo USER/jepa-rlhf
    modal run modal_app.py::grpo --hf-repo USER/jepa-rlhf --rm bt
    modal run modal_app.py::grpo --hf-repo USER/jepa-rlhf --rm jepa
    modal run modal_app.py::evaluate --hf-repo USER/jepa-rlhf
    modal run modal_app.py::dpo --hf-repo USER/jepa-rlhf
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml")
    .env({"PYTHONPATH": "/root/src", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("src", remote_path="/root/src")
)
app = modal.App("jepa-rlhf")

GPU = "A100-80GB"
SECRETS = [
    modal.Secret.from_name("huggingface"),  # HF_TOKEN with write access
    modal.Secret.from_name("wandb"),  # WANDB_API_KEY, optionally WANDB_ENTITY / WANDB_PROJECT
]
# cache model downloads so reruns don't fetch the same weights again
VOLUMES = {"/root/.cache/huggingface": modal.Volume.from_name("hf-cache", create_if_missing=True)}
HOUR = 60 * 60


def _setup() -> None:
    import logging

    from transformers import set_seed

    from jepa_rm.config import SEED

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    set_seed(SEED)


@app.function(image=image, gpu=GPU, secrets=SECRETS, volumes=VOLUMES, timeout=2 * HOUR)
def reward_models(hf_repo: str, max_pairs: int = 0, jepa_pooling: str = "mean"):
    """Train every reward model kind (BT, its length-penalized fix, the frozen-feature control, JEPA
    and its two ablations) and compare them on held-out pairs.

    `--max-pairs` caps the training pairs for a quick test; the default (0) uses all 4,601.

    For the pooling ablation, use a separate repo per setting, e.g. `--jepa-pooling last
    --hf-repo USER/jepa-rlhf-last`, so the two experiments never overwrite each other.
    """
    _setup()
    import logging
    import random
    from dataclasses import asdict

    from jepa_rm import tracking
    from jepa_rm.artifacts import HubStore, rm_dir
    from jepa_rm.config import SEED, BTConfig, JepaConfig
    from jepa_rm.data import load_pairs
    from jepa_rm.evaluation.reward_models import compare_reward_models, format_table
    from jepa_rm.rewards import REWARD_KINDS
    from jepa_rm.rewards.suite import train_reward_models

    store = HubStore(hf_repo)
    train = load_pairs("train")
    random.Random(SEED).shuffle(train)
    train = train[:max_pairs] if max_pairs else train
    val = load_pairs("validation")
    logging.info("%d train pairs, %d validation pairs", len(train), len(val))

    bt_cfg, jepa_cfg = BTConfig(), JepaConfig(pooling=jepa_pooling)
    config = {"hf_repo": hf_repo, "max_pairs": max_pairs, "bt": asdict(bt_cfg), "jepa": asdict(jepa_cfg)}
    run_name = "reward-models" if jepa_pooling == "mean" else f"reward-models-{jepa_pooling}"  # ablations stand out
    with tracking.run(run_name, job_type="reward_models", config=config):
        tracking.define_axis("bt", "bt/step")
        for kind in ("frozen_bt", "jepa", "jepa_pred", "jepa_rank"):
            tracking.define_axis(kind, f"{kind}/epoch")

        models = train_reward_models(train, "/tmp/reward_models", bt_cfg, jepa_cfg)
        for kind in REWARD_KINDS:
            store.upload_dir(f"/tmp/reward_models/{rm_dir(kind)}", rm_dir(kind))

        results = compare_reward_models(val, models)
        store.save_result("reward_models.json", {**config, **results})
        logging.info("\n%s", format_table(results))

        tracking.log_table("val/reward_models", [{"rm": name, **m} for name, m in results["models"].items()])
        tracking.summary({f"val/{name}/{k}": v for name, m in results["models"].items() for k, v in m.items()
                          if not isinstance(v, list)})
        tracking.summary({"val/humans_prefer_longer": results["humans_prefer_longer"]})


@app.function(image=image, gpu=GPU, secrets=SECRETS, volumes=VOLUMES, timeout=6 * HOUR)
def grpo(
    hf_repo: str, rm: str = "bt", policy: str = "qwen", max_steps: int = 200, seed: int = 0, lr: float = 1e-5
):
    """GRPO on the policy model against one reward model (any kind; see jepa_rm/rewards).

    Pilot first: `--max-steps 60` against bt, then check KL and gold-vs-proxy in W&B. If the BT run
    isn't being over-optimized, raise `--lr` or `--max-steps` before running the full comparison;
    there's nothing to compare if the standard reward model never gets hacked.
    """
    _setup()
    from dataclasses import asdict

    from jepa_rm import tracking
    from jepa_rm.artifacts import HubStore
    from jepa_rm.config import GrpoRunConfig
    from jepa_rm.training.grpo import train_grpo

    cfg = GrpoRunConfig(policy=policy, max_steps=max_steps, seed=seed, lr=lr)
    config = {"hf_repo": hf_repo, "rm": rm, **asdict(cfg)}
    with tracking.run(f"grpo-{policy}-{rm}-s{seed}", job_type="grpo", config=config):
        train_grpo(HubStore(hf_repo), rm, cfg)


@app.function(image=image, gpu=GPU, secrets=SECRETS, volumes=VOLUMES, timeout=2 * HOUR)
def dpo(hf_repo: str, policy: str = "qwen", max_pairs: int = 3000, seed: int = 0):
    """DPO baseline, trained on the same preference data with no reward model."""
    _setup()
    from dataclasses import asdict

    from jepa_rm import tracking
    from jepa_rm.artifacts import HubStore
    from jepa_rm.config import DpoRunConfig
    from jepa_rm.training.dpo import train_dpo

    cfg = DpoRunConfig(policy=policy, max_pairs=max_pairs, seed=seed)
    with tracking.run(f"dpo-{policy}-s{seed}", job_type="dpo", config={"hf_repo": hf_repo, **asdict(cfg)}):
        train_dpo(HubStore(hf_repo), cfg)


@app.function(image=image, gpu=GPU, secrets=SECRETS, volumes=VOLUMES, timeout=3 * HOUR)
def evaluate(hf_repo: str, policy: str = "qwen", runs: str = "bt,jepa", seeds: str = "0", n_prompts: int = 200):
    """Score the base model and every checkpoint with the gold judge and both reward models."""
    _setup()
    from dataclasses import asdict

    from jepa_rm import tracking
    from jepa_rm.artifacts import HubStore
    from jepa_rm.config import EvalConfig
    from jepa_rm.evaluation.policies import evaluate_policies

    run_names = tuple(r.strip() for r in runs.split(",") if r.strip())
    seed_list = tuple(int(s) for s in seeds.split(",") if s.strip())
    cfg = EvalConfig(policy=policy, runs=run_names, seeds=seed_list, n_prompts=n_prompts)
    with tracking.run(f"eval-{policy}", job_type="evaluate", config={"hf_repo": hf_repo, **asdict(cfg)}):
        evaluate_policies(HubStore(hf_repo), cfg)


@app.function(image=image, gpu=GPU, secrets=SECRETS, volumes=VOLUMES, timeout=HOUR)
def stress_test(hf_repo: str, n_examples: int = 200, with_gold: bool = True):
    """Edit good answers in ways that shouldn't help (padding, flattery, ...) and see which scorers rise.

    Needs only the reward models from `reward_models`, so it can run before any RL.
    """
    _setup()
    import logging
    from dataclasses import asdict

    from jepa_rm import tracking
    from jepa_rm.artifacts import HubStore, rm_dir
    from jepa_rm.config import StressConfig
    from jepa_rm.data import load_pairs
    from jepa_rm.evaluation.plots import plot_stress_test
    from jepa_rm.evaluation.policies import GoldJudge
    from jepa_rm.evaluation.stress_test import format_table, pick_examples, run_stress_test, table_rows
    from jepa_rm.rewards import REWARD_KINDS, load_reward_model

    store = HubStore(hf_repo)
    cfg = StressConfig(n_examples=n_examples, with_gold=with_gold)
    with tracking.run("stress-test", job_type="stress_test", config={"hf_repo": hf_repo, **asdict(cfg)}):
        examples = pick_examples(load_pairs("validation"), cfg.n_examples)
        scorers = {kind: (lambda k=kind: load_reward_model(k, store.download_dir(rm_dir(k)))) for kind in REWARD_KINDS}
        if cfg.with_gold:
            scorers["gold"] = lambda: GoldJudge(cfg.gold_rm)

        results = run_stress_test(examples, scorers)
        png = plot_stress_test(results)
        store.save_result("stress_test.json", {"config": asdict(cfg), **results})
        store.save_result("stress_test.png", png)
        logging.info("\n%s", format_table(results))

        tracking.log_image("stress/figure", png)
        rows = table_rows(results)
        tracking.log_table("stress/results", rows)
        tracking.summary({f"stress/{r['perturbation']}/{r['scorer']}": r["delta_sigma"] for r in rows})


@app.function(image=image, gpu=GPU, secrets=SECRETS, volumes=VOLUMES, timeout=4 * HOUR)
def best_of_n(hf_repo: str, policy: str = "qwen", n_prompts: int = 64, n_samples: int = 64, extend: bool = False):
    """Reward hacking without training: pick each reward model's favourite of n samples from the
    base model and score it with the gold judge. Needs only the reward models.

    `--extend` grows the previous run to `--n-samples` per prompt, on its prompts, reusing the
    samples it already scored (`--n-prompts` is then ignored).
    """
    _setup()
    from dataclasses import asdict

    from jepa_rm import tracking
    from jepa_rm.artifacts import HubStore
    from jepa_rm.config import BonConfig
    from jepa_rm.evaluation.best_of_n import run_best_of_n

    ns = tuple(n for n in (2, 4, 8, 16, 32, 64) if n <= n_samples)
    cfg = BonConfig(policy=policy, n_prompts=n_prompts, n_samples=n_samples, ns=ns)
    with tracking.run(f"best-of-n-{policy}", job_type="best_of_n", config={"hf_repo": hf_repo, **asdict(cfg)}):
        run_best_of_n(HubStore(hf_repo), cfg, extend=extend)
