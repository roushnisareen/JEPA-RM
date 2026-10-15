"""Rebuild the figures from downloaded results on your laptop. No GPU, no Modal.

    huggingface-cli download USER/jepa-rlhf --include "results/*" --local-dir .
    python scripts/make_figures.py results --policy qwen

Writes PNG (for the blog) and PDF (for print) versions of each figure to figures/.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from jepa_rm.evaluation.best_of_n import best_of_n_figures
from jepa_rm.evaluation.plots import plot_reward_hacking, plot_stress_test, plot_training_curves
from jepa_rm.rewards import REWARD_KINDS


def write(out_dir: Path, name: str, render) -> None:
    for fmt in ("png", "pdf"):
        path = out_dir / f"{name}.{fmt}"
        path.write_bytes(render(fmt))
        print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="folder with the downloaded results/*.json")
    parser.add_argument("--policy", default="qwen")
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    stress_path = args.results / "stress_test.json"
    if stress_path.exists():
        stress = json.loads(stress_path.read_text())
        write(args.out, "stress_test", lambda fmt: plot_stress_test(stress, fmt=fmt))
    else:
        print(f"skipping stress test figure: {stress_path} not found")

    bon_path = args.results / f"best_of_n_{args.policy}.json"
    if bon_path.exists():
        bon = json.loads(bon_path.read_text())
        for name in ("best_of_n", "best_of_n_ablation"):
            write(args.out, f"{name}_{args.policy}",
                  lambda fmt, name=name: best_of_n_figures(bon["summary"], bon["base_std"], fmt=fmt)[name])
    else:
        print(f"skipping best-of-n figure: {bon_path} not found")

    eval_path = args.results / f"eval_{args.policy}.json"
    if eval_path.exists():
        result = json.loads(eval_path.read_text())
        write(args.out, f"reward_hacking_{args.policy}",
              lambda fmt: plot_reward_hacking(result["summary"], result["base_std"], fmt=fmt))
        if any(r.get("kl") is not None for r in result["summary"] if r["run"] != "base"):
            write(args.out, f"reward_hacking_kl_{args.policy}",
                  lambda fmt: plot_reward_hacking(result["summary"], result["base_std"], x="kl", fmt=fmt))
    else:
        print(f"skipping reward hacking figure: {eval_path} not found")

    logs: dict[str, list] = {}  # run -> one log history per seed
    for run in REWARD_KINDS:
        pattern = re.compile(rf"grpo_{re.escape(args.policy)}_{run}(_s\d+)?\.json")
        for path in sorted(args.results.iterdir()):
            if pattern.fullmatch(path.name):
                logs.setdefault(run, []).append(json.loads(path.read_text())["log"])
    if logs:
        write(args.out, f"training_{args.policy}", lambda fmt: plot_training_curves(logs, fmt=fmt))
    else:
        print("skipping training curves: no grpo_*.json found")


if __name__ == "__main__":
    main()
