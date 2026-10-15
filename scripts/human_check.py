"""Blind human spot check of the gold judge. No GPU.

    # 1. make a sheet of 50 blind pairs (base vs. the BT run's checkpoint at step 200)
    python scripts/human_check.py export results/eval_qwen.json --run bt --step 200 --n 50

    # 2. open human_check.csv in any spreadsheet app and fill the "better" column with A, B or tie.
    #    Don't open human_check_key.json until you're done: it says which answer is which.

    # 3. score it
    python scripts/human_check.py score human_check.csv human_check_key.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_rm.evaluation.human_check import agreement, make_blind_pairs, read_csv, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="write a blind rating sheet")
    export.add_argument("eval_json", type=Path, help="results/eval_{policy}.json from the evaluate job")
    export.add_argument("--run", required=True, help="bt, jepa or dpo")
    export.add_argument("--step", type=int, required=True, help="checkpoint step to compare with the base model")
    export.add_argument("--seed", type=int, default=0, help="which training seed's checkpoint")
    export.add_argument("--n", type=int, default=50)
    export.add_argument("--out", type=Path, default=Path("human_check.csv"))

    score = sub.add_parser("score", help="compare your ratings with the judge")
    score.add_argument("ratings_csv", type=Path)
    score.add_argument("key_json", type=Path)

    args = parser.parse_args()
    if args.command == "export":
        generations = json.loads(args.eval_json.read_text())["generations"]
        rows, key = make_blind_pairs(generations, args.run, args.step, args.n, run_seed=args.seed)
        write_csv(args.out, rows)
        key_path = args.out.with_name(args.out.stem + "_key.json")
        key_path.write_text(json.dumps(key, indent=2))
        print(f"wrote {len(rows)} pairs to {args.out} (rate them before opening {key_path})")
    else:
        try:
            result = agreement(read_csv(args.ratings_csv), json.loads(args.key_json.read_text()))
        except ValueError as e:
            raise SystemExit(f"error: {e}") from None
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
