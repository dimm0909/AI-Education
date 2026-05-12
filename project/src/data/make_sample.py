from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic sample from hour.csv")
    parser.add_argument("--input", type=Path, default=Path("data/raw/hour.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/sample/hour_sample.csv"))
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    sample = df.sample(n=min(args.rows, len(df)), random_state=args.seed).sort_values(["dteday", "hr"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False)
    print(f"Saved sample with shape {sample.shape} to {args.output}")


if __name__ == "__main__":
    main()
