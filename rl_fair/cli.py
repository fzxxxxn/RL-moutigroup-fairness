from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from .config import DatasetSpec, TrainConfig, RewardConfig
from .rl.train import train_reweighter


def main():
    p = argparse.ArgumentParser(prog="rl-fair", description="RL-based dataset reweighting for fairness/accuracy trade-offs.")
    p.add_argument("--csv", type=str, required=True, help="Path to a CSV dataset.")
    p.add_argument("--features", type=str, required=True, help="Comma-separated feature columns.")
    p.add_argument("--label", type=str, required=True, help="Label column.")
    p.add_argument("--protected", type=str, required=True, help="Comma-separated protected columns used for grouping.")
    p.add_argument("--perf", type=str, default="auc", choices=["auc", "acc"], help="Performance metric.")
    p.add_argument("--disparity", type=str, default="dp", choices=["dp", "eo", "abroca"], help="Disparity metric.")
    p.add_argument("--steps", type=int, default=200, help="RL steps.")
    p.add_argument("--out", type=str, default="best_weights.json", help="Output path for best weights JSON.")

    args = p.parse_args()

    df = pd.read_csv(args.csv)
    feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    protected_cols = [c.strip() for c in args.protected.split(",") if c.strip()]

    spec = DatasetSpec(feature_cols=feature_cols, label_col=args.label, protected_cols=protected_cols)
    train_cfg = TrainConfig(num_steps=args.steps, verbose=True)

    res = train_reweighter(df, spec, train_cfg=train_cfg, reward_cfg=RewardConfig(), perf_metric=args.perf, disparity_metric=args.disparity)

    Path(args.out).write_text(json.dumps({
        "best_weights": res.best_weights.tolist(),
        "best_perf": res.best_perf,
        "best_disparity": res.best_disparity,
        "baseline_perf": res.baseline_perf,
        "baseline_disparity": res.baseline_disparity,
    }, indent=2), encoding="utf-8")

    print(f"Wrote: {args.out}")
