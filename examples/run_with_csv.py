"""Example runner for a local CSV (do not commit real data).

Usage:
  python examples/run_with_csv.py --csv your.csv --features f1,f2 --label y --protected p0,p1

This mirrors the CLI but is easier to customize in code.
"""
import argparse
import pandas as pd
from rl_fair import DatasetSpec, TrainConfig, RewardConfig, train_reweighter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--protected", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    features = [c.strip() for c in args.features.split(",") if c.strip()]
    protected = [c.strip() for c in args.protected.split(",") if c.strip()]

    spec = DatasetSpec(feature_cols=features, label_col=args.label, protected_cols=protected)

    train_cfg = TrainConfig(num_steps=200, verbose=True, model_name="logreg")
    reward_cfg = RewardConfig(lambda_fair=2.0, lambda_acc=1.0)

    res = train_reweighter(df, spec, train_cfg=train_cfg, reward_cfg=reward_cfg, perf_metric="auc", disparity_metric="dp")
    print(res)

if __name__ == "__main__":
    main()
