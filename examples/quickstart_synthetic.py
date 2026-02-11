"""Quickstart example using a synthetic dataset.

This example intentionally uses generic protected columns (p0, p1) to avoid
demonstrating any real-world sensitive attributes.
"""
import numpy as np
import pandas as pd

from rl_fair import DatasetSpec, TrainConfig, RewardConfig, train_reweighter


def make_synth(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    p0 = rng.integers(0, 2, size=n)
    p1 = rng.integers(0, 2, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # label: depends on x plus group-specific bias (creates disparity)
    logits = 0.8*x1 - 0.6*x2 + 0.7*(p0==1) - 0.5*(p1==1)
    prob = 1/(1+np.exp(-logits))
    y = (rng.random(n) < prob).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "p0": p0, "p1": p1, "y": y})
    return df


if __name__ == "__main__":
    df = make_synth()

    spec = DatasetSpec(
        feature_cols=["x1", "x2"],
        label_col="y",
        protected_cols=["p0", "p1"],  # intersectional groups
    )

    train_cfg = TrainConfig(
        num_steps=120,
        delta=0.07,
        verbose=True,
        model_name="logreg",
    )
    reward_cfg = RewardConfig(lambda_fair=2.0, lambda_acc=1.0, lambda_entropy=0.01)

    res = train_reweighter(df, spec, train_cfg=train_cfg, reward_cfg=reward_cfg, perf_metric="auc", disparity_metric="eo")

    print("\nBaseline perf/disparity:", res.baseline_perf, res.baseline_disparity)
    print("Best     perf/disparity:", res.best_perf, res.best_disparity)
    print("Best weights:", res.best_weights)
