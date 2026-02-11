from __future__ import annotations

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd


def normalize_weights(w: np.ndarray, min_w: float = 0.0, max_w: float = 1.0) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be 1D")
    w = np.clip(w, min_w, max_w)
    s = w.sum()
    if s <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / s
    return w


def resample_by_group(
    df: pd.DataFrame,
    group_ids: np.ndarray,
    weights: np.ndarray,
    n_samples: int,
    replace: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Resample rows to match desired group weights.

    Args:
      df: full training dataframe
      group_ids: integer group ids aligned with df rows
      weights: desired sampling distribution over groups (len = n_groups)
      n_samples: total sample count
      replace: sample with replacement by default
    """
    if rng is None:
        rng = np.random.default_rng()

    group_ids = np.asarray(group_ids, dtype=np.int64)
    weights = np.asarray(weights, dtype=float)

    n_groups = int(group_ids.max()) + 1
    if len(weights) != n_groups:
        raise ValueError(f"weights length {len(weights)} != n_groups {n_groups}")

    # Determine per-group sample counts
    counts = np.floor(weights * n_samples).astype(int)
    # Fix rounding to hit exact n_samples
    deficit = n_samples - counts.sum()
    if deficit > 0:
        # add remaining samples to groups with largest fractional parts
        frac = weights * n_samples - np.floor(weights * n_samples)
        order = np.argsort(-frac)
        for k in order[:deficit]:
            counts[k] += 1
    elif deficit < 0:
        # remove extras from groups with largest counts
        order = np.argsort(-counts)
        for k in order[: (-deficit)]:
            if counts[k] > 0:
                counts[k] -= 1

    idx_all = []
    for g in range(n_groups):
        idx = np.where(group_ids == g)[0]
        if len(idx) == 0:
            continue
        take = counts[g]
        if take <= 0:
            continue
        chosen = rng.choice(idx, size=take, replace=replace)
        idx_all.append(chosen)

    if len(idx_all) == 0:
        raise ValueError("No samples selected; check your group_ids / weights")

    idx_all = np.concatenate(idx_all)
    # Shuffle to mix groups
    rng.shuffle(idx_all)
    return df.iloc[idx_all].reset_index(drop=True)
