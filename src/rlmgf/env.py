from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

class WeightingEnv:
    """Environment that exposes actions as per-slice multipliers to reweight training data."""
    def __init__(self, df: pd.DataFrame, subgroup_col: str):
        self.df = df
        self.subgroup_col = subgroup_col
        self.subgroups = sorted(df[subgroup_col].unique().tolist())
        self.n_slices = len(self.subgroups)

    def apply_weights(self, base_weights: np.ndarray, action_weights: np.ndarray) -> np.ndarray:
        multipliers = action_weights.clip(0.1, 10.0)
        w = base_weights.copy()
        for i, g in enumerate(self.subgroups):
            w[self.df[self.subgroup_col].values == g] *= multipliers[i]
        # normalize
        w = w / (w.mean() + 1e-8)
        return w
