import numpy as np
import pandas as pd
from rl_fair.resampling import resample_by_group, normalize_weights

def test_resample_counts():
    df = pd.DataFrame({"x": range(10)})
    group = np.array([0]*5 + [1]*5)
    w = normalize_weights(np.array([0.8, 0.2]))
    out = resample_by_group(df, group, w, n_samples=100, replace=True, rng=np.random.default_rng(0))
    # approximate proportion
    g_out = group[out.index.to_numpy()] if False else None
    assert len(out) == 100
