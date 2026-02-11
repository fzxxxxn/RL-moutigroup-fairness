from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.metrics import roc_curve


def abroca_disparity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    protected_binary: np.ndarray,
    grid_points: int = 200,
) -> float:
    """Compute ABROCA-style disparity for a *binary* protected attribute.

    ABROCA approximates the area between two ROC curves (protected=0 vs protected=1).
    Lower is better (0 means identical ROC curves).

    Notes:
      - This implementation is dependency-free (does not require the `abroca` package).
      - If one subgroup lacks positives/negatives, returns 0.0 (no reliable estimate).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    p = np.asarray(protected_binary)

    vals = np.unique(p)
    if len(vals) != 2:
        raise ValueError("protected_binary must have exactly 2 unique values")

    m0 = p == vals[0]
    m1 = p == vals[1]
    # Need both classes present in each subgroup to build ROC
    def ok(mask):
        yt = y_true[mask]
        return (np.any(yt==0) and np.any(yt==1))

    if not ok(m0) or not ok(m1):
        return 0.0

    fpr0, tpr0, _ = roc_curve(y_true[m0], y_score[m0])
    fpr1, tpr1, _ = roc_curve(y_true[m1], y_score[m1])

    grid = np.linspace(0.0, 1.0, grid_points)
    t0 = np.interp(grid, fpr0, tpr0)
    t1 = np.interp(grid, fpr1, tpr1)
    return float(np.trapz(np.abs(t0 - t1), grid))
