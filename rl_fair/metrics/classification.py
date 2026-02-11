from __future__ import annotations

from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # binary or multiclass (ovo) handled by sklearn
    try:
        return float(roc_auc_score(y_true, y_score, multi_class="ovo" if len(np.unique(y_true))>2 else "raise"))
    except Exception:
        # Fallback: if y_score is 2D probs in multiclass
        return float(roc_auc_score(y_true, y_score, multi_class="ovo"))


def group_confusion_rates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """Per-group confusion-derived rates for binary classification."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group = np.asarray(group, dtype=np.int64)
    out: Dict[int, Dict[str, float]] = {}

    for g in np.unique(group):
        mask = group == g
        yt = y_true[mask]
        yp = y_pred[mask]
        # Handle degenerate groups
        if yt.size == 0:
            continue
        # confusion_matrix order: [[tn, fp],[fn,tp]]
        cm = confusion_matrix(yt, yp, labels=[0,1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        pr  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        p_hat = float(np.mean(yp == 1))
        out[int(g)] = {"tpr": float(tpr), "fpr": float(fpr), "precision": float(pr), "p_hat": p_hat}
    return out


def demographic_parity_diff(
    y_pred: np.ndarray,
    group: np.ndarray,
) -> float:
    """Max pairwise difference in positive prediction rates across groups (binary y_pred)."""
    group = np.asarray(group, dtype=np.int64)
    y_pred = np.asarray(y_pred)
    rates = []
    for g in np.unique(group):
        mask = group == g
        if mask.sum() == 0:
            continue
        rates.append(float(np.mean(y_pred[mask] == 1)))
    if len(rates) <= 1:
        return 0.0
    return float(np.max(rates) - np.min(rates))


def equal_opportunity_diff(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
) -> float:
    """Max pairwise difference in TPR across groups (binary classification)."""
    rates = group_confusion_rates(y_true, y_pred, group)
    tprs = [v["tpr"] for v in rates.values()]
    if len(tprs) <= 1:
        return 0.0
    return float(np.max(tprs) - np.min(tprs))
