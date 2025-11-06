from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score

def _macro_one_vs_all_probs(y_true: np.ndarray, scores: np.ndarray, n_classes: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for c in range(1, n_classes + 1):
        y_bin = (y_true == c).astype(int)
        # assume scores is shape [N, n_classes] of probabilities
        pairs.append((y_bin, scores[:, c-1]))
    return pairs

def bpsn_auc_for_subgroup(y_true: np.ndarray, y_scores: np.ndarray, subgroup_mask: np.ndarray, n_classes: int) -> float:
    # Background Positives (all positives outside subgroup), Subgroup Negatives
    aucs = []
    for c in range(1, n_classes+1):
        pos_mask = (y_true == c) & (~subgroup_mask)
        neg_mask = (y_true != c) & (subgroup_mask)
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue
        y = np.concatenate([np.ones(pos_mask.sum()), np.zeros(neg_mask.sum())])
        s = np.concatenate([y_scores[pos_mask, c-1], y_scores[neg_mask, c-1]])
        try:
            aucs.append(roc_auc_score(y, s))
        except Exception:
            continue
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))

def dispersion(values: List[float]) -> Dict[str, float]:
    v = np.array(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"range": float("nan"), "std": float("nan"), "var": float("nan")}
    return {"range": float(np.max(v) - np.min(v)), "std": float(np.std(v)), "var": float(np.var(v))}
