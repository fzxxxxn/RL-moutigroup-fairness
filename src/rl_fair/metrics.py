from typing import Dict
import numpy as np
import pandas as pd

def demographic_parity_diff(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    df = pd.DataFrame({'y_pred': y_pred, 'g': groups})
    rates = df.groupby('g')['y_pred'].mean()
    return float(rates.max() - rates.min())

def tpr_fpr_by_group(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> Dict:
    out = {}
    for g in np.unique(groups):
        idx = (groups == g)
        yt, yp = y_true[idx], y_pred[idx]
        tp = ((yt==1)&(yp==1)).sum()
        fp = ((yt==0)&(yp==1)).sum()
        tn = ((yt==0)&(yp==0)).sum()
        fn = ((yt==1)&(yp==0)).sum()
        tpr = tp / max(1, (tp+fn))
        fpr = fp / max(1, (fp+tn))
        out[g] = {'tpr': float(tpr), 'fpr': float(fpr)}
    return out

def equalized_odds_gap(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    stats = tpr_fpr_by_group(y_true, y_pred, groups)
    tprs = [v['tpr'] for v in stats.values()]
    fprs = [v['fpr'] for v in stats.values()]
    return (max(tprs) - min(tprs)) + (max(fprs) - min(fprs))

def aggregate_fairness(y_true, y_pred, groups, weights: Dict[str, float] = None) -> float:
    """Weighted sum of disparity metrics (the larger, the worse)."""
    if weights is None:
        weights = {'dp': 1.0, 'eo': 1.0}
    dp = demographic_parity_diff(y_true, y_pred, groups)
    eo = equalized_odds_gap(y_true, y_pred, groups)
    return weights['dp']*dp + weights['eo']*eo
