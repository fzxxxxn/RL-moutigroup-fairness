from typing import Dict, List
import json
import numpy as np
import pandas as pd
from .metrics import bpsn_auc_for_subgroup, dispersion

def evaluate_fairness(y_true, scores, subgroups: np.ndarray, n_classes: int = 5) -> Dict:
    results = {}
    uniq = sorted(np.unique(subgroups).tolist())
    aucs = []
    for g in uniq:
        m = (subgroups == g)
        auc = bpsn_auc_for_subgroup(y_true, scores, m, n_classes=n_classes)
        results[str(g)] = auc
        if not np.isnan(auc):
            aucs.append(auc)
    fair_disp = dispersion(aucs)
    fair_mean = float(np.nanmean(aucs)) if aucs else float("nan")
    return {"per_group_bpsn_auc": results, "mean_bpsn_auc": fair_mean, "dispersion": fair_disp}
