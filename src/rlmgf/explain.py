from typing import Dict, List, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from importlib import import_module

def _import(path: str):
    m, n = path.rsplit(".", 1)
    return getattr(import_module(m), n)

def explain_permutation_importance(model_path: str, model_params: Dict,
                                   X_train, y_train, X_test, y_test,
                                   feature_names: List[str], n_repeats: int = 10) -> Dict:
    Model = _import(model_path)
    clf = Model(**model_params)
    clf.fit(X_train, y_train)
    result = permutation_importance(clf, X_test, y_test, n_repeats=n_repeats, random_state=42)
    importances = {feature_names[i]: float(result.importances_mean[i]) for i in range(len(feature_names))}
    return {"perm_importance": importances}

def slice_report(y_true: np.ndarray, y_pred: np.ndarray, subgroup: np.ndarray) -> Dict:
    out: Dict[str, Dict] = {}
    for g in sorted(np.unique(subgroup).tolist()):
        m = subgroup == g
        acc = float((y_true[m] == y_pred[m]).mean()) if m.any() else float("nan")
        out[str(g)] = {"n": int(m.sum()), "accuracy": acc}
    return out
