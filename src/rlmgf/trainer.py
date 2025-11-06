from typing import Dict, Tuple
import numpy as np
import pandas as pd
from importlib import import_module
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

def import_by_path(path: str):
    module, name = path.rsplit(".", 1)
    return getattr(import_module(module), name)

def fit_predict_classifier(model_path: str, model_params: Dict, X_train, y_train, X_test):
    Model = import_by_path(model_path)
    clf = Model(**model_params)
    clf.fit(X_train, y_train)
    try:
        # predict_proba shape [N, C]
        scores = clf.predict_proba(X_test)
    except Exception:
        # some models may not implement predict_proba; fallback via decision_function
        dec = clf.decision_function(X_test)
        if dec.ndim == 1:
            dec = np.stack([1-dec, dec], axis=1)
        # convert to pseudo-probabilities
        from scipy.special import softmax
        scores = softmax(dec, axis=1)
    preds = np.argmax(scores, axis=1) + 1
    return clf, preds, scores

def compute_basic_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro"))
    }
