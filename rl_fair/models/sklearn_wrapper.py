from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


def make_sklearn_model(model_name: str, **kwargs):
    """Factory for simple sklearn classifiers (safe defaults)."""
    model_name = model_name.lower().strip()
    if model_name in {"logreg", "lr", "logistic"}:
        # Use saga for robustness with large/sparse data; adjust as needed.
        return LogisticRegression(max_iter=500, solver="lbfgs", **kwargs)
    if model_name in {"mlp"}:
        return MLPClassifier(max_iter=300, **kwargs)
    if model_name in {"rf", "random_forest"}:
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, **kwargs)
    raise ValueError(f"Unknown model_name: {model_name}")


def fit_predict(model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray):
    """Fit and return (y_pred, y_score) on validation set."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    # Scores for AUC/ABROCA: use predict_proba if present else decision_function.
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_val)
        # binary: proba[:,1], multiclass: full matrix
        y_score = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_val)
    else:
        # As a fallback, use predicted labels (not ideal).
        y_score = y_pred
    return y_pred, y_score
