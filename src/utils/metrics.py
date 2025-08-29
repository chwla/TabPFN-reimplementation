from __future__ import annotations
import numpy as np
from typing import Dict, Union
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score

Array = Union[np.ndarray, list]

def classification_metrics(y_true: Array, y_pred_proba: Array, y_pred_label: Array) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred_label = np.asarray(y_pred_label)
    out = {"accuracy": float(accuracy_score(y_true, y_pred_label))}
    try:
        proba = np.asarray(y_pred_proba)
        if proba.ndim == 1:  # binary probabilities for positive class
            out["roc_auc_ovr"] = float(roc_auc_score(y_true, proba))
        else:  # multiclass probabilities
            out["roc_auc_ovr"] = float(roc_auc_score(y_true, proba, multi_class="ovr"))
    except Exception:
        pass
    return out

def regression_metrics(y_true: Array, y_pred: Array) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "r2": r2}