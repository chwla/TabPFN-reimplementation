from __future__ import annotations
import time
import numpy as np
from typing import Dict, Any
from xgboost import XGBClassifier, XGBRegressor
from src.utils.metrics import classification_metrics, regression_metrics

def run_xgb(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    task_type: str = "cls",
    random_state: int = 42,
    **xgb_kwargs,
) -> Dict[str, Any]:
    t0 = time.time()
    if task_type == "cls":
        n_classes = int(len(np.unique(y_train)))
        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            random_state=random_state, tree_method="hist",
            **xgb_kwargs
        )
        model.fit(X_train, y_train)
        if n_classes > 2:
            proba = model.predict_proba(X_val)  # (n_samples, n_classes)
            y_pred = np.argmax(proba, axis=1)
        else:
            proba1 = model.predict_proba(X_val)[:, 1]  # positive class prob
            proba = proba1
            y_pred = (proba1 >= 0.5).astype(int)
        metrics = classification_metrics(y_val, proba, y_pred)
    elif task_type == "reg":
        model = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=random_state, tree_method="hist",
            **xgb_kwargs
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        metrics = regression_metrics(y_val, y_pred)
    else:
        raise ValueError("task_type must be 'cls' or 'reg'")

    elapsed = time.time() - t0
    return {"model": model, "metrics": metrics, "elapsed_s": float(elapsed)}