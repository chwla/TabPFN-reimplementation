from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal
from sklearn.datasets import make_blobs, make_classification, make_circles

Array = np.ndarray

@dataclass
class Task:
    X_support: Array
    y_support: Array
    X_query: Array
    y_query: Array  # kept for evaluation

def _inject_missing(X: Array, missing_prob: float, rng: np.random.Generator) -> Array:
    if missing_prob <= 0.0:
        return X
    X = X.copy()
    mask = rng.random(X.shape) < missing_prob
    X[mask] = np.nan
    return X

def make_classification_task(
    n_support: int = 128,
    n_query: int = 128,
    n_features: int = 10,
    classes: int = 3,
    rule: Literal["blobs", "linear", "xor", "circles"] = "blobs",
    noise: float = 0.05,
    class_imbalance: float = 0.0,
    flip_y: float = 0.05,
    missing_prob: float = 0.0,
    random_state: Optional[int] = None,
) -> Task:
    rng = np.random.default_rng(random_state)
    n_total = n_support + n_query

    if rule == "blobs":
        X, y = make_blobs(n_samples=n_total, centers=classes, n_features=n_features, cluster_std=1.5, random_state=random_state)
    elif rule == "linear":
        weights = [1.0 - class_imbalance] + [class_imbalance/(classes-1)] * (classes-1) if classes > 1 else [1.0]
        X, y = make_classification(
            n_samples=n_total, n_features=n_features, n_informative=min(n_features, 6),
            n_redundant=0, n_repeated=0, n_classes=classes, n_clusters_per_class=1,
            weights=weights if classes == 2 else None, flip_y=flip_y, class_sep=1.5, random_state=random_state
        )
    elif rule == "xor":
        base = rng.normal(size=(n_total, n_features))
        x0, x1 = base[:, 0], base[:, 1]
        y = ((x0 > 0) ^ (x1 > 0)).astype(int)
        if classes != 2:
            raise ValueError("XOR supports classes=2")
        X = base + rng.normal(scale=noise, size=base.shape)
    elif rule == "circles":
        if classes != 2:
            raise ValueError("circles supports classes=2")
        X2, y = make_circles(n_samples=n_total, noise=noise, factor=0.5, random_state=random_state)
        if n_features > 2:
            pad = rng.normal(size=(n_total, n_features - 2))
            X = np.concatenate([X2, pad], axis=1)
        else:
            X = X2[:, :n_features]
    else:
        raise ValueError(f"Unknown rule: {rule}")

    X = _inject_missing(X, missing_prob, rng)
    idx = rng.permutation(n_total)
    s_idx, q_idx = idx[:n_support], idx[n_support:]
    return Task(X_support=X[s_idx], y_support=y[s_idx], X_query=X[q_idx], y_query=y[q_idx])

def make_regression_task(
    n_support: int = 128,
    n_query: int = 128,
    n_features: int = 10,
    kind: Literal["linear", "poly", "sin", "piecewise"] = "linear",
    noise: float = 0.1,
    missing_prob: float = 0.0,
    random_state: Optional[int] = None,
) -> Task:
    rng = np.random.default_rng(random_state)
    n_total = n_support + n_query
    X = rng.normal(size=(n_total, n_features))
    w = rng.normal(size=(n_features,))

    if kind == "linear":
        y = X @ w + 0.1 * rng.normal(size=n_total)
    elif kind == "poly":
        y = 2.0*X[:,0]**2 - 1.5*X[:,1]**2 + X[:,2:] @ w[2:] + noise * rng.normal(size=n_total)
    elif kind == "sin":
        y = np.sin(2.0*X[:,0]) + 0.5*np.sin(3.0*X[:,1]) + X[:,2:] @ w[2:] + noise * rng.normal(size=n_total)
    elif kind == "piecewise":
        y = np.where(X[:,0] > 0, 2.0*X[:,0] + X[:,1], -1.5*X[:,0] + 0.5*X[:,1]) + X[:,2:] @ w[2:] + noise * rng.normal(size=n_total)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    X = _inject_missing(X, missing_prob, rng)
    idx = rng.permutation(n_total)
    s_idx, q_idx = idx[:n_support], idx[n_support:]
    return Task(X_support=X[s_idx], y_support=y[s_idx], X_query=X[q_idx], y_query=y[q_idx])