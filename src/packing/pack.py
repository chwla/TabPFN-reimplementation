from __future__ import annotations
"""
Utilities to pack support+query into sequences a Transformer can read.
We follow the Week-2 plan:
- For classification: token = [features..., onehot(label) or zeros if query, known_flag]
- For regression:    token = [features..., label_or_0_if_query, known_flag]
We also provide batching with left-padding to the max length in the batch.
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Literal, Optional

Array = np.ndarray

def _impute_missing(X: Array) -> Array:
    """
    Impute NaNs column-wise with median (numeric only). Returns a copy.
    """
    Xc = X.copy()
    for j in range(Xc.shape[1]):
        col = Xc[:, j]
        if np.isnan(col).any():
            med = np.nanmedian(col)
            col[np.isnan(col)] = med
            Xc[:, j] = col
    return Xc

def pack_task_classification(
    X_support: Array, y_support: Array, X_query: Array, y_query: Array,
    num_classes: int,
) -> Dict[str, Array]:
    """
    Builds a single-task sequence for classification.
    Returns:
      tokens: (L, D_in)  where D_in = n_feat + num_classes + 1(known_flag)
      target: (L,) class indices with -100 on non-query positions
      is_query: (L,) bool mask
    """
    Xs = _impute_missing(X_support)
    Xq = _impute_missing(X_query)

    n_feat = Xs.shape[1]
    onehot_support = np.zeros((Xs.shape[0], num_classes), dtype=float)
    for i, yi in enumerate(y_support.astype(int)):
        if yi < 0 or yi >= num_classes:
            raise ValueError(f"class index {yi} out of range for num_classes={num_classes}")
        onehot_support[i, yi] = 1.0

    onehot_query = np.zeros((Xq.shape[0], num_classes), dtype=float)

    known_s = np.ones((Xs.shape[0], 1), dtype=float)
    known_q = np.zeros((Xq.shape[0], 1), dtype=float)

    toks_support = np.concatenate([Xs, onehot_support, known_s], axis=1)
    toks_query   = np.concatenate([Xq, onehot_query, known_q], axis=1)
    tokens = np.concatenate([toks_support, toks_query], axis=0)

    Ls, Lq = len(y_support), len(y_query)
    target = np.full((Ls + Lq,), fill_value=-100, dtype=int)  # -100 ignored by CE
    target[Ls:] = y_query.astype(int)
    is_query = np.zeros((Ls + Lq,), dtype=bool)
    is_query[Ls:] = True

    return {"tokens": tokens, "target": target, "is_query": is_query}

def pack_task_regression(
    X_support: Array, y_support: Array, X_query: Array, y_query: Array,
) -> Dict[str, Array]:
    """
    Builds a single-task sequence for regression.
    token = [features..., label_or_0, known_flag]
    Returns:
      tokens: (L, D_in) with last dim = n_feat + 1(label slot) + 1(known_flag)
      target: (L,) float with NaN on non-query positions
      is_query: (L,) bool mask
    """
    Xs = _impute_missing(X_support)
    Xq = _impute_missing(X_query)

    n_feat = Xs.shape[1]
    label_s = y_support.reshape(-1, 1).astype(float)
    label_q = np.zeros((Xq.shape[0], 1), dtype=float)

    known_s = np.ones((Xs.shape[0], 1), dtype=float)
    known_q = np.zeros((Xq.shape[0], 1), dtype=float)

    toks_support = np.concatenate([Xs, label_s, known_s], axis=1)
    toks_query   = np.concatenate([Xq, label_q, known_q], axis=1)
    tokens = np.concatenate([toks_support, toks_query], axis=0)

    Ls, Lq = len(y_support), len(y_query)
    target = np.full((Ls + Lq,), fill_value=np.nan, dtype=float)
    target[Ls:] = y_query.astype(float)
    is_query = np.zeros((Ls + Lq,), dtype=bool)
    is_query[Ls:] = True

    return {"tokens": tokens, "target": target, "is_query": is_query}

def batchify(
    packed_list: List[Dict[str, Array]]
) -> Dict[str, torch.Tensor]:
    """
    Pad a list of packed tasks to the same length and convert to torch tensors.
    Returns tensors with shapes:
      tokens:  (B, T, D)
      target:  (B, T)   (float; cast later as needed)
      is_query:(B, T)   (bool)
      attn_mask: (B, T) (bool; True for valid tokens)
    """
    B = len(packed_list)
    max_len = max(p["tokens"].shape[0] for p in packed_list)
    D = packed_list[0]["tokens"].shape[1]

    tokens = np.zeros((B, max_len, D), dtype=float)
    target = np.zeros((B, max_len), dtype=float)
    is_query = np.zeros((B, max_len), dtype=bool)
    attn_mask = np.zeros((B, max_len), dtype=bool)

    for i, p in enumerate(packed_list):
        L = p["tokens"].shape[0]
        tokens[i, :L, :] = p["tokens"]
        target[i, :L] = p["target"]
        is_query[i, :L] = p["is_query"]
        attn_mask[i, :L] = True

    return {
        "tokens": torch.tensor(tokens, dtype=torch.float32),
        "target": torch.tensor(target, dtype=torch.float32),
        "is_query": torch.tensor(is_query, dtype=torch.bool),
        "attn_mask": torch.tensor(attn_mask, dtype=torch.bool),
    }