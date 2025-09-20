from __future__ import annotations
import pandas as pd
import numpy as np
import hashlib
from typing import List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder

def _hash_to_bucket(s: str, n_bins: int, seed: int = 0) -> Tuple[int, int]:
    """Deterministic hash -> (bucket in [0..n_bins-1], sign in {-1, +1})."""
    key = str(seed).encode("utf-8")
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8, key=key).digest()
    val = int.from_bytes(h, "little", signed=False)
    bucket = val % n_bins
    sign = 1 if (val >> 63) & 1 == 0 else -1
    return bucket, sign

class HashedCSVPreprocessor:
    """
    Fixed-length featurization that does NOT depend on the raw number of columns.
    - Numeric columns: z-score using support mean/std, then signed-hash each column name into n_bins.
    - Categorical columns: token per (col, value) hashed into n_bins with +1/-1 sign.
    - Target: regression float or classification label-encoded.
    """
    def __init__(self, task_type: str, target_col: str, n_bins: int = 128, seed: int = 0):
        assert task_type in {"cls", "reg"}
        self.task_type = task_type
        self.target_col = target_col
        self.n_bins = int(n_bins)
        self.seed = int(seed)

        self.cat_cols: List[str] = []
        self.num_cols: List[str] = []
        self.means_: Optional[np.ndarray] = None
        self.stds_: Optional[np.ndarray] = None
        self.lbl: Optional[LabelEncoder] = None

    def _split_cols(self, df: pd.DataFrame) -> None:
        cols = [c for c in df.columns if c != self.target_col]
        self.cat_cols = [c for c in cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
        self.num_cols = [c for c in cols if c not in self.cat_cols]

    def fit(self, df_support: pd.DataFrame) -> None:
        self._split_cols(df_support)

        # numeric stats
        if self.num_cols:
            Xn = df_support[self.num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=True)
            means = np.nanmean(Xn, axis=0)
            stds = np.nanstd(Xn, axis=0); stds[stds == 0] = 1.0
            self.means_ = means
            self.stds_ = stds

        # label encoder for classification
        if self.task_type == "cls":
            self.lbl = LabelEncoder()
            self.lbl.fit(df_support[self.target_col].astype(str))

    def _hash_row(self, row: pd.Series) -> np.ndarray:
        x = np.zeros(self.n_bins, dtype=np.float32)
        # numeric
        for i, col in enumerate(self.num_cols):
            v = pd.to_numeric(row[col], errors="coerce")
            if pd.isna(v):
                continue
            if self.means_ is not None and self.stds_ is not None:
                v = (float(v) - float(self.means_[i])) / float(self.stds_[i])
            bucket, sign = _hash_to_bucket(f"num|{col}", self.n_bins, self.seed)
            x[bucket] += sign * float(v)
        # categorical
        for col in self.cat_cols:
            val = row[col]
            if pd.isna(val):
                continue
            token = f"cat|{col}|{str(val)}"
            bucket, sign = _hash_to_bucket(token, self.n_bins, self.seed)
            x[bucket] += float(sign)  # presence (+/-1)
        return x

    def transform_xy(self, df: pd.DataFrame):
        X = np.vstack([self._hash_row(df.iloc[i]) for i in range(len(df))])
        if self.target_col in df.columns:
            y_raw = df[self.target_col]
            if self.task_type == "cls":
                y = self.lbl.transform(y_raw.astype(str))
            else:
                y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=np.float32)
            return X, y
        else:
            return X, None