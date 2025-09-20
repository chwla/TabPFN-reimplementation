
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

class CSVPreprocessor:
    """
    Fit on support (train) data; transform both support and query consistently.
    - Categorical columns -> OneHotEncoder(handle_unknown='ignore')
    - Numeric columns     -> StandardScaler (fit on support)
    - Target:
        * classification -> LabelEncoder on y_support
        * regression     -> float
    """
    def __init__(self, task_type: str, target_col: str):
        assert task_type in {"cls", "reg"}
        self.task_type = task_type
        self.target_col = target_col
        self.cat_cols: List[str] = []
        self.num_cols: List[str] = []
        self.ohe: Optional[OneHotEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.lbl: Optional[LabelEncoder] = None
        self.feature_names_: Optional[List[str]] = None

    def _split_cols(self, df: pd.DataFrame) -> None:
        cols = [c for c in df.columns if c != self.target_col]
        self.cat_cols = [c for c in cols if df[c].dtype == 'object' or str(df[c].dtype).startswith('category')]
        self.num_cols = [c for c in cols if c not in self.cat_cols]

    def fit(self, df_support: pd.DataFrame) -> None:
        self._split_cols(df_support)

        if self.cat_cols:
            self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.ohe.fit(df_support[self.cat_cols])
        else:
            self.ohe = None

        if self.num_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(df_support[self.num_cols])
        else:
            self.scaler = None

        if self.task_type == "cls":
            self.lbl = LabelEncoder()
            self.lbl.fit(df_support[self.target_col].astype(str))
        else:
            self.lbl = None

        feat_names = []
        if self.num_cols:
            feat_names += self.num_cols
        if self.cat_cols and self.ohe is not None:
            ohe_names = self.ohe.get_feature_names_out(self.cat_cols).tolist()
            feat_names += ohe_names
        self.feature_names_ = feat_names

    def transform_xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_parts = []
        if self.num_cols:
            Xn = df[self.num_cols].copy()
            for c in self.num_cols:
                Xn[c] = pd.to_numeric(Xn[c], errors='coerce')
            if self.scaler is not None:
                Xn = self.scaler.transform(Xn)
            X_parts.append(Xn.astype(np.float32))
        if self.cat_cols and self.ohe is not None:
            Xc = self.ohe.transform(df[self.cat_cols])
            X_parts.append(Xc.astype(np.float32))
        if X_parts:
            X = np.concatenate(X_parts, axis=1).astype(np.float32)
        else:
            X = np.empty((len(df), 0), dtype=np.float32)

        if self.target_col in df.columns:
            y_raw = df[self.target_col]
            if self.task_type == "cls":
                y = self.lbl.transform(y_raw.astype(str)) if self.lbl is not None else y_raw.to_numpy()
            else:
                y = pd.to_numeric(y_raw, errors='coerce').to_numpy(dtype=np.float32)
            return X, y
        else:
            return X, None

def load_csv_support_query(
    csv_path: Optional[str],
    target_col: str,
    task_type: str,
    query_ratio: float = 0.2,
    seed: int = 42,
    support_csv: Optional[str] = None,
    query_csv: Optional[str] = None,
):
    """
    Returns:
      preproc, X_support, y_support, X_query, y_query (or None), df_support_raw, df_query_raw
    """
    assert task_type in {"cls", "reg"}
    rng = np.random.default_rng(seed)

    if support_csv is not None and query_csv is not None:
        df_s = pd.read_csv(support_csv)
        df_q = pd.read_csv(query_csv)
    else:
        assert csv_path is not None, "Provide either a single --csv or both --support_csv and --query_csv"
        df = pd.read_csv(csv_path)
        assert target_col in df.columns, f"Target column '{target_col}' not found in CSV"
        idx = np.arange(len(df))
        rng.shuffle(idx)
        split = int((1.0 - query_ratio) * len(df))
        s_idx, q_idx = idx[:split], idx[split:]
        df_s = df.iloc[s_idx].reset_index(drop=True)
        df_q = df.iloc[q_idx].reset_index(drop=True)

    pre = CSVPreprocessor(task_type=task_type, target_col=target_col)
    pre.fit(df_s)

    Xs, ys = pre.transform_xy(df_s)
    Xq, yq = pre.transform_xy(df_q)

    return pre, Xs, ys, Xq, yq, df_s, df_q
