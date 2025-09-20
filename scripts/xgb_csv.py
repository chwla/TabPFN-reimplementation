from __future__ import annotations
import argparse, os, time, sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from src.dataio.csv_data import load_csv_support_query
from src.utils.metrics import classification_metrics, regression_metrics

def main():
    ap = argparse.ArgumentParser(description="Train XGBoost on support CSV and write predictions for query CSV.")
    ap.add_argument("--task_type", choices=["cls","reg"], required=True)
    ap.add_argument("--target", required=True, help="Target column in CSVs")
    ap.add_argument("--csv", type=str, help="Single labeled CSV; we split into support/query with --split")
    ap.add_argument("--support_csv", type=str, help="Labeled support CSV path")
    ap.add_argument("--query_csv", type=str, help="Query CSV path (target may be absent)")
    ap.add_argument("--split", type=float, default=0.2, help="If using --csv, fraction to reserve as query")
    ap.add_argument("--out", type=str, default=None, help="Output predictions CSV path (default: results/xgb_predictions_<timestamp>.csv)")
    # Lightly exposed hyperparams
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--learning_rate", type=float, default=0.1)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample_bytree", type=float, default=0.9)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    assert (args.csv is not None) ^ (args.support_csv is not None and args.query_csv is not None), \
        "Provide either --csv OR (--support_csv AND --query_csv)"

    os.makedirs("results", exist_ok=True)

    # Load and preprocess data consistently with TabPFN pipeline (one-hot + scaling; labels encoded)
    pre, Xs, ys, Xq, yq, df_s_raw, df_q_raw = load_csv_support_query(
        csv_path=args.csv, target_col=args.target, task_type=args.task_type,
        query_ratio=args.split, support_csv=args.support_csv, query_csv=args.query_csv
    )

    if args.task_type == "cls":
        n_classes = int(len(np.unique(ys)))
        model = XGBClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate,
            subsample=args.subsample, colsample_bytree=args.colsample_bytree, reg_lambda=1.0,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            random_state=args.random_state, tree_method="hist"
        )
        model.fit(Xs, ys)
        if n_classes > 2:
            proba = model.predict_proba(Xq)
            pred_ids = np.argmax(proba, axis=1)
        else:
            proba_pos = model.predict_proba(Xq)[:, 1]
            pred_ids = (proba_pos >= 0.5).astype(int)
        # Map IDs back to original labels
        pred_labels = pre.lbl.inverse_transform(pred_ids.astype(int))
        df_out = df_q_raw.copy()
        df_out["prediction"] = pred_labels
        # If ground truth exists in query CSV, add it
        if yq is not None:
            df_out["target_true"] = pre.lbl.inverse_transform(yq.astype(int))
            mets = classification_metrics(yq, pred_ids if n_classes==2 else proba, pred_ids)
            print("[XGB/CLS] metrics:", mets)
    else:
        model = XGBRegressor(
            n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate,
            subsample=args.subsample, colsample_bytree=args.colsample_bytree, reg_lambda=1.0,
            random_state=args.random_state, tree_method="hist"
        )
        model.fit(Xs, ys)
        preds = model.predict(Xq)
        df_out = df_q_raw.copy()
        df_out["prediction"] = preds
        if yq is not None:
            df_out["target_true"] = yq
            mets = regression_metrics(yq, preds)
            print("[XGB/REG] metrics:", mets)

    out_path = args.out or f"results/xgb_predictions_{int(time.time())}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"[OK] Wrote XGBoost predictions to {out_path}")

if __name__ == "__main__":
    main()