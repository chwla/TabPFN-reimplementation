from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def main():
    ap = argparse.ArgumentParser(description="Compare two prediction CSVs (TabPFN vs XGBoost).")
    ap.add_argument("--task_type", choices=["cls","reg"], required=True)
    ap.add_argument("--tabpfn_csv", required=True, help="Predictions CSV from scripts/predict_csv.py")
    ap.add_argument("--xgb_csv", required=True, help="Predictions CSV from scripts/xgb_csv.py")
    ap.add_argument("--key", type=str, default=None,
                    help="Optional column to join on (must exist in both CSVs). If omitted, compares by row order.")
    ap.add_argument("--out", type=str, default=None, help="Output merged comparison CSV (default: results/compare_<ts>.csv)")
    args = ap.parse_args()

    df_t = pd.read_csv(args.tabpfn_csv)
    df_x = pd.read_csv(args.xgb_csv)

    # Normalize column names we care about
    def pick_pred(df):
        if "prediction" in df.columns: return "prediction"
        # fallback: first column containing "prediction"
        cand = [c for c in df.columns if "prediction" in c.lower()]
        if not cand:
            print("[ERROR] No prediction column found. Expected a column named 'prediction'.", file=sys.stderr)
            sys.exit(2)
        return cand[0]

    def pick_truth(df):
        if "target_true" in df.columns: return "target_true"
        cand = [c for c in df.columns if "target" in c.lower() and "true" in c.lower()]
        return cand[0] if cand else None

    pred_col_t = pick_pred(df_t)
    pred_col_x = pick_pred(df_x)
    truth_t = pick_truth(df_t)
    truth_x = pick_truth(df_x)

    if args.key:
        if args.key not in df_t.columns or args.key not in df_x.columns:
            print(f"[ERROR] key '{args.key}' not present in both files.", file=sys.stderr)
            sys.exit(2)
        # Only keep the needed columns from each side; suffix to disambiguate
        left = df_t[[args.key, pred_col_t] + ([truth_t] if truth_t else [])].rename(
            columns={pred_col_t: "prediction_tabpfn", (truth_t or "target_true"): "target_true"}
        )
        right = df_x[[args.key, pred_col_x] + ([truth_x] if truth_x else [])].rename(
            columns={pred_col_x: "prediction_xgb", (truth_x or "target_true"): "target_true_xgb"}
        )
        merged = pd.merge(left, right, on=args.key, how="inner")
        # Prefer TabPFN truth if both exist, else XGB truth
        if "target_true" not in merged.columns and "target_true_xgb" in merged.columns:
            merged = merged.rename(columns={"target_true_xgb": "target_true"})
        elif "target_true_xgb" in merged.columns:
            merged = merged.drop(columns=["target_true_xgb"])
    else:
        # Must be same length if aligning by row order
        if len(df_t) != len(df_x):
            print("[ERROR] CSVs have different lengths and no --key provided for alignment.", file=sys.stderr)
            sys.exit(2)
        # Build merged by row order and keep both predictions explicitly
        merged = df_t.copy()
        merged = merged.rename(columns={pred_col_t: "prediction_tabpfn"})
        merged["prediction_xgb"] = df_x[pred_col_x].values
        # Bring over truth if present (prefer TabPFN file)
        if truth_t:
            merged = merged.rename(columns={truth_t: "target_true"})
        elif truth_x:
            merged["target_true"] = df_x[truth_x].values

    # Agreement column
    merged["agree"] = (merged["prediction_tabpfn"].astype(str).values ==
                       merged["prediction_xgb"].astype(str).values)

    # Metrics
    if args.task_type == "cls":
        lines = []
        if "target_true" in merged.columns:
            y_true = merged["target_true"].astype(str)
            acc_tab = (merged["prediction_tabpfn"].astype(str) == y_true).mean()
            acc_xgb = (merged["prediction_xgb"].astype(str) == y_true).mean()
            lines.append(f"Accuracy (TabPFN): {acc_tab:.4f}")
            lines.append(f"Accuracy (XGBoost): {acc_xgb:.4f}")
            # Confusion matrices (optional, prints nicely)
            labels = sorted(pd.unique(pd.concat([y_true,
                                                 merged["prediction_tabpfn"].astype(str),
                                                 merged["prediction_xgb"].astype(str)])))
            cm_tab = confusion_matrix(y_true, merged["prediction_tabpfn"].astype(str), labels=labels)
            cm_xgb = confusion_matrix(y_true, merged["prediction_xgb"].astype(str), labels=labels)
            lines.append(f"Agreement rate: {merged['agree'].mean():.4f}")
            lines.append("Labels: " + ", ".join(labels))
            lines.append("Confusion (TabPFN vs True):\n" + pd.DataFrame(cm_tab, index=labels, columns=labels).to_string())
            lines.append("Confusion (XGB vs True):\n" + pd.DataFrame(cm_xgb, index=labels, columns=labels).to_string())
        else:
            lines.append(f"Agreement rate (no ground truth): {merged['agree'].mean():.4f}")
        print("\n".join(lines))
    else:
        # regression
        if "target_true" in merged.columns:
            y_true = pd.to_numeric(merged["target_true"], errors="coerce")
            tab = pd.to_numeric(merged["prediction_tabpfn"], errors="coerce")
            xgb = pd.to_numeric(merged["prediction_xgb"], errors="coerce")
            rmse_tab = float(np.sqrt(((tab - y_true) ** 2).mean()))
            rmse_xgb = float(np.sqrt(((xgb - y_true) ** 2).mean()))
            print(f"RMSE (TabPFN): {rmse_tab:.4f}")
            print(f"RMSE (XGBoost): {rmse_xgb:.4f}")
        else:
            tab = pd.to_numeric(merged["prediction_tabpfn"], errors="coerce")
            xgb = pd.to_numeric(merged["prediction_xgb"], errors="coerce")
            mae_between = float(np.mean(np.abs(tab - xgb)))
            print(f"No ground truth; mean absolute difference between models: {mae_between:.4f}")

    out_path = args.out or f"results/compare_{int(time.time())}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[OK] Wrote merged comparison CSV to {out_path}")

if __name__ == "__main__":
    main()
