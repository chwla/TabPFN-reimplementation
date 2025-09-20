
from __future__ import annotations
import argparse, os, sys, json, time
import numpy as np
import pandas as pd
import torch

from src.dataio.csv_data import load_csv_support_query
from src.packing.pack import pack_task_classification, pack_task_regression, batchify
from src.model.transformer import TransformerTabPFN

def main():
    ap = argparse.ArgumentParser(description="Load a Week-2 checkpoint, pack support+query from CSV, and write predictions CSV.")
    ap.add_argument("--task_type", choices=["cls","reg"], required=True)
    ap.add_argument("--target", required=True, help="Target column name in CSV (ignored for query_csv if it has no labels)")
    ap.add_argument("--csv", type=str, help="Single labeled CSV; we will split into support/query (use --split)")
    ap.add_argument("--support_csv", type=str, help="Labeled support CSV path")
    ap.add_argument("--query_csv", type=str, help="Query CSV path; target column may be absent")
    ap.add_argument("--split", type=float, default=0.2, help="If using --csv, fraction to reserve as query")
    ap.add_argument("--ckpt", type=str, default="results/checkpoints/week2.pt", help="Path to .pt checkpoint")
    ap.add_argument("--device", type=str, default=None, help="cpu|cuda|mps (defaults to auto)")
    ap.add_argument("--out", type=str, default=None, help="Output CSV path (default: results/predictions_<timestamp>.csv)")
    ap.add_argument("--classes", type=int, default=None, help="Override num classes if checkpoint meta missing")
    args = ap.parse_args()

    assert (args.csv is not None) ^ (args.support_csv is not None and args.query_csv is not None), \
        "Provide either --csv OR (--support_csv AND --query_csv)"

    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    os.makedirs("results", exist_ok=True)

    pre, Xs, ys, Xq, yq, df_s_raw, df_q_raw = load_csv_support_query(
        csv_path=args.csv, target_col=args.target, task_type=args.task_type,
        query_ratio=args.split, support_csv=args.support_csv, query_csv=args.query_csv
    )
    n_features = Xs.shape[1]
    n_support, n_query = Xs.shape[0], Xq.shape[0]

    ckpt = torch.load(args.ckpt, map_location="cpu")
    meta = ckpt.get("meta", {})
    task_type_ckpt = meta.get("task_type", args.task_type)
    if task_type_ckpt != args.task_type:
        print(f"[ERROR] Checkpoint task_type={task_type_ckpt} != requested {args.task_type}. Use a matching checkpoint.", file=sys.stderr)
        sys.exit(1)

    num_classes = meta.get("num_classes", args.classes)
    if args.task_type == "cls" and num_classes is None:
        num_classes = int(len(np.unique(ys)))
        print(f"[INFO] Inferred num_classes={num_classes} from support labels.")

    d_in_ckpt = meta.get("d_in", None)
    d_model = meta.get("d_model", 128)
    n_layers = meta.get("n_layers", 4)
    n_heads = meta.get("n_heads", 8)
    ffn = meta.get("ffn", 256)

    if args.task_type == "cls":
        d_in_expected = n_features + num_classes + 1
    else:
        d_in_expected = n_features + 1 + 1

    if d_in_ckpt is not None and d_in_ckpt != d_in_expected:
        print(f"[ERROR] Feature mismatch.\n"
              f"  CSV (after preprocessing) -> n_features={n_features}\n"
              f"  Expected d_in = {d_in_expected} for task={args.task_type}\n"
              f"  But checkpoint d_in = {d_in_ckpt}\n\n"
              f"👉 Retrain Week-2 with matching dims, e.g.:\n"
              f"   python3 -m scripts.train_synth --task_type {args.task_type} "
              f"{'--classes '+str(num_classes) if args.task_type=='cls' else ''} "
              f"--n_features {n_features} --steps 5000\n",
              file=sys.stderr)
        sys.exit(2)

    model = TransformerTabPFN(
        d_in=d_in_expected, d_model=d_model, nhead=n_heads, num_layers=n_layers,
        dim_feedforward=ffn, dropout=0.1, task_type=args.task_type,
        num_classes=(num_classes if args.task_type=='cls' else None),
        max_len=n_support + n_query
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    if args.task_type == "cls":
        packed = pack_task_classification(Xs, ys, Xq, yq if yq is not None else np.zeros(len(Xq), dtype=int), num_classes=num_classes)
    else:
        packed = pack_task_regression(Xs, ys, Xq, yq if yq is not None else np.zeros(len(Xq), dtype=float))
    batch = batchify([packed])
    tokens = batch["tokens"].to(device)
    attn = batch["attn_mask"].to(device)
    with torch.no_grad():
        out = model(tokens, attn)["out"][0]
    is_query = batch["is_query"][0].cpu().numpy()

    if args.task_type == "cls":
        logits = out[is_query, :].cpu().numpy()
        pred_ids = logits.argmax(axis=-1)
        pred_labels = pre.lbl.inverse_transform(pred_ids.astype(int))
        df_out = df_q_raw.copy()
        df_out["prediction"] = pred_labels
        if yq is not None:
            df_out["target_true"] = pre.lbl.inverse_transform(yq.astype(int))
    else:
        preds = out[is_query].cpu().numpy()
        df_out = df_q_raw.copy()
        df_out["prediction"] = preds
        if yq is not None:
            df_out["target_true"] = yq

    out_path = args.out or f"results/predictions_{int(time.time())}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"[OK] Wrote predictions to {out_path}")
    print(f"[INFO] Support rows: {n_support} | Query rows: {n_query} | n_features(after preprocess): {n_features}")

if __name__ == "__main__":
    main()
