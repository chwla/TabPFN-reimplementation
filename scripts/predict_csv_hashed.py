from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import pandas as pd
import torch

from src.dataio.hash_preproc import HashedCSVPreprocessor
from src.packing.pack import pack_task_classification, pack_task_regression, batchify
from src.model.transformer import TransformerTabPFN

def main():
    ap = argparse.ArgumentParser(description="Predict on CSV using a feature-hashed checkpoint (fixed n_bins).")
    ap.add_argument("--task_type", choices=["cls","reg"], required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--csv", type=str)
    ap.add_argument("--support_csv", type=str)
    ap.add_argument("--query_csv", type=str)
    ap.add_argument("--split", type=float, default=0.2)
    ap.add_argument("--ckpt", type=str, default="results/checkpoints/week2_hashed.pt")
    ap.add_argument("--n_bins", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    assert (args.csv is not None) ^ (args.support_csv is not None and args.query_csv is not None), \
        "Provide either --csv OR (--support_csv AND --query_csv)"

    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ckpt = torch.load(args.ckpt, map_location="cpu")
    meta = ckpt.get("meta", {})
    task_type_ckpt = meta.get("task_type", args.task_type)
    if task_type_ckpt != args.task_type:
        print(f"[ERROR] Checkpoint task_type={task_type_ckpt} != requested {args.task_type}.", file=sys.stderr)
        sys.exit(1)
    n_bins = args.n_bins or int(meta.get("hash_bins", 128))

    # load CSVs
    if args.support_csv and args.query_csv:
        df_s = pd.read_csv(args.support_csv)
        df_q = pd.read_csv(args.query_csv)
    else:
        df = pd.read_csv(args.csv)
        if args.target not in df.columns:
            print(f"[ERROR] Target column '{args.target}' not found.", file=sys.stderr)
            sys.exit(2)
        split = int((1.0 - args.split) * len(df))
        df_s = df.iloc[:split].reset_index(drop=True)
        df_q = df.iloc[split:].reset_index(drop=True)

    pre = HashedCSVPreprocessor(task_type=args.task_type, target_col=args.target, n_bins=n_bins, seed=42)
    pre.fit(df_s)
    Xs, ys = pre.transform_xy(df_s)
    Xq, yq = pre.transform_xy(df_q)

    # build model with fixed d_in from n_bins
    if args.task_type == "cls":
        num_classes = len(np.unique(ys))
        d_in = n_bins + num_classes + 1
        model = TransformerTabPFN(d_in=d_in, d_model=meta.get("d_model",128), nhead=meta.get("n_heads",8),
                                  num_layers=meta.get("n_layers",4), dim_feedforward=meta.get("ffn",256),
                                  dropout=0.1, task_type="cls", num_classes=num_classes,
                                  max_len=len(df_s)+len(df_q)).to(device)
        model.load_state_dict(ckpt["model"], strict=False)  # allow head-size differences
        model.eval()
        packed = pack_task_classification(Xs, ys, Xq, yq if yq is not None else np.zeros(len(Xq), dtype=int),
                                          num_classes=num_classes)
        batch = batchify([packed])
        tokens = batch["tokens"].to(device); attn = batch["attn_mask"].to(device)
        with torch.no_grad():
            out = model(tokens, attn)["out"][0]
        is_query = batch["is_query"][0].cpu().numpy()
        logits = out[is_query, :].cpu().numpy()
        pred_ids = logits.argmax(axis=-1)
        inv = pre.lbl.inverse_transform(pred_ids.astype(int))
        df_out = df_q.copy(); df_out["prediction"] = inv
        if yq is not None:
            df_out["target_true"] = pre.lbl.inverse_transform(yq.astype(int))
    else:
        d_in = n_bins + 1 + 1
        model = TransformerTabPFN(d_in=d_in, d_model=meta.get("d_model",128), nhead=meta.get("n_heads",8),
                                  num_layers=meta.get("n_layers",4), dim_feedforward=meta.get("ffn",256),
                                  dropout=0.1, task_type="reg", num_classes=None,
                                  max_len=len(df_s)+len(df_q)).to(device)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        packed = pack_task_regression(Xs, ys, Xq, yq if yq is not None else np.zeros(len(Xq), dtype=float))
        batch = batchify([packed])
        tokens = batch["tokens"].to(device); attn = batch["attn_mask"].to(device)
        with torch.no_grad():
            out = model(tokens, attn)["out"][0]
        is_query = batch["is_query"][0].cpu().numpy()
        preds = out[is_query].cpu().numpy()
        df_out = df_q.copy(); df_out["prediction"] = preds
        if yq is not None:
            df_out["target_true"] = yq

    out_path = args.out or f"results/predictions_hashed_{int(time.time())}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"[OK] Wrote predictions to {out_path} (n_bins={n_bins})")

if __name__ == "__main__":
    main()