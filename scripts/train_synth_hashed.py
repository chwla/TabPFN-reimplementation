from __future__ import annotations
import argparse
from src.train_hashed import train_week2_hashed  # ensures module is importable

def main():
    p = argparse.ArgumentParser(description="Train a universal (hashed) checkpoint usable for any CSV.")
    p.add_argument("--task_type", choices=["cls","reg"], default="reg")
    p.add_argument("--n_bins", type=int, default=128)
    p.add_argument("--n_features_max", type=int, default=50)
    p.add_argument("--n_support", type=int, default=128)
    p.add_argument("--n_query", type=int, default=128)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch_tasks", type=int, default=16)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--ffn", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--ckpt_path", type=str, default="results/checkpoints/week2_hashed.pt")
    args = p.parse_args()

    train_week2_hashed(
        task_type=args.task_type, n_bins=args.n_bins, n_features_max=args.n_features_max,
        n_support=args.n_support, n_query=args.n_query, steps=args.steps, batch_tasks=args.batch_tasks,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, ffn=args.ffn, dropout=args.dropout,
        lr=args.lr, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        device=args.device, seed=args.seed, log_every=args.log_every, ckpt_path=args.ckpt_path
    )

if __name__ == "__main__":
    main()