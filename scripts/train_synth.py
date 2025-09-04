from __future__ import annotations
import argparse
from src.train import train_week2

def main():
    p = argparse.ArgumentParser(description="Week 2 training: Transformer on synthetic support+query tasks.")
    p.add_argument("--task_type", choices=["cls","reg"], default="cls")
    p.add_argument("--classes", type=int, default=3, help="num classes (classification)")
    p.add_argument("--n_features", type=int, default=10)
    p.add_argument("--n_support", type=int, default=128)
    p.add_argument("--n_query", type=int, default=128)

    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--ffn", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_tasks", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--ckpt_path", type=str, default="results/checkpoints/week2.pt")
    args = p.parse_args()

    train_week2(
        task_type=args.task_type,
        num_classes=args.classes,
        n_features=args.n_features,
        n_support=args.n_support,
        n_query=args.n_query,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn=args.ffn,
        dropout=args.dropout,
        steps=args.steps,
        batch_tasks=args.batch_tasks,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        label_smoothing=args.label_smoothing,
        device=args.device,
        seed=args.seed,
        log_every=args.log_every,
        ckpt_path=args.ckpt_path,
    )

if __name__ == "__main__":
    main()