from __future__ import annotations
import argparse
from src.synth.tasks import make_classification_task, make_regression_task
from src.baselines.xgb import run_xgb
from src.utils.logger import log_result

def main():
    parser = argparse.ArgumentParser(description="Run XGBoost on a synthetic task and log results.")
    parser.add_argument("--task_type", choices=["cls", "reg"], default="cls")
    parser.add_argument("--rule", choices=["blobs", "linear", "xor", "circles"], default="blobs",
                        help="Classification rule (if task_type=cls)")
    parser.add_argument("--kind", choices=["linear", "poly", "sin", "piecewise"], default="linear",
                        help="Regression kind (if task_type=reg)")
    parser.add_argument("--n_support", type=int, default=128)
    parser.add_argument("--n_query", type=int, default=128)
    parser.add_argument("--n_features", type=int, default=10)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--flip_y", type=float, default=0.05)
    parser.add_argument("--missing_prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.task_type == "cls":
        task = make_classification_task(
            n_support=args.n_support, n_query=args.n_query, n_features=args.n_features,
            classes=args.classes, rule=args.rule, noise=args.noise, flip_y=args.flip_y,
            missing_prob=args.missing_prob, random_state=args.seed
        )
        res = run_xgb(task.X_support, task.y_support, task.X_query, task.y_query, task_type="cls", random_state=args.seed)
        print("Metrics:", res["metrics"])
        log_result({
            "dataset": f"synth_cls::{args.rule}",
            "task_type": "classification",
            "n_train": len(task.y_support),
            "n_val": len(task.y_query),
            "metric": list(res["metrics"].keys())[0] if res["metrics"] else "n/a",
            "metric_value": list(res["metrics"].values())[0] if res["metrics"] else "n/a",
            "elapsed_s": f"{res['elapsed_s']:.3f}",
            "notes": f"features={args.n_features}; classes={args.classes}; noise={args.noise}; flip_y={args.flip_y}; missing={args.missing_prob}"
        })
    else:
        task = make_regression_task(
            n_support=args.n_support, n_query=args.n_query, n_features=args.n_features,
            kind=args.kind, noise=args.noise, missing_prob=args.missing_prob, random_state=args.seed
        )
        res = run_xgb(task.X_support, task.y_support, task.X_query, task.y_query, task_type="reg", random_state=args.seed)
        print("Metrics:", res["metrics"])
        log_result({
            "dataset": f"synth_reg::{args.kind}",
            "task_type": "regression",
            "n_train": len(task.y_support),
            "n_val": len(task.y_query),
            "metric": list(res["metrics"].keys())[0] if res["metrics"] else "n/a",
            "metric_value": list(res["metrics"].values())[0] if res["metrics"] else "n/a",
            "elapsed_s": f"{res['elapsed_s']:.3f}",
            "notes": f"features={args.n_features}; noise={args.noise}; missing={args.missing_prob}"
        })

if __name__ == "__main__":
    main()