from __future__ import annotations
import os, time, math, random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Literal, Optional
from src.synth.tasks import make_classification_task, make_regression_task
from src.packing.pack import pack_task_classification, pack_task_regression, batchify
from src.model.transformer import TransformerTabPFN
from src.utils.logger import log_result

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate_once(model, device, task_type: str, num_classes: int, n_support: int, n_query: int, n_features: int, seed: int = 123):
    # quick sanity eval on one synthetic task
    if task_type == "cls":
        t = make_classification_task(n_support=n_support, n_query=n_query, n_features=n_features, classes=num_classes, rule="linear", random_state=seed)
        p = pack_task_classification(t.X_support, t.y_support, t.X_query, t.y_query, num_classes=num_classes)
        batch = batchify([p])
        tokens = batch["tokens"].to(device)
        attn = batch["attn_mask"].to(device)
        out = model(tokens, attn)["out"]  # (1, T, C)
        # only query positions
        is_query = batch["is_query"][0].cpu().numpy()
        logits = out[0, is_query, :].cpu()
        y = torch.tensor(t.y_query, dtype=torch.long)
        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean().item()
        return {"acc": acc}
    else:
        t = make_regression_task(n_support=n_support, n_query=n_query, n_features=n_features, kind="linear", random_state=seed)
        p = pack_task_regression(t.X_support, t.y_support, t.X_query, t.y_query)
        batch = batchify([p])
        tokens = batch["tokens"].to(device)
        attn = batch["attn_mask"].to(device)
        out = model(tokens, attn)["out"][0]  # (T,)
        is_query = batch["is_query"][0].cpu().numpy()
        pred = out[is_query].cpu().numpy()
        rmse = float(np.sqrt(((pred - t.y_query)**2).mean()))
        return {"rmse": rmse}

def train_week2(
    # data/task
    task_type: Literal["cls","reg"] = "cls",
    num_classes: int = 3,
    n_features: int = 10,
    n_support: int = 128,
    n_query: int = 128,
    # model
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 8,
    ffn: int = 256,
    dropout: float = 0.1,
    # opt
    steps: int = 10000,
    batch_tasks: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    label_smoothing: float = 0.0,
    # misc
    device: Optional[str] = None,
    seed: int = 42,
    log_every: int = 100,
    ckpt_path: str = "results/checkpoints/week2.pt",
) -> None:
    """
    Train a small Transformer on synthetic tasks (Week 2). Generates tasks on the fly.
    """
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Input dim depends on task type
    if task_type == "cls":
        d_in = n_features + num_classes + 1  # features + onehot(label) + known_flag
    else:
        d_in = n_features + 1 + 1            # features + scalar label + known_flag

    model = TransformerTabPFN(
        d_in=d_in, d_model=d_model, nhead=n_heads, num_layers=n_layers,
        dim_feedforward=ffn, dropout=dropout, task_type=task_type, num_classes=(num_classes if task_type=="cls" else None),
        max_len=n_support + n_query
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if task_type == "cls":
        ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=-100)
    else:
        mse = nn.MSELoss()

    t0 = time.time()
    running = 0.0

    for step in range(1, steps + 1):
        packed_batch = []
        for _ in range(batch_tasks):
            if task_type == "cls":
                rule = random.choice(["blobs", "linear"])  # keep it simple for Week 2
                t = make_classification_task(n_support=n_support, n_query=n_query, n_features=n_features, classes=num_classes, rule=rule, random_state=random.randint(0, 10_000))
                p = pack_task_classification(t.X_support, t.y_support, t.X_query, t.y_query, num_classes=num_classes)
            else:
                kind = random.choice(["linear", "poly"])
                t = make_regression_task(n_support=n_support, n_query=n_query, n_features=n_features, kind=kind, random_state=random.randint(0, 10_000))
                p = pack_task_regression(t.X_support, t.y_support, t.X_query, t.y_query)
            packed_batch.append(p)

        batch = batchify(packed_batch)
        tokens = batch["tokens"].to(device)          # (B, T, D_in)
        target = batch["target"].to(device)          # (B, T) float; cast later
        is_query = batch["is_query"].to(device)      # (B, T) bool
        attn = batch["attn_mask"].to(device)         # (B, T) bool

        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(tokens, attn)["out"]             # (B, T, C) or (B, T)

        if task_type == "cls":
            logits = out[is_query]                   # (Nq, C)
            y = target.long()[is_query]              # (Nq,)
            loss = ce(logits, y)
        else:
            preds = out[is_query]                    # (Nq,)
            y = target[is_query]                     # (Nq,)
            loss = mse(preds, y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        running += float(loss.item())
        if step % log_every == 0:
            avg = running / log_every
            running = 0.0
            elapsed = time.time() - t0
            if task_type == "cls":
                eval_metrics = evaluate_once(model, device, task_type, num_classes, n_support, n_query, n_features)
                print(f"[{step:6d}] loss={avg:.4f} | acc@probe={eval_metrics['acc']:.3f} | {elapsed/60:.1f} min")
                log_result({
                    "phase": "train",
                    "step": step,
                    "task_type": "classification",
                    "loss": round(avg, 6),
                    "probe_acc": round(eval_metrics["acc"], 6),
                    "elapsed_min": round(elapsed/60, 3)
                })
            else:
                eval_metrics = evaluate_once(model, device, task_type, num_classes, n_support, n_query, n_features)
                print(f"[{step:6d}] loss={avg:.4f} | rmse@probe={eval_metrics['rmse']:.3f} | {elapsed/60:.1f} min")
                log_result({
                    "phase": "train",
                    "step": step,
                    "task_type": "regression",
                    "loss": round(avg, 6),
                    "probe_rmse": round(eval_metrics["rmse"], 6),
                    "elapsed_min": round(elapsed/60, 3)
                })

        # Save a checkpoint halfway and at the end
        if step in {steps//2, steps}:
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save({"model": model.state_dict(),
                        "meta": {
                            "task_type": task_type,
                            "num_classes": num_classes,
                            "n_features": n_features,
                            "d_in": d_in,
                            "d_model": d_model,
                            "n_layers": n_layers,
                            "n_heads": n_heads,
                            "ffn": ffn,
                        }}, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")