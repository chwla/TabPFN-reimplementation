from __future__ import annotations
import argparse, random, numpy as np, torch, os, time
from typing import Literal, Optional
from src.synth.tasks import make_classification_task, make_regression_task
from src.packing.pack import pack_task_classification, pack_task_regression, batchify
from src.model.transformer import TransformerTabPFN
import torch.nn as nn

def _hash_to_bucket_int(i: int, n_bins: int, seed: int = 0):
    bucket = (i + seed) % n_bins
    sign = 1 if ((i + seed) % 2 == 0) else -1
    return bucket, sign

def hash_numpy_numeric(X: np.ndarray, n_bins: int, seed: int = 0) -> np.ndarray:
    X = X.astype(np.float32, copy=True)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True); sd[sd == 0] = 1.0
    X = (X - mu) / sd
    H = np.zeros((X.shape[0], n_bins), dtype=np.float32)
    for j in range(X.shape[1]):
        b, s = _hash_to_bucket_int(j, n_bins, seed)
        H[:, b] += s * X[:, j]
    return H

def train_week2_hashed(
    task_type: Literal["cls","reg"] = "reg",
    n_bins: int = 128,
    n_features_max: int = 50,
    n_support: int = 128,
    n_query: int = 128,
    steps: int = 30000,
    batch_tasks: int = 16,
    d_model: int = 128, n_layers: int = 4, n_heads: int = 8, ffn: int = 256, dropout: float = 0.1,
    lr: float = 1e-3, weight_decay: float = 1e-2, grad_clip: float = 1.0,
    device: Optional[str] = None, seed: int = 42, log_every: int = 200,
    ckpt_path: str = "results/checkpoints/week2_hashed.pt",
) -> None:
    rng = np.random.default_rng(seed); random.seed(seed); torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    if task_type == "cls":
        num_classes = 3
        d_in = n_bins + num_classes + 1
    else:
        num_classes = None
        d_in = n_bins + 1 + 1

    model = TransformerTabPFN(
        d_in=d_in, d_model=d_model, nhead=n_heads, num_layers=n_layers,
        dim_feedforward=ffn, dropout=dropout, task_type=task_type, num_classes=num_classes,
        max_len=n_support + n_query
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100) if task_type == "cls" else nn.MSELoss()

    t0 = time.time(); run = 0.0

    for step in range(1, steps + 1):
        packed = []
        for _ in range(batch_tasks):
            if task_type == "cls":
                f = int(rng.integers(5, n_features_max+1))
                rule = random.choice(["blobs","linear","xor","circles"])
                t = make_classification_task(n_support=n_support, n_query=n_query, n_features=f, classes=3, rule=rule, random_state=int(rng.integers(1e9)))
                Xs = hash_numpy_numeric(t.X_support, n_bins=n_bins, seed=seed)
                Xq = hash_numpy_numeric(t.X_query,   n_bins=n_bins, seed=seed)
                p = pack_task_classification(Xs, t.y_support, Xq, t.y_query, num_classes=3)
            else:
                f = int(rng.integers(5, n_features_max+1))
                kind = random.choice(["linear","poly","sin","piecewise"])
                t = make_regression_task(n_support=n_support, n_query=n_query, n_features=f, kind=kind, random_state=int(rng.integers(1e9)))
                Xs = hash_numpy_numeric(t.X_support, n_bins=n_bins, seed=seed)
                Xq = hash_numpy_numeric(t.X_query,   n_bins=n_bins, seed=seed)
                p = pack_task_regression(Xs, t.y_support, Xq, t.y_query)
            packed.append(p)

        batch = batchify(packed)
        tokens = batch["tokens"].to(device); target = batch["target"].to(device)
        is_query = batch["is_query"].to(device); attn = batch["attn_mask"].to(device)
        model.train(); opt.zero_grad(set_to_none=True)
        out = model(tokens, attn)["out"]

        loss = loss_fn(out[is_query], (target.long() if task_type=="cls" else target)[is_query])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        run += float(loss.item())
        if step % log_every == 0:
            avg = run / log_every; run = 0.0
            elapsed = (time.time() - t0) / 60
            print(f"[{step:6d}] loss={avg:.4f} | mins={elapsed:.1f}")

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model": model.state_dict(),
                "meta": {
                    "task_type": task_type,
                    "hash_bins": n_bins,
                    "d_in": d_in,
                    "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads, "ffn": ffn
                }}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")