"""
A tiny "hello ML" PyTorch script:
- Builds a synthetic linear regression dataset
- Trains a 2-layer MLP for a few epochs
- Prints the train loss to show the setup works
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def make_toy_regression(n=512, d=10, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=(d,)).astype(np.float32)
    y = X @ w + noise * rng.normal(size=(n,)).astype(np.float32)
    return X, y

class MLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def main():
    X, y = make_toy_regression(n=1024, d=10, noise=0.2, seed=123)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    model = MLP(d_in=X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.MSELoss()

    model.train()
    t0 = time.time()
    for epoch in range(5):
        running = 0.0
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1} | MSE: {running/len(ds):.4f}")
    print(f"Finished in {time.time()-t0:.2f}s. ✅ If loss decreased, PyTorch is working.")

if __name__ == "__main__":
    main()