# TabPFN-Style — Week 1 Skeleton (Synthetic Tasks + XGBoost Baseline)

Beginner-friendly repo scaffold for **Week 1**:
- Synthetic task generator (classification + regression)
- XGBoost baseline runner
- Simple metrics + CSV logger
- "Hello PyTorch" sanity script

## 📦 Layout

```
tabpfn_synth_week1/
├─ src/
│  ├─ baselines/
│  │  └─ xgb.py
│  ├─ synth/
│  │  └─ tasks.py
│  └─ utils/
│     ├─ metrics.py
│     └─ logger.py
├─ scripts/
│  ├─ run_xgb_on_synth.py
│  ├─ hello_torch.py
│  └─ smoke_test.py
├─ data/              # (empty; place real datasets here later)
├─ experiments/       # (empty; store experiment configs here)
├─ results/           # (empty; logs and outputs will appear here)
├─ requirements.txt
├─ .gitignore
└─ LICENSE
```

## 🧰 Setup (one-time)

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## ✅ Smoke test: everything wired up

This trains a tiny PyTorch model on a toy regression **and** runs the XGBoost baseline on a synthetic classification task, then logs to `results/run_log.csv`.

```bash
python -m scripts.smoke_test
```

## ▶️ Run the XGBoost baseline on synthetic data

Classification:
```bash
python -m scripts.run_xgb_on_synth --task_type cls --rule blobs --n_support 128 --n_query 128 --n_features 10 --classes 3
```

Regression:
```bash
python -m scripts.run_xgb_on_synth --task_type reg --kind linear --n_support 128 --n_query 128 --n_features 10
```

## 🧪 Week 1 synthetic tasks

- **Classification**: Gaussian blobs, linear rules, XOR, circles  
- **Regression**: linear, polynomial, sinusoid, piecewise linear  
- Options: label noise, class imbalance, missing values.

Each call returns a **support set** (few labeled rows) and a **query set** (rows to predict).

---

**Glossary**
- **Support set** = labeled examples your model sees.
- **Query set** = rows to predict.
- **Baseline** = reference model (XGBoost) you’ll compare against.