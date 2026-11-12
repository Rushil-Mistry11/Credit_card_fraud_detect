"""Create a SYNTHETIC dataset with the same schema as the Kaggle file.

Only for testing the pipeline before you download the real data.
Usage:  python -m ml.make_demo_data   ->  data/creditcard_demo.csv
Then point the project at it:
    Windows (PowerShell):  $env:FRAUD_DATA="data/creditcard_demo.csv"
    macOS/Linux:           export FRAUD_DATA=data/creditcard_demo.csv
"""
import numpy as np
import pandas as pd

from ml.config import FEATURES, ROOT

rng = np.random.RandomState(0)
n, fraud_rate = 30000, 0.015
y = (rng.rand(n) < fraud_rate).astype(int)
X = rng.randn(n, 28)
X[y == 1] += rng.randn(28) * 1.5        # fraud rows drift away from the legit cloud
time_col = np.sort(rng.randint(0, 172800, n)).astype(float)
amount = np.abs(rng.lognormal(3.5, 1.2, n))

df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
df.insert(0, "Time", time_col)
df["Amount"] = amount
df["Class"] = y
df = df[FEATURES + ["Class"]]
out = ROOT / "data" / "creditcard_demo.csv"
df.to_csv(out, index=False)
print(f"Wrote {out}  rows={len(df)}  fraud={y.sum()}")
