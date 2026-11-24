"""Find REAL held-out transactions that each model scores as MEDIUM risk.

Run once, after training:   python -m ml.make_borderline_samples
Creates data/borderline_samples.csv, used by the "Borderline behaviour" option in
the Payment Simulator tab. Takes a few minutes (SVC scoring is the slow part).

Rows come from the held-out test split (never trained on). A row counts as
borderline for a model if its fraud probability is 35%-65%, safely inside the
app's MEDIUM band (30%-70%).
"""
import time

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config import (DATA_PATH, DISPLAY_NAMES, FEATURES, MODEL_NAMES,
                       MODELS_DIR, RANDOM_STATE)

LOW, HIGH = 0.35, 0.65   # margin inside the 0.30-0.70 MEDIUM band


def score(model, scaler, X):
    scaled = pd.DataFrame(scaler.transform(X[FEATURES]), columns=FEATURES)
    return model.predict_proba(scaled)[:, 1]


def main():
    df = pd.read_csv(DATA_PATH)
    # Same split as training (same params + seed) -> these rows were never trained on
    _, test = train_test_split(df, test_size=0.2, stratify=df["Class"],
                               random_state=RANDOM_STATE)
    test = test.reset_index(drop=True)
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    models = {n: joblib.load(MODELS_DIR / f"{n}.pkl")
              for n in MODEL_NAMES if (MODELS_DIR / f"{n}.pkl").exists()}
    print(f"Scoring {len(test)} held-out transactions with {len(models)} models ...")

    for name, model in models.items():
        t0 = time.time()
        p = score(model, scaler, test)
        test[f"p_{name}"] = p.round(4)
        test[f"medium_{name}"] = ((p >= LOW) & (p < HIGH)).astype(int)
        print(f"  {DISPLAY_NAMES[name]:26} {time.time() - t0:4.0f}s   "
              f"borderline rows: {int(test[f'medium_{name}'].sum())}")

    keep = test[[f"medium_{n}" for n in models]].sum(axis=1) > 0
    out = test[keep]
    path = DATA_PATH.parent / "borderline_samples.csv"
    out.to_csv(path, index=False)
    print(f"\nSaved {len(out)} rows to {path}")
    for name in models:
        if out[f"medium_{name}"].sum() == 0:
            print(f"  note: no borderline rows for {DISPLAY_NAMES[name]}")


if __name__ == "__main__":
    main()
