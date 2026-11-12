import argparse
import json
import time

import joblib
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml.config import (DATA_PATH, DECISION_THRESHOLD, DISPLAY_NAMES, FEATURES,
                       MODELS_DIR, RANDOM_STATE)
from ml.preprocess import load_and_prepare


def downsample_curve(x, y, n=300):
    """Thin a curve while keeping its shape (even spacing along the curve length)."""
    x, y = np.asarray(x), np.asarray(y)
    d = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    idx = np.unique(np.searchsorted(d, np.linspace(0, d[-1], n)).clip(0, len(x) - 1))
    return [round(float(v), 5) for v in x[idx]], [round(float(v), 5) for v in y[idx]]


def evaluate(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    return proba, pred


def main(args):
    MODELS_DIR.mkdir(exist_ok=True)
    print("Loading data, splitting, scaling, applying SMOTE ...")
    d = load_and_prepare(args.data)
    X_train, y_train, X_test, y_test = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    X_res, y_res = d["X_res"], d["y_res"]
    print(f"  rows={d['n_rows']}  fraud={d['n_fraud']}  "
          f"train(before SMOTE)={len(X_train)}  train(after SMOTE)={len(X_res)}")

    # SVC training is O(n^2)-O(n^3): train it on a random subsample of the SMOTE'd set
    n_svc = min(args.svc_samples, len(X_res))
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_res), n_svc, replace=False)
    X_svc, y_svc = X_res.iloc[idx], y_res.iloc[idx]

    # Train-set subsample (original distribution) used for the bias/variance gap
    n_eval = min(50000, len(X_train))
    eval_idx = rng.choice(len(X_train), n_eval, replace=False)
    X_tr_eval, y_tr_eval = X_train.iloc[eval_idx], y_train.iloc[eval_idx]

    models = {
        "svc": (SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                    random_state=RANDOM_STATE), X_svc, y_svc),
        # criterion="entropy" -> information-gain splits (ID3/C4.5 style; sklearn uses CART)
        "decision_tree": (DecisionTreeClassifier(criterion="entropy", max_depth=10,
                                                 random_state=RANDOM_STATE), X_res, y_res),
        "naive_bayes": (GaussianNB(), X_res, y_res),
        "mlp": (MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                              max_iter=200, early_stopping=True,
                              random_state=RANDOM_STATE), X_res, y_res),
    }

    results = {}
    for name, (model, Xt, yt) in models.items():
        print(f"\nTraining {DISPLAY_NAMES[name]} on {len(Xt)} rows ...")
        t0 = time.time()
        model.fit(Xt, yt)
        train_time = time.time() - t0

        proba, pred = evaluate(model, X_test, y_test)
        _, tr_pred = evaluate(model, X_tr_eval, y_tr_eval)
        fpr, tpr, _ = roc_curve(y_test, proba)
        prec, rec, _ = precision_recall_curve(y_test, proba)

        fpr, tpr = downsample_curve(fpr, tpr)
        pr_rec, pr_prec = downsample_curve(rec, prec)

        results[name] = {
            "display_name": DISPLAY_NAMES[name],
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "pr_auc": float(average_precision_score(y_test, proba)),
            "train_f1": float(f1_score(y_tr_eval, tr_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "train_time_sec": round(train_time, 1),
            "train_rows": int(len(Xt)),
            "roc_curve": {"fpr": fpr, "tpr": tpr},
            "pr_curve": {"recall": pr_rec, "precision": pr_prec},
        }
        r = results[name]
        print(f"  P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
              f"ROC-AUC={r['roc_auc']:.3f} PR-AUC={r['pr_auc']:.3f} "
              f"(train F1={r['train_f1']:.3f}, {train_time:.0f}s)")
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")

    # Global feature importance from the decision tree (for the dashboard)
    tree = models["decision_tree"][0]
    imp = sorted(zip(FEATURES, tree.feature_importances_), key=lambda t: -t[1])[:15]
    payload = {
        "meta": {"n_rows": d["n_rows"], "n_fraud": d["n_fraud"],
                 "fraud_pct": round(100 * d["n_fraud"] / d["n_rows"], 4),
                 "train_rows_after_smote": int(len(X_res)),
                 "test_rows": int(len(X_test)),
                 "threshold": DECISION_THRESHOLD},
        "models": results,
        "feature_importance": {k: round(float(v), 5) for k, v in imp},
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("\nSaved models + models/metrics.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DATA_PATH)
    p.add_argument("--svc-samples", type=int, default=40000)
    main(p.parse_args())
