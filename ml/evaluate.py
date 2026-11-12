import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from ml.config import DATA_PATH, FIGURES_DIR, MODELS_DIR, RANDOM_STATE
from ml.preprocess import load_and_prepare


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=150)
    plt.close(fig)
    print("saved", FIGURES_DIR / name)


def main(args):
    with open(MODELS_DIR / "metrics.json") as f:
        data = json.load(f)
    models = data["models"]

    # 1. ROC curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in models.values():
        ax.plot(m["roc_curve"]["fpr"], m["roc_curve"]["tpr"],
                label=f"{m['display_name']} (AUC={m['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves")
    ax.legend(loc="lower right")
    save(fig, "roc_curves.png")

    # 2. Precision-Recall curves (more informative under heavy imbalance)
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in models.values():
        ax.plot(m["pr_curve"]["recall"], m["pr_curve"]["precision"],
                label=f"{m['display_name']} (AP={m['pr_auc']:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    ax.legend(loc="upper right")
    save(fig, "pr_curves.png")

    # 3. Confusion matrices
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.8))
    for ax, m in zip(np.atleast_1d(axes), models.values()):
        cm = np.array(m["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set(title=m["display_name"], xticks=[0, 1], yticks=[0, 1],
               xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"],
               xlabel="Predicted", ylabel="Actual")
    save(fig, "confusion_matrices.png")

    # 4. Metric comparison
    names = [m["display_name"] for m in models.values()]
    metric_keys = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    x = np.arange(len(names))
    w = 0.16
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, k in enumerate(metric_keys):
        ax.bar(x + (i - 2) * w, [m[k] for m in models.values()], w, label=k)
    ax.set_xticks(x, names, rotation=10)
    ax.set(ylim=(0, 1.25), title="Model Comparison")
    ax.legend(ncol=5, loc="upper center")
    save(fig, "metric_comparison.png")

    # 5. Bias-variance view: train F1 vs test F1
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - 0.2, [m["train_f1"] for m in models.values()], 0.4, label="Train F1")
    ax.bar(x + 0.2, [m["f1"] for m in models.values()], 0.4, label="Test F1")
    ax.set_xticks(x, names, rotation=10)
    ax.set(ylim=(0, 1.05), title="Bias-Variance: Train vs Test F1 "
                                 "(low both = high bias, big gap = high variance)")
    ax.title.set_fontsize(9)
    ax.legend()
    save(fig, "bias_variance.png")

    # 6. Feature importance (decision tree)
    fi = data["feature_importance"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(list(fi.keys())[::-1], list(fi.values())[::-1], color="#2a6fdb")
    ax.set(title="Decision Tree Feature Importance (top 15)")
    save(fig, "feature_importance.png")

    # 7. Model-complexity curve: Decision Tree depth sweep (bias-variance tradeoff)
    if not args.skip_depth:
        print("Running decision-tree depth sweep ...")
        d = load_and_prepare(args.data, apply_smote=True, save_scaler=False)
        rng = np.random.RandomState(RANDOM_STATE)
        n = min(100000, len(d["X_res"]))
        idx = rng.choice(len(d["X_res"]), n, replace=False)
        Xr, yr = d["X_res"].iloc[idx], d["y_res"].iloc[idx]
        ne = min(50000, len(d["X_train"]))
        ei = rng.choice(len(d["X_train"]), ne, replace=False)
        Xe, ye = d["X_train"].iloc[ei], d["y_train"].iloc[ei]
        depths = [1, 2, 3, 4, 6, 8, 10, 14, 18, 25]
        tr, te = [], []
        for dep in depths:
            t = DecisionTreeClassifier(criterion="entropy", max_depth=dep,
                                       random_state=RANDOM_STATE).fit(Xr, yr)
            tr.append(f1_score(ye, t.predict(Xe)))
            te.append(f1_score(d["y_test"], t.predict(d["X_test"])))
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(depths, tr, "o-", label="Train F1")
        ax.plot(depths, te, "s-", label="Test F1")
        ax.set(xlabel="max_depth", ylabel="F1", title="Decision Tree: Underfit -> Overfit")
        ax.legend()
        save(fig, "decision_tree_depth_curve.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DATA_PATH)
    p.add_argument("--skip-depth", action="store_true")
    main(p.parse_args())
