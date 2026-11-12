import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv("FRAUD_DATA", ROOT / "data" / "creditcard.csv"))
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
DB_PATH = ROOT / "fraud_logs.db"

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
MODEL_NAMES = ["svc", "decision_tree", "naive_bayes", "mlp"]
DISPLAY_NAMES = {
    "svc": "SVC (RBF)",
    "decision_tree": "Decision Tree (Entropy)",
    "naive_bayes": "Naive Bayes",
    "mlp": "MLP (ANN)",
}

RANDOM_STATE = 42
DECISION_THRESHOLD = 0.50   # probability >= this -> flagged as fraud
LOW_THRESHOLD = 0.30        # < 0.30 LOW, 0.30-0.70 MEDIUM, >= 0.70 HIGH
HIGH_THRESHOLD = 0.70
