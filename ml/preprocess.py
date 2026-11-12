import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.config import DATA_PATH, FEATURES, MODELS_DIR, RANDOM_STATE


def load_and_prepare(data_path=DATA_PATH, apply_smote=True, save_scaler=True):
    df = pd.read_csv(data_path)
    missing = [c for c in FEATURES + ["Class"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    X, y = df[FEATURES], df["Class"]

    # 1) Split FIRST so synthetic samples never leak into the test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 2) Fit scaler on training data only
    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=FEATURES)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=FEATURES)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    if save_scaler:
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    # 3) SMOTE on training data only
    X_res = y_res = None
    if apply_smote:
        X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "X_res": X_res, "y_res": y_res,
        "n_rows": len(df), "n_fraud": int(y.sum()),
    }
