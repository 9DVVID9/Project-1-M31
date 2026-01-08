import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

DATA_PATH = Path("TRAFFIC_CLEANED.csv")
MODEL_PATH = Path("model_random_forest.pkl")
FEATURE_COLUMNS = ["hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos"]
LABEL_COLUMN = "traffic_level"


def train_random_forest() -> None:
    """Train a Random Forest classifier on the cleaned traffic dataset."""
    print("--- MODEL 4: RANDOM FOREST ---")

    if not DATA_PATH.exists():
        raise FileNotFoundError("Missing TRAFFIC_CLEANED.csv. Run src/clean_data.py first.")

    df = pd.read_csv(DATA_PATH)

    # Add cyclic encoding for hour to capture wrap-around smoothly.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset missing required feature columns: {missing_features}")
    if LABEL_COLUMN not in df.columns:
        raise ValueError("Dataset missing 'traffic_level' column. Re-run src/clean_data.py.")

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Light parameter search to pick a stronger forest without heavy tuning.
    candidates = [
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": None},
        {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced"},
        {"n_estimators": 300, "max_depth": 14, "min_samples_leaf": 1, "max_features": None, "class_weight": None},
        {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 2, "max_features": "log2", "class_weight": "balanced"},
    ]

    best_params = None
    best_cv = -1.0
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for params in candidates:
        model = RandomForestClassifier(random_state=42, **params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        mean_score = scores.mean()
        print(f"CV accuracy {mean_score * 100:.1f}% with params {params}")
        if mean_score > best_cv:
            best_cv = mean_score
            best_params = params

    assert best_params is not None, "No model candidates evaluated."
    final_model = RandomForestClassifier(random_state=42, **best_params)
    final_model.fit(X_train, y_train)

    preds = final_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Best params: {best_params}")
    print(f"Holdout accuracy: {acc * 100:.1f}%")

    joblib.dump(final_model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train_random_forest()
