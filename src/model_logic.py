import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with hour, dow, and lag features created when missing."""
    d = df.copy()

    # Normalize timestamp to build temporal features if needed.
    ts = None
    for col in ["timestamp", "utc_ts"]:
        if col in d.columns:
            ts = pd.to_datetime(d[col], errors="coerce")
            break
    if ts is not None:
        d["_ts"] = ts

    # Hour feature.
    if "hour" not in d.columns:
        if "_ts" in d.columns:
            d["hour"] = d["_ts"].dt.hour
        else:
            raise KeyError("Column 'hour' missing and no timestamp to derive it.")

    # Day-of-week feature (0=Mon).
    if "dow" not in d.columns:
        if "day_of_week" in d.columns:
            d["dow"] = d["day_of_week"]
        elif "_ts" in d.columns:
            d["dow"] = d["_ts"].dt.dayofweek
        else:
            raise KeyError("Column 'dow' missing and no timestamp to derive it.")

    # Lag features.
    for name, lag in [("lag_1", 1), ("lag_3", 3), ("lag_6", 6)]:
        if name not in d.columns:
            d[name] = d["total_vehicles"].shift(lag)

    # Drop rows with missing inputs/target to keep training clean.
    d = d.dropna(subset=["hour", "dow", "lag_1", "lag_3", "lag_6", "total_vehicles"])
    return d


def get_X_y(df):
    """
    Features (X):
    - hour
    - dow
    - lag_1
    - lag_3
    - lag_6

    Target (y):
    - total_vehicles
    """
    d = _ensure_features(df)
    X = d[["hour", "dow", "lag_1", "lag_3", "lag_6"]]
    y = d["total_vehicles"]
    return X, y


def train_random_forest(X, y):
    """
    Train a tuned Random Forest model
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    return model, mae, X_test, y_test
