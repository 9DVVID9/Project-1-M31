import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split

BASE_FEATURE_COLUMNS = ["hour", "day_of_week", "is_weekend"]
CYCLIC_FEATURE_COLUMNS = ["hour_sin", "hour_cos"]
CLASS_LABEL = "traffic_level"
REG_LABEL = "total_vehicles"
DATA_FILE = Path("TRAFFIC_CLEANED.csv")

DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
DAY_TO_INDEX = {name: idx for idx, name in enumerate(DAY_NAMES)}

MODEL_REGISTRY = [
    ("Linear Regression", Path("model_lin_reg.pkl"), "regression"),
    ("KNN Classifier", Path("model_knn.pkl"), "classification"),
    ("Decision Tree", Path("model_decision_tree.pkl"), "classification"),
    ("Random Forest", Path("model_random_forest.pkl"), "classification"),
]


class ModelManager:
    """Load trained models and compute quick evaluation metrics."""

    def __init__(self) -> None:
        self.models: Dict[str, Dict[str, object]] = {}
        self.metrics: Dict[str, float] = {}
        self.messages: List[str] = []
        self.load_models()

    def load_models(self) -> None:
        self.models.clear()
        self.metrics.clear()
        self.messages = []

        for display_name, path, model_type in MODEL_REGISTRY:
            if not path.exists():
                self.messages.append(f"Missing model file: {path.name}")
                continue
            try:
                model = joblib.load(path)
            except Exception as exc:  # noqa: BLE001 - show friendly info in GUI
                self.messages.append(f"Failed to load {path.name}: {exc}")
                continue
            self.models[display_name] = {
                "model": model,
                "type": model_type,
                "path": path,
            }

        if not self.models:
            self.messages.append("No models loaded. Run the training scripts first.")
            return

        if not DATA_FILE.exists():
            self.messages.append("TRAFFIC_CLEANED.csv not found; metrics unavailable.")
            return

        try:
            df = pd.read_csv(DATA_FILE)
        except Exception as exc:  # noqa: BLE001 - surface read issues to the user
            self.messages.append(f"Could not read TRAFFIC_CLEANED.csv: {exc}")
            return

        # Ensure dataset has required columns before computing metrics.
        for column in BASE_FEATURE_COLUMNS + [CLASS_LABEL, REG_LABEL]:
            if column not in df.columns:
                self.messages.append(
                    f"Dataset missing required column '{column}'; metrics unavailable."
                )
                return

        df = self._add_cyclic_time(df)

        X_full = df[BASE_FEATURE_COLUMNS + CYCLIC_FEATURE_COLUMNS]
        y_class = df[CLASS_LABEL]
        y_reg = df[REG_LABEL]

        try:
            _, X_test_class, _, y_test_class = train_test_split(
                X_full,
                y_class,
                test_size=0.2,
                random_state=42,
                stratify=y_class,
            )
        except ValueError as exc:  # noqa: BLE001 - not enough samples for stratify
            self.messages.append(f"Not enough data for classification metric: {exc}")
            return

        try:
            _, X_test_reg, _, y_test_reg = train_test_split(
                X_full,
                y_reg,
                test_size=0.2,
                random_state=42,
            )
        except ValueError as exc:
            self.messages.append(f"Not enough data for regression metric: {exc}")
            return

        for display_name, info in self.models.items():
            model = info["model"]
            if info["type"] == "classification":
                aligned = self._align_features(X_test_class, model)
                preds = model.predict(aligned)
                self.metrics[display_name] = accuracy_score(y_test_class, preds)
            else:
                aligned = self._align_features(X_test_reg, model)
                preds = model.predict(aligned)
                self.metrics[display_name] = mean_absolute_error(y_test_reg, preds)

        self.messages.append("Models ready. Metrics computed where possible.")

    @staticmethod
    def _add_cyclic_time(df: pd.DataFrame) -> pd.DataFrame:
        if "hour" in df.columns:
            df = df.copy()
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        return df

    @staticmethod
    def _align_features(df: pd.DataFrame, model) -> pd.DataFrame:
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is None:
            return df
        return df.reindex(columns=feature_names)


class TrafficPredictorGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Traffic Predictor")

        self.manager = ModelManager()

        self.model_var = tk.StringVar()
        self.hour_var = tk.StringVar(value="12")
        self.day_var = tk.StringVar(value=DAY_NAMES[0])
        self.metric_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value=self._status_text())
        self.result_var = tk.StringVar(value="Pick a model and enter inputs.")

        self._build_layout()
        self._bind_events()
        self._populate_models()
        self._update_metric_label()

    def _build_layout(self) -> None:
        padding = {"padx": 10, "pady": 6}
        frame = ttk.Frame(self.root)
        frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        ttk.Label(frame, text="Model").grid(row=0, column=0, sticky="w", **padding)
        self.model_combo = ttk.Combobox(
            frame,
            textvariable=self.model_var,
            state="readonly",
            width=28,
        )
        self.model_combo.grid(row=0, column=1, sticky="ew", **padding)

        ttk.Label(frame, text="Hour (0-23)").grid(row=1, column=0, sticky="w", **padding)
        self.hour_spin = ttk.Spinbox(
            frame,
            from_=0,
            to=23,
            wrap=True,
            textvariable=self.hour_var,
            width=5,
        )
        self.hour_spin.grid(row=1, column=1, sticky="w", **padding)

        ttk.Label(frame, text="Day of Week").grid(row=2, column=0, sticky="w", **padding)
        self.day_combo = ttk.Combobox(
            frame,
            textvariable=self.day_var,
            values=DAY_NAMES,
            state="readonly",
            width=20,
        )
        self.day_combo.grid(row=2, column=1, sticky="w", **padding)

        ttk.Label(frame, textvariable=self.metric_var).grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="w",
            **padding,
        )

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=4, column=0, columnspan=2, sticky="ew", **padding)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        ttk.Button(button_frame, text="Predict", command=self._predict).grid(
            row=0, column=0, sticky="ew", padx=4
        )
        ttk.Button(button_frame, text="Refresh Models", command=self._refresh_models).grid(
            row=0, column=1, sticky="ew", padx=4
        )

        ttk.Label(frame, textvariable=self.result_var, wraplength=360).grid(
            row=5,
            column=0,
            columnspan=2,
            sticky="w",
            **padding,
        )

        ttk.Label(frame, textvariable=self.status_var, wraplength=360).grid(
            row=6,
            column=0,
            columnspan=2,
            sticky="w",
            **padding,
        )

    def _bind_events(self) -> None:
        self.model_var.trace_add("write", lambda *_: self._update_metric_label())

    def _populate_models(self) -> None:
        models = list(self.manager.models.keys())
        self.model_combo["values"] = models
        if models:
            self.model_var.set(models[0])
        else:
            self.model_var.set("")

    def _update_metric_label(self) -> None:
        name = self.model_var.get()
        metric = self.manager.metrics.get(name)
        info = self.manager.models.get(name)
        if not info:
            self.metric_var.set("Metric: model not loaded.")
            return
        if metric is None:
            self.metric_var.set("Metric: unavailable (missing dataset)")
            return
        if info["type"] == "classification":
            self.metric_var.set(f"Accuracy: {metric * 100:.1f}% on holdout set")
        else:
            self.metric_var.set(f"MAE: {metric:.2f} vehicles on holdout set")

    def _refresh_models(self) -> None:
        self.manager.load_models()
        self._populate_models()
        self._update_metric_label()
        self.status_var.set(self._status_text())

    def _status_text(self) -> str:
        if not self.manager.messages:
            return ""
        return " | ".join(self.manager.messages)

    def _predict(self) -> None:
        model_name = self.model_var.get()
        info = self.manager.models.get(model_name)
        if not info:
            messagebox.showerror("Prediction Error", "Please load a model before predicting.")
            return

        try:
            hour = int(self.hour_var.get())
        except ValueError:
            messagebox.showerror("Invalid Hour", "Hour must be an integer between 0 and 23.")
            return

        if hour < 0 or hour > 23:
            messagebox.showerror("Invalid Hour", "Hour must be between 0 and 23.")
            return

        day_name = self.day_var.get()
        if day_name not in DAY_TO_INDEX:
            messagebox.showerror("Invalid Day", "Select a valid day of the week.")
            return

        # --- QUESTA PARTE DEFINISCE 'features' (NON RIMUOVERLA) ---
        day_idx = DAY_TO_INDEX[day_name]
        weekend_flag = 1 if day_idx >= 5 else 0
        features = pd.DataFrame(
            [{
                "hour": hour,
                "day_of_week": day_idx,
                "is_weekend": weekend_flag,
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
            }]
        )

        model = info["model"]
        features_aligned = self.manager._align_features(features, model)

        if info["type"] == "classification":
            prediction = model.predict(features_aligned)[0]
            confidence = self._class_probability(model, features_aligned, prediction)
            
            # TRUCCO: Proviamo a recuperare il numero dal Linear Regression
            # per mostrare sia il conteggio che il livello
            try:
                reg_info = self.manager.models.get("Linear Regression")
                if reg_info:
                    reg_model = reg_info["model"]
                    num_features = self.manager._align_features(features, reg_model)
                    count = reg_model.predict(num_features)[0]
                    res_text = f"Count: {count:.1f} veh | Level: {prediction}"
                else:
                    res_text = f"Level: {prediction}"
                
                if confidence is not None:
                    res_text += f" (conf {confidence * 100:.1f}%)"
                self.result_var.set(res_text)
            except:
                self.result_var.set(f"Level: {prediction}")
        else:
            prediction = model.predict(features_aligned)[0]
            self.result_var.set(f"Predicted vehicle count: {prediction:.1f}")
    @staticmethod
    def _class_probability(model, features: pd.DataFrame, label: str) -> Optional[float]:
        if not hasattr(model, "predict_proba"):
            return None
        try:
            probabilities = model.predict_proba(features)[0]
            classes = getattr(model, "classes_", None)
            if classes is None:
                return None
            index = list(classes).index(label)
            return probabilities[index]
        except Exception:  # noqa: BLE001 - fallback to no-confidence display
            return None


def main() -> None:
    root = tk.Tk()
    TrafficPredictorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
