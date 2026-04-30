"""
ML Model Registry - manages all available models
"""
import os
import json
import joblib
import numpy as np
from typing import Any, Dict, List, Optional
from pathlib import Path
from sklearn.datasets import load_iris, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def train_and_save_demo_models():
    """Train and save demo models if they don't exist."""

    # ── Iris Classifier (Random Forest) ──────────────────────────────────────
    iris_path = MODELS_DIR / "iris_classifier.pkl"
    if not iris_path.exists():
        iris = load_iris()
        X_train, _, y_train, _ = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ])
        pipeline.fit(X_train, y_train)
        joblib.dump(
            {"pipeline": pipeline, "classes": iris.target_names.tolist()},
            iris_path,
        )
        print("[OK] iris_classifier.pkl saved")

    # ── Digits Classifier (SVM) ───────────────────────────────────────────────
    digits_path = MODELS_DIR / "digits_classifier.pkl"
    if not digits_path.exists():
        digits = load_digits()
        X_train, _, y_train, _ = train_test_split(
            digits.data, digits.target, test_size=0.2, random_state=42
        )
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
        ])
        pipeline.fit(X_train, y_train)
        joblib.dump(
            {"pipeline": pipeline, "classes": [str(i) for i in range(10)]},
            digits_path,
        )
        print("[OK] digits_classifier.pkl saved")


class ModelRegistry:
    """Central registry for all ML models."""

    CATALOG: Dict[str, Dict[str, Any]] = {
        "iris_classifier": {
            "id": "iris_classifier",
            "name": "Iris Flower Classifier",
            "description": "Classifies iris flowers into 3 species (Setosa, Versicolor, Virginica) using 4 measurements.",
            "type": "classification",
            "file": "iris_classifier.pkl",
            "inputs": [
                {"name": "sepal_length", "label": "Sepal Length (cm)", "min": 0, "max": 10, "default": 5.1},
                {"name": "sepal_width",  "label": "Sepal Width (cm)",  "min": 0, "max": 10, "default": 3.5},
                {"name": "petal_length", "label": "Petal Length (cm)", "min": 0, "max": 10, "default": 1.4},
                {"name": "petal_width",  "label": "Petal Width (cm)",  "min": 0, "max": 10, "default": 0.2},
            ],
            "tags": ["scikit-learn", "classification", "RandomForest"],
        },
        "digits_classifier": {
            "id": "digits_classifier",
            "name": "Handwritten Digit Recognizer",
            "description": "Recognizes handwritten digits (0-9) from an 8×8 pixel grayscale image (64 features).",
            "type": "classification",
            "file": "digits_classifier.pkl",
            "inputs": "canvas",  # special — handled by frontend draw canvas
            "tags": ["scikit-learn", "classification", "SVM"],
        },
    }

    def __init__(self):
        train_and_save_demo_models()
        self._cache: Dict[str, Any] = {}

    def list_models(self) -> List[Dict]:
        return [
            {k: v for k, v in meta.items() if k != "file"}
            for meta in self.CATALOG.values()
        ]

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self.CATALOG.get(model_id)

    def _load(self, model_id: str) -> Optional[Any]:
        if model_id in self._cache:
            return self._cache[model_id]
        meta = self.CATALOG.get(model_id)
        if not meta:
            return None
        path = MODELS_DIR / meta["file"]
        if not path.exists():
            return None
        obj = joblib.load(path)
        self._cache[model_id] = obj
        return obj

    def predict(self, model_id: str, features: List[float]) -> Dict[str, Any]:
        obj = self._load(model_id)
        if obj is None:
            raise ValueError(f"Model '{model_id}' not found or not loaded.")

        pipeline = obj["pipeline"]
        classes  = obj["classes"]
        X = np.array(features).reshape(1, -1)

        pred_idx  = int(pipeline.predict(X)[0])
        pred_label = classes[pred_idx]

        # Probabilities (if supported)
        proba: Optional[List[float]] = None
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X)[0].tolist()

        return {
            "prediction": pred_label,
            "prediction_index": pred_idx,
            "probabilities": (
                [{"label": c, "probability": round(p, 4)} for c, p in zip(classes, proba)]
                if proba else None
            ),
        }


# Singleton
registry = ModelRegistry()
