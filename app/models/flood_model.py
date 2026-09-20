"""
Hybrid physics + ML flood-risk model.

The ML component is trained on labelled historical flood observations
(Mumbai flood events dataset). If the model file is not yet on disk,
it is automatically trained on startup so that model_status is always "trained".
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence
import joblib
import numpy as np

FEATURES = [
    "rain_15m_mm",
    "rain_60m_mm",
    "rain_180m_mm",
    "rain_intensity_mm_h",
    "elevation_m",
    "slope_pct",
    "distance_to_drain_m",
    "drainage_capacity_ratio",
    "tide_m",
    "historical_flood_frequency",
    "impervious_fraction",
]

MODEL_PATH = Path(__file__).resolve().parent / "flood_risk.joblib"

CANDIDATE_MODEL_PATHS = [
    MODEL_PATH,
    Path(__file__).resolve().parents[1] / "data" / "models" / "flood_risk.joblib",
    Path(__file__).resolve().parents[2] / "data" / "models" / "flood_risk.joblib",
    Path("data/models/flood_risk.joblib"),
    Path("app/models/flood_risk.joblib"),
]

CANDIDATE_TRAINING_CSVS = [
    Path(__file__).resolve().parents[1] / "data" / "training" / "real_flood_events_mumbai.csv",
    Path(__file__).resolve().parents[2] / "data" / "training" / "real_flood_events_mumbai.csv",
    Path("data/training/real_flood_events_mumbai.csv"),
    Path("app/data/training/real_flood_events_mumbai.csv"),
]

_model_cache = None


def physics_score(x: dict) -> float:
    """
    Explainable baseline. It is not a trained ML prediction.
    """
    rain = min(1.0, float(x.get("rain_60m_mm", 0)) / 100.0)
    short = min(1.0, float(x.get("rain_15m_mm", 0)) / 40.0)
    drain = min(1.0, max(0.0, float(x.get("drainage_capacity_ratio", 0))))
    tide = min(1.0, max(0.0, float(x.get("tide_m", 0)) / 2.5))
    impervious = min(1.0, max(0.0, float(x.get("impervious_fraction", 0))))
    historical = min(1.0, max(0.0, float(x.get("historical_flood_frequency", 0))))

    score = (
        0.30 * rain
        + 0.15 * short
        + 0.25 * drain
        + 0.10 * tide
        + 0.10 * impervious
        + 0.10 * historical
    )
    return float(np.clip(score, 0, 1))


def auto_train_model():
    """
    Automatically trains RandomForest on the real Mumbai flood dataset
    and caches + saves it to MODEL_PATH.
    """
    global _model_cache

    csv_path = None
    for p in CANDIDATE_TRAINING_CSVS:
        if p.exists():
            csv_path = p
            break

    if not csv_path:
        print("[flood_model] No training CSV found, skipping auto-training.")
        return None

    try:
        X, y = [], []
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_val = int(float(row.get("label", 0)))
                feat_dict = {k: float(row.get(k, 0.0)) for k in FEATURES}
                X.append(feat_dict)
                y.append(label_val)

        if len(X) > 0 and len(set(y)) >= 2:
            print(f"[flood_model] Auto-training Random Forest on {len(X)} historical Mumbai flood records...")
            result = train(X, y)
            print(f"[flood_model] Training complete! ROC-AUC: {result.get('roc_auc')}, F1: {result.get('f1')}")
            if MODEL_PATH.exists():
                _model_cache = joblib.load(MODEL_PATH)
                return _model_cache
    except Exception as exc:
        print(f"[flood_model] Auto-training failed: {exc}")

    return None


def load_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    for p in CANDIDATE_MODEL_PATHS:
        if p.exists():
            try:
                _model_cache = joblib.load(p)
                return _model_cache
            except Exception as exc:
                print(f"[flood_model] Failed loading from {p}: {exc}")

    # Not found on disk: automatically train on the fly
    return auto_train_model()


def predict(features: dict) -> dict:
    model = load_model()

    vector = np.array([[float(features.get(k, 0.0)) for k in FEATURES]], dtype=float)

    if model is None:
        score = physics_score(features)
        return {
            "probability": round(score, 4),
            "model_status": "untrained",
            "method": "physics_baseline",
            "confidence": 55.0,
        }

    probability = float(model.predict_proba(vector)[0, 1])
    confidence = float(max(model.predict_proba(vector)[0]) * 100)
    return {
        "probability": round(probability, 4),
        "model_status": "trained",
        "method": "supervised_ml_plus_physics_features",
        "confidence": round(confidence, 1),
    }


def train(X: Sequence[dict], y: Sequence[int]) -> dict:
    """
    Train a RandomForest classifier on labelled historical cells/events.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score

    matrix = np.array([[float(row.get(k, 0.0)) for k in FEATURES] for row in X])
    labels = np.array(y, dtype=int)

    if len(set(labels.tolist())) < 2:
        raise ValueError("Training data must contain both flood and non-flood labels.")

    X_train, X_test, y_train, y_test = train_test_split(
        matrix, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    global _model_cache
    _model_cache = model

    return {
        "saved_to": str(MODEL_PATH),
        "samples": int(len(labels)),
        "roc_auc": round(float(roc_auc_score(y_test, p)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "features": FEATURES,
    }
