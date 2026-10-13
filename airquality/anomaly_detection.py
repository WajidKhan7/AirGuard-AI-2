"""Anomaly detection with Isolation Forest.

Unsupervised: needs only the numeric feature matrix, no label at all, so it
runs on every dataset that has enough numeric columns regardless of whether
classification or forecasting are possible.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
DEFAULT_CONTAMINATION = 0.05


def train_anomaly_detector(X: np.ndarray, contamination: float = DEFAULT_CONTAMINATION) -> Pipeline:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(
            n_estimators=300, contamination=contamination,
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])
    model.fit(X)
    return model


def score_anomalies(model: Pipeline, X: np.ndarray, feature_names: list[str] | None = None) -> pd.DataFrame:
    """Return one row per observation: Normal/Anomaly label and a raw score."""
    raw_pred = model.predict(X)                 # -1 = anomaly, 1 = normal
    scores = model.named_steps["iforest"].score_samples(
        model.named_steps["scaler"].transform(X)
    )
    df = pd.DataFrame({
        "status": np.where(raw_pred == -1, "Anomaly", "Normal"),
        "anomaly_score": np.round(scores, 5),   # lower = more anomalous
    })
    return df


def summarise(results: pd.DataFrame) -> dict:
    total = len(results)
    anomalies = int((results["status"] == "Anomaly").sum())
    return {
        "total_observations": total,
        "normal_observations": total - anomalies,
        "anomalies": anomalies,
        "anomaly_percentage": round(100.0 * anomalies / total, 2) if total else 0.0,
    }


def evaluate_against_labels(results: pd.DataFrame, y_true: np.ndarray) -> dict | None:
    """If the dataset happens to carry a real anomaly label, score against it.

    Most air-quality CSVs won't have one — this is opportunistic, not assumed.
    """
    if y_true is None:
        return None
    y_pred = (results["status"] == "Anomaly").astype(int).to_numpy()
    y_true = np.asarray(y_true).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f1), 4)}
