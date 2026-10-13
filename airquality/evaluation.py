"""Evaluation: confusion matrix, precision/recall/F1, and all saved plots."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless: always write a file, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def classification_metrics(y_true, y_pred, labels: list[str]) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4),
    }


def per_class_report(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    rows = []
    for cls in labels:
        r = report.get(cls, {})
        rows.append({"class": cls, "precision": round(r.get("precision", 0.0), 4),
                     "recall": round(r.get("recall", 0.0), 4),
                     "f1": round(r.get("f1-score", 0.0), 4),
                     "support": int(r.get("support", 0))})
    for avg_key, label in (("macro avg", "macro avg"), ("weighted avg", "weighted avg")):
        r = report.get(avg_key, {})
        rows.append({"class": label, "precision": round(r.get("precision", 0.0), 4),
                     "recall": round(r.get("recall", 0.0), 4),
                     "f1": round(r.get("f1-score", 0.0), 4),
                     "support": int(r.get("support", 0))})
    return pd.DataFrame(rows)


def confusion_table(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"Actual {l}" for l in labels],
                        columns=[f"Predicted {l}" for l in labels])


def plot_confusion_matrix(y_true, y_pred, labels: list[str], out_path: Path) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2, 1.4 * len(labels) + 2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > vmax / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_feature_importance(importance_df: pd.DataFrame, out_path: Path, top_n: int = 15) -> Path:
    top = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(top))))
    ax.barh(top["feature"], top["importance"], color="#3a7ca5")
    ax.set_xlabel("Importance"); ax.set_title("Feature Importance (Random Forest)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_model_comparison(comparison: pd.DataFrame, out_path: Path) -> Path:
    metrics = [c for c in ("val_accuracy", "val_precision_macro", "val_recall_macro", "val_f1_macro")
               if c in comparison.columns]
    fig, ax = plt.subplots(figsize=(1.8 * len(comparison) + 3, 4.5))
    x = np.arange(len(comparison))
    width = 0.8 / max(1, len(metrics))
    for k, m in enumerate(metrics):
        ax.bar(x + k * width, comparison[m], width, label=m.replace("val_", "").replace("_", " "))
    ax.set_xticks(x + width * (len(metrics) - 1) / 2, comparison["model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05); ax.legend(); ax.set_title("Model Comparison (validation set)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_forecast(test_true, test_pred, future=None, out_path: Path = None) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(range(len(test_true)), test_true, label="Actual", color="#333333")
    ax.plot(range(len(test_pred)), test_pred, label="Predicted", color="#e0592a", linestyle="--")
    if future is not None and len(future):
        fx = range(len(test_true), len(test_true) + len(future))
        ax.plot(fx, future, label="Future forecast", color="#2a9d8f", linestyle=":")
    ax.set_xlabel("Time step"); ax.set_ylabel("Value"); ax.set_title("LSTM Forecast vs Actual")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
