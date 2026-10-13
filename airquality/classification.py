"""Classification: model zoo, tuning, cross-validation, feature importance.

Dataset-size aware: RandomizedSearchCV is used above a row-count threshold so
a huge CSV doesn't trigger an exhaustive grid; a small dataset gets an
exhaustive GridSearchCV instead since it's cheap and slightly more thorough.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .validation import SchemaError

logger = logging.getLogger(__name__)

GRIDSEARCH_ROW_LIMIT = 20_000       # above this, use RandomizedSearchCV instead
TUNING_SAMPLE_CAP = 60_000          # tuning searches see at most this many rows
RANDOM_STATE = 42


def build_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                           random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }


def _rf_search_space(randomised: bool) -> dict:
    if randomised:
        return {
            "clf__n_estimators": randint(150, 500),
            "clf__max_depth": [None, 8, 12, 20, 30],
            "clf__min_samples_split": randint(2, 12),
            "clf__min_samples_leaf": randint(1, 8),
            "clf__max_features": ["sqrt", "log2", 0.5],
        }
    return {
        "clf__n_estimators": [150, 300],
        "clf__max_depth": [None, 15, 30],
        "clf__min_samples_split": [2, 6],
        "clf__min_samples_leaf": [1, 3],
        "clf__max_features": ["sqrt", 0.5],
    }


def cross_validate_models(X: np.ndarray, y: np.ndarray, cv_splits: int = 5) -> pd.DataFrame:
    """k-fold CV for every model in the zoo, reported per-fold and as mean/std."""
    n_per_class = pd.Series(y).value_counts().min()
    splits = max(2, min(cv_splits, int(n_per_class)))
    if splits < cv_splits:
        logger.warning("reducing CV folds to %d (rarest class has only %d samples)",
                       splits, n_per_class)
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for name, model in build_models().items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        row = {"model": name, "cv_mean_f1_macro": round(float(scores.mean()), 4),
               "cv_std_f1_macro": round(float(scores.std()), 4)}
        for i, s in enumerate(scores, start=1):
            row[f"fold_{i}"] = round(float(s), 4)
        rows.append(row)
        logger.info("CV %-20s mean=%.4f std=%.4f", name, scores.mean(), scores.std())
    return pd.DataFrame(rows).sort_values("cv_mean_f1_macro", ascending=False).reset_index(drop=True)


def tune_random_forest(X_train: np.ndarray, y_train: np.ndarray, n_rows: int) -> dict:
    """Hyperparameter-tune the Random Forest, the system's primary classifier.

    Uses GridSearchCV below ``GRIDSEARCH_ROW_LIMIT`` rows, RandomizedSearchCV
    above it, and tunes on a capped subsample for very large datasets while
    still fitting the final chosen configuration on the full training set.
    """
    X_fit, y_fit = X_train, y_train
    subsampled = False
    if len(X_train) > TUNING_SAMPLE_CAP:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train), TUNING_SAMPLE_CAP, replace=False)
        X_fit, y_fit = X_train[idx], np.asarray(y_train)[idx]
        subsampled = True

    randomised = n_rows > GRIDSEARCH_ROW_LIMIT
    base = Pipeline([("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                                     class_weight="balanced_subsample"))])
    space = _rf_search_space(randomised)
    splits = max(2, min(5, int(pd.Series(y_fit).value_counts().min())))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)

    t0 = time.perf_counter()
    if randomised:
        search = RandomizedSearchCV(base, space, n_iter=25, scoring="f1_macro", cv=cv,
                                    random_state=RANDOM_STATE, n_jobs=-1, refit=False)
        method = "RandomizedSearchCV (dataset > %d rows)" % GRIDSEARCH_ROW_LIMIT
    else:
        search = GridSearchCV(base, space, scoring="f1_macro", cv=cv, n_jobs=-1, refit=False)
        method = "GridSearchCV"
    search.fit(X_fit, y_fit)
    elapsed = time.perf_counter() - t0

    # Refit the winning configuration on the FULL training set (not the
    # tuning subsample), so the final model doesn't waste available data.
    best_model = Pipeline([("clf", RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced_subsample",
        **{k.replace("clf__", ""): v for k, v in search.best_params_.items()},
    ))])
    best_model.fit(X_train, y_train)

    return {
        "method": method,
        "subsampled_for_tuning": subsampled,
        "tuning_rows_used": len(X_fit),
        "best_params": {k.replace("clf__", ""): _jsonable(v) for k, v in search.best_params_.items()},
        "best_cv_score": round(float(search.best_score_), 4),
        "seconds": round(elapsed, 1),
        "model": best_model,
    }


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 6)
    return v


def train_and_compare(
    X_train, y_train, X_val, y_val,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Fit every baseline model, score each on the validation split.

    Model selection happens here, on validation data only — the test split
    stays untouched until the final evaluation step.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    fitted, rows = {}, []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        rows.append({
            "model": name,
            "val_accuracy": round(float(accuracy_score(y_val, pred)), 4),
            "val_precision_macro": round(float(precision_score(y_val, pred, average="macro", zero_division=0)), 4),
            "val_recall_macro": round(float(recall_score(y_val, pred, average="macro", zero_division=0)), 4),
            "val_f1_macro": round(float(f1_score(y_val, pred, average="macro", zero_division=0)), 4),
        })
        fitted[name] = model
    comparison = pd.DataFrame(rows).sort_values("val_f1_macro", ascending=False).reset_index(drop=True)
    return fitted, comparison


def feature_importance(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    clf = model.named_steps.get("clf")
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_).mean(axis=0)
        importances = importances / importances.sum()
    else:
        return pd.DataFrame({"feature": feature_names, "importance": np.nan})
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)
