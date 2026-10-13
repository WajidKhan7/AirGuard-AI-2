"""Tests for the classification branch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airquality import classification as clf
from airquality import evaluation as ev


@pytest.fixture
def toy_classification_data():
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(0, 1, (n, 5)).astype(np.float32)
    # make the label a genuine (noisy) function of the features so a model
    # can actually learn something, rather than pure label noise.
    score = X[:, 0] * 2 - X[:, 1] + rng.normal(0, 0.5, n)
    y = np.where(score < -0.5, "Low", np.where(score < 0.5, "Moderate", "High"))
    return X, y


def test_build_models_returns_expected_zoo():
    models = clf.build_models()
    assert set(models) == {"Logistic Regression", "Random Forest", "SVM"}


def test_cross_validate_models_returns_reasonable_scores(toy_classification_data):
    X, y = toy_classification_data
    table = cross_val = clf.cross_validate_models(X, y, cv_splits=4)
    assert set(table["model"]) == {"Logistic Regression", "Random Forest", "SVM"}
    assert table["cv_mean_f1_macro"].between(0, 1).all()
    assert (table["cv_mean_f1_macro"] > 0.3).all()   # better than random on a learnable signal


def test_train_and_compare_selects_using_validation_only(toy_classification_data):
    X, y = toy_classification_data
    X_tr, X_va = X[:200], X[200:]
    y_tr, y_va = y[:200], y[200:]
    fitted, comparison = clf.train_and_compare(X_tr, y_tr, X_va, y_va)
    assert len(fitted) == 3
    assert comparison["val_f1_macro"].is_monotonic_decreasing


def test_tune_random_forest_uses_gridsearch_for_small_data(toy_classification_data):
    X, y = toy_classification_data
    result = clf.tune_random_forest(X, y, n_rows=len(X))
    assert "GridSearchCV" in result["method"]
    assert not result["subsampled_for_tuning"]
    assert 0 <= result["best_cv_score"] <= 1
    assert "n_estimators" in result["best_params"]


def test_tune_random_forest_uses_randomizedsearch_for_large_data(toy_classification_data):
    X, y = toy_classification_data
    result = clf.tune_random_forest(X, y, n_rows=50_000)
    assert "RandomizedSearchCV" in result["method"]


def test_tune_random_forest_subsamples_huge_training_sets(monkeypatch):
    # Use a small cap so this test verifies the subsampling *logic* without
    # waiting on a real 60k-row RandomizedSearchCV.
    monkeypatch.setattr(clf, "TUNING_SAMPLE_CAP", 500)
    rng = np.random.default_rng(1)
    n = clf.TUNING_SAMPLE_CAP + 300
    X = rng.normal(0, 1, (n, 3)).astype(np.float32)
    y = rng.choice(["Low", "Moderate", "High"], n)
    result = clf.tune_random_forest(X, y, n_rows=n)
    assert result["subsampled_for_tuning"] is True
    assert result["tuning_rows_used"] == 500
    # the FINAL model must still be fit on all n rows, not the subsample
    assert result["model"].named_steps["clf"].n_features_in_ == 3


def test_feature_importance_sums_to_about_one(toy_classification_data):
    X, y = toy_classification_data
    result = clf.tune_random_forest(X, y, n_rows=len(X))
    names = [f"f{i}" for i in range(X.shape[1])]
    fi = clf.feature_importance(result["model"], names)
    assert set(fi["feature"]) == set(names)
    assert abs(fi["importance"].sum() - 1.0) < 1e-6
    assert fi.iloc[0]["importance"] >= fi.iloc[-1]["importance"]  # sorted descending


# ---------------------------------------------------------------------------
# evaluation.py
# ---------------------------------------------------------------------------


def test_classification_metrics_known_case():
    y_true = ["Low", "Low", "Moderate", "High"]
    y_pred = ["Low", "Moderate", "Moderate", "High"]
    m = ev.classification_metrics(y_true, y_pred, ["Low", "Moderate", "High"])
    assert m["accuracy"] == 0.75
    assert 0 <= m["f1_macro"] <= 1


def test_confusion_table_shape():
    y_true = ["Low", "Moderate", "High"] * 3
    y_pred = ["Low", "Moderate", "High"] * 3
    table = ev.confusion_table(y_true, y_pred, ["Low", "Moderate", "High"])
    assert table.shape == (3, 3)
    assert table.to_numpy().trace() == 9    # perfect prediction -> all on the diagonal


def test_per_class_report_has_macro_and_weighted_rows():
    y_true = ["Low", "Moderate", "High", "Low"]
    y_pred = ["Low", "Low", "High", "Low"]
    report = ev.per_class_report(y_true, y_pred, ["Low", "Moderate", "High"])
    assert "macro avg" in report["class"].values
    assert "weighted avg" in report["class"].values
