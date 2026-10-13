"""Tests for the anomaly-detection branch."""

from __future__ import annotations

import numpy as np
import pytest

from airquality import anomaly_detection as ad


@pytest.fixture
def data_with_outliers():
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, (190, 4))
    outliers = rng.normal(0, 1, (10, 4)) + 12   # clearly separated cluster
    X = np.vstack([normal, outliers]).astype(np.float32)
    return X, np.array([0] * 190 + [1] * 10)    # ground truth, for testing only


def test_train_and_score_anomalies(data_with_outliers):
    X, _ = data_with_outliers
    model = ad.train_anomaly_detector(X, contamination=0.05)
    results = ad.score_anomalies(model, X)
    assert set(results["status"].unique()) <= {"Normal", "Anomaly"}
    assert len(results) == len(X)


def test_anomaly_percentage_matches_contamination_roughly(data_with_outliers):
    X, _ = data_with_outliers
    model = ad.train_anomaly_detector(X, contamination=0.05)
    results = ad.score_anomalies(model, X)
    summary = ad.summarise(results)
    assert summary["total_observations"] == len(X)
    assert 0 < summary["anomaly_percentage"] < 15


def test_anomaly_detector_actually_flags_the_injected_outliers(data_with_outliers):
    X, y_true = data_with_outliers
    model = ad.train_anomaly_detector(X, contamination=0.05)
    results = ad.score_anomalies(model, X)
    metrics = ad.evaluate_against_labels(results, y_true)
    assert metrics is not None
    assert metrics["recall"] > 0.5   # the injected cluster should mostly be caught


def test_evaluate_against_labels_returns_none_without_labels(data_with_outliers):
    X, _ = data_with_outliers
    model = ad.train_anomaly_detector(X)
    results = ad.score_anomalies(model, X)
    assert ad.evaluate_against_labels(results, None) is None
