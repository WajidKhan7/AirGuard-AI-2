"""Tests for LSTM forecasting — small synthetic series so the suite stays fast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airquality import lstm_forecasting as lstmf
from airquality import schema as sch


@pytest.fixture
def hourly_series_df():
    n = 400
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    hours = dates.hour.values
    pm25 = 30 + 15 * np.sin(hours / 24 * 2 * np.pi) + rng.normal(0, 2, n)
    return pd.DataFrame({"Date": dates.date, "Time": dates.strftime("%H:%M:%S"),
                         "PM2.5": pm25, "PM10": pm25 * 1.6 + rng.normal(0, 3, n)})


def test_build_datetime_index_combines_date_and_time(hourly_series_df):
    profile = sch.inspect_dataset(hourly_series_df)
    dt = lstmf.build_datetime_index(hourly_series_df, profile)
    assert dt is not None
    assert dt.notna().mean() > 0.95
    assert dt.is_monotonic_increasing


def test_build_datetime_index_returns_none_without_datetime_columns():
    df = pd.DataFrame({"PM2.5": [1, 2, 3], "PM10": [4, 5, 6]})
    profile = sch.inspect_dataset(df)
    assert lstmf.build_datetime_index(df, profile) is None


def test_infer_lookback_hourly_vs_daily():
    hourly = pd.Series(pd.date_range("2024-01-01", periods=48, freq="h"))
    daily = pd.Series(pd.date_range("2024-01-01", periods=48, freq="D"))
    assert lstmf.infer_lookback(hourly) == 24
    assert lstmf.infer_lookback(daily) == 7


def test_make_sequences_shapes():
    values = np.arange(50, dtype=np.float32)
    X, y = lstmf.make_sequences(values, lookback=10)
    assert X.shape == (40, 10, 1)
    assert len(y) == 40
    assert np.allclose(X[0, :, 0], values[:10])
    assert y[0] == values[10]


def test_make_sequences_rejects_too_short_series():
    with pytest.raises(Exception):
        lstmf.make_sequences(np.arange(5, dtype=np.float32), lookback=10)


def test_find_forecast_targets(hourly_series_df):
    profile = sch.inspect_dataset(hourly_series_df)
    targets = lstmf.find_forecast_targets(profile, hourly_series_df)
    assert "PM2.5" in targets and "PM10" in targets


def test_lstm_forecaster_fit_predict_roundtrip():
    rng = np.random.default_rng(0)
    series = 20 + 5 * np.sin(np.arange(200) / 10) + rng.normal(0, 0.3, 200)
    model = lstmf.LSTMForecaster(lookback=12, epochs=3, hidden_size=8)
    model.fit(series[:150].astype(np.float32), series[150:180].astype(np.float32), verbose=False)
    preds = model.predict(series[150:180].astype(np.float32), context=series[:150].astype(np.float32))
    assert len(preds) == 30
    assert np.isfinite(preds).all()


def test_lstm_forecaster_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    series = (10 + rng.normal(0, 1, 150)).astype(np.float32)
    model = lstmf.LSTMForecaster(lookback=8, epochs=2, hidden_size=6)
    model.fit(series[:120], series[120:], verbose=False)
    preds_before = model.predict(series[120:], context=series[:120])

    path = tmp_path / "lstm_model.pt"
    model.save(path)
    reloaded = lstmf.LSTMForecaster.load(path)
    preds_after = reloaded.predict(series[120:], context=series[:120])
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-4)


def test_evaluate_forecast_metrics():
    y_true = np.array([10.0, 12.0, 9.0, 11.0])
    y_pred = np.array([10.0, 12.0, 9.0, 11.0])
    m = lstmf.evaluate_forecast(y_true, y_pred)
    assert m["mae"] == 0.0 and m["rmse"] == 0.0
    assert m["r2"] == 1.0


def test_run_lstm_pipeline_full_flow(hourly_series_df):
    profile = sch.inspect_dataset(hourly_series_df)
    result = lstmf.run_lstm_pipeline(hourly_series_df, profile, target="PM2.5", epochs=3)
    assert result["available"] is True
    assert set(result["test_metrics"]) == {"mae", "rmse", "r2"}
    assert result["split_sizes"]["train"] > result["split_sizes"]["test"]


def test_run_lstm_pipeline_reports_reason_when_unavailable():
    df = pd.DataFrame({"PM2.5": np.random.default_rng(0).normal(30, 5, 300)})
    profile = sch.inspect_dataset(df)
    result = lstmf.run_lstm_pipeline(df, profile, target="PM2.5")
    assert result["available"] is False
    assert "datetime" in result["reason"].lower()
