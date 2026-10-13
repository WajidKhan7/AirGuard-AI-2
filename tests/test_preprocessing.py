"""Tests for schema detection and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airquality import schema as sch
from airquality import preprocessing as prep


@pytest.fixture
def messy_df():
    rng = np.random.default_rng(0)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "Date": dates.date, "Time": dates.strftime("%H:%M:%S"),
        "Station_ID": rng.integers(1000, 1005, n),
        "PM2.5": np.clip(rng.normal(40, 15, n), 1, None),
        "PM10": np.clip(rng.normal(70, 20, n), 1, None),
        "CO": np.clip(rng.normal(1, 0.3, n), 0.05, None),
        "NO2": np.clip(rng.normal(20, 6, n), 0.5, None),
        "Temperature": rng.normal(22, 5, n),
        "Humidity": np.clip(rng.normal(55, 15, n), 0, 100),
        "Comments": ["ok"] * n,
    })
    df.loc[df.sample(15, random_state=1).index, "PM2.5"] = np.nan
    df.loc[df.sample(5, random_state=2).index, "Humidity"] = -10
    df = pd.concat([df, df.sample(8, random_state=3)], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# schema.py
# ---------------------------------------------------------------------------


def test_normalise_column_name_variants():
    assert sch.normalise_column_name("PM2.5") == "pm2_5"
    assert sch.normalise_column_name("pm_2_5") == "pm2_5"
    assert sch.normalise_column_name("Particulate Matter 2.5") == "pm2_5"
    assert sch.normalise_column_name("PM25") == "pm2_5"


def test_inspect_dataset_detects_roles(messy_df):
    profile = sch.inspect_dataset(messy_df)
    assert set(profile.datetime_columns) == {"Date", "Time"}
    assert "PM2.5" in profile.pollutant_columns
    assert "Temperature" in profile.meteorological_columns
    assert "Humidity" in profile.meteorological_columns  # must NOT be caught as an identifier


def test_inspect_dataset_excludes_identifiers_and_text(messy_df):
    profile = sch.inspect_dataset(messy_df)
    assert "Station_ID" in profile.identifier_columns
    assert "Station_ID" not in profile.feature_columns
    assert "Comments" not in profile.feature_columns
    assert profile.excluded_columns["Station_ID"] == "identifier"


def test_inspect_dataset_handles_arbitrary_extra_columns(messy_df):
    df = messy_df.copy()
    df["SomeNewSensor"] = np.random.default_rng(4).normal(5, 1, len(df))
    profile = sch.inspect_dataset(df)
    assert "SomeNewSensor" in profile.feature_columns


def test_schema_signature_changes_with_feature_set(messy_df):
    p1 = sch.inspect_dataset(messy_df)
    df2 = messy_df.drop(columns=["Humidity"])
    p2 = sch.inspect_dataset(df2)
    assert p1.signature() != p2.signature()


def test_compare_schemas_detects_added_and_removed_columns(messy_df):
    p1 = sch.inspect_dataset(messy_df)
    df2 = messy_df.drop(columns=["Humidity"]).copy()
    df2["WindSpeed"] = np.random.default_rng(9).normal(8, 2, len(df2))
    p2 = sch.inspect_dataset(df2)
    diff = sch.compare_schemas(p1.to_dict(), p2)
    assert "WindSpeed" in diff["added_columns"]
    assert "Humidity" in diff["removed_columns"]
    assert diff["compatible_for_prediction"] is False  # a required feature vanished


def test_detect_long_format_positive():
    long_df = pd.DataFrame({
        "Date": ["2024-01-01"] * 6,
        "City": ["Lahore"] * 6,
        "Type": ["PM2.5", "PM10", "CO", "PM2.5", "PM10", "CO"],
        "Count": [55.0, 90.0, 1.1, 60.0, 95.0, 1.3],
    })
    result = sch.detect_long_format(long_df)
    assert result == ("Type", "Count")


def test_detect_long_format_negative(messy_df):
    assert sch.detect_long_format(messy_df) is None


def test_reshape_long_to_wide():
    long_df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "Type": ["PM2.5", "PM10", "PM2.5", "PM10"],
        "Count": [55.0, 90.0, 60.0, 95.0],
    })
    wide = sch.reshape_long_to_wide(long_df, "Type", "Count")
    assert set(wide.columns) == {"Date", "PM2.5", "PM10"}
    assert len(wide) == 2


# ---------------------------------------------------------------------------
# preprocessing.py
# ---------------------------------------------------------------------------


def test_clean_dataset_removes_duplicates_and_fixes_invalid_values(messy_df):
    profile = sch.inspect_dataset(messy_df)
    cleaned, report = prep.clean_dataset(messy_df, profile)
    assert report.duplicate_rows_removed >= 8
    assert cleaned["Humidity"].min() >= 0
    assert cleaned["PM2.5"].isna().sum() == 0    # imputed, not left missing
    assert report.final_rows == len(cleaned)


def test_clean_dataset_does_not_delete_rows_for_ordinary_missing_values(messy_df):
    profile = sch.inspect_dataset(messy_df)
    cleaned, report = prep.clean_dataset(messy_df, profile)
    # 15 missing PM2.5 values should be imputed, not dropped
    assert report.final_rows >= len(messy_df) - report.duplicate_rows_removed - 5


def test_derive_label_from_pm25_when_no_aqi_or_category(messy_df):
    profile = sch.inspect_dataset(messy_df)
    cleaned, report = prep.clean_dataset(messy_df, profile)
    cleaned, label_col = prep.derive_label(cleaned, profile, report)
    assert label_col == "derived_air_quality_label"
    assert set(cleaned[label_col].unique()) <= {"Low", "Moderate", "High"}
    assert "PM2.5" in report.label_rule or "pm2_5" in report.label_source.lower() or "PM2.5" in report.label_source


def test_derive_label_uses_existing_category_column(messy_df):
    df = messy_df.copy()
    df["AQI_Category"] = np.random.default_rng(0).choice(["Low", "Moderate", "High"], len(df))
    profile = sch.inspect_dataset(df)
    cleaned, report = prep.clean_dataset(df, profile)
    cleaned, label_col = prep.derive_label(cleaned, profile, report)
    assert label_col == "AQI_Category"
    assert "existing column" in report.label_source


def test_derive_label_returns_none_when_nothing_usable():
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=40, freq="D"),
        "Random1": np.random.default_rng(0).normal(0, 1, 40),
        "Random2": np.random.default_rng(1).normal(0, 1, 40),
    })
    profile = sch.inspect_dataset(df)
    cleaned, report = prep.clean_dataset(df, profile)
    cleaned, label_col = prep.derive_label(cleaned, profile, report)
    assert label_col is None
    assert any("No classification label" in n for n in report.notes)


def test_select_classification_features_excludes_leakage_column(messy_df):
    df = messy_df.copy()
    df["AQI"] = np.random.default_rng(0).normal(80, 20, len(df))
    profile = sch.inspect_dataset(df)
    cleaned, report = prep.clean_dataset(df, profile)
    cleaned, label_col = prep.derive_label(cleaned, profile, report)
    feats = prep.select_classification_features(cleaned, profile, label_col)
    assert "AQI" not in feats
