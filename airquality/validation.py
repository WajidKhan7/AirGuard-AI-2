"""Custom exceptions and dataset-content validation.

CSV path handling and loading live in ``data_loader.py``; this module covers
what happens once a dataframe is in hand — plus the exception types every
other module raises so failures are typed and give the user something
readable, never a raw traceback for an expected problem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class AirQualityError(Exception):
    """Base class for every error raised by this package."""


class DatasetError(AirQualityError):
    """Problem with the dataset itself: missing file, bad format, unusable content."""


class SchemaError(AirQualityError):
    """The dataset's columns don't support the requested operation."""


class NotFittedError(AirQualityError):
    """predict() was called before fit(), or no trained artifacts exist yet."""


MIN_ROWS_FOR_ML = 30
MIN_NUMERIC_FEATURES = 2


def validate_dataset_basic(df: pd.DataFrame) -> None:
    """Cheap checks that must pass before any inspection or cleaning runs."""
    if df is None or df.empty:
        raise DatasetError("Dataset contains no usable records.")
    if df.shape[1] < 2:
        raise DatasetError("Dataset has fewer than 2 columns — nothing to build a model from.")
    if df.dropna(how="all").empty:
        raise DatasetError("Dataset contains no usable records (all rows are empty).")


def validate_numeric_features(numeric_cols: list[str], min_features: int = MIN_NUMERIC_FEATURES) -> None:
    if len(numeric_cols) < min_features:
        raise SchemaError(
            f"Not enough numerical features for machine learning.\n"
            f"Found {len(numeric_cols)} numeric column(s), need at least {min_features}.\n"
            f"Detected numeric columns: {numeric_cols}"
        )


def clip_infinite(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    n_inf = 0
    for c in numeric_cols:
        mask = np.isinf(df[c].to_numpy(dtype="float64", na_value=0.0))
        n_inf += int(mask.sum())
        if mask.any():
            df.loc[mask, c] = np.nan
    return df, n_inf
