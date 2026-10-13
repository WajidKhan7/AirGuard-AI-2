"""Dataset loading: path validation, CSV reading with graceful fallbacks,
and large-dataset size reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .validation import DatasetError

LARGE_DATASET_ROWS = 200_000
HUGE_DATASET_ROWS = 1_000_000


def validate_csv_path(path: str) -> Path:
    """Check a user-supplied path before attempting to read it."""
    if not path or not str(path).strip():
        raise DatasetError("No dataset path was provided.")
    p = Path(str(path).strip().strip('"'))
    if not p.exists():
        raise DatasetError(f"Dataset file not found:\n{p}\n\nPlease select a valid CSV path.")
    if p.is_dir():
        raise DatasetError(f"{p} is a folder, not a CSV file.")
    if p.suffix.lower() != ".csv":
        raise DatasetError(f"Unsupported file format: '{p.suffix}'.\nPlease provide a CSV dataset.")
    if p.stat().st_size == 0:
        raise DatasetError("The selected file is empty (0 bytes).")
    return p


def load_csv_safely(path: Path) -> pd.DataFrame:
    """Read a CSV with a few graceful fallbacks for messy real-world files."""
    last_error = None
    for kwargs in ({}, {"encoding": "latin-1"}, {"sep": ";"},
                   {"engine": "python", "on_bad_lines": "skip"}):
        try:
            df = pd.read_csv(path, **kwargs)
            if df.shape[1] >= 2:     # a wrong separator often yields 1 giant column
                return df
        except Exception as exc:  # noqa: BLE001 - deliberately broad, we retry
            last_error = exc
    raise DatasetError(f"Could not read '{path.name}' as a CSV file.\nUnderlying error: {last_error}")


def load_dataset(path_str: str) -> tuple[pd.DataFrame, Path]:
    """Validate a path and load it. The single entry point train.py/predict.py use."""
    path = validate_csv_path(path_str)
    df = load_csv_safely(path)
    return df, path


def dataset_size_report(n_rows: int) -> str:
    if n_rows >= HUGE_DATASET_ROWS:
        return (f"Dataset size detected: {n_rows:,} rows\n\n"
                f"Large dataset mode enabled (>{HUGE_DATASET_ROWS:,} rows).\n"
                f"Hyperparameter search will use a bounded random search and a "
                f"tuning subsample; final models are still trained on the full "
                f"cleaned dataset.")
    if n_rows >= LARGE_DATASET_ROWS:
        return (f"Dataset size detected: {n_rows:,} rows\n\n"
                f"Large dataset mode enabled. Using memory-efficient processing.")
    return f"Dataset size detected: {n_rows:,} rows"
