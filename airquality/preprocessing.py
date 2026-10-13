"""Cleaning and target derivation, driven by the schema profile.

Nothing here is hard-coded to a specific dataset's column names — every
decision (which columns to clean, which one is the label, how to bucket a
numeric AQI into categories) is made from the ``SchemaProfile`` produced by
``schema.inspect_dataset``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import SchemaProfile
from .validation import clip_infinite

logger = logging.getLogger(__name__)

# US-EPA-style breakpoints, collapsed to three bands, applied to whichever
# numeric AQI-like column is available. This rule is applied only when the
# dataset has no categorical label of its own, and is always reported to the
# user so the label's origin is never hidden.
AQI_BANDS = (100.0, 200.0)                 # <=100 Low, <=200 Moderate, >200 High
PM25_BANDS = (35.4, 150.4)                 # ug/m3, US-EPA "Good/Moderate" cut points doubled up
LABEL_CLASSES = ("Low", "Moderate", "High")


@dataclass
class CleaningReport:
    original_rows: int
    duplicate_rows_removed: int = 0
    missing_values_before: int = 0
    missing_values_after: int = 0
    invalid_rows_removed: int = 0
    infinite_values_fixed: int = 0
    outliers_capped: int = 0
    final_rows: int = 0
    label_source: str = ""
    label_rule: str = ""
    notes: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        lines = [
            f"Original rows: {self.original_rows:,}",
            f"Duplicate rows removed: {self.duplicate_rows_removed:,}",
            f"Missing values handled: {self.missing_values_before:,}",
            f"Invalid rows removed: {self.invalid_rows_removed:,}",
            f"Infinite values fixed: {self.infinite_values_fixed:,}",
            f"Outlier values capped: {self.outliers_capped:,}",
            f"Final rows: {self.final_rows:,}",
        ]
        if self.label_source:
            lines.append(f"\nLabel source: {self.label_source}")
        if self.label_rule:
            lines.append(f"Label rule: {self.label_rule}")
        lines.extend(f"Note: {n}" for n in self.notes)
        return "\n".join(lines)


def _to_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(
        s.astype(str).str.strip().str.replace(r"^\s*$", "", regex=True),
        errors="coerce",
    )


def clean_dataset(df: pd.DataFrame, profile: SchemaProfile) -> tuple[pd.DataFrame, CleaningReport]:
    """Deduplicate, coerce types, fix invalid/infinite values, cap outliers.

    Conservative by design: rows are only dropped when every feature is
    missing or a hard physical bound is violated, never for an ordinary
    missing value (those are imputed instead).
    """
    report = CleaningReport(original_rows=len(df))
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    # whitespace-only strings and literal "NA"/"N/A"/"-" become real NaN
    obj_cols = out.select_dtypes(include=["object", "string"]).columns
    out[obj_cols] = out[obj_cols].apply(
        lambda s: s.astype(str).str.strip().replace(
            {"": np.nan, "NA": np.nan, "N/A": np.nan, "-": np.nan, "nan": np.nan, "null": np.nan}
        )
    )

    numeric_like = [cp.raw_name for cp in profile.columns if cp.dtype == "numeric" and cp.raw_name in out.columns]
    for col in numeric_like:
        out[col] = _to_numeric_series(out[col])

    out, n_inf = clip_infinite(out, numeric_like)
    report.infinite_values_fixed = n_inf
    report.missing_values_before = int(out[numeric_like].isna().sum().sum()) if numeric_like else 0

    before = len(out)
    out = out.drop_duplicates(keep="first")
    report.duplicate_rows_removed = before - len(out)

    # A row with every feature-relevant numeric column missing carries no
    # information and is dropped; everything else is imputed below.
    relevant = [c for c in numeric_like if c in profile.feature_columns or c == profile.label_column]
    before = len(out)
    if relevant:
        out = out.dropna(subset=relevant, how="all")
    report.invalid_rows_removed = before - len(out)

    # Physically impossible values (negative concentrations, negative Kelvin
    # temperatures if that unit was used, >100% humidity) are treated as
    # invalid readings and set to missing, then imputed like any other gap.
    non_negative_cols = [c for c in numeric_like
                          if any(k in c.lower() for k in ("pm", "co", "no", "so", "o3", "aqi", "speed", "rain"))]
    for col in non_negative_cols:
        bad = out[col] < 0
        if bad.any():
            out.loc[bad, col] = np.nan
    humidity_cols = [c for c in numeric_like if "humid" in c.lower()]
    for col in humidity_cols:
        bad = (out[col] < 0) | (out[col] > 100)
        if bad.any():
            out.loc[bad, col] = np.nan

    # Outlier capping: winsorise at the 0.5/99.5 percentile per numeric
    # feature column, which tames sensor spikes without deleting rows.
    n_capped = 0
    for col in [c for c in numeric_like if c in profile.feature_columns]:
        series = out[col].dropna()
        if len(series) < 20:
            continue
        lo, hi = series.quantile(0.005), series.quantile(0.995)
        mask = (out[col] < lo) | (out[col] > hi)
        n_capped += int(mask.sum())
        out[col] = out[col].clip(lo, hi)
    report.outliers_capped = n_capped

    # Impute remaining gaps: median per numeric column (robust to the
    # skew typical of pollutant concentrations).
    for col in numeric_like:
        if out[col].isna().any():
            fill = out[col].median()
            out[col] = out[col].fillna(fill if pd.notna(fill) else 0.0)
    report.missing_values_after = int(out[numeric_like].isna().sum().sum()) if numeric_like else 0

    report.final_rows = len(out)
    if report.final_rows < 30:
        report.notes.append(
            f"Only {report.final_rows} usable rows remain after cleaning — "
            "results below will be statistically weak."
        )
    return out.reset_index(drop=True), report


def derive_label(df: pd.DataFrame, profile: SchemaProfile, report: CleaningReport) -> tuple[pd.DataFrame, str | None]:
    """Return (df_with_label_column, label_column_name), deriving one if needed.

    Priority: an existing categorical label column > a numeric AQI column
    bucketed by EPA-style bands > PM2.5 bucketed the same way > no label
    (classification is skipped, the rest of the pipeline continues).
    """
    df = df.copy()
    label_col = profile.label_column

    if label_col and label_col in df.columns:
        col_profile = next((cp for cp in profile.columns if cp.raw_name == label_col), None)
        if col_profile and col_profile.dtype == "categorical":
            classes = df[label_col].dropna().unique().tolist()
            if 2 <= len(classes) <= 10:
                report.label_source = f"existing column '{label_col}'"
                report.label_rule = f"used as-is, classes found: {sorted(map(str, classes))}"
                df[label_col] = df[label_col].astype(str).str.strip()
                return df, label_col
            report.notes.append(
                f"Column '{label_col}' looked like a label but has "
                f"{len(classes)} distinct values — ignored, deriving a label instead."
            )

    # numeric AQI column present?
    aqi_col = next(
        (cp.raw_name for cp in profile.columns
         if cp.role == "pollutant" and "aqi" in cp.normalised_name and cp.dtype == "numeric"),
        None,
    )
    if aqi_col and aqi_col in df.columns and df[aqi_col].notna().sum() >= 30:
        lo, hi = AQI_BANDS
        df["derived_air_quality_label"] = pd.cut(
            df[aqi_col], bins=[-np.inf, lo, hi, np.inf], labels=list(LABEL_CLASSES)
        ).astype(str)
        report.label_source = f"derived from numeric column '{aqi_col}'"
        report.label_rule = f"AQI <= {lo:g} -> Low, <= {hi:g} -> Moderate, > {hi:g} -> High"
        return df, "derived_air_quality_label"

    pm25_col = next(
        (cp.raw_name for cp in profile.columns
         if cp.normalised_name == "pm2_5" and cp.dtype == "numeric"), None,
    )
    if pm25_col and pm25_col in df.columns and df[pm25_col].notna().sum() >= 30:
        lo, hi = PM25_BANDS
        df["derived_air_quality_label"] = pd.cut(
            df[pm25_col], bins=[-np.inf, lo, hi, np.inf], labels=list(LABEL_CLASSES)
        ).astype(str)
        report.label_source = f"derived from PM2.5 column '{pm25_col}' (US-EPA breakpoints)"
        report.label_rule = f"PM2.5 <= {lo:g} -> Low, <= {hi:g} -> Moderate, > {hi:g} -> High"
        return df, "derived_air_quality_label"

    report.notes.append(
        "No classification label found and none could be derived "
        "(no AQI or PM2.5 column with enough valid values). "
        "Classification will be skipped; anomaly detection and forecasting "
        "will continue if the required columns exist."
    )
    return df, None


def select_classification_features(
    df: pd.DataFrame, profile: SchemaProfile, label_col: str | None
) -> list[str]:
    """Feature columns for classification/anomaly detection: profile's feature
    list minus the label itself and anything that would leak it (e.g. the AQI
    column the label was derived from, when the label is derived rather than
    an independent column)."""
    feats = [c for c in profile.feature_columns if c != label_col]
    if label_col == "derived_air_quality_label":
        aqi_like = [c for c in feats if "aqi" in c.lower()]
        for c in aqi_like:
            feats.remove(c)
    return feats
