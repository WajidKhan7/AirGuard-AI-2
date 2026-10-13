"""Schema detection and dataset fingerprinting.

This module is what lets the system work on "any suitable air-quality CSV"
instead of one fixed dataset: it normalises column-name variants, guesses
each column's role (pollutant, meteorological, datetime, identifier, target),
and records a signature of the dataset structure so a saved model can tell
whether a later dataset is compatible with it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Column-name normalisation
# ----------------------------------------------------------------------------

#: canonical_name -> regex fragments that match common variants of that column.
#: Matching is done against a lowercased, punctuation-stripped version of each
#: header, so "PM2.5", "pm_2_5", "PM25 (ug/m3)" and "Particulate Matter 2.5"
#: all normalise to "pm2_5".
POLLUTANT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "pm2_5": (r"^pm ?2 ?5$", r"^pm25$", r"particulate matter 2 ?5"),
    "pm10": (r"^pm ?10$", r"particulate matter 10"),
    "co": (r"^co$", r"carbon monoxide"),
    "no2": (r"^no2$", r"nitrogen dioxide"),
    "so2": (r"^so2$", r"sulphur dioxide", r"sulfur dioxide"),
    "o3": (r"^o3$", r"^ozone$"),
    "nox": (r"^nox$",),
    "no": (r"^no$", r"^nitric oxide$"),
    "aqi": (r"^aqi$", r"air quality index"),
}

METEOROLOGICAL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "temperature": (r"temp", r"^temperature.*"),
    "humidity": (r"humid",),
    "wind_speed": (r"wind ?speed", r"^ws$"),
    "wind_direction": (r"wind ?dir",),
    "pressure": (r"pressure", r"^press$"),
    "rainfall": (r"rain", r"precip"),
    "noise_level": (r"noise",),
    "visibility": (r"visib",),
}

DATETIME_SYNONYMS: dict[str, tuple[str, ...]] = {
    "date": (r"^date$", r"^day$"),
    "time": (r"^time$", r"^hour$", r"^clock$"),
    "datetime": (r"date ?time", r"^timestamp$"),
}

LABEL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "aqi_category": (r"aqi.?category", r"air ?quality ?category", r"aq ?class",
                     r"^category$", r"^class$", r"^label$", r"^quality$",
                     r"pollution ?level", r"air ?quality ?level"),
}

#: matched against the space-normalised key (e.g. "station_id" -> "station id"),
#: so patterns use a literal space, never an underscore.
IDENTIFIER_PATTERNS: tuple[str, ...] = (
    r"^id$", r".* id$", r"^index$", r"^row ?number$", r"^record ?number$",
    r"^s ?no$", r"^serial", r"^unnamed",
)
TEXT_ONLY_PATTERNS: tuple[str, ...] = (
    r"comment", r"remark", r"note", r"description", r"station ?name",
    r"^city$", r"^location$", r"^station$", r"^address$",
)

_ALL_SYNONYMS = {**POLLUTANT_SYNONYMS, **METEOROLOGICAL_SYNONYMS, **DATETIME_SYNONYMS}


def _norm_key(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(name).strip().lower()).strip()


def normalise_column_name(name: str) -> str:
    """Best-effort canonical name for a raw header, or the cleaned original."""
    key = _norm_key(name)
    compact = key.replace(" ", "")
    for canonical, patterns in _ALL_SYNONYMS.items():
        for pat in patterns:
            if re.search(pat, key) or re.search(pat.replace(" ?", ""), compact):
                return canonical
    return re.sub(r"\s+", "_", key) or str(name)


def matches_any(name: str, synonyms: dict[str, tuple[str, ...]]) -> str | None:
    key = _norm_key(name)
    compact = key.replace(" ", "")
    for canonical, patterns in synonyms.items():
        for pat in patterns:
            if re.search(pat, key) or re.search(pat.replace(" ?", ""), compact):
                return canonical
    return None


def matches_pattern_list(name: str, patterns: tuple[str, ...]) -> bool:
    key = _norm_key(name)
    return any(re.search(p, key) for p in patterns)


# ----------------------------------------------------------------------------
# Schema inspection
# ----------------------------------------------------------------------------


@dataclass
class ColumnProfile:
    raw_name: str
    normalised_name: str
    dtype: str                       # "numeric" | "datetime" | "categorical" | "text"
    role: str                        # "pollutant" | "meteorological" | "datetime" | "label" |
                                      # "identifier" | "text" | "feature_candidate" | "constant"
    missing_frac: float
    n_unique: int
    is_constant: bool = False
    is_high_cardinality_id: bool = False


@dataclass
class SchemaProfile:
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    pollutant_columns: list[str] = field(default_factory=list)
    meteorological_columns: list[str] = field(default_factory=list)
    label_column: str | None = None
    identifier_columns: list[str] = field(default_factory=list)
    excluded_columns: dict[str, str] = field(default_factory=dict)   # name -> reason
    feature_columns: list[str] = field(default_factory=list)

    def signature(self) -> str:
        """Short hash of the structural shape of the dataset (not its values)."""
        payload = json.dumps(
            {"cols": sorted(self.feature_columns), "label": self.label_column,
             "datetime": sorted(self.datetime_columns)},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows, "n_cols": self.n_cols,
            "datetime_columns": self.datetime_columns,
            "pollutant_columns": self.pollutant_columns,
            "meteorological_columns": self.meteorological_columns,
            "label_column": self.label_column,
            "identifier_columns": self.identifier_columns,
            "excluded_columns": self.excluded_columns,
            "feature_columns": self.feature_columns,
            "signature": self.signature(),
        }


def _looks_like_id(series: pd.Series, n_rows: int) -> bool:
    if n_rows == 0:
        return False
    n_unique = series.nunique(dropna=True)
    return n_unique >= 0.98 * n_rows and n_unique > 20


def _try_parse_datetime(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.85


def inspect_dataset(df: pd.DataFrame) -> SchemaProfile:
    """Profile every column: type, role, and whether it belongs in the model.

    This is the core "understand an arbitrary CSV" step. It never mutates
    ``df`` — cleaning happens afterwards in ``preprocessing.py`` using this
    profile as a guide.
    """
    n_rows, n_cols = df.shape
    profile = SchemaProfile(n_rows=n_rows, n_cols=n_cols)
    label_candidates: list[str] = []

    for raw in df.columns:
        series = df[raw]
        norm = normalise_column_name(raw)
        missing_frac = float(series.isna().mean())
        n_unique = int(series.nunique(dropna=True))

        is_numeric = pd.api.types.is_numeric_dtype(series)
        if not is_numeric:
            coerced = pd.to_numeric(series, errors="coerce")
            if coerced.notna().mean() > 0.85 and series.notna().any():
                is_numeric = True  # numbers stored as strings ("12.3", "N/A")

        # Name-based match only (word-boundary safe — "Humidity" must never
        # match an "id" pattern). A numeric column is never called an
        # identifier on cardinality alone; only its name earns that label.
        is_id_pattern = matches_pattern_list(raw, IDENTIFIER_PATTERNS)
        is_id_like = is_id_pattern or (not is_numeric and _looks_like_id(series, n_rows))
        is_constant = n_unique <= 1
        is_text_meta = matches_pattern_list(raw, TEXT_ONLY_PATTERNS) and not is_numeric

        dt_hit = matches_any(raw, DATETIME_SYNONYMS)
        is_datetime = bool(dt_hit) or (not is_numeric and _try_parse_datetime(series))

        pollutant_hit = matches_any(raw, POLLUTANT_SYNONYMS)
        meteo_hit = matches_any(raw, METEOROLOGICAL_SYNONYMS)
        label_hit = matches_any(raw, LABEL_SYNONYMS)

        if is_datetime:
            dtype, role = "datetime", "datetime"
            profile.datetime_columns.append(raw)
        elif is_id_like:
            dtype = "numeric" if is_numeric else "categorical"
            role = "identifier"
            profile.identifier_columns.append(raw)
        elif is_constant:
            dtype = "numeric" if is_numeric else "categorical"
            role = "constant"
        elif is_text_meta:
            dtype, role = "text", "text"
        elif label_hit and not is_numeric:
            dtype, role = "categorical", "label"
            label_candidates.append(raw)
        elif pollutant_hit and pollutant_hit != "aqi":
            dtype, role = "numeric", "pollutant"
            profile.pollutant_columns.append(raw)
        elif pollutant_hit == "aqi":
            dtype, role = "numeric", "pollutant"
            profile.pollutant_columns.append(raw)
            if is_numeric:
                label_candidates.append(raw)  # numeric AQI can also seed a label
        elif meteo_hit:
            dtype, role = "numeric", "meteorological"
            profile.meteorological_columns.append(raw)
        elif is_numeric:
            dtype, role = "numeric", "feature_candidate"
        elif series.nunique(dropna=True) <= max(10, int(0.05 * n_rows)):
            dtype, role = "categorical", "feature_candidate"
        else:
            dtype, role = "text", "text"

        profile.columns.append(ColumnProfile(
            raw_name=raw, normalised_name=norm, dtype=dtype, role=role,
            missing_frac=missing_frac, n_unique=n_unique,
            is_constant=is_constant, is_high_cardinality_id=is_id_like,
        ))

    # Prefer an explicit categorical label column; fall back to a numeric AQI
    # column (preprocessing.py will bucket it into Low/Moderate/High).
    text_labels = [c for c in label_candidates
                   if next(cp for cp in profile.columns if cp.raw_name == c).dtype == "categorical"]
    profile.label_column = text_labels[0] if text_labels else (label_candidates[0] if label_candidates else None)

    for cp in profile.columns:
        if cp.role == "identifier":
            profile.excluded_columns[cp.raw_name] = "identifier"
        elif cp.role == "constant":
            profile.excluded_columns[cp.raw_name] = "constant column (no variation)"
        elif cp.role == "text":
            profile.excluded_columns[cp.raw_name] = "unsupported text field"
        elif cp.missing_frac > 0.60:
            profile.excluded_columns[cp.raw_name] = f"too much missing data ({cp.missing_frac:.0%})"
        elif cp.role == "datetime":
            profile.excluded_columns[cp.raw_name] = "datetime (used for time features, not as a raw feature)"
        elif cp.raw_name == profile.label_column:
            profile.excluded_columns[cp.raw_name] = "classification target"

    profile.feature_columns = [
        cp.raw_name for cp in profile.columns
        if cp.raw_name not in profile.excluded_columns and cp.dtype in ("numeric", "categorical")
    ]
    return profile


def compare_schemas(previous: dict, current: SchemaProfile) -> dict:
    """Compare a saved schema signature against a freshly inspected dataset.

    Returns a dict describing compatibility, used both to warn on prediction
    with a mismatched dataset and to report what changed on retraining.
    """
    prev_features = set(previous.get("feature_columns", []))
    curr_features = set(current.feature_columns)
    added = sorted(curr_features - prev_features)
    removed = sorted(prev_features - curr_features)
    return {
        "same_signature": previous.get("signature") == current.signature(),
        "added_columns": added,
        "removed_columns": removed,
        "previous_feature_count": len(prev_features),
        "current_feature_count": len(curr_features),
        "previous_row_count": previous.get("n_rows"),
        "current_row_count": current.n_rows,
        "compatible_for_prediction": len(removed) == 0,  # missing features block prediction
    }


# ----------------------------------------------------------------------------
# Long ("melted") format support
# ----------------------------------------------------------------------------

#: Many public open-data air-quality exports (Pakistan's included) publish
#: one row per (date, station, pollutant) rather than one column per
#: pollutant: e.g. ``Date, City, Type, Count`` where ``Type`` holds a value
#: like "PM2.5" and ``Count`` holds the reading. This is a structurally
#: different shape from what the rest of the pipeline expects, so it is
#: detected and pivoted to wide format before schema inspection runs.
TYPE_COLUMN_PATTERNS = (r"^type$", r"^parameter$", r"^pollutant$", r"^species$",
                        r"^measurement$", r"^param$")
VALUE_COLUMN_PATTERNS = (r"^count$", r"^value$", r"^concentration$", r"^reading$",
                         r"^result$", r"^measurement ?value$", r"^obs ?value$")
_KNOWN_POLLUTANT_TOKENS = {
    "pm2.5", "pm25", "pm2 5", "pm10", "co", "no2", "so2", "o3", "ozone", "aqi",
    "nox", "no", "voc", "pb",
}


def detect_long_format(df: pd.DataFrame) -> tuple[str, str] | None:
    """Return (type_column, value_column) if the dataframe looks like a
    melted pollutant-per-row export, else None."""
    type_candidates = [c for c in df.columns if matches_pattern_list(c, TYPE_COLUMN_PATTERNS)]
    value_candidates = [c for c in df.columns if matches_pattern_list(c, VALUE_COLUMN_PATTERNS)]
    if not type_candidates or not value_candidates:
        return None
    for type_col in type_candidates:
        values = set(df[type_col].dropna().astype(str).str.strip().str.lower().unique())
        overlap = sum(1 for v in values if v in _KNOWN_POLLUTANT_TOKENS or _norm_key(v) in _KNOWN_POLLUTANT_TOKENS)
        if overlap >= 2:
            value_col = next((v for v in value_candidates if pd.api.types.is_numeric_dtype(df[v])
                              or pd.to_numeric(df[v], errors="coerce").notna().mean() > 0.7), value_candidates[0])
            return type_col, value_col
    return None


def reshape_long_to_wide(df: pd.DataFrame, type_col: str, value_col: str) -> pd.DataFrame:
    """Pivot a melted (Date, City, Type, Count)-style frame to one column per
    pollutant. Duplicate (id, pollutant) combinations are averaged."""
    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    id_cols = [c for c in df.columns if c not in (type_col, value_col)]
    wide = df.pivot_table(index=id_cols, columns=type_col, values=value_col, aggfunc="mean")
    wide = wide.reset_index()
    wide.columns = [str(c).strip() for c in wide.columns]
    return wide

