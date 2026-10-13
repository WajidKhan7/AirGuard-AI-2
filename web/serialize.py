"""Turn the pipeline's Python-native result dicts (pandas DataFrames,
dataclasses, sklearn/torch model objects, numpy arrays) into plain JSON-safe
structures for the HTTP API — without touching the pipeline itself.

Nothing here changes what the models compute; it only reshapes the same
results the Tkinter GUI already displays.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _clean_value(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, np.ndarray):
        return [_clean_value(x) for x in v.tolist()]
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    return v


def df_records(df: pd.DataFrame | None, limit: int | None = None) -> list[dict]:
    if df is None:
        return []
    out = df.reset_index()
    if limit is not None:
        out = out.head(limit)
    return [{k: _clean_value(v) for k, v in row.items()} for row in out.to_dict(orient="records")]


def _artifact_url(path_str: str | None) -> str | None:
    if not path_str:
        return None
    return f"/api/artifacts/{Path(path_str).name}"


def serialize_classification(cls: dict | None) -> dict | None:
    if not cls:
        return None
    if not cls.get("available"):
        return {"available": False, "reason": cls.get("reason")}
    return {
        "available": True,
        "label_column": cls["label_column"],
        "classes": cls["classes"],
        "final_model_name": cls["final_model_name"],
        "split_sizes": cls["split_sizes"],
        "tuning": {k: _clean_value(v) for k, v in cls["tuning"].items()},
        "cross_validation": df_records(cls["cross_validation"]),
        "validation_comparison": df_records(cls["validation_comparison"]),
        "test_metrics": {k: _clean_value(v) for k, v in cls["test_metrics"].items()},
        "per_class_report": df_records(cls["per_class_report"]),
        "confusion_matrix": {
            "labels": cls["classes"],
            "rows": df_records(cls["confusion_matrix"]),
        },
        "confusion_matrix_image": _artifact_url(cls.get("confusion_matrix_path")),
        "feature_importance": df_records(cls["feature_importance"], limit=25),
        "feature_importance_image": _artifact_url(cls.get("feature_importance_path")),
        "model_comparison_image": _artifact_url(cls.get("model_comparison_path")),
    }


def serialize_anomaly(an: dict | None) -> dict | None:
    if not an:
        return None
    if not an.get("available"):
        return {"available": False, "reason": an.get("reason")}
    flagged = an["results"][an["results"]["status"] == "Anomaly"]
    return {
        "available": True,
        "summary": {k: _clean_value(v) for k, v in an["summary"].items()},
        "features_used": an["features_used"],
        "sample_anomalies": df_records(flagged, limit=100),
    }


def serialize_lstm(l: dict | None) -> dict | None:
    if not l:
        return None
    if not l.get("available"):
        return {"available": False, "reason": l.get("reason"),
                "available_targets": l.get("available_targets", [])}
    return {
        "available": True,
        "target": l["target"],
        "lookback": l["lookback"],
        "n_rows_used": l["n_rows_used"],
        "split_sizes": l["split_sizes"],
        "validation_metrics": {k: _clean_value(v) for k, v in l["validation_metrics"].items()},
        "test_metrics": {k: _clean_value(v) for k, v in l["test_metrics"].items()},
        "forecast_image": _artifact_url(l.get("forecast_plot_path")),
    }


def serialize_training_results(r: dict) -> dict:
    """Full training-pipeline result -> JSON-safe dict for the API response."""
    schema_change = r.get("schema_change")
    performance_change = r.get("performance_change")
    long_format = r.get("long_format_reshape")
    cleaning = r["cleaning_report"]
    cleaning_dict = dataclasses.asdict(cleaning) if dataclasses.is_dataclass(cleaning) else dict(cleaning)

    return {
        "dataset_path": r["dataset_path"],
        "raw_shape": list(r["raw_shape"]),
        "size_report": r["size_report"],
        "schema": r["schema"],
        "long_format_reshape": (
            {"type_column": long_format["type_column"], "value_column": long_format["value_column"],
             "pollutants_found": long_format["pollutants_found"], "new_shape": list(long_format["new_shape"])}
            if long_format else None
        ),
        "cleaning_report": {k: _clean_value(v) for k, v in cleaning_dict.items()},
        "label_column": r.get("label_column"),
        "feature_analysis": r["feature_analysis"],
        "schema_change": schema_change,
        "performance_change": performance_change,
        "branches": {
            "classification": serialize_classification(r["branches"].get("classification")),
            "anomaly": serialize_anomaly(r["branches"].get("anomaly")),
            "lstm": serialize_lstm(r["branches"].get("lstm")),
        },
        "runtime_seconds": r.get("runtime_seconds"),
    }


def serialize_prediction_results(r: dict) -> dict:
    if not r.get("compatible", True):
        return {
            "compatible": False,
            "missing_required_features": r.get("missing_required_features", []),
            "extra_columns": r.get("extra_columns", []),
        }
    out: dict[str, Any] = {
        "compatible": True,
        "n_rows": r["n_rows"],
        "extra_columns": r.get("extra_columns", []),
    }
    if "classification" in r:
        c = r["classification"]
        preds = list(c["predictions"])
        conf = [_clean_value(x) for x in c["confidence"]]
        out["classification"] = {
            "classes": c["classes"],
            "predictions_preview": preds[:200],
            "confidence_preview": conf[:200],
            "total_predictions": len(preds),
            "class_counts": {cls: int(sum(1 for p in preds if p == cls)) for cls in c["classes"]},
        }
    if "anomaly" in r:
        out["anomaly"] = {"summary": {k: _clean_value(v) for k, v in r["anomaly"]["summary"].items()}}
    if "lstm" in r:
        l = r["lstm"]
        preds = [_clean_value(x) for x in l["predictions"]]
        out["lstm"] = {"target": l["target"], "predictions_preview": preds[:200],
                      "total_predictions": len(preds), "last_actual": _clean_value(l["last_actual"])}
    return out
