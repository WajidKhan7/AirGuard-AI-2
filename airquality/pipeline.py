"""End-to-end orchestration for both modes.

``run_training_pipeline`` implements the flow in requirement 51 (TRAINING
MODE): inspect -> clean -> derive label -> split -> classify -> tune -> CV ->
anomaly detect -> forecast -> evaluate -> save. Every dataset that comes in
is re-inspected from scratch — nothing about a previous run is assumed to
still be true, which is what makes dynamic dataset modification (added rows,
added/removed columns) work without touching this code.

``run_prediction_pipeline`` implements PREDICTION MODE: load saved artifacts,
check the new dataset's schema against what the model was trained on, and
only proceed where it's actually compatible.

Both take an optional ``on_step(name, status, detail)`` callback so a caller
(the Tkinter GUI, or a test) can observe progress without this module
knowing anything about how progress is displayed.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import anomaly_detection as ad
from . import classification as clf
from . import data_loader
from . import evaluation as ev
from . import lstm_forecasting as lstmf
from . import preprocessing as prep
from . import schema as sch
from .validation import DatasetError, NotFittedError, SchemaError, validate_dataset_basic, validate_numeric_features

logger = logging.getLogger(__name__)

StepCallback = Callable[[str, str, str], None]
RANDOM_STATE = 42


def _noop(step: str, status: str, detail: str = "") -> None:
    pass


def _encode_features(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Numeric columns pass through; low-cardinality categorical columns are
    one-hot encoded. Returns the encoded frame and the final column list."""
    parts = [df[numeric_cols]] if numeric_cols else []
    encoded_cols = list(numeric_cols)
    if categorical_cols:
        dummies = pd.get_dummies(df[categorical_cols].astype(str), prefix=categorical_cols)
        parts.append(dummies)
        encoded_cols += list(dummies.columns)
    out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    return out, encoded_cols


# ----------------------------------------------------------------------------
# TRAINING MODE
# ----------------------------------------------------------------------------


def run_training_pipeline(
    csv_path: str,
    artifacts_dir: str = "artifacts",
    forecast_target: str | None = None,
    on_step: StepCallback = _noop,
) -> dict:
    t_start = time.perf_counter()
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    results: dict = {"branches": {}}

    # ---- load -----------------------------------------------------------
    on_step("Dataset Loading", "RUNNING")
    df, path = data_loader.load_dataset(csv_path)
    validate_dataset_basic(df)
    on_step("Dataset Loading", "DONE", f"{path.name}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    results["dataset_path"] = str(path)
    results["raw_shape"] = df.shape
    results["size_report"] = data_loader.dataset_size_report(len(df))

    # ---- reshape long/melted exports (e.g. Date, City, Type, Count) ------
    long_format = sch.detect_long_format(df)
    if long_format:
        on_step("Dataset Reshaping", "RUNNING")
        type_col, value_col = long_format
        pollutants = sorted(df[type_col].dropna().astype(str).str.strip().unique())
        df = sch.reshape_long_to_wide(df, type_col, value_col)
        on_step("Dataset Reshaping", "DONE",
                f"Detected long-format data ('{type_col}' / '{value_col}') and pivoted to wide "
                f"format: {len(pollutants)} pollutant column(s) created ({', '.join(pollutants[:8])}"
                f"{'...' if len(pollutants) > 8 else ''})")
        results["long_format_reshape"] = {"type_column": type_col, "value_column": value_col,
                                          "pollutants_found": pollutants, "new_shape": df.shape}
    else:
        on_step("Dataset Reshaping", "SKIPPED", "dataset is already in wide (one column per pollutant) format")

    # ---- inspect ----------------------------------------------------------
    on_step("Dataset Inspection", "RUNNING")
    profile = sch.inspect_dataset(df)
    results["schema"] = profile.to_dict()
    results["profile"] = profile
    on_step("Dataset Inspection", "DONE",
            f"{len(profile.pollutant_columns)} pollutant, "
            f"{len(profile.meteorological_columns)} meteorological, "
            f"{len(profile.datetime_columns)} datetime column(s) detected")

    # ---- schema-change comparison against a previous run -----------------
    prev_meta_path = artifacts / "metadata.json"
    schema_change = None
    if prev_meta_path.exists():
        try:
            prev_meta = json.loads(prev_meta_path.read_text())
            schema_change = sch.compare_schemas(prev_meta.get("schema", {}), profile)
            schema_change["previous_test_f1"] = (
                prev_meta.get("classification", {}).get("test_metrics", {}).get("f1_macro")
            )
        except Exception as exc:  # noqa: BLE001 - a corrupt old metadata file shouldn't block retraining
            logger.warning("could not read previous metadata.json: %s", exc)
    results["schema_change"] = schema_change

    # ---- clean --------------------------------------------------------
    on_step("Data Preprocessing", "RUNNING")
    cleaned, clean_report = prep.clean_dataset(df, profile)
    cleaned, label_col = prep.derive_label(cleaned, profile, clean_report)
    results["cleaning_report"] = clean_report
    results["label_column"] = label_col
    on_step("Data Preprocessing", "DONE", f"{clean_report.final_rows:,} usable rows remain")

    # ---- feature analysis --------------------------------------------
    on_step("Feature Analysis", "RUNNING")
    feature_cols = prep.select_classification_features(cleaned, profile, label_col)
    numeric_feat = [c for c in feature_cols
                    if next(cp for cp in profile.columns if cp.raw_name == c).dtype == "numeric"]
    categorical_feat = [c for c in feature_cols if c not in numeric_feat]
    results["feature_analysis"] = {
        "used_features": feature_cols,
        "numeric_features": numeric_feat,
        "categorical_features": categorical_feat,
        "excluded_columns": profile.excluded_columns,
    }
    on_step("Feature Analysis", "DONE",
            f"{len(feature_cols)} feature(s) selected, {len(profile.excluded_columns)} column(s) excluded")

    try:
        validate_numeric_features(numeric_feat)
    except SchemaError as exc:
        on_step("Feature Analysis", "ERROR", str(exc))
        raise

    X_all, encoded_cols = _encode_features(cleaned, numeric_feat, categorical_feat)
    results["encoded_feature_columns"] = encoded_cols

    # ---- classification branch --------------------------------------
    on_step("Train/Validation/Test Split", "RUNNING")
    classification_result = None
    if label_col:
        y_all = cleaned[label_col]
        class_counts = y_all.value_counts()
        if len(class_counts) < 2:
            results["branches"]["classification"] = {
                "available": False, "reason": f"Only one class present ('{class_counts.index[0]}')."
            }
            on_step("Train/Validation/Test Split", "SKIPPED", "classification: only one class present")
        elif class_counts.min() < 2:
            rare = class_counts[class_counts < 2].index.tolist()
            results["branches"]["classification"] = {
                "available": False,
                "reason": f"Class(es) {rare} have fewer than 2 samples and can't be split/stratified."
            }
            on_step("Train/Validation/Test Split", "SKIPPED", "classification: a class has <2 samples")
        else:
            X_tr, X_tmp, y_tr, y_tmp = train_test_split(
                X_all.to_numpy(dtype=np.float32), y_all.to_numpy(),
                train_size=0.70, stratify=y_all, random_state=RANDOM_STATE,
            )
            X_va, X_te, y_va, y_te = train_test_split(
                X_tmp, y_tmp, train_size=0.5, stratify=y_tmp, random_state=RANDOM_STATE,
            )
            on_step("Train/Validation/Test Split", "DONE",
                    f"train={len(X_tr):,} val={len(X_va):,} test={len(X_te):,}")

            on_step("Classification Training", "RUNNING")
            fitted_models, val_comparison = clf.train_and_compare(X_tr, y_tr, X_va, y_va)
            on_step("Classification Training", "DONE", f"{len(fitted_models)} baseline model(s) trained")

            on_step("Hyperparameter Tuning", "RUNNING")
            tuning = clf.tune_random_forest(X_tr, y_tr, n_rows=len(df))
            on_step("Hyperparameter Tuning", "DONE",
                    f"{tuning['method']}, best CV f1_macro={tuning['best_cv_score']}")

            on_step("Cross-Validation", "RUNNING")
            cv_table = clf.cross_validate_models(X_tr, y_tr)
            on_step("Cross-Validation", "DONE", f"best: {cv_table.iloc[0]['model']}")

            # Final model: whichever of (best baseline, tuned RF) wins on the
            # validation split — never chosen using the test set.
            tuned_pred_val = tuning["model"].predict(X_va)
            tuned_val_f1 = ev.classification_metrics(y_va, tuned_pred_val, sorted(class_counts.index))["f1_macro"]
            best_baseline_f1 = val_comparison.iloc[0]["val_f1_macro"]
            if tuned_val_f1 >= best_baseline_f1:
                final_model, final_name = tuning["model"], "Random Forest (tuned)"
            else:
                final_name = val_comparison.iloc[0]["model"]
                final_model = fitted_models[final_name]

            labels = sorted(class_counts.index.tolist())
            test_pred = final_model.predict(X_te)
            test_metrics = ev.classification_metrics(y_te, test_pred, labels)
            per_class = ev.per_class_report(y_te, test_pred, labels)
            cm_table = ev.confusion_table(y_te, test_pred, labels)
            cm_path = ev.plot_confusion_matrix(y_te, test_pred, labels, artifacts / "confusion_matrix.png")

            importance_df = clf.feature_importance(
                tuning["model"] if final_name.startswith("Random Forest") else fitted_models["Random Forest"],
                encoded_cols,
            )
            fi_path = ev.plot_feature_importance(importance_df, artifacts / "feature_importance.png")
            cmp_path = ev.plot_model_comparison(val_comparison, artifacts / "model_comparison.png")

            on_step("Evaluation", "DONE",
                    f"final model: {final_name}, test f1_macro={test_metrics['f1_macro']}")

            classification_result = {
                "available": True,
                "label_column": label_col,
                "classes": labels,
                "final_model_name": final_name,
                "final_model": final_model,
                "split_sizes": {"train": len(X_tr), "val": len(X_va), "test": len(X_te)},
                "tuning": {k: v for k, v in tuning.items() if k != "model"},
                "cross_validation": cv_table,
                "validation_comparison": val_comparison,
                "test_metrics": test_metrics,
                "per_class_report": per_class,
                "confusion_matrix": cm_table,
                "confusion_matrix_path": str(cm_path),
                "feature_importance": importance_df,
                "feature_importance_path": str(fi_path),
                "model_comparison_path": str(cmp_path),
            }
            results["branches"]["classification"] = classification_result
    else:
        on_step("Train/Validation/Test Split", "SKIPPED", "no classification label available")
        results["branches"]["classification"] = {
            "available": False, "reason": "No classification label was found or could be derived."
        }

    # ---- anomaly detection branch -------------------------------------
    on_step("Anomaly Detection", "RUNNING")
    X_numeric = cleaned[numeric_feat].to_numpy(dtype=np.float32) if numeric_feat else None
    anomaly_result = None
    if X_numeric is not None and len(X_numeric) >= 20:
        contamination = float(np.clip(0.05, 0.005, 0.5))
        anomaly_model = ad.train_anomaly_detector(X_numeric, contamination=contamination)
        anomaly_scores = ad.score_anomalies(anomaly_model, X_numeric, numeric_feat)
        summary = ad.summarise(anomaly_scores)
        anomaly_scores.to_csv(artifacts / "anomaly_results.csv", index=False)
        anomaly_result = {"available": True, "model": anomaly_model, "results": anomaly_scores,
                          "summary": summary, "features_used": numeric_feat}
        on_step("Anomaly Detection", "DONE",
                f"{summary['anomalies']:,} anomalies of {summary['total_observations']:,} "
                f"({summary['anomaly_percentage']}%)")
    else:
        anomaly_result = {"available": False, "reason": "Not enough numeric data for anomaly detection."}
        on_step("Anomaly Detection", "SKIPPED", anomaly_result["reason"])
    results["branches"]["anomaly"] = anomaly_result

    # ---- LSTM forecasting branch ---------------------------------------
    on_step("LSTM Training", "RUNNING")
    targets = lstmf.find_forecast_targets(profile, cleaned)
    chosen_target = forecast_target if forecast_target in targets else (targets[0] if targets else None)
    if chosen_target is None:
        lstm_result = {"available": False,
                       "reason": "No valid datetime column and/or no numeric pollutant target with "
                                 "enough values was detected."}
        on_step("LSTM Training", "SKIPPED", lstm_result["reason"])
    else:
        lstm_result = lstmf.run_lstm_pipeline(cleaned, profile, chosen_target)
        lstm_result["available_targets"] = targets
        if lstm_result["available"]:
            fpath = ev.plot_forecast(lstm_result["test_true"], lstm_result["test_pred"],
                                     lstm_result["future_forecast"], artifacts / "lstm_forecast.png")
            lstm_result["forecast_plot_path"] = str(fpath)
            on_step("LSTM Training", "DONE",
                    f"target={chosen_target} test MAE={lstm_result['test_metrics']['mae']} "
                    f"RMSE={lstm_result['test_metrics']['rmse']} R2={lstm_result['test_metrics']['r2']}")
        else:
            on_step("LSTM Training", "SKIPPED", lstm_result["reason"])
    results["branches"]["lstm"] = lstm_result

    # ---- end-to-end sanity check ---------------------------------------
    on_step("End-to-End Testing", "RUNNING")
    completed = [k for k, v in results["branches"].items() if v and v.get("available")]
    on_step("End-to-End Testing", "DONE", f"branches completed: {completed or 'none'}")

    # ---- save artifacts --------------------------------------------
    on_step("Saving Artifacts", "RUNNING")
    if classification_result:
        joblib.dump(classification_result["final_model"], artifacts / "classification_model.pkl")
        classification_result["per_class_report"].to_csv(artifacts / "classification_report.csv", index=False)
        (artifacts / "classification_report.txt").write_text(
            f"Final model: {classification_result['final_model_name']}\n\n"
            f"Test metrics:\n{json.dumps(classification_result['test_metrics'], indent=2)}\n\n"
            f"Per-class report:\n{classification_result['per_class_report'].to_string(index=False)}\n\n"
            f"Confusion matrix:\n{classification_result['confusion_matrix'].to_string()}\n"
        )
        classification_result["validation_comparison"].to_csv(artifacts / "model_comparison.csv", index=False)
    if anomaly_result and anomaly_result.get("available"):
        joblib.dump(anomaly_result["model"], artifacts / "anomaly_model.pkl")
    if lstm_result and lstm_result.get("available"):
        lstm_result["model"].save(artifacts / "lstm_model.pt")

    joblib.dump({
        "numeric_features": numeric_feat, "categorical_features": categorical_feat,
        "encoded_columns": encoded_cols, "label_column": label_col,
    }, artifacts / "preprocessing.pkl")

    metadata = {
        "created_utc": pd.Timestamp.now("UTC").isoformat(),
        "dataset_path": str(path),
        "schema": profile.to_dict(),
        "classification": ({"available": True, "final_model_name": classification_result["final_model_name"],
                            "test_metrics": classification_result["test_metrics"],
                            "classes": classification_result["classes"]}
                           if classification_result else {"available": False}),
        "anomaly": {"available": bool(anomaly_result and anomaly_result.get("available"))},
        "lstm": ({"available": True, "target": lstm_result["target"], "lookback": lstm_result["lookback"]}
                if lstm_result and lstm_result.get("available") else {"available": False}),
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    (artifacts / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    on_step("Saving Artifacts", "DONE", f"artifacts written to {artifacts.resolve()}")

    results["metadata"] = metadata
    results["runtime_seconds"] = metadata["runtime_seconds"]

    # performance-change reporting (requirement 37)
    if schema_change is not None and classification_result:
        prev_f1 = schema_change.get("previous_test_f1")
        if prev_f1 is not None:
            results["performance_change"] = {
                "previous_test_f1": prev_f1,
                "new_test_f1": classification_result["test_metrics"]["f1_macro"],
                "delta": round(classification_result["test_metrics"]["f1_macro"] - prev_f1, 4),
            }
    return results


# ----------------------------------------------------------------------------
# PREDICTION MODE
# ----------------------------------------------------------------------------


def load_trained_artifacts(artifacts_dir: str = "artifacts") -> dict:
    artifacts = Path(artifacts_dir)
    meta_path = artifacts / "metadata.json"
    if not meta_path.exists():
        raise NotFittedError(
            f"No trained model found in {artifacts.resolve()}.\nRun training first (python train.py)."
        )
    metadata = json.loads(meta_path.read_text())
    preprocessing = joblib.load(artifacts / "preprocessing.pkl")
    bundle = {"metadata": metadata, "preprocessing": preprocessing, "artifacts_dir": artifacts}

    if metadata.get("classification", {}).get("available") and (artifacts / "classification_model.pkl").exists():
        bundle["classification_model"] = joblib.load(artifacts / "classification_model.pkl")
    if metadata.get("anomaly", {}).get("available") and (artifacts / "anomaly_model.pkl").exists():
        bundle["anomaly_model"] = joblib.load(artifacts / "anomaly_model.pkl")
    if metadata.get("lstm", {}).get("available") and (artifacts / "lstm_model.pt").exists():
        bundle["lstm_model"] = lstmf.LSTMForecaster.load(artifacts / "lstm_model.pt")
    return bundle


def check_prediction_compatibility(bundle: dict, profile: sch.SchemaProfile) -> dict:
    return sch.compare_schemas(bundle["metadata"]["schema"], profile)


def run_prediction_pipeline(csv_path: str, artifacts_dir: str = "artifacts",
                            on_step: StepCallback = _noop) -> dict:
    on_step("Loading Trained Model", "RUNNING")
    bundle = load_trained_artifacts(artifacts_dir)
    on_step("Loading Trained Model", "DONE",
            f"model trained on {bundle['metadata'].get('dataset_path', 'unknown dataset')}")

    on_step("Dataset Loading", "RUNNING")
    df, path = data_loader.load_dataset(csv_path)
    validate_dataset_basic(df)
    on_step("Dataset Loading", "DONE", f"{path.name}: {df.shape[0]:,} rows x {df.shape[1]} columns")

    long_format = sch.detect_long_format(df)
    if long_format:
        on_step("Dataset Reshaping", "RUNNING")
        type_col, value_col = long_format
        df = sch.reshape_long_to_wide(df, type_col, value_col)
        on_step("Dataset Reshaping", "DONE", f"pivoted long-format data using '{type_col}' / '{value_col}'")
    else:
        on_step("Dataset Reshaping", "SKIPPED", "already wide format")

    on_step("Compatibility Check", "RUNNING")
    profile = sch.inspect_dataset(df)
    compat = check_prediction_compatibility(bundle, profile)
    prep_info = bundle["preprocessing"]
    required = set(prep_info["numeric_features"]) | set(prep_info["categorical_features"])
    present = set(df.columns)
    missing_required = sorted(required - present)
    extra_present = sorted(present - required - {prep_info.get("label_column")})

    if missing_required:
        on_step("Compatibility Check", "ERROR", f"missing required feature(s): {missing_required}")
        return {
            "compatible": False,
            "missing_required_features": missing_required,
            "extra_columns": extra_present,
        }
    on_step("Compatibility Check", "DONE",
            "PASS" if not extra_present else f"PASS (extra columns ignored: {extra_present})")

    cleaned, clean_report = prep.clean_dataset(df, profile)
    X_all, _ = _encode_features(cleaned, prep_info["numeric_features"], prep_info["categorical_features"])
    for col in prep_info["encoded_columns"]:
        if col not in X_all.columns:
            X_all[col] = 0.0            # a one-hot category seen in training but absent here
    X_all = X_all[prep_info["encoded_columns"]]

    result = {"compatible": True, "extra_columns": extra_present, "n_rows": len(cleaned)}

    if "classification_model" in bundle:
        on_step("Classification", "RUNNING")
        model = bundle["classification_model"]
        preds = model.predict(X_all.to_numpy(dtype=np.float32))
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_all.to_numpy(dtype=np.float32))
            confidence = proba.max(axis=1)
        else:
            confidence = np.full(len(preds), np.nan)
        result["classification"] = {
            "predictions": preds, "confidence": confidence,
            "classes": bundle["metadata"]["classification"]["classes"],
        }
        on_step("Classification", "DONE", f"{len(preds)} row(s) classified")
    else:
        on_step("Classification", "SKIPPED", "no classification model in this trained run")

    if "anomaly_model" in bundle:
        on_step("Anomaly Detection", "RUNNING")
        numeric_feat = prep_info["numeric_features"]
        X_num = cleaned[numeric_feat].to_numpy(dtype=np.float32) if numeric_feat else None
        if X_num is not None and len(X_num):
            scores = ad.score_anomalies(bundle["anomaly_model"], X_num)
            result["anomaly"] = {"results": scores, "summary": ad.summarise(scores)}
            on_step("Anomaly Detection", "DONE", str(result["anomaly"]["summary"]))
        else:
            on_step("Anomaly Detection", "SKIPPED", "no numeric features available")
    else:
        on_step("Anomaly Detection", "SKIPPED", "no anomaly model in this trained run")

    if "lstm_model" in bundle:
        on_step("LSTM Forecast", "RUNNING")
        target = bundle["metadata"]["lstm"]["target"]
        datetimes = lstmf.build_datetime_index(cleaned, profile)
        if datetimes is not None and target in cleaned.columns:
            work = cleaned.copy()
            work["_dt"] = datetimes
            work = work.dropna(subset=["_dt", target]).sort_values("_dt")
            series = work[target].to_numpy(dtype=np.float32)
            lookback = bundle["lstm_model"].lookback
            if len(series) > lookback:
                preds = bundle["lstm_model"].predict(series)
                result["lstm"] = {"target": target, "predictions": preds,
                                  "last_actual": float(series[-1])}
                on_step("LSTM Forecast", "DONE", f"{len(preds)} forecast point(s) for '{target}'")
            else:
                on_step("LSTM Forecast", "SKIPPED", f"not enough rows ({len(series)}) for lookback {lookback}")
        else:
            on_step("LSTM Forecast", "SKIPPED", "no usable datetime/target column in this dataset")
    else:
        on_step("LSTM Forecast", "SKIPPED", "no LSTM model in this trained run")

    return result
