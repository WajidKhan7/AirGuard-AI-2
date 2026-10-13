#!/usr/bin/env python3
"""AirGuard AI — Training Mode (desktop app).

    python train.py

Opens a desktop window: pick a CSV, click Start Training, watch the
pipeline run, then browse the results dashboard. No terminal interaction
and no web technology anywhere — this is a plain Tkinter application.
"""

from __future__ import annotations

import logging
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from airquality.gui import (
    ThreadedRunner, StatusPanel, TRAIN_STEPS, dataframe_to_treeview,
    load_image_scaled, pick_csv_file, show_error, show_info,
)
from airquality.pipeline import run_training_pipeline
from airquality.validation import AirQualityError

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-7s %(message)s")

ARTIFACTS_DIR = "artifacts"


class TrainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AirGuard AI - Air Quality Monitoring")
        self.geometry("980x720")
        self.minsize(860, 620)
        self.csv_path = tk.StringVar(value="")
        self._active_runner = None
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_select_screen()

    # ------------------------------------------------------------------
    def _build_select_screen(self) -> None:
        for w in self.winfo_children():
            w.destroy()
        frame = ttk.Frame(self, padding=30)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="AIRGUARD AI", font=("Segoe UI", 22, "bold")).pack(pady=(10, 0))
        ttk.Label(frame, text="Air Quality Machine Learning System",
                 font=("Segoe UI", 12)).pack(pady=(0, 30))

        ttk.Button(frame, text="Select CSV Dataset", command=self._on_select).pack(pady=6)
        self.path_label = ttk.Label(frame, textvariable=self.csv_path, foreground="#333333",
                                    wraplength=700)
        self.path_label.pack(pady=(4, 20))

        ttk.Label(frame, text=(
            "Works with any air-quality-style CSV: pollutant readings, "
            "meteorological columns and/or a date/time column, in any naming style.\n"
            "Missing columns simply mean the corresponding branch (classification / "
            "anomaly detection / forecasting) is skipped with a clear reason."
        ), foreground="#666666", wraplength=760, justify="center").pack(pady=(0, 20))

        self.start_btn = ttk.Button(frame, text="Start Training", command=self._on_start,
                                    state="disabled")
        self.start_btn.pack(pady=10)

        forecast_frame = ttk.LabelFrame(frame, text="Optional: LSTM forecast target", padding=10)
        forecast_frame.pack(pady=10, fill="x")
        ttk.Label(forecast_frame,
                 text="Leave on Auto to let the system pick automatically after inspecting the dataset.",
                 foreground="#666666").pack(anchor="w")

    def _on_select(self) -> None:
        path = pick_csv_file(self)
        if path:
            self.csv_path.set(path)
            self.start_btn.configure(state="normal")

    def _on_start(self) -> None:
        self._build_training_screen()

    # ------------------------------------------------------------------
    def _build_training_screen(self) -> None:
        for w in self.winfo_children():
            w.destroy()
        frame = ttk.Frame(self, padding=20)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Training in progress...", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(frame, text=Path(self.csv_path.get()).name, foreground="#666666").pack(anchor="w", pady=(0, 10))

        self.status_panel = StatusPanel(frame, TRAIN_STEPS)
        self.status_panel.pack(fill="both", expand=True)

        def target_fn(on_step):
            return run_training_pipeline(self.csv_path.get(), artifacts_dir=ARTIFACTS_DIR, on_step=on_step)

        runner = ThreadedRunner(self, target_fn, self.status_panel,
                                on_done=self._on_training_done, on_error=self._on_training_error)
        self._active_runner = runner
        runner.start()

    def _on_close(self) -> None:
        if self._active_runner is not None:
            self._active_runner.cancel()
        self.destroy()

    def _on_training_error(self, message: str) -> None:
        show_error(self, "Training Error", message)
        self._build_select_screen()

    def _on_training_done(self, results: dict) -> None:
        self.results = results
        self._build_results_screen()

    # ------------------------------------------------------------------
    def _build_results_screen(self) -> None:
        for w in self.winfo_children():
            w.destroy()
        r = self.results

        top = ttk.Frame(self, padding=(20, 12, 20, 0))
        top.pack(fill="x")
        ttk.Label(top, text="Training Complete", font=("Segoe UI", 16, "bold")).pack(side="left")
        ttk.Button(top, text="Train With Another Dataset",
                  command=self._build_select_screen).pack(side="right")
        ttk.Button(top, text="Retrain (same dataset)",
                  command=lambda: self._retrain(r["dataset_path"])).pack(side="right", padx=8)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=20, pady=12)

        self._tab_overview(notebook, r)
        self._tab_classification(notebook, r)
        self._tab_confusion(notebook, r)
        self._tab_feature_importance(notebook, r)
        self._tab_anomaly(notebook, r)
        self._tab_lstm(notebook, r)
        self._tab_comparison(notebook, r)

        bottom = ttk.Frame(self, padding=(20, 0, 20, 14))
        bottom.pack(fill="x")
        ttk.Button(bottom, text="Save Results",
                  command=lambda: show_info(self, "Saved",
                                            f"All artifacts are saved in:\n{Path(ARTIFACTS_DIR).resolve()}")
                 ).pack(side="left")
        ttk.Button(bottom, text="Close", command=self.destroy).pack(side="right")

    def _retrain(self, csv_path: str) -> None:
        self.csv_path.set(csv_path)
        self._build_training_screen()

    # ------------------------------------------------------------------
    def _tab_overview(self, notebook, r) -> None:
        tab = ttk.Frame(notebook, padding=16)
        notebook.add(tab, text="Overview")
        schema = r["schema"]
        lines = [
            f"Dataset: {Path(r['dataset_path']).name}",
            f"Rows: {r['raw_shape'][0]:,}    Columns: {r['raw_shape'][1]}",
            f"Cleaned rows: {r['cleaning_report'].final_rows:,}",
            "",
            f"Detected pollutant columns: {', '.join(schema['pollutant_columns']) or '(none)'}",
            f"Detected meteorological columns: {', '.join(schema['meteorological_columns']) or '(none)'}",
            f"Detected datetime columns: {', '.join(schema['datetime_columns']) or '(none)'}",
            f"Label column: {r.get('label_column') or '(derived / none)'}",
            "",
            r["size_report"],
        ]
        if r.get("schema_change") is not None:
            sc = r["schema_change"]
            lines += ["", "--- Compared to the previous training run ---",
                     f"Columns added: {sc['added_columns'] or 'none'}",
                     f"Columns removed: {sc['removed_columns'] or 'none'}",
                     f"Row count: {sc['previous_row_count']:,} -> {sc['current_row_count']:,}"]
            if r.get("performance_change"):
                pc = r["performance_change"]
                lines.append(f"Test F1 (macro): {pc['previous_test_f1']} -> {pc['new_test_f1']} "
                             f"({'+' if pc['delta']>=0 else ''}{pc['delta']})")
        cls = r["branches"].get("classification") or {}
        if cls.get("available"):
            lines += ["", f"Train / Val / Test samples: {cls['split_sizes']['train']:,} / "
                          f"{cls['split_sizes']['val']:,} / {cls['split_sizes']['test']:,}"]

        text = tk.Text(tab, wrap="word", height=26, font=("Consolas", 10))
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")
        text.pack(fill="both", expand=True)

    def _tab_classification(self, notebook, r) -> None:
        tab = ttk.Frame(notebook, padding=16)
        notebook.add(tab, text="Classification")
        cls = r["branches"].get("classification") or {}
        if not cls.get("available"):
            ttk.Label(tab, text=f"Classification: NOT AVAILABLE\n\nReason: {cls.get('reason', 'unknown')}",
                     foreground=("#c1121f")).pack(anchor="w")
            return
        m = cls["test_metrics"]
        lines = [
            f"Final model: {cls['final_model_name']}",
            f"Classes: {', '.join(cls['classes'])}",
            "",
            "TEST-SET PERFORMANCE (previously unseen data)",
            f"  Accuracy:            {m['accuracy']}",
            f"  Precision (macro):   {m['precision_macro']}",
            f"  Recall (macro):      {m['recall_macro']}",
            f"  F1-score (macro):    {m['f1_macro']}",
            f"  Precision (weighted):{m['precision_weighted']}",
            f"  Recall (weighted):   {m['recall_weighted']}",
            f"  F1-score (weighted): {m['f1_weighted']}",
            "",
            f"Cross-validation (training split, best model: "
            f"{cls['cross_validation'].iloc[0]['model']}):",
            f"  Mean F1 (macro): {cls['cross_validation'].iloc[0]['cv_mean_f1_macro']}"
            f"  Std: {cls['cross_validation'].iloc[0]['cv_std_f1_macro']}",
            "",
            f"Hyperparameter tuning ({cls['tuning']['method']}):",
            f"  Best params: {cls['tuning']['best_params']}",
            f"  Best CV score: {cls['tuning']['best_cv_score']}",
        ]
        text = tk.Text(tab, wrap="word", height=20, font=("Consolas", 10))
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")
        text.pack(fill="both", expand=True)

        ttk.Label(tab, text="Per-class report:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 2))
        tree_frame = ttk.Frame(tab)
        tree_frame.pack(fill="x")
        tree = dataframe_to_treeview(tree_frame, cls["per_class_report"], height=len(cls["classes"]) + 2)
        tree.pack(fill="x")

    def _tab_confusion(self, notebook, r) -> None:
        tab = ttk.Frame(notebook, padding=16)
        notebook.add(tab, text="Confusion Matrix")
        cls = r["branches"].get("classification") or {}
        if not cls.get("available"):
            ttk.Label(tab, text="Not available (no classification model).").pack(anchor="w")
            return
        img = load_image_scaled(cls["confusion_matrix_path"])
        if img:
            lbl = ttk.Label(tab, image=img)
            lbl.image = img
            lbl.pack(pady=10)
        text = tk.Text(tab, height=8, font=("Consolas", 9))
        text.insert("1.0", cls["confusion_matrix"].to_string())
        text.configure(state="disabled")
        text.pack(fill="x", pady=10)

    def _tab_feature_importance(self, notebook, r) -> None:
        tab = ttk.Frame(notebook, padding=16)
        notebook.add(tab, text="Feature Importance")
        cls = r["branches"].get("classification") or {}
        if not cls.get("available"):
            ttk.Label(tab, text="Not available (no classification model).").pack(anchor="w")
            return
        img = load_image_scaled(cls["feature_importance_path"])
        if img:
            lbl = ttk.Label(tab, image=img)
            lbl.image = img
            lbl.pack(pady=10)

        fa = r["feature_analysis"]
        note = (f"Useful features ({len(fa['used_features'])}): {', '.join(fa['used_features'])}\n\n"
               f"Ignored columns:\n" +
               "\n".join(f"  {c} -> {reason}" for c, reason in fa["excluded_columns"].items()))
        text = tk.Text(tab, wrap="word", height=8, font=("Consolas", 9))
        text.insert("1.0", note)
        text.configure(state="disabled")
        text.pack(fill="x", pady=10)

    def _tab_anomaly(self, notebook, r) -> None:
        tab = ttk.Frame(notebook, padding=16)
        notebook.add(tab, text="Anomaly Detection")
        an = r["branches"].get("anomaly") or {}
        if not an.get("available"):
            ttk.Label(tab, text=f"Anomaly detection: NOT AVAILABLE\n\nReason: {an.get('reason', 'unknown')}",
                     foreground="#c1121f").pack(anchor="w")
            return
        s = an["summary"]
        lines = [
            f"Total observations: {s['total_observations']:,}",
            f"Normal: {s['normal_observations']:,}",
            f"Anomalies: {s['anomalies']:,}",
            f"Anomaly percentage: {s['anomaly_percentage']}%",
            "",
            f"Features used: {', '.join(an['features_used'])}",
        ]
        ttk.Label(tab, text="\n".join(lines), font=("Consolas", 10), justify="left").pack(anchor="w")

        ttk.Label(tab, text="Sample of flagged results:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(14, 2))
        flagged = an["results"][an["results"]["status"] == "Anomaly"].head(30)
        if not flagged.empty:
            tree_frame = ttk.Frame(tab)
            tree_frame.pack(fill="both", expand=True)
            dataframe_to_treeview(tree_frame, flagged.reset_index().rename(columns={"index": "row"}),
                                  height=10).pack(fill="both", expand=True)

    def _tab_lstm(self, notebook, r) -> None:
        tab = ttk.Frame(notebook, padding=16)
        notebook.add(tab, text="LSTM Forecasting")
        l = r["branches"].get("lstm") or {}
        if not l.get("available"):
            ttk.Label(tab, text=f"LSTM: NOT AVAILABLE\n\nReason: {l.get('reason', 'unknown')}",
                     foreground="#c1121f").pack(anchor="w")
            available_targets = l.get("available_targets") or []
            if available_targets:
                ttk.Label(tab, text=f"Available forecast targets in this dataset: "
                                    f"{', '.join(available_targets)}", foreground="#666666").pack(anchor="w", pady=(8,0))
            return
        img = load_image_scaled(l["forecast_plot_path"])
        if img:
            lbl = ttk.Label(tab, image=img)
            lbl.image = img
            lbl.pack(pady=10)
        vm, tm = l["validation_metrics"], l["test_metrics"]
        lines = [
            f"Forecast target: {l['target']}     Lookback window: {l['lookback']} steps",
            f"Rows used: {l['n_rows_used']:,}   "
            f"(train={l['split_sizes']['train']:,} val={l['split_sizes']['val']:,} test={l['split_sizes']['test']:,})",
            "",
            f"Validation  MAE={vm['mae']}  RMSE={vm['rmse']}  R2={vm['r2']}",
            f"Test        MAE={tm['mae']}  RMSE={tm['rmse']}  R2={tm['r2']}",
        ]
        ttk.Label(tab, text="\n".join(lines), font=("Consolas", 10), justify="left").pack(anchor="w")

    def _tab_comparison(self, notebook, r) -> None:
        tab = ttk.Frame(notebook, padding=16)
        notebook.add(tab, text="Model Comparison")
        cls = r["branches"].get("classification") or {}
        if not cls.get("available"):
            ttk.Label(tab, text="Not available (no classification model).").pack(anchor="w")
            return
        img = load_image_scaled(cls["model_comparison_path"])
        if img:
            lbl = ttk.Label(tab, image=img)
            lbl.image = img
            lbl.pack(pady=10)
        tree_frame = ttk.Frame(tab)
        tree_frame.pack(fill="x")
        dataframe_to_treeview(tree_frame, cls["validation_comparison"], height=4).pack(fill="x")


def main() -> int:
    try:
        app = TrainApp()
        app.mainloop()
        return 0
    except AirQualityError as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
