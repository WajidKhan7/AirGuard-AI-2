#!/usr/bin/env python3
"""AirGuard AI — Prediction Mode (desktop app).

    python predict.py

Loads the model(s) saved by train.py and applies them to a new CSV, after
checking the new dataset's schema is actually compatible with what the
model was trained on.
"""

from __future__ import annotations

import logging
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import pandas as pd

from airquality.gui import (
    ThreadedRunner, StatusPanel, PREDICT_STEPS, dataframe_to_treeview,
    pick_csv_file, show_error, show_info,
)
from airquality.pipeline import run_prediction_pipeline
from airquality.validation import AirQualityError, NotFittedError

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-7s %(message)s")

ARTIFACTS_DIR = "artifacts"


class PredictApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AirGuard AI Prediction")
        self.geometry("880x680")
        self.minsize(760, 560)
        self.csv_path = tk.StringVar(value="")
        self._active_runner = None
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_select_screen()

    def _build_select_screen(self) -> None:
        for w in self.winfo_children():
            w.destroy()
        frame = ttk.Frame(self, padding=30)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="AIRGUARD AI PREDICTION", font=("Segoe UI", 20, "bold")).pack(pady=(10, 0))
        try:
            from airquality.pipeline import load_trained_artifacts
            meta = load_trained_artifacts(ARTIFACTS_DIR)["metadata"]
            ttk.Label(frame, text=f"Loaded model trained on: {Path(meta['dataset_path']).name}",
                     foreground="#666666").pack(pady=(4, 20))
        except NotFittedError as exc:
            ttk.Label(frame, text=str(exc), foreground="#c1121f", wraplength=700,
                     justify="center").pack(pady=(4, 20))
            return

        ttk.Button(frame, text="Select New CSV", command=self._on_select).pack(pady=6)
        ttk.Label(frame, textvariable=self.csv_path, foreground="#333333",
                 wraplength=700).pack(pady=(4, 20))

        self.run_btn = ttk.Button(frame, text="Run Prediction", command=self._on_run, state="disabled")
        self.run_btn.pack(pady=10)

    def _on_select(self) -> None:
        path = pick_csv_file(self)
        if path:
            self.csv_path.set(path)
            self.run_btn.configure(state="normal")

    def _on_run(self) -> None:
        self._build_progress_screen()

    def _build_progress_screen(self) -> None:
        for w in self.winfo_children():
            w.destroy()
        frame = ttk.Frame(self, padding=20)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Running prediction...", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(frame, text=Path(self.csv_path.get()).name, foreground="#666666").pack(anchor="w", pady=(0, 10))

        self.status_panel = StatusPanel(frame, PREDICT_STEPS)
        self.status_panel.pack(fill="both", expand=True)

        def target_fn(on_step):
            return run_prediction_pipeline(self.csv_path.get(), artifacts_dir=ARTIFACTS_DIR, on_step=on_step)

        runner = ThreadedRunner(self, target_fn, self.status_panel,
                                on_done=self._on_done, on_error=self._on_error)
        self._active_runner = runner
        runner.start()

    def _on_close(self) -> None:
        if self._active_runner is not None:
            self._active_runner.cancel()
        self.destroy()

    def _on_error(self, message: str) -> None:
        show_error(self, "Prediction Error", message)
        self._build_select_screen()

    def _on_done(self, result: dict) -> None:
        self.result = result
        self._build_results_screen()

    def _build_results_screen(self) -> None:
        for w in self.winfo_children():
            w.destroy()
        r = self.result

        top = ttk.Frame(self, padding=(20, 12, 20, 0))
        top.pack(fill="x")
        ttk.Label(top, text="Prediction Results", font=("Segoe UI", 16, "bold")).pack(side="left")
        ttk.Button(top, text="Select New CSV", command=self._build_select_screen).pack(side="right")

        body = ttk.Frame(self, padding=20)
        body.pack(fill="both", expand=True)

        if not r.get("compatible", True):
            lines = ["Dataset compatibility: FAILED", "",
                    f"Missing required features:\n  " + "\n  ".join(r.get("missing_required_features", [])),
                    "", "Prediction cannot continue.",
                    "Please provide a compatible dataset or retrain the model with this schema."]
            ttk.Label(body, text="\n".join(lines), foreground="#c1121f", justify="left",
                     font=("Consolas", 11)).pack(anchor="w")
            return

        if r.get("extra_columns"):
            ttk.Label(body, text=(
                f"Additional columns detected (not used by the trained model): "
                f"{', '.join(r['extra_columns'])}\nRetraining is recommended if you want them to contribute."
            ), foreground="#a15c00", wraplength=780, justify="left").pack(anchor="w", pady=(0, 14))

        if "classification" in r:
            c = r["classification"]
            counts = pd.Series(c["predictions"]).value_counts()
            mean_conf = np.nanmean(c["confidence"]) if len(c["confidence"]) else float("nan")
            box = ttk.LabelFrame(body, text="Classification", padding=10)
            box.pack(fill="x", pady=6)
            summary = "\n".join(f"  {cls}: {counts.get(cls, 0):,}" for cls in c["classes"])
            ttk.Label(box, text=(
                f"Predicted category counts (of {len(c['predictions']):,} rows):\n{summary}\n\n"
                f"Average confidence: {mean_conf:.1%}" if not np.isnan(mean_conf) else "\nConfidence: n/a"
            ), justify="left", font=("Consolas", 10)).pack(anchor="w")

        if "anomaly" in r:
            a = r["anomaly"]["summary"]
            box = ttk.LabelFrame(body, text="Anomaly Detection", padding=10)
            box.pack(fill="x", pady=6)
            ttk.Label(box, text=(
                f"Normal observations: {a['normal_observations']:,}\n"
                f"Anomalies: {a['anomalies']:,}\n"
                f"Anomaly percentage: {a['anomaly_percentage']}%"
            ), justify="left", font=("Consolas", 10)).pack(anchor="w")

        if "lstm" in r:
            l = r["lstm"]
            box = ttk.LabelFrame(body, text="LSTM Forecast", padding=10)
            box.pack(fill="x", pady=6)
            preview = ", ".join(f"{v:.2f}" for v in l["predictions"][:8])
            ttk.Label(box, text=(
                f"Target: {l['target']}\nForecast generated successfully.\n"
                f"First predictions: {preview}{' ...' if len(l['predictions']) > 8 else ''}"
            ), justify="left", font=("Consolas", 10)).pack(anchor="w")

        bottom = ttk.Frame(self, padding=(20, 0, 20, 14))
        bottom.pack(fill="x")
        ttk.Button(bottom, text="Close", command=self.destroy).pack(side="right")


def main() -> int:
    try:
        app = PredictApp()
        app.mainloop()
        return 0
    except AirQualityError as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
