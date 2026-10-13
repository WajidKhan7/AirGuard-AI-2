"""Tkinter desktop GUI for both train.py and predict.py.

Deliberately thin: every real decision (schema detection, cleaning, model
selection, metrics) happens in the airquality.* pipeline modules, which know
nothing about Tkinter and are fully testable without a display. This module
only wires a file dialog and a results dashboard on top of them, running the
pipeline in a background thread so the window never freezes.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .validation import AirQualityError

APP_BG = "#f4f6f8"
ACCENT = "#2a6f97"
DONE_COLOR = "#2a9d8f"
ERROR_COLOR = "#c1121f"
SKIP_COLOR = "#8d8d8d"

TRAIN_STEPS = [
    "Dataset Loading", "Dataset Reshaping", "Dataset Inspection", "Data Preprocessing", "Feature Analysis",
    "Train/Validation/Test Split", "Classification Training", "Hyperparameter Tuning",
    "Cross-Validation", "Evaluation", "Anomaly Detection", "LSTM Training",
    "End-to-End Testing", "Saving Artifacts",
]
PREDICT_STEPS = [
    "Loading Trained Model", "Dataset Loading", "Dataset Reshaping", "Compatibility Check",
    "Classification", "Anomaly Detection", "LSTM Forecast",
]


class StatusPanel(ttk.Frame):
    """The "Dataset Loading........ DONE" style progress list."""

    def __init__(self, master, steps: list[str]):
        super().__init__(master, padding=10)
        self.rows: dict[str, tuple[ttk.Label, ttk.Label]] = {}
        for i, step in enumerate(steps):
            name = ttk.Label(self, text=step, width=32, anchor="w", font=("Segoe UI", 10))
            name.grid(row=i, column=0, sticky="w", pady=2)
            dots = ttk.Label(self, text="." * 20, foreground="#bbbbbb")
            dots.grid(row=i, column=1, sticky="w")
            status = ttk.Label(self, text="", width=12, anchor="w", font=("Segoe UI", 10, "bold"))
            status.grid(row=i, column=2, sticky="w")
            self.rows[step] = (dots, status)
        self.detail = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.detail, foreground="#555555",
                 wraplength=520, justify="left").grid(
            row=len(steps), column=0, columnspan=3, sticky="w", pady=(10, 0))

    def update_step(self, step: str, status: str, detail: str = "") -> None:
        colors = {"RUNNING": ACCENT, "DONE": DONE_COLOR, "ERROR": ERROR_COLOR, "SKIPPED": SKIP_COLOR}
        try:
            if step in self.rows:
                _, label = self.rows[step]
                label.configure(text=status, foreground=colors.get(status, "black"))
            if detail:
                self.detail.set(f"{step}: {detail}")
            self.update_idletasks()
        except tk.TclError:
            pass  # panel was destroyed (window closed) while an update was in flight


class ThreadedRunner:
    """Runs a pipeline function on a background thread and pumps its
    on_step() callbacks (and the final result / any error) back to the
    Tkinter main thread through a queue, polled with `after()`."""

    def __init__(self, root: tk.Tk, target_fn, status_panel: StatusPanel,
                on_done, on_error):
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.status_panel = status_panel
        self.on_done = on_done
        self.on_error = on_error
        self._after_id = None
        self._thread = threading.Thread(
            target=self._run, args=(target_fn,), daemon=True
        )

    def start(self) -> None:
        self._thread.start()
        self._schedule_poll()

    def _schedule_poll(self) -> None:
        try:
            if self.root.winfo_exists():
                self._after_id = self.root.after(80, self._poll)
        except tk.TclError:
            pass  # window was closed while the background thread was still running

    def cancel(self) -> None:
        """Stop polling. Called when the window is closed while a background
        pipeline run is still in flight, so Tk never tries to invoke a
        callback on a destroyed widget."""
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _run(self, target_fn) -> None:
        def on_step(step, status, detail=""):
            self.q.put(("step", step, status, detail))
        try:
            result = target_fn(on_step)
            self.q.put(("done", result))
        except AirQualityError as exc:
            self.q.put(("error", str(exc)))
        except Exception as exc:  # noqa: BLE001 - last-resort catch for the GUI thread
            self.q.put(("error", f"Unexpected error: {exc}"))

    def _poll(self) -> None:
        try:
            if not self.root.winfo_exists():
                return  # window closed; drop any remaining queued updates
            while True:
                item = self.q.get_nowait()
                if item[0] == "step":
                    _, step, status, detail = item
                    self.status_panel.update_step(step, status, detail)
                elif item[0] == "done":
                    self.on_done(item[1])
                    return
                elif item[0] == "error":
                    self.on_error(item[1])
                    return
        except queue.Empty:
            pass
        except tk.TclError:
            return  # widgets were destroyed mid-update
        self._schedule_poll()


def pick_csv_file(parent) -> str:
    return filedialog.askopenfilename(
        parent=parent, title="Select CSV Dataset",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )


def load_image_scaled(path: str, max_width: int = 620) -> ImageTk.PhotoImage | None:
    p = Path(path)
    if not p.exists():
        return None
    img = Image.open(p)
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)))
    return ImageTk.PhotoImage(img)


def dataframe_to_treeview(master, df, height: int = 8) -> ttk.Treeview:
    tree = ttk.Treeview(master, columns=list(df.columns), show="headings", height=height)
    for col in df.columns:
        tree.heading(col, text=str(col))
        tree.column(col, width=max(70, min(160, 10 * len(str(col)))), anchor="center")
    for _, row in df.head(500).iterrows():
        tree.insert("", "end", values=[row[c] for c in df.columns])
    return tree


def show_error(parent, title: str, message: str) -> None:
    messagebox.showerror(title, message, parent=parent)


def show_info(parent, title: str, message: str) -> None:
    messagebox.showinfo(title, message, parent=parent)
