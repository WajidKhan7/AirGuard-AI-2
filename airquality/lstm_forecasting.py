"""LSTM time-series forecasting.

Implemented in PyTorch rather than TensorFlow/Keras (see README for why) —
same architecture and the same evaluation metrics (MAE, RMSE, R2) either way.
Model files are saved as ``.pt`` instead of ``.keras``.

Everything here is driven by what the dataset actually contains: the
datetime column(s) and the forecasting target are detected, not assumed, and
the lookback window is chosen from the data's own sampling frequency rather
than a fixed constant.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from .schema import SchemaProfile
from .validation import DatasetError, SchemaError

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
MIN_ROWS_FOR_LSTM = 60


def set_seed(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Datetime assembly and lookback inference
# ----------------------------------------------------------------------------


def build_datetime_index(df: pd.DataFrame, profile: SchemaProfile) -> pd.Series | None:
    """Combine whatever date/time columns exist into one parsed datetime series."""
    cols = profile.datetime_columns
    if not cols:
        return None
    date_col = next((c for c in cols if "date" in c.lower() and "time" not in c.lower()), None)
    time_col = next((c for c in cols if c.lower() == "time" or "time" in c.lower() and c != date_col), None)
    combo_col = next((c for c in cols if c not in (date_col, time_col)), None)

    if date_col and time_col and date_col != time_col:
        combined = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
        parsed = pd.to_datetime(combined, errors="coerce", format="mixed")
    elif combo_col:
        parsed = pd.to_datetime(df[combo_col], errors="coerce", format="mixed")
    else:
        col = date_col or cols[0]
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")

    if parsed.notna().mean() < 0.5:
        return None
    return parsed


def infer_lookback(datetimes: pd.Series) -> int:
    """Pick a sequence length from the data's own sampling interval.

    Roughly one day of history for sub-daily data, a week for daily data,
    capped so it never asks for more history than a small dataset has.
    """
    diffs = datetimes.sort_values().diff().dropna()
    if diffs.empty:
        return 24
    median_seconds = diffs.dt.total_seconds().median()
    if median_seconds <= 0:
        return 24
    if median_seconds <= 3600:          # hourly or finer -> one day of history
        return 24
    if median_seconds <= 86400:         # daily -> one week
        return 7
    return 12                           # monthly or coarser -> one year-ish


# ----------------------------------------------------------------------------
# Sequence construction
# ----------------------------------------------------------------------------


def find_forecast_targets(profile: SchemaProfile, df: pd.DataFrame, min_valid: int = 30) -> list[str]:
    """Numeric pollutant/AQI columns with enough valid values to forecast."""
    candidates = [c for c in profile.pollutant_columns if c in df.columns]
    return [c for c in candidates if df[c].notna().sum() >= min_valid]


def make_sequences(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= lookback:
        raise DatasetError(
            f"Not enough chronological rows ({len(values)}) for a lookback of "
            f"{lookback} — need at least {lookback + 1}."
        )
    X = np.array([values[i:i + lookback] for i in range(len(values) - lookback)], dtype=np.float32)
    y = np.array([values[i + lookback] for i in range(len(values) - lookback)], dtype=np.float32)
    return X[..., None], y  # (n, lookback, 1)


# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------


class _LSTMNet(nn.Module):
    def __init__(self, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMForecaster:
    """scikit-learn-flavoured wrapper: fit / predict / evaluate / save / load."""

    def __init__(self, lookback: int, hidden_size: int = 32, num_layers: int = 1,
                epochs: int = 30, batch_size: int = 32, learning_rate: float = 1e-3,
                patience: int = 6):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.scaler = MinMaxScaler()
        self.net: _LSTMNet | None = None
        self.history: list[dict] = []

    def fit(self, train_series: np.ndarray, val_series: np.ndarray | None = None, verbose: bool = True):
        set_seed()
        scaled_train = self.scaler.fit_transform(train_series.reshape(-1, 1)).ravel()
        X_tr, y_tr = make_sequences(scaled_train, self.lookback)

        have_val = val_series is not None and len(val_series) > self.lookback
        if have_val:
            context = np.concatenate([train_series[-self.lookback:], val_series])
            scaled_val = self.scaler.transform(context.reshape(-1, 1)).ravel()
            X_va, y_va = make_sequences(scaled_val, self.lookback)

        device = _device()
        self.net = _LSTMNet(self.hidden_size, self.num_layers).to(device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                            batch_size=self.batch_size, shuffle=True)

        best_state, best_val, stale = None, np.inf, 0
        for epoch in range(1, self.epochs + 1):
            self.net.train()
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                total += loss.item() * len(xb)
            train_loss = total / len(X_tr)
            record = {"epoch": epoch, "train_loss": round(train_loss, 6)}

            if have_val:
                self.net.eval()
                with torch.no_grad():
                    val_pred = self.net(torch.from_numpy(X_va).to(device)).cpu().numpy()
                val_loss = float(np.mean((val_pred - y_va) ** 2))
                record["val_loss"] = round(val_loss, 6)
                if val_loss < best_val - 1e-6:
                    best_val, best_state, stale = val_loss, copy.deepcopy(self.net.state_dict()), 0
                else:
                    stale += 1
            self.history.append(record)
            if verbose:
                logger.info("LSTM epoch %02d %s", epoch,
                            "  ".join(f"{k}={v}" for k, v in record.items() if k != "epoch"))
            if have_val and stale >= self.patience:
                logger.info("early stopping at epoch %d", epoch)
                break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict(self, series: np.ndarray, context: np.ndarray | None = None) -> np.ndarray:
        """Predict the next value after each window in ``series``.

        ``context`` (e.g. the tail of the training set) supplies history for
        the first few points of ``series`` so no predictions are lost to
        warm-up.
        """
        if self.net is None:
            raise SchemaError("LSTM model has not been trained yet.")
        full = series if context is None else np.concatenate([context[-self.lookback:], series])
        scaled = self.scaler.transform(full.reshape(-1, 1)).ravel()
        X, _ = make_sequences(scaled, self.lookback)
        self.net.eval()
        pred_scaled = self.net(torch.from_numpy(X).to(_device())).cpu().numpy()
        return self.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    def forecast_future(self, history: np.ndarray, steps: int) -> np.ndarray:
        """Autoregressive multi-step forecast beyond the end of ``history``."""
        if self.net is None:
            raise SchemaError("LSTM model has not been trained yet.")
        scaled_hist = list(self.scaler.transform(history.reshape(-1, 1)).ravel()[-self.lookback:])
        device = _device()
        self.net.eval()
        preds = []
        with torch.no_grad():
            for _ in range(steps):
                window = np.array(scaled_hist[-self.lookback:], dtype=np.float32)[None, :, None]
                next_scaled = float(self.net(torch.from_numpy(window).to(device)).cpu().numpy()[0])
                preds.append(next_scaled)
                scaled_hist.append(next_scaled)
        return self.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "lookback": self.lookback, "hidden_size": self.hidden_size,
            "num_layers": self.num_layers, "scaler_min": self.scaler.data_min_,
            "scaler_max": self.scaler.data_max_, "history": self.history,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "LSTMForecaster":
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        model = cls(blob["lookback"], blob["hidden_size"], blob["num_layers"])
        model.scaler.data_min_ = blob["scaler_min"]
        model.scaler.data_max_ = blob["scaler_max"]
        model.scaler.data_range_ = model.scaler.data_max_ - model.scaler.data_min_
        model.scaler.scale_ = 1.0 / model.scaler.data_range_
        model.scaler.min_ = -model.scaler.data_min_ * model.scaler.scale_
        model.scaler.n_features_in_ = 1
        model.history = blob.get("history", [])
        model.net = _LSTMNet(blob["hidden_size"], blob["num_layers"])
        model.net.load_state_dict(blob["state_dict"])
        model.net.to(_device()).eval()
        return model


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4) if len(y_true) > 1 else float("nan"),
    }


def run_lstm_pipeline(
    df: pd.DataFrame,
    profile: SchemaProfile,
    target: str,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    epochs: int = 30,
    forecast_horizon: int = 24,
) -> dict | None:
    """End-to-end forecasting branch. Returns a dict with available=False (and
    a reason) if the dataset can't support it — the rest of the pipeline
    continues either way."""
    datetimes = build_datetime_index(df, profile)
    if datetimes is None:
        logger.warning("LSTM skipped: no usable datetime column detected")
        return {"available": False, "reason": "No valid datetime column was detected."}

    work = df.copy()
    work["_dt"] = datetimes
    work = work.dropna(subset=["_dt", target]).sort_values("_dt")
    work = work.drop_duplicates(subset="_dt", keep="last")
    if len(work) < MIN_ROWS_FOR_LSTM:
        logger.warning("LSTM skipped: only %d chronological rows with a valid target", len(work))
        return {"available": False,
                "reason": f"Only {len(work)} usable chronological rows for '{target}' "
                          f"(need at least {MIN_ROWS_FOR_LSTM})."}

    lookback = infer_lookback(work["_dt"])
    lookback = min(lookback, max(2, len(work) // 10))
    series = work[target].to_numpy(dtype=np.float32)

    n = len(series)
    i_tr = int(n * train_frac)
    i_va = int(n * (train_frac + val_frac))
    if i_va - i_tr < lookback + 5 or n - i_va < lookback + 5:
        return {"available": False,
                "reason": f"Not enough chronological rows ({n}) to form train/validation/test "
                          f"windows with a lookback of {lookback}."}

    train, val, test = series[:i_tr], series[i_tr:i_va], series[i_va:]

    model = LSTMForecaster(lookback=lookback, epochs=epochs)
    model.fit(train, val, verbose=False)

    # predict() is given context (the preceding split), so it returns one
    # prediction per point in val/test rather than losing the first
    # `lookback` points to warm-up.
    val_pred = model.predict(val, context=train)
    test_pred = model.predict(test, context=np.concatenate([train, val]))
    val_true = val
    test_true = test

    val_metrics = evaluate_forecast(val_true, val_pred)
    test_metrics = evaluate_forecast(test_true, test_pred)
    future = model.forecast_future(series, steps=min(forecast_horizon, 48))

    return {
        "available": True,
        "target": target,
        "lookback": lookback,
        "n_rows_used": n,
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model": model,
        "test_true": test_true,
        "test_pred": test_pred,
        "future_forecast": future,
        "last_timestamp": work["_dt"].iloc[-1],
        "inferred_frequency_seconds": float(work["_dt"].diff().dt.total_seconds().median()),
    }
