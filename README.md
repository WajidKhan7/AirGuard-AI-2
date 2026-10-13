# AirGuard AI — Air Quality Machine Learning System

A Python-only desktop application that trains and applies three ML models —
classification, anomaly detection, and LSTM time-series forecasting — on
**any suitable air-quality CSV**. No web framework, no HTML/JS, no browser:
the interface is a Tkinter desktop window, and the underlying pipeline is
plain scikit-learn / PyTorch.

## Two disclosures up front

**1. LSTM is implemented in PyTorch, not TensorFlow/Keras.** Same
architecture (an LSTM layer + a linear head), same training loop, and the
same evaluation metrics (MAE, RMSE, R²) either way — only the library and the
model file extension differ (`lstm_model.pt` instead of `.keras`). This was a
practical call: PyTorch was already installed for an earlier iteration of
this project and disk space was tight; installing a second deep-learning
framework alongside it wasn't worth the risk. If you specifically need a
`.keras` file, this is the one place in the codebase that would need
rewriting — `airquality/lstm_forecasting.py` — and everything that calls it
(`pipeline.py`) would keep working unchanged since it only calls
`.fit()` / `.predict()` / `.save()` / `.load()`.

**2. Automatic column and format detection is a heuristic, not a guarantee.**
The system recognizes common naming variants (`PM2.5`, `pm_2_5`,
`Particulate Matter 2.5`) and both wide format (one column per pollutant) and
long/melted format (`Date, City, Type, Count`, common in Pakistani open-data
exports). It cannot read your mind about a column named something it's never
seen — when that happens, the column falls through to "feature_candidate" (used
if numeric) or gets excluded with a stated reason, and the Feature Analysis /
Overview tabs always show you exactly what was used and what was ignored, so
nothing is silently wrong.

## What each requirement maps to

| Requirement | Where |
|---|---|
| Train/Validation/Test split | `pipeline.py` (stratified 70/15/15 for classification; chronological for LSTM in `lstm_forecasting.py`) |
| Confusion matrix | `evaluation.py::confusion_table`, `plot_confusion_matrix` — shown in the Confusion Matrix tab |
| Hyperparameter tuning | `classification.py::tune_random_forest` — GridSearchCV under 20k rows, RandomizedSearchCV above, tuning-subsample cap above 60k rows |
| Cross-validation | `classification.py::cross_validate_models` — StratifiedKFold, per-fold + mean/std |
| Precision/Recall/F1 comparison | `evaluation.py::classification_metrics`, `per_class_report`; `classification.py::train_and_compare` for the model-vs-model table |
| Anomaly detection | `anomaly_detection.py` — Isolation Forest, Normal/Anomaly output |
| LSTM forecasting | `lstm_forecasting.py` — auto datetime detection, auto lookback from sampling frequency, MAE/RMSE/R² |
| Dynamic dataset modification | `schema.py::compare_schemas`, `pipeline.py` schema-change and performance-change reporting |
| Feature selection / importance | `schema.py::inspect_dataset` (selection), `classification.py::feature_importance` |
| Input validation / error handling | `validation.py`, `data_loader.py` — typed exceptions, readable messages, no raw tracebacks for expected problems |
| End-to-end testing | `tests/test_pipeline.py` |
| Desktop GUI | `train.py`, `predict.py`, `airquality/gui.py` (Tkinter, threaded so it never freezes) |
| Large dataset handling | `data_loader.py::dataset_size_report`, tuning subsample cap in `classification.py` |
| Model saving/loading | `pipeline.py::run_training_pipeline` (save), `load_trained_artifacts` (load) |

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux/macOS: `source .venv/bin/activate` instead of the `Activate.ps1` line.

Tkinter ships with most Python installers on Windows and macOS. On Linux, if
`import tkinter` fails, install it via your package manager first (e.g.
`sudo apt install python3-tk` on Debian/Ubuntu) — it's a system package, not a
pip package.

## Run

```powershell
python train.py
```

A window opens. Select a CSV, click **Start Training**, watch the status
list update live, then browse the results across tabs (Overview,
Classification, Confusion Matrix, Feature Importance, Anomaly Detection,
LSTM Forecasting, Model Comparison). Artifacts are saved to `artifacts/`
automatically regardless of what you do in the window.

```powershell
python predict.py
```

Loads the model saved by the last training run, lets you pick a new CSV,
checks it's compatible with what the model expects, and shows classification
/ anomaly / forecast results for it.

A `sample_data/sample_air_quality.csv` is included so you can try the whole
flow immediately without your own file.

## Web frontend (optional)

A browser-based frontend is also included, wrapping the exact same
`airquality/pipeline.py` functions the desktop app uses — nothing in the ML
code changes between the two. This is a genuine addition on top of what the
project brief originally specified (which asked for desktop-only, no web
technology at all); it exists because it was requested afterward. If you're
being graded against the original written requirements, it's worth a line in
your report noting the desktop app (`train.py` / `predict.py`) is the
spec-compliant deliverable and the web frontend is an added extra.

```powershell
pip install -r requirements.txt      # now includes fastapi/uvicorn
python web\app.py
```

Then open **http://127.0.0.1:8000** in a browser. Everything runs locally —
no data leaves your machine, there's no cloud dependency, and closing the
terminal stops the server. The page has the same two modes as the desktop
app (Train / Predict), the same live step-by-step progress, and the same
results dashboard (Overview, Classification, Confusion Matrix, Feature
Importance, Anomaly Detection, LSTM Forecasting, Model Comparison), just
rendered as a web page instead of a Tkinter window.

Architecture: `web/app.py` is a small FastAPI backend — it uploads the CSV,
runs the pipeline on a background thread (so the HTTP request returns
immediately), and the page polls `/api/jobs/{id}` for progress, matching how
the desktop app's threaded runner works. `web/serialize.py` converts the
pipeline's Python-native results (DataFrames, dataclasses) into JSON; it's a
formatting layer only; no metric is computed there. `web/static/` is plain
HTML/CSS/JS with no build step and no framework.

## How it adapts to different datasets

1. **Column names**: normalized against a synonym table (`schema.py`), so
   `PM2.5`, `PM 2.5`, `pm25`, and `Particulate Matter 2.5` all resolve to the
   same pollutant.
2. **Format**: if the CSV looks like a melted export (`Type` column holding
   pollutant names, a `Count`/`Value` column holding readings), it's detected
   and pivoted to wide format automatically before anything else runs.
3. **Label**: uses an existing category column if one looks right (2–10
   distinct classes); otherwise derives Low/Moderate/High from a numeric AQI
   column, or from PM2.5 using US-EPA-style breakpoints, if either exists
   with enough valid values. If neither is available, classification is
   skipped — with the reason stated — while anomaly detection and forecasting
   continue.
4. **Features**: identifiers, constant columns, free-text fields, and
   high-missingness columns are excluded automatically; everything else
   numeric or low-cardinality-categorical becomes a feature. The excluded
   list and its reasons are always shown.
5. **Retraining on a modified dataset**: run `train.py` again pointed at the
   new file. The system re-inspects it from scratch — it doesn't assume
   anything about the previous run — and if `artifacts/metadata.json` exists
   from an earlier run, it reports what changed (columns added/removed, row
   count, and the old vs. new test F1) rather than silently reusing stale
   assumptions.

## Methodology notes

- **No data leakage**: scalers and encoders are fit on the training split
  only; hyperparameter tuning and model selection use the training/validation
  splits only; the test split is touched exactly once, at final evaluation.
  When a numeric AQI column is used to derive the label, that column is
  excluded from the feature set (it would otherwise leak the answer).
- **Classification split is stratified random** (not chronological) — most
  air-quality datasets mix multiple stations/dates with no meaningful
  sequential order for classification purposes. **LSTM forecasting uses a
  strictly chronological split** — the model never trains on data from after
  the point it's being validated or tested on.
- **Anomaly detection needs no label** — Isolation Forest runs on whatever
  numeric features survive cleaning, on the full cleaned dataset.
- **Nothing is hard-coded.** Every accuracy/F1/MAE/RMSE/R² number in the
  results comes from evaluating the trained model on that run's actual test
  split. Re-run on different data and you will get different numbers.

## Testing

```powershell
python -m pytest tests -q
```

58 tests across five files: `test_preprocessing.py` (schema detection,
cleaning, label derivation, long-format pivoting), `test_classification.py`
(tuning, cross-validation, feature importance), `test_anomaly.py`,
`test_lstm.py` (small synthetic series, kept fast), and `test_pipeline.py`
(input validation, error handling, full end-to-end training and prediction
runs, dynamic dataset modification, schema-compatibility checking).

Note: the classification and end-to-end tests genuinely run
GridSearchCV/cross-validation, so the suite takes a few minutes — it isn't
hung, real model fitting is happening.

## Project structure

```
airquality_ml/
├── train.py                    # Tkinter training app
├── predict.py                  # Tkinter prediction app
├── requirements.txt
├── README.md
├── airquality/
│   ├── data_loader.py          # CSV loading, path validation, size detection
│   ├── schema.py                # column normalization, role detection, long-format pivot
│   ├── preprocessing.py         # cleaning, label derivation, feature selection
│   ├── classification.py        # RF/LogReg/SVM, tuning, cross-validation, importance
│   ├── anomaly_detection.py     # Isolation Forest
│   ├── lstm_forecasting.py      # PyTorch LSTM forecasting
│   ├── evaluation.py            # metrics, confusion matrix, all plots
│   ├── validation.py            # exceptions, dataset-content checks
│   ├── pipeline.py              # orchestrates training mode + prediction mode
│   └── gui.py                   # shared Tkinter widgets, threaded runner
├── tests/                       # pytest suite, 58 tests
├── web/
│   ├── app.py                   # FastAPI backend (optional web frontend)
│   ├── serialize.py             # pipeline results -> JSON for the API
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── app.js
└── sample_data/
    ├── sample_air_quality.csv          # wide format
    └── pak_epa_long_format.csv         # long/melted format example
```

## Limitations

- Automatic column-role detection is heuristic. An unusually named column
  (no recognizable synonym, not numeric-looking) is still used as a feature
  if it's numeric, but won't get special pollutant/meteorological treatment.
- Long-format detection requires a `Type`/`Parameter`/`Pollutant`-style
  column whose values substantially overlap common pollutant names (PM2.5,
  PM10, CO, NO2, SO2, O3, AQI, ...). A long-format file using entirely
  different pollutant naming won't be recognized as long-format automatically.
- LSTM forecasting needs a genuine, mostly-parseable datetime column (or a
  Date+Time pair) and at least ~60 chronological rows for the chosen target.
  Without one, the GUI explains why and still completes classification and
  anomaly detection.
- Categorical feature one-hot encoding is fit against the full dataset's
  category vocabulary (not just the training split) before splitting — this
  doesn't leak target information, but a prediction-time dataset with a
  brand-new category value the model never saw will encode that value as
  all-zeros rather than erroring.
- This was built and tested on a Linux sandbox; the Windows PowerShell
  commands above are written to the spec but I could not run them on
  Windows directly — if you hit a platform-specific issue (most likely
  around Tkinter's DPI scaling or file-dialog behavior), it should be a
  small fix, not a rewrite.
