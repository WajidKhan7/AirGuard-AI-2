// AirGuard AI — frontend. Talks only to the local FastAPI backend (web/app.py),
// which wraps the same airquality.pipeline functions the desktop app uses.

const TRAIN_STEPS = [
  "Dataset Loading", "Dataset Reshaping", "Dataset Inspection", "Data Preprocessing",
  "Feature Analysis", "Train/Validation/Test Split", "Classification Training",
  "Hyperparameter Tuning", "Cross-Validation", "Evaluation", "Anomaly Detection",
  "LSTM Training", "End-to-End Testing", "Saving Artifacts",
];
const PREDICT_STEPS = [
  "Loading Trained Model", "Dataset Loading", "Dataset Reshaping",
  "Compatibility Check", "Classification", "Anomaly Detection", "LSTM Forecast",
];

// ---------------------------------------------------------------------------
// Mode tabs (Train / Predict)
// ---------------------------------------------------------------------------

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`panel-${btn.dataset.mode}`).classList.add("active");
    if (btn.dataset.mode === "predict") refreshModelBanner();
  });
});

// ---------------------------------------------------------------------------
// Dropzone helper
// ---------------------------------------------------------------------------

function wireDropzone(zoneId, inputId, filenameId, onFile) {
  const zone = document.getElementById(zoneId);
  const input = document.getElementById(inputId);
  const label = document.getElementById(filenameId);
  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files[0]) {
      label.textContent = input.files[0].name;
      onFile(input.files[0]);
    }
  });
  ["dragover", "dragenter"].forEach((evt) =>
    zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.add("drag-over"); })
  );
  ["dragleave", "drop"].forEach((evt) =>
    zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.remove("drag-over"); })
  );
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) {
      input.files = e.dataTransfer.files;
      label.textContent = file.name;
      onFile(file);
    }
  });
}

// ---------------------------------------------------------------------------
// Status list rendering (mirrors the Tkinter status panel)
// ---------------------------------------------------------------------------

function buildStatusList(listEl, stepNames) {
  listEl.innerHTML = "";
  const rows = {};
  stepNames.forEach((name) => {
    const li = document.createElement("li");
    const nameSpan = document.createElement("span");
    nameSpan.className = "step-name";
    nameSpan.textContent = name;
    const statusSpan = document.createElement("span");
    statusSpan.className = "step-status";
    li.append(nameSpan, statusSpan);
    listEl.appendChild(li);
    rows[name] = statusSpan;
  });
  const detail = document.createElement("div");
  detail.className = "status-detail";
  listEl.parentElement.appendChild(detail);
  return { rows, detail };
}

function applySteps(handle, steps) {
  let lastDetail = "";
  steps.forEach(({ step, status, detail }) => {
    const el = handle.rows[step];
    if (el) { el.textContent = status; el.className = `step-status ${status}`; }
    if (detail) lastDetail = `${step}: ${detail}`;
  });
  handle.detail.textContent = lastDetail;
}

// ---------------------------------------------------------------------------
// Job polling
// ---------------------------------------------------------------------------

async function pollJob(jobId, statusHandle, onDone, onError) {
  const seenCount = { n: 0 };
  const tick = async () => {
    let res;
    try {
      res = await fetch(`/api/jobs/${jobId}`);
    } catch (e) {
      onError("Could not reach the backend. Is `python app.py` still running?");
      return;
    }
    if (!res.ok) { onError(`Server error (${res.status}).`); return; }
    const job = await res.json();
    if (job.steps.length > seenCount.n) {
      applySteps(statusHandle, job.steps.slice(seenCount.n));
      seenCount.n = job.steps.length;
    }
    if (job.status === "DONE") { onDone(job.result); return; }
    if (job.status === "ERROR") { onError(job.error || "Unknown error."); return; }
    setTimeout(tick, 500);
  };
  tick();
}

// ---------------------------------------------------------------------------
// Small render helpers
// ---------------------------------------------------------------------------

function el(tag, opts = {}, children = []) {
  const e = document.createElement(tag);
  Object.entries(opts).forEach(([k, v]) => {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k === "html") e.innerHTML = v;
    else e.setAttribute(k, v);
  });
  children.forEach((c) => e.appendChild(c));
  return e;
}

function metricGrid(pairs) {
  const grid = el("div", { class: "metric-grid" });
  pairs.forEach(([label, value]) => {
    grid.appendChild(el("div", { class: "metric-box" }, [
      el("div", { class: "label", text: label }),
      el("div", { class: "value", text: value === null || value === undefined ? "n/a" : value }),
    ]));
  });
  return grid;
}

function tableFromRecords(records, columns) {
  if (!records || records.length === 0) return el("p", { class: "muted", text: "No rows." });
  const cols = columns || Object.keys(records[0]).filter((c) => c !== "index");
  const table = el("table");
  const thead = el("tr");
  cols.forEach((c) => thead.appendChild(el("th", { text: c })));
  table.appendChild(el("thead", {}, [thead]));
  const tbody = el("tbody");
  records.forEach((row) => {
    const tr = el("tr");
    cols.forEach((c) => tr.appendChild(el("td", { text: row[c] === null || row[c] === undefined ? "" : row[c] })));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  return table;
}

function resultsTabs(container, tabs) {
  // tabs: [{label, render: () => HTMLElement}]
  const nav = el("div", { class: "results-tabs" });
  const body = el("div", { class: "results-tab-body" });
  container.append(nav, body);
  tabs.forEach((tab, i) => {
    const btn = el("button", { class: `results-tab-btn${i === 0 ? " active" : ""}`, text: tab.label });
    const panel = el("div", { class: `results-tab-panel${i === 0 ? " active" : ""}` });
    panel.appendChild(tab.render());
    btn.addEventListener("click", () => {
      nav.querySelectorAll(".results-tab-btn").forEach((b) => b.classList.remove("active"));
      body.querySelectorAll(".results-tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      panel.classList.add("active");
    });
    nav.appendChild(btn);
    body.appendChild(panel);
  });
}

// ---------------------------------------------------------------------------
// TRAIN mode
// ---------------------------------------------------------------------------

let trainSelectedFile = null;

wireDropzone("train-dropzone", "train-file-input", "train-filename", (file) => {
  trainSelectedFile = file;
  document.getElementById("train-start-btn").disabled = false;
});

document.getElementById("train-start-btn").addEventListener("click", () => startTraining(trainSelectedFile));
document.getElementById("train-error-retry").addEventListener("click", () => {
  document.getElementById("train-error").classList.add("hidden");
  document.getElementById("train-upload").classList.remove("hidden");
});

async function startTraining(file) {
  if (!file) return;
  document.getElementById("train-upload").classList.add("hidden");
  document.getElementById("train-results").classList.add("hidden");
  document.getElementById("train-error").classList.add("hidden");
  const progressCard = document.getElementById("train-progress");
  progressCard.classList.remove("hidden");
  document.getElementById("train-progress-file").textContent = file.name;
  const listEl = document.getElementById("train-status-list");
  const handle = buildStatusList(listEl, TRAIN_STEPS);

  const form = new FormData();
  form.append("file", file);
  let jobId;
  try {
    const res = await fetch("/api/train", { method: "POST", body: form });
    if (!res.ok) throw new Error(await res.text());
    ({ job_id: jobId } = await res.json());
  } catch (e) {
    showTrainError(String(e));
    return;
  }

  pollJob(jobId, handle,
    (result) => { progressCard.classList.add("hidden"); renderTrainResults(result); },
    (message) => showTrainError(message));
}

function showTrainError(message) {
  document.getElementById("train-progress").classList.add("hidden");
  document.getElementById("train-error").classList.remove("hidden");
  document.getElementById("train-error-text").textContent = message;
}

function renderTrainResults(r) {
  const container = document.getElementById("train-results");
  container.innerHTML = "";
  container.classList.remove("hidden");

  const retrainBar = el("div", { class: "retrain-actions" }, [
    el("button", { class: "btn primary", text: "Train With Another Dataset" }),
  ]);
  retrainBar.firstChild.addEventListener("click", () => {
    container.classList.add("hidden");
    document.getElementById("train-upload").classList.remove("hidden");
    trainSelectedFile = null;
    document.getElementById("train-start-btn").disabled = true;
    document.getElementById("train-filename").textContent = "";
  });
  container.appendChild(retrainBar);

  const cls = r.branches.classification;
  const an = r.branches.anomaly;
  const lstm = r.branches.lstm;

  resultsTabs(container, [
    { label: "Overview", render: () => renderOverview(r) },
    { label: "Classification", render: () => renderClassification(cls) },
    { label: "Confusion Matrix", render: () => renderConfusionMatrix(cls) },
    { label: "Feature Importance", render: () => renderFeatureImportance(cls, r.feature_analysis) },
    { label: "Anomaly Detection", render: () => renderAnomaly(an) },
    { label: "LSTM Forecasting", render: () => renderLstm(lstm) },
    { label: "Model Comparison", render: () => renderModelComparison(cls) },
  ]);
}

function renderOverview(r) {
  const wrap = el("div");
  wrap.appendChild(el("h2", { text: "Dataset" }));
  wrap.appendChild(metricGrid([
    ["Rows", r.raw_shape[0].toLocaleString()],
    ["Columns", r.raw_shape[1]],
    ["Cleaned rows", r.cleaning_report.final_rows.toLocaleString()],
    ["Label column", r.label_column || "(none)"],
  ]));
  if (r.long_format_reshape) {
    wrap.appendChild(el("div", {
      class: "info-banner",
      text: `Detected long-format data ('${r.long_format_reshape.type_column}' / ` +
            `'${r.long_format_reshape.value_column}') and pivoted to wide format: ` +
            `${r.long_format_reshape.pollutants_found.length} pollutant column(s) created.`,
    }));
  }
  wrap.appendChild(el("p", { class: "muted", text: r.size_report }));

  wrap.appendChild(el("h2", { text: "Detected columns" }));
  wrap.appendChild(el("p", {}, [
    document.createTextNode(`Pollutant: ${r.schema.pollutant_columns.join(", ") || "(none)"}`),
  ]));
  wrap.appendChild(el("p", { text: `Meteorological: ${r.schema.meteorological_columns.join(", ") || "(none)"}` }));
  wrap.appendChild(el("p", { text: `Datetime: ${r.schema.datetime_columns.join(", ") || "(none)"}` }));

  if (r.schema_change) {
    const sc = r.schema_change;
    wrap.appendChild(el("h2", { text: "Compared to the previous training run" }));
    wrap.appendChild(el("p", { text: `Columns added: ${sc.added_columns.length ? sc.added_columns.join(", ") : "none"}` }));
    wrap.appendChild(el("p", { text: `Columns removed: ${sc.removed_columns.length ? sc.removed_columns.join(", ") : "none"}` }));
    wrap.appendChild(el("p", { text: `Row count: ${sc.previous_row_count?.toLocaleString?.() ?? sc.previous_row_count} → ${sc.current_row_count.toLocaleString()}` }));
    if (r.performance_change) {
      const pc = r.performance_change;
      const sign = pc.delta >= 0 ? "+" : "";
      wrap.appendChild(el("p", {
        html: `Test F1 (macro): <strong>${pc.previous_test_f1}</strong> → <strong>${pc.new_test_f1}</strong> ` +
              `(${sign}${pc.delta})`,
      }));
    }
  }
  return wrap;
}

function renderClassification(cls) {
  if (!cls || !cls.available) return unavailableBox("Classification", cls?.reason);
  const wrap = el("div");
  wrap.appendChild(el("p", { html: `Final model: <strong>${cls.final_model_name}</strong> &nbsp;|&nbsp; Classes: ${cls.classes.join(", ")}` }));
  wrap.appendChild(el("h2", { text: "Test-set performance (previously unseen data)" }));
  const m = cls.test_metrics;
  wrap.appendChild(metricGrid([
    ["Accuracy", m.accuracy], ["Precision (macro)", m.precision_macro],
    ["Recall (macro)", m.recall_macro], ["F1 (macro)", m.f1_macro],
    ["Precision (weighted)", m.precision_weighted], ["Recall (weighted)", m.recall_weighted],
    ["F1 (weighted)", m.f1_weighted],
  ]));

  wrap.appendChild(el("h2", { text: `Cross-validation (best: ${cls.cross_validation[0]?.model ?? "n/a"})` }));
  wrap.appendChild(tableFromRecords(cls.cross_validation));

  wrap.appendChild(el("h2", { text: `Hyperparameter tuning (${cls.tuning.method})` }));
  wrap.appendChild(el("p", { text: `Best CV score: ${cls.tuning.best_cv_score}` }));
  wrap.appendChild(el("pre", { text: JSON.stringify(cls.tuning.best_params, null, 2) }));

  wrap.appendChild(el("h2", { text: "Per-class report" }));
  wrap.appendChild(tableFromRecords(cls.per_class_report));
  return wrap;
}

function renderConfusionMatrix(cls) {
  if (!cls || !cls.available) return unavailableBox("Confusion matrix", cls?.reason);
  const wrap = el("div");
  if (cls.confusion_matrix_image) wrap.appendChild(el("img", { class: "plot-img", src: cls.confusion_matrix_image }));
  wrap.appendChild(tableFromRecords(cls.confusion_matrix.rows));
  return wrap;
}

function renderFeatureImportance(cls, featureAnalysis) {
  if (!cls || !cls.available) return unavailableBox("Feature importance", cls?.reason);
  const wrap = el("div");
  if (cls.feature_importance_image) wrap.appendChild(el("img", { class: "plot-img", src: cls.feature_importance_image }));
  wrap.appendChild(el("h2", { text: "Feature selection" }));
  wrap.appendChild(el("p", { text: `Used: ${featureAnalysis.used_features.join(", ")}` }));
  const excluded = Object.entries(featureAnalysis.excluded_columns);
  if (excluded.length) {
    const list = el("ul");
    excluded.forEach(([col, reason]) => list.appendChild(el("li", { text: `${col} → ${reason}` })));
    wrap.appendChild(el("h2", { text: "Ignored columns" }));
    wrap.appendChild(list);
  }
  return wrap;
}

function renderAnomaly(an) {
  if (!an || !an.available) return unavailableBox("Anomaly detection", an?.reason);
  const wrap = el("div");
  const s = an.summary;
  wrap.appendChild(metricGrid([
    ["Total observations", s.total_observations.toLocaleString()],
    ["Normal", s.normal_observations.toLocaleString()],
    ["Anomalies", s.anomalies.toLocaleString()],
    ["Anomaly %", `${s.anomaly_percentage}%`],
  ]));
  wrap.appendChild(el("p", { class: "muted", text: `Features used: ${an.features_used.join(", ")}` }));
  wrap.appendChild(el("h2", { text: "Sample flagged observations" }));
  wrap.appendChild(tableFromRecords(an.sample_anomalies));
  return wrap;
}

function renderLstm(lstm) {
  if (!lstm || !lstm.available) {
    const box = unavailableBox("LSTM forecasting", lstm?.reason);
    if (lstm?.available_targets?.length) {
      box.appendChild(el("p", { class: "muted", text: `Available forecast targets in this dataset: ${lstm.available_targets.join(", ")}` }));
    }
    return box;
  }
  const wrap = el("div");
  if (lstm.forecast_image) wrap.appendChild(el("img", { class: "plot-img", src: lstm.forecast_image }));
  wrap.appendChild(el("p", { html: `Target: <strong>${lstm.target}</strong> &nbsp;|&nbsp; Lookback window: ${lstm.lookback} steps &nbsp;|&nbsp; Rows used: ${lstm.n_rows_used.toLocaleString()}` }));
  wrap.appendChild(el("h2", { text: "Validation" }));
  wrap.appendChild(metricGrid([["MAE", lstm.validation_metrics.mae], ["RMSE", lstm.validation_metrics.rmse], ["R²", lstm.validation_metrics.r2]]));
  wrap.appendChild(el("h2", { text: "Test" }));
  wrap.appendChild(metricGrid([["MAE", lstm.test_metrics.mae], ["RMSE", lstm.test_metrics.rmse], ["R²", lstm.test_metrics.r2]]));
  return wrap;
}

function renderModelComparison(cls) {
  if (!cls || !cls.available) return unavailableBox("Model comparison", cls?.reason);
  const wrap = el("div");
  if (cls.model_comparison_image) wrap.appendChild(el("img", { class: "plot-img", src: cls.model_comparison_image }));
  wrap.appendChild(tableFromRecords(cls.validation_comparison));
  return wrap;
}

function unavailableBox(title, reason) {
  return el("div", { class: "unavailable", html: `<strong>${title}: NOT AVAILABLE</strong><br>Reason: ${reason || "unknown"}` });
}

// ---------------------------------------------------------------------------
// PREDICT mode
// ---------------------------------------------------------------------------

let predictSelectedFile = null;

wireDropzone("predict-dropzone", "predict-file-input", "predict-filename", (file) => {
  predictSelectedFile = file;
  document.getElementById("predict-start-btn").disabled = false;
});

document.getElementById("predict-start-btn").addEventListener("click", () => startPrediction(predictSelectedFile));
document.getElementById("predict-error-retry").addEventListener("click", () => {
  document.getElementById("predict-error").classList.add("hidden");
  document.getElementById("predict-upload").classList.remove("hidden");
});

async function refreshModelBanner() {
  const banner = document.getElementById("predict-model-banner");
  const uploadCard = document.getElementById("predict-upload");
  try {
    const res = await fetch("/api/model-status");
    const status = await res.json();
    if (!status.trained) {
      banner.innerHTML = "";
      banner.appendChild(el("div", {
        class: "unavailable",
        text: "No trained model found yet. Go to the Train tab and train on a dataset first.",
      }));
      uploadCard.classList.add("hidden");
      return;
    }
    uploadCard.classList.remove("hidden");
    banner.innerHTML = "";
    const datasetName = (status.dataset_path || "").split(/[\\/]/).pop();
    banner.appendChild(el("p", { text: `Loaded model trained on: ${datasetName}` }));
    const avail = [];
    if (status.classification_available) avail.push("Classification");
    if (status.anomaly_available) avail.push("Anomaly Detection");
    if (status.lstm_available) avail.push("LSTM Forecasting");
    banner.appendChild(el("p", { class: "muted", text: `Available: ${avail.join(", ") || "none"}` }));
  } catch (e) {
    banner.innerHTML = "";
    banner.appendChild(el("div", { class: "unavailable", text: "Could not reach the backend. Is `python app.py` running?" }));
  }
}

async function startPrediction(file) {
  if (!file) return;
  document.getElementById("predict-upload").classList.add("hidden");
  document.getElementById("predict-results").classList.add("hidden");
  document.getElementById("predict-error").classList.add("hidden");
  const progressCard = document.getElementById("predict-progress");
  progressCard.classList.remove("hidden");
  const listEl = document.getElementById("predict-status-list");
  const handle = buildStatusList(listEl, PREDICT_STEPS);

  const form = new FormData();
  form.append("file", file);
  let jobId;
  try {
    const res = await fetch("/api/predict", { method: "POST", body: form });
    if (!res.ok) throw new Error(await res.text());
    ({ job_id: jobId } = await res.json());
  } catch (e) {
    showPredictError(String(e));
    return;
  }

  pollJob(jobId, handle,
    (result) => { progressCard.classList.add("hidden"); renderPredictResults(result); },
    (message) => showPredictError(message));
}

function showPredictError(message) {
  document.getElementById("predict-progress").classList.add("hidden");
  document.getElementById("predict-error").classList.remove("hidden");
  document.getElementById("predict-error-text").textContent = message;
}

function renderPredictResults(r) {
  const container = document.getElementById("predict-results");
  container.innerHTML = "";
  container.classList.remove("hidden");

  const again = el("div", { class: "retrain-actions" }, [el("button", { class: "btn primary", text: "Select New CSV" })]);
  again.firstChild.addEventListener("click", () => {
    container.classList.add("hidden");
    document.getElementById("predict-upload").classList.remove("hidden");
    predictSelectedFile = null;
    document.getElementById("predict-start-btn").disabled = true;
    document.getElementById("predict-filename").textContent = "";
  });
  container.appendChild(again);

  if (!r.compatible) {
    container.appendChild(el("div", {
      class: "unavailable",
      html: `<strong>Dataset compatibility: FAILED</strong><br><br>Missing required features:<br>` +
            r.missing_required_features.map((f) => `&nbsp;&nbsp;${f}`).join("<br>") +
            `<br><br>Please provide a compatible dataset or retrain the model with this schema.`,
    }));
    return;
  }

  if (r.extra_columns && r.extra_columns.length) {
    container.appendChild(el("div", {
      class: "warn-banner",
      text: `Additional columns detected (not used by the trained model): ${r.extra_columns.join(", ")}. ` +
            `Retraining is recommended if you want them to contribute.`,
    }));
  }

  if (r.classification) {
    const c = r.classification;
    const box = el("div", { class: "card" }, [el("h2", { text: "Classification" })]);
    const grid = metricGrid(c.classes.map((cls) => [cls, (c.class_counts[cls] || 0).toLocaleString()]));
    box.appendChild(grid);
    box.appendChild(el("p", { class: "muted", text: `${c.total_predictions.toLocaleString()} row(s) classified` +
      (c.total_predictions > c.predictions_preview.length ? ` (showing first ${c.predictions_preview.length})` : "") }));
    container.appendChild(box);
  }

  if (r.anomaly) {
    const s = r.anomaly.summary;
    const box = el("div", { class: "card" }, [el("h2", { text: "Anomaly Detection" })]);
    box.appendChild(metricGrid([
      ["Normal", s.normal_observations.toLocaleString()],
      ["Anomalies", s.anomalies.toLocaleString()],
      ["Anomaly %", `${s.anomaly_percentage}%`],
    ]));
    container.appendChild(box);
  }

  if (r.lstm) {
    const l = r.lstm;
    const box = el("div", { class: "card" }, [el("h2", { text: "LSTM Forecast" })]);
    box.appendChild(el("p", { text: `Target: ${l.target}` }));
    box.appendChild(el("p", { text: `Forecast generated successfully (${l.total_predictions} point(s)).` }));
    box.appendChild(el("p", { class: "muted", text: `First predictions: ${l.predictions_preview.slice(0, 8).map((v) => v.toFixed(2)).join(", ")}${l.predictions_preview.length > 8 ? " ..." : ""}` }));
    container.appendChild(box);
  }
}

// initial load
refreshModelBanner();
