"""AirGuard AI — web backend.

This is a thin HTTP wrapper around the exact same functions the Tkinter app
uses (``airquality.pipeline.run_training_pipeline`` /
``run_prediction_pipeline``). It adds nothing to the ML pipeline itself —
only a way to trigger it from a browser and poll its progress, since
training genuinely takes real compute time and can't run inside the page.

Run with:
    python app.py
then open http://127.0.0.1:8000 in a browser.
"""

from __future__ import annotations

import shutil
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `import airquality` works when run from web/

from airquality.pipeline import run_prediction_pipeline, run_training_pipeline, load_trained_artifacts
from airquality.validation import AirQualityError, NotFittedError
from web.serialize import serialize_prediction_results, serialize_training_results

UPLOAD_DIR = ROOT / "web" / "_uploads"
ARTIFACTS_DIR = ROOT / "artifacts"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AirGuard AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ----------------------------------------------------------------------------
# In-memory job registry. A "job" is one training or prediction run; the
# heavy pipeline call runs on a background thread so the HTTP request that
# started it returns immediately, and the browser polls /api/jobs/{id}.
# ----------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _new_job(kind: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"id": job_id, "kind": kind, "status": "RUNNING", "steps": [],
                         "result": None, "error": None}
    return job_id


def _job_on_step(job_id: str):
    def cb(step: str, status: str, detail: str = "") -> None:
        with _jobs_lock:
            _jobs[job_id]["steps"].append({"step": step, "status": status, "detail": detail})
    return cb


def _run_training_job(job_id: str, csv_path: str) -> None:
    try:
        result = run_training_pipeline(csv_path, artifacts_dir=str(ARTIFACTS_DIR),
                                       on_step=_job_on_step(job_id))
        payload = serialize_training_results(result)
        with _jobs_lock:
            _jobs[job_id]["status"] = "DONE"
            _jobs[job_id]["result"] = payload
    except AirQualityError as exc:
        with _jobs_lock:
            _jobs[job_id]["status"] = "ERROR"
            _jobs[job_id]["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - last-resort catch on the worker thread
        with _jobs_lock:
            _jobs[job_id]["status"] = "ERROR"
            _jobs[job_id]["error"] = f"Unexpected error: {exc}"


def _run_prediction_job(job_id: str, csv_path: str) -> None:
    try:
        result = run_prediction_pipeline(csv_path, artifacts_dir=str(ARTIFACTS_DIR),
                                         on_step=_job_on_step(job_id))
        payload = serialize_prediction_results(result)
        with _jobs_lock:
            _jobs[job_id]["status"] = "DONE"
            _jobs[job_id]["result"] = payload
    except AirQualityError as exc:
        with _jobs_lock:
            _jobs[job_id]["status"] = "ERROR"
            _jobs[job_id]["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        with _jobs_lock:
            _jobs[job_id]["status"] = "ERROR"
            _jobs[job_id]["error"] = f"Unexpected error: {exc}"


def _save_upload(file: UploadFile, job_id: str) -> str:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file.")
    dest = UPLOAD_DIR / f"{job_id}_{Path(file.filename).name}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return str(dest)


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/model-status")
def model_status() -> dict:
    """Does a trained model already exist? The frontend uses this to decide
    whether the Predict tab is usable."""
    try:
        bundle = load_trained_artifacts(str(ARTIFACTS_DIR))
        meta = bundle["metadata"]
        return {
            "trained": True,
            "dataset_path": meta.get("dataset_path"),
            "trained_at": meta.get("created_utc"),
            "classification_available": meta.get("classification", {}).get("available", False),
            "anomaly_available": meta.get("anomaly", {}).get("available", False),
            "lstm_available": meta.get("lstm", {}).get("available", False),
        }
    except NotFittedError:
        return {"trained": False}


@app.post("/api/train")
async def start_training(file: UploadFile = File(...)) -> dict:
    job_id = _new_job("train")
    csv_path = _save_upload(file, job_id)
    thread = threading.Thread(target=_run_training_job, args=(job_id, csv_path), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.post("/api/predict")
async def start_prediction(file: UploadFile = File(...)) -> dict:
    job_id = _new_job("predict")
    csv_path = _save_upload(file, job_id)
    thread = threading.Thread(target=_run_prediction_job, args=(job_id, csv_path), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job id.")
        return dict(job)  # shallow copy — safe to serialize outside the lock


@app.get("/api/artifacts/{filename}")
def get_artifact(filename: str):
    path = ARTIFACTS_DIR / filename
    if ".." in filename or not path.resolve().is_relative_to(ARTIFACTS_DIR.resolve()):
        raise HTTPException(400, "Invalid filename.")
    if not path.exists():
        raise HTTPException(404, "Artifact not found.")
    return FileResponse(path)


static_dir = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
