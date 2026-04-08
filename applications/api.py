"""CREDIT Forecast API
--------------------
FastAPI application that loads a CREDIT v2 model once at startup and serves
autoregressive forecasts over HTTP.

Usage
-----
    # Set the config path, then launch with uvicorn:
    export CREDIT_CONFIG=/path/to/config.yml
    uvicorn applications.api:app --host 0.0.0.0 --port 8000

    # With GPU and multiple workers (inference is per-request, model is shared):
    uvicorn applications.api:app --host 0.0.0.0 --port 8000 --workers 1

    # Health check:
    curl http://localhost:8000/health

    # Run a forecast:
    curl -X POST http://localhost:8000/forecast \\
        -H "Content-Type: application/json" \\
        -d '{"init_time": "2024-01-15T00", "steps": 40}'

Notes
-----
- Use --workers 1 only. The model is loaded into GPU memory at startup; multiple
  workers would each load their own copy.
- Requests block until the rollout completes. A 40-step 1-degree rollout takes
  roughly 30–60 s on an A100. For longer runs consider increasing the client
  timeout or using a task queue (Celery, RQ).
- Output NetCDF files are written to `save_dir` (from request body) or the
  `predict.save_forecast` path in the config. The response JSON tells you where.
"""

import logging
import multiprocessing as mp
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from fastapi import FastAPI, HTTPException
from multiprocessing.shared_memory import SharedMemory
from pydantic import BaseModel

from credit.applications.rollout_realtime_gen2 import (
    _build_output_denorm,
    _inject_flat_schema,
    _inject_tracer_inds,
    _sample_to_batch,
    _save_worker,
)
from credit.datasets.era5 import ERA5Dataset
from credit.models import load_model
from credit.output import load_metadata
from credit.postblock import GlobalEnergyFixer, GlobalMassFixer, GlobalWaterFixer
from credit.preblock import build_preblocks, apply_preblocks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Startup: load model and all static assets once
# ---------------------------------------------------------------------------

_STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.environ.get("CREDIT_CONFIG")
    if not config_path:
        raise RuntimeError("Set CREDIT_CONFIG=/path/to/config.yml before launching.")

    logger.info("Loading config from %s", config_path)
    with open(config_path) as f:
        conf = yaml.safe_load(f)

    conf["save_loc"] = os.path.expandvars(conf["save_loc"])
    _inject_flat_schema(conf)
    conf.setdefault("predict", {})["mode"] = "none"
    _inject_tracer_inds(conf)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Model — loaded once, stays in memory for the lifetime of the server
    logger.info("Loading model weights …")
    model = load_model(conf, load_weights=True).to(device)
    model.eval()
    logger.info("Model ready.")

    # Preblocks — driven from config preblocks: section
    preblocks = build_preblocks(conf.get("preblocks", {}))

    # Inverse-normalisation stats (for physical-unit output)
    denorm_mean, denorm_std = _build_output_denorm(conf, device)

    # Lat/lon grid and output metadata
    latlons = xr.open_dataset(conf["loss"]["latitude_weights"]).load()
    lat = latlons.latitude.values
    lon = latlons.longitude.values
    meta_data = load_metadata(conf)

    # Channel counts — ERA5Dataset input insertion order: [dynfrc | static | prog]
    src = conf["data"]["source"]["ERA5"]
    v = src["variables"]
    prog = v.get("prognostic") or {}
    dyn = v.get("dynamic_forcing") or {}
    static_v = v.get("static") or {}
    n_levels = len(src["levels"])
    n_prog = len(prog.get("vars_3D", [])) * n_levels + len(prog.get("vars_2D", []))
    n_dyn = len(dyn.get("vars_2D", []))
    n_static = len(static_v.get("vars_2D", []))
    static_dim_size = n_dyn + n_static

    _STATE.update(
        conf=conf,
        device=device,
        model=model,
        preblocks=preblocks,
        denorm_mean=denorm_mean,
        denorm_std=denorm_std,
        lat=lat,
        lon=lon,
        meta_data=meta_data,
        n_prog=n_prog,
        n_dyn=n_dyn,
        static_dim_size=static_dim_size,
    )

    yield  # ← server is running

    _STATE.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CREDIT Forecast API",
    description="Autoregressive global weather forecast from a CREDIT v2 model.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class ForecastRequest(BaseModel):
    init_time: str
    """Forecast initialisation time. ISO format: YYYY-MM-DDTHH (e.g. '2024-01-15T00')."""

    steps: int = 40
    """Number of autoregressive steps to run."""

    save_dir: Optional[str] = None
    """
    Directory to write output NetCDF files.
    Defaults to predict.save_forecast in the config, or a system temp dir.
    """

    save_workers: int = 4
    """Number of CPU workers for async NetCDF writes."""


class ForecastResponse(BaseModel):
    status: str
    init_time: str
    steps: int
    lead_time_hours: int
    save_dir: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Liveness check")
def health():
    """Returns 200 if the server is up and the model is loaded."""
    device = _STATE.get("device")
    return {
        "status": "ok",
        "model_loaded": bool(_STATE),
        "device": str(device) if device else "not loaded",
    }


@app.post("/forecast", response_model=ForecastResponse, summary="Run a forecast")
def forecast(req: ForecastRequest):
    """
    Run an autoregressive forecast from `init_time` for `steps` steps.

    Output NetCDF files are written to `save_dir` (one file per step):

        <save_dir>/<YYYY-MM-DDTHH>Z/pred_<YYYY-MM-DDTHH>Z_<FHR:03d>.nc

    The response JSON tells you the output path.
    """
    if not _STATE:
        raise HTTPException(503, "Model not loaded — server is still starting up.")

    conf = _STATE["conf"]
    device = _STATE["device"]
    model = _STATE["model"]
    preblocks = _STATE["preblocks"]
    denorm_mean = _STATE["denorm_mean"]
    denorm_std = _STATE["denorm_std"]
    lat = _STATE["lat"]
    lon = _STATE["lon"]
    meta_data = _STATE["meta_data"]
    n_prog = _STATE["n_prog"]
    n_dyn = _STATE["n_dyn"]
    static_dim_size = _STATE["static_dim_size"]

    # ---- Parse init time ----
    try:
        init_time = pd.Timestamp(req.init_time)
    except Exception:
        raise HTTPException(422, f"Cannot parse init_time: {req.init_time!r}. Use YYYY-MM-DDTHH.")

    dt = pd.Timedelta(conf["data"]["timestep"])
    lead_time_periods = conf["data"]["lead_time_periods"]

    # ---- Output directory ----
    save_dir = req.save_dir or conf.get("predict", {}).get("save_forecast") or tempfile.mkdtemp(prefix="credit_fcst_")
    save_dir = os.path.expandvars(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Patch save path into a shallow copy so we don't mutate global state
    conf = {**conf, "predict": {**conf.get("predict", {}), "save_forecast": save_dir}}

    # ---- Dataset (covers the requested init-time window) ----
    dataset_conf = dict(conf["data"])
    dataset_conf["start_datetime"] = str(init_time.date())
    dataset_conf["end_datetime"] = str((init_time + req.steps * dt).date())
    dataset_conf["forecast_len"] = 1
    try:
        dataset = ERA5Dataset(dataset_conf, return_target=False)
    except Exception as e:
        raise HTTPException(500, f"Failed to open dataset: {e}")

    init_str = init_time.strftime("%Y-%m-%dT%HZ")
    logger.info("Forecast %s  steps=%d  save_dir=%s", init_str, req.steps, save_dir)

    # ---- Postblocks (optional physical conservation fixers) ----
    post_conf = conf["model"].get("post_conf", {})
    flag_mass = flag_water = flag_energy = False
    if post_conf.get("activate", False):
        if post_conf.get("global_mass_fixer", {}).get("activate_outside_model", False):
            flag_mass = True
            opt_mass = GlobalMassFixer(post_conf)
        if post_conf.get("global_water_fixer", {}).get("activate_outside_model", False):
            flag_water = True
            opt_water = GlobalWaterFixer(post_conf)
        if post_conf.get("global_energy_fixer", {}).get("activate_outside_model", False):
            flag_energy = True
            opt_energy = GlobalEnergyFixer(post_conf)

    # ---- Initial state ----
    try:
        sample = dataset[(init_time, 0)]
    except Exception as e:
        raise HTTPException(422, f"Could not load initial state for {init_str}: {e}")

    batch = _sample_to_batch(sample)
    x, _ = apply_preblocks(preblocks, batch)
    x = x.to(device).float()  # (1, C_in, 1, H, W)

    # ---- Autoregressive rollout ----
    x_init = None
    with mp.Pool(req.save_workers) as pool:
        results = []

        with torch.no_grad():
            for step in range(1, req.steps + 1):
                y_pred = model(x)

                if flag_mass:
                    if step == 1:
                        x_init = x.clone()
                    y_pred = opt_mass({"y_pred": y_pred, "x": x_init})["y_pred"]
                if flag_water:
                    y_pred = opt_water({"y_pred": y_pred, "x": x})["y_pred"]
                if flag_energy:
                    y_pred = opt_energy({"y_pred": y_pred, "x": x})["y_pred"]

                # Inverse-normalise → physical units; drop T dim for output
                y_phys = (y_pred * denorm_std + denorm_mean).squeeze(2).cpu().numpy()  # (1, C, H, W)

                # Async NetCDF write via SharedMemory (avoids pickling the array)
                shm = SharedMemory(create=True, size=y_phys.nbytes)
                np.ndarray(y_phys.shape, dtype=y_phys.dtype, buffer=shm.buf)[:] = y_phys
                results.append(
                    pool.apply_async(
                        _save_worker,
                        (
                            shm.name,
                            y_phys.shape,
                            y_phys.dtype,
                            init_str,
                            step,
                            lead_time_periods,
                            lat,
                            lon,
                            meta_data,
                            conf,
                        ),
                    )
                )
                logger.debug("step %d/%d done", step, req.steps)

                # Advance state for next step
                if step < req.steps:
                    t_next = init_time + step * dt
                    sample_frc = dataset[(t_next, 1)]  # dynamic_forcing only
                    batch_frc = _sample_to_batch(sample_frc)
                    x_frc, _ = apply_preblocks(preblocks, batch_frc)
                    x_frc = x_frc.to(device).float()

                    # ERA5Dataset insertion order: [dynfrc | static | prog]
                    x[:, :n_dyn, ...] = x_frc
                    x[:, static_dim_size:, ...] = torch.from_numpy(y_phys[:, :n_prog, np.newaxis]).to(device)
                    # x[:, n_dyn:static_dim_size, ...] — static unchanged

        for r in results:
            r.get()
        pool.close()
        pool.join()

    logger.info("Forecast complete: %s → %s", init_str, save_dir)
    return ForecastResponse(
        status="ok",
        init_time=init_str,
        steps=req.steps,
        lead_time_hours=req.steps * lead_time_periods,
        save_dir=save_dir,
    )
