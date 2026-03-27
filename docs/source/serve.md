# Forecast API Server

CREDIT ships a lightweight FastAPI server that loads a trained model once and
serves autoregressive forecasts over HTTP. It is designed for scenarios where
you want to run forecasts on demand without the overhead of submitting a PBS
job each time — for example, a demo service, a shared inference node, or a
container in a Kubernetes cluster.

## Quick start

```bash
# Install the extra dependencies
pip install miles-credit[serve]    # fastapi + uvicorn

# Point at your config and launch
export CREDIT_CONFIG=/path/to/my_run.yml
uvicorn applications.api:app --host 0.0.0.0 --port 8000
```

The server loads the model and all normalisation statistics at startup and
keeps them in GPU memory. Each `/forecast` request runs the rollout and writes
output NetCDF files to disk.

## Endpoints

### `GET /health`

Returns 200 immediately. Use this as your liveness/readiness probe.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model_loaded": true, "device": "cuda:0"}
```

`model_loaded` is `false` until the lifespan startup finishes (model weights
are on disk — this can take 10–30 s).

---

### `POST /forecast`

Run an autoregressive forecast.

**Request body (JSON):**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `init_time` | string | required | ISO init time, e.g. `"2024-01-15T00"` |
| `steps` | int | `40` | Number of autoregressive steps |
| `save_dir` | string | from config | Directory to write output NetCDF files |
| `save_workers` | int | `4` | CPU workers for async NetCDF writes |

**Response:**

```json
{
  "status": "ok",
  "init_time": "2024-01-15T00Z",
  "steps": 40,
  "lead_time_hours": 240,
  "save_dir": "/path/to/output"
}
```

Output files follow the same naming convention as `credit realtime`:

```
<save_dir>/<YYYY-MM-DDTHH>Z/pred_<YYYY-MM-DDTHH>Z_<FHR:03d>.nc
```

**Example:**

```bash
curl -X POST http://localhost:8000/forecast \
    -H "Content-Type: application/json" \
    -d '{"init_time": "2024-01-15T00", "steps": 40}'
```

```bash
# Custom output directory
curl -X POST http://localhost:8000/forecast \
    -H "Content-Type: application/json" \
    -d '{"init_time": "2024-01-15T00", "steps": 40, "save_dir": "/scratch/me/forecasts"}'
```

---

## Interactive docs

FastAPI generates interactive API docs automatically. With the server running,
open your browser at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## Configuration

The server reads a single CREDIT v2 YAML config set via the `CREDIT_CONFIG`
environment variable. All model, data, and normalisation settings come from
that file — no extra config needed.

```bash
export CREDIT_CONFIG=/glade/work/$USER/my_run/config.yml
uvicorn applications.api:app --host 0.0.0.0 --port 8000
```

The config must have `predict.mode: none` (single-GPU inference). Multi-GPU
DDP/FSDP serving is not supported via the API; use `credit rollout` for that.

---

## Deployment notes

**Workers**: always use `--workers 1`. The model is loaded into GPU memory once
at startup; multiple workers would each load their own copy and quickly exhaust
VRAM.

**Timeouts**: requests block until the rollout finishes. A 40-step 1-degree
rollout takes roughly 30–60 s on an A100. Set your client and reverse-proxy
timeouts accordingly (e.g. `--timeout-keep-alive 300` for uvicorn behind
nginx).

**GPU**: the server automatically uses `cuda:0` if a GPU is available,
otherwise falls back to CPU (much slower — not recommended for production).

**On NCAR clusters**: run the server on a login node or an interactive Casper
job. Do not run long-lived servers inside PBS batch jobs.

**Docker / Kubernetes**: see the [Quickstart](quickstart.md) for the path to
containerisation. The server is the natural target for a Kubernetes Deployment
with a GPU node selector and a liveness probe pointed at `/health`.
