"""Tests for the CREDIT Forecast API (applications/api.py).

All tests run without a real model, checkpoint, or GPU.
The lifespan (model loading) is never triggered — we either test with
_STATE empty (pre-startup behaviour) or patch _STATE directly with a
minimal mock to exercise the request/response logic.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Skip entire module if FastAPI is not installed (optional dependency)
fastapi = pytest.importorskip("fastapi", reason="fastapi not installed — pip install miles-credit[serve]")
from fastapi.testclient import TestClient  # noqa: E402

from applications.api import app, ForecastRequest  # noqa: E402


# ---------------------------------------------------------------------------
# Client fixture — does NOT trigger lifespan (model not loaded)
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """TestClient without lifespan — _STATE is empty."""
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Minimal mock state (replaces what lifespan normally populates)
# ---------------------------------------------------------------------------


def _make_mock_state(n_prog=3, n_dyn=1, h=4, w=8, c_out=4):
    """Build a minimal _STATE dict with a trivial identity-like model."""
    device = torch.device("cpu")

    class _TinyModel(torch.nn.Module):
        def forward(self, x):
            # Return zeros shaped like (B, c_out, 1, H, W)
            B, _, T, H, W = x.shape
            return torch.zeros(B, c_out, T, H, W)

    conf = {
        "data": {
            "timestep": "6h",
            "lead_time_periods": 6,
            "scaler_type": "std_new",
            "source": {
                "ERA5": {
                    "levels": [500, 850],
                    "level_coord": "level",
                    "variables": {
                        "prognostic": {"vars_3D": ["T"], "vars_2D": ["SP"]},
                        "diagnostic": {"vars_2D": []},
                        "dynamic_forcing": {"vars_2D": ["cos_lat"]},
                    },
                }
            },
            "mean_path": "/fake/mean.nc",
            "std_path": "/fake/std.nc",
        },
        "model": {"post_conf": {"activate": False}},
        "predict": {"mode": "none", "save_forecast": "/tmp/credit_test_fcst"},
        "save_loc": "/tmp",
    }

    return {
        "conf": conf,
        "device": device,
        "model": _TinyModel().eval(),
        "preblocks": torch.nn.ModuleDict(),
        "denorm_mean": torch.zeros(1, c_out, 1, 1, 1),
        "denorm_std": torch.ones(1, c_out, 1, 1, 1),
        "lat": np.linspace(90, -90, h),
        "lon": np.linspace(0, 360, w, endpoint=False),
        "meta_data": {},
        "n_prog": n_prog,
        "n_dyn": n_dyn,
    }


@pytest.fixture()
def patched_client():
    """TestClient with _STATE patched to a minimal mock — no real model needed."""
    mock_state = _make_mock_state()
    with patch.dict("applications.api._STATE", mock_state, clear=False):
        yield TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_model_not_loaded_before_startup(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is False

    def test_model_loaded_after_state_patch(self, patched_client):
        data = patched_client.get("/health").json()
        assert data["model_loaded"] is True

    def test_status_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_device_field_present(self, client):
        assert "device" in client.get("/health").json()


# ---------------------------------------------------------------------------
# /forecast — pre-startup (empty _STATE)
# ---------------------------------------------------------------------------


class TestForecastNotReady:
    def test_503_when_state_empty(self, client):
        resp = client.post("/forecast", json={"init_time": "2024-01-15T00", "steps": 2})
        assert resp.status_code == 503

    def test_503_message_informative(self, client):
        resp = client.post("/forecast", json={"init_time": "2024-01-15T00", "steps": 2})
        assert "not loaded" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# /forecast — schema validation (no model needed)
# ---------------------------------------------------------------------------


class TestForecastSchema:
    def test_missing_init_time_is_422(self, patched_client):
        resp = patched_client.post("/forecast", json={"steps": 10})
        assert resp.status_code == 422

    def test_invalid_init_time_is_422(self, patched_client):
        resp = patched_client.post("/forecast", json={"init_time": "not-a-date", "steps": 2})
        assert resp.status_code == 422

    def test_default_steps(self):
        req = ForecastRequest(init_time="2024-01-15T00")
        assert req.steps == 40

    def test_default_save_dir_is_none(self):
        req = ForecastRequest(init_time="2024-01-15T00")
        assert req.save_dir is None

    def test_default_save_workers(self):
        req = ForecastRequest(init_time="2024-01-15T00")
        assert req.save_workers == 4


# ---------------------------------------------------------------------------
# /forecast — full round-trip with mocked dataset + save worker
# ---------------------------------------------------------------------------


class TestForecastRoundTrip:
    """Run one forecast step end-to-end with a mock dataset and no disk IO."""

    def _make_fake_sample(self, n_in=5, h=4, w=8):
        """Return a minimal ERA5Dataset-style sample dict."""
        return {
            "input": {
                "era5": torch.zeros(n_in, 1, h, w),  # (C, T, H, W)
            },
            "metadata": {},
        }

    def test_returns_200_with_mocked_dataset_and_worker(self, patched_client, tmp_path):
        fake_sample = self._make_fake_sample()

        def _fake_dataset_getitem(key):
            return fake_sample

        mock_dataset = MagicMock()
        mock_dataset.__getitem__ = MagicMock(side_effect=_fake_dataset_getitem)

        def _fake_apply_preblocks(preblocks, batch):
            # Return a minimal x tensor: (1, C, 1, H, W)
            return {"x": torch.zeros(1, 5, 1, 4, 8)}

        def _fake_save_worker(*args, **kwargs):
            pass  # no disk IO

        with (
            patch("applications.api.ERA5Dataset", return_value=mock_dataset),
            patch("applications.api.apply_preblocks", side_effect=_fake_apply_preblocks),
            patch("applications.api._save_worker", side_effect=_fake_save_worker),
            patch("multiprocessing.pool.Pool.apply_async", return_value=MagicMock(get=lambda: None)),
        ):
            resp = patched_client.post(
                "/forecast",
                json={
                    "init_time": "2024-01-15T00",
                    "steps": 1,
                    "save_dir": str(tmp_path),
                },
            )

        assert resp.status_code == 200

    def test_response_fields_present(self, patched_client, tmp_path):
        fake_sample = {"input": {"era5": torch.zeros(5, 1, 4, 8)}, "metadata": {}}
        mock_dataset = MagicMock()
        mock_dataset.__getitem__ = MagicMock(return_value=fake_sample)

        with (
            patch("applications.api.ERA5Dataset", return_value=mock_dataset),
            patch("applications.api.apply_preblocks", return_value={"x": torch.zeros(1, 5, 1, 4, 8)}),
            patch("applications.api._save_worker"),
            patch("multiprocessing.pool.Pool.apply_async", return_value=MagicMock(get=lambda: None)),
        ):
            data = patched_client.post(
                "/forecast",
                json={
                    "init_time": "2024-01-15T00",
                    "steps": 1,
                    "save_dir": str(tmp_path),
                },
            ).json()

        for field in ("status", "init_time", "steps", "lead_time_hours", "save_dir"):
            assert field in data, f"Missing field: {field}"

    def test_lead_time_hours_correct(self, patched_client, tmp_path):
        """lead_time_hours = steps × lead_time_periods (6 h per step in mock conf)."""
        fake_sample = {"input": {"era5": torch.zeros(5, 1, 4, 8)}, "metadata": {}}
        mock_dataset = MagicMock()
        mock_dataset.__getitem__ = MagicMock(return_value=fake_sample)

        # Call 0 → initial state (5 channels); subsequent calls → dynamic forcing (n_dyn=1 channel)
        call_count = {"n": 0}

        def _preblocks(preblocks, batch):
            t = torch.zeros(1, 1, 1, 4, 8) if call_count["n"] > 0 else torch.zeros(1, 5, 1, 4, 8)
            call_count["n"] += 1
            return {"x": t}

        with (
            patch("applications.api.ERA5Dataset", return_value=mock_dataset),
            patch("applications.api.apply_preblocks", side_effect=_preblocks),
            patch("applications.api._save_worker"),
            patch("multiprocessing.pool.Pool.apply_async", return_value=MagicMock(get=lambda: None)),
        ):
            data = patched_client.post(
                "/forecast",
                json={
                    "init_time": "2024-01-15T00",
                    "steps": 3,
                    "save_dir": str(tmp_path),
                },
            ).json()

        # 3 steps × 6 h lead_time_periods = 18 h
        assert data["lead_time_hours"] == 18

    def test_dataset_open_failure_raises_500(self, patched_client):
        with patch("applications.api.ERA5Dataset", side_effect=FileNotFoundError("no data")):
            resp = patched_client.post(
                "/forecast",
                json={
                    "init_time": "2024-01-15T00",
                    "steps": 1,
                },
            )
        assert resp.status_code == 500
