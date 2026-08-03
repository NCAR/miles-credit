"""Fast unit tests for the Gen2 GFS dataset and downloader."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import obstore
import pandas as pd
import pytest
import torch
import xarray as xr
from credit.datasets.gen_2.gfs import GFSDataset, _file_paths
from credit.datasets.gen_2.gfs_download import download_gfs


class _FakeBytes:
    def __init__(self, value: bytes) -> None:
        self.value = value

    def to_bytes(self) -> bytes:
        return self.value

    def bytes(self) -> bytes:
        return self.value


class _FakeReader:
    def __init__(self, value: bytes) -> None:
        self.buffer = io.BytesIO(value)

    def read(self, size: int = -1) -> _FakeBytes:
        return _FakeBytes(self.buffer.read(size))

    def readline(self, size: int = -1) -> _FakeBytes:
        return _FakeBytes(self.buffer.readline(size))

    def readlines(self, hint: int = -1) -> list[_FakeBytes]:
        return [_FakeBytes(line) for line in self.buffer.readlines(hint)]

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self.buffer.seek(offset, whence)

    def tell(self) -> int:
        return self.buffer.tell()

    def seekable(self) -> bool:
        return True

    def close(self) -> None:
        self.buffer.close()


class _FakeStore:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def get(self, path: str) -> _FakeBytes:
        return _FakeBytes(self.files[path])


def _make_dataset_bytes() -> tuple[bytes, bytes]:
    shape_3d = (1, 4, 2, 3)
    shape_2d = (1, 2, 3)
    coords = {
        "time": [np.datetime64("2024-01-01T00:00")],
        "pfull": [100.0, 500.0, 850.0, 1000.0],
        "grid_yt": [45.0, 15.0],
        "grid_xt": [0.0, 120.0, 240.0],
    }
    atm = xr.Dataset(
        {
            "lon": (("grid_yt", "grid_xt"), np.zeros((2, 3), dtype=np.float64)),
            "lat": (("grid_yt", "grid_xt"), np.zeros((2, 3), dtype=np.float64)),
            "tmp": (("time", "pfull", "grid_yt", "grid_xt"), np.full(shape_3d, 250.0, dtype=np.float32)),
            "ugrd": (("time", "pfull", "grid_yt", "grid_xt"), np.full(shape_3d, 12.0, dtype=np.float32)),
            "pressfc": (("time", "grid_yt", "grid_xt"), np.full(shape_2d, 100000.0, dtype=np.float32)),
        },
        coords=coords,
    )
    sfc = xr.Dataset(
        {
            "lon": (("grid_yt", "grid_xt"), np.zeros((2, 3), dtype=np.float64)),
            "lat": (("grid_yt", "grid_xt"), np.zeros((2, 3), dtype=np.float64)),
            "tmp2m": (("time", "grid_yt", "grid_xt"), np.full(shape_2d, 280.0, dtype=np.float32)),
            "land": (("time", "grid_yt", "grid_xt"), np.ones(shape_2d, dtype=np.float32)),
        },
        coords={key: value for key, value in coords.items() if key != "pfull"},
    )
    return bytes(atm.to_netcdf(engine="h5netcdf")), bytes(sfc.to_netcdf(engine="h5netcdf"))


@pytest.fixture
def fake_remote(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    atm_bytes, sfc_bytes = _make_dataset_bytes()
    files: dict[str, bytes] = {}
    timestamps = pd.date_range("2024-01-01", periods=2, freq="6h")
    for system in ("gdas", "gfs"):
        for forecast_hour in (None, 1):
            for timestamp in timestamps:
                atm_path, sfc_path = _file_paths(system, timestamp, forecast_hour)
                files[atm_path] = atm_bytes
                files[sfc_path] = sfc_bytes

    store = _FakeStore(files)
    monkeypatch.setattr(obstore.store, "GCSStore", lambda **kwargs: store)
    monkeypatch.setattr(
        obstore,
        "open_reader",
        lambda current_store, path: _FakeReader(current_store.files[path]),
    )
    monkeypatch.setattr(
        obstore,
        "list_with_delimiter",
        lambda current_store, prefix=None: {
            "common_prefixes": [],
            "objects": [{"path": path} for path in current_store.files if path.startswith(prefix or "")],
        },
    )
    return files


def _config(
    *,
    system: str = "gdas",
    mode: str = "remote",
    base_path: str | None = None,
    forecast_hour: int | None = None,
    level_type: str = "model",
    levels: list[int | float] | None = None,
    check_availability: bool = True,
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "dataset_type": "gfs",
        "system": system,
        "mode": mode,
        "level_type": level_type,
        "levels": levels,
        "forecast_hour": forecast_hour,
        "check_availability": check_availability,
        "variables": {
            "prognostic": {"vars_3D": ["tmp", "ugrd"], "vars_2D": ["pressfc", "tmp2m"]},
            "static": {"vars_2D": ["land"]},
        },
    }
    if base_path is not None:
        source["base_path"] = base_path
    return {
        "source": {"GFS": source},
        "start_datetime": "2024-01-01T00:00",
        "end_datetime": "2024-01-01T06:00",
        "timestep": "6h",
        "forecast_len": 0,
    }


def _assert_finite(sample: dict[str, Any]) -> None:
    for data_type in ("input", "target"):
        for tensor in sample.get(data_type, {}).values():
            assert torch.isfinite(tensor).all()


@pytest.mark.parametrize(
    ("system", "forecast_hour"),
    [("gdas", None), ("gfs", 1)],
)
def test_remote_read_shapes_and_finite_values(fake_remote, system, forecast_hour):
    dataset = GFSDataset(_config(system=system, forecast_hour=forecast_hour, levels=[1, 3]))

    assert len(dataset) == 2
    sample = dataset[(dataset.datetimes[0], 0)]
    assert sample["input"]["GFS/prognostic/3d/tmp"].shape == (2, 1, 2, 3)
    assert sample["input"]["GFS/prognostic/3d/ugrd"].shape == (2, 1, 2, 3)
    assert sample["input"]["GFS/prognostic/2d/pressfc"].shape == (1, 1, 2, 3)
    assert sample["input"]["GFS/prognostic/2d/tmp2m"].shape == (1, 1, 2, 3)
    assert sample["input"]["GFS/static/2d/land"].shape == (1, 1, 2, 3)
    _assert_finite(sample)


def test_remote_pressure_levels_and_missing_runs_are_filtered(fake_remote):
    dataset = GFSDataset(_config(level_type="pressure", levels=[850, 500]))

    assert len(dataset) == 2
    sample = dataset[(dataset.datetimes[0], 0)]
    tensor = sample["input"]["GFS/prognostic/3d/tmp"]
    assert tensor.shape == (2, 1, 2, 3)
    assert torch.equal(tensor[:, 0, 0, 0], torch.tensor([250.0, 250.0]))
    _assert_finite(sample)


def test_remote_availability_check_excludes_a_dropped_run(fake_remote, monkeypatch):
    dropped_path, _ = _file_paths("gdas", pd.Timestamp("2024-01-01 06:00"), None)
    fake_remote.pop(dropped_path)
    dataset = GFSDataset(_config())

    assert list(dataset.datetimes) == [pd.Timestamp("2024-01-01 00:00")]


def test_local_download_and_local_read(fake_remote, tmp_path: Path):
    config = _config(system="gfs", forecast_hour=1, base_path=str(tmp_path), mode="local")
    download_gfs(config, num_workers=1)

    dataset = GFSDataset(config)
    assert len(dataset) == 2
    assert all(path.is_file() for path in tmp_path.rglob("*.nc"))
    sample = dataset[(dataset.datetimes[0], 0)]
    assert sample["input"]["GFS/prognostic/3d/tmp"].shape == (4, 1, 2, 3)
    _assert_finite(sample)


@pytest.mark.parametrize(
    ("source_update", "message"),
    [
        ({"system": "unknown"}, "Expected 'gdas' or 'gfs'"),
        ({"level_type": "hybrid"}, "Expected 'model' or 'pressure'"),
    ],
)
def test_invalid_gfs_settings_raise(fake_remote, source_update, message):
    config = _config(check_availability=False)
    config["source"]["GFS"].update(source_update)
    with pytest.raises(ValueError, match=message):
        GFSDataset(config)
