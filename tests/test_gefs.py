"""Fast mocked tests for the Gen2 GEFS dataset and downloader."""

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
from credit.datasets.gen_2.gefs import GEFSDataset, _member_file_paths
from credit.datasets.gen_2.gefs_download import download_gefs


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


def _netcdf_bytes(member_offset: float) -> tuple[bytes, bytes, bytes]:
    coords = {
        "lev": [1, 2, 3],
        "levp": [1, 2, 3, 4],
        "lat": [0, 1],
        "lon": [0, 1],
        "latp": [0, 1, 2],
        "lonp": [0, 1, 2],
    }
    atm = xr.Dataset(
        {
            "geolat": (("lat", "lon"), np.array([[10, 11], [12, 13]], dtype=np.float32)),
            "geolon": (("lat", "lon"), np.array([[20, 21], [22, 23]], dtype=np.float32)),
            "ps": (("lat", "lon"), np.full((2, 2), 1000 + member_offset, dtype=np.float32)),
            "t": (("lev", "lat", "lon"), np.full((3, 2, 2), 250 + member_offset, dtype=np.float32)),
            "zh": (("levp", "lat", "lon"), np.arange(16, dtype=np.float32).reshape(4, 2, 2)),
            "u_s": (("lev", "latp", "lon"), np.full((3, 3, 2), 3 + member_offset, dtype=np.float32)),
            "v_w": (("lev", "lat", "lonp"), np.full((3, 2, 3), 4 + member_offset, dtype=np.float32)),
            "u_w": (("lev", "lat", "lonp"), np.full((3, 2, 3), 5 + member_offset, dtype=np.float32)),
            "v_s": (("lev", "latp", "lon"), np.full((3, 3, 2), 6 + member_offset, dtype=np.float32)),
        },
        coords=coords,
    )
    surface = xr.Dataset(
        {
            "geolat": (("yaxis_1", "xaxis_1"), np.array([[10, 11], [12, 13]], dtype=np.float32)),
            "geolon": (("yaxis_1", "xaxis_1"), np.array([[20, 21], [22, 23]], dtype=np.float32)),
            "t2m": (("Time", "yaxis_1", "xaxis_1"), np.full((1, 2, 2), 280 + member_offset, dtype=np.float32)),
            "slmsk": (("Time", "yaxis_1", "xaxis_1"), np.ones((1, 2, 2), dtype=np.float32)),
        },
        coords={"Time": [1], "yaxis_1": [1, 2], "xaxis_1": [1, 2]},
    )
    control = xr.Dataset({"vcoord": (("nvcoord", "levsp"), np.arange(8, dtype=np.float64).reshape(2, 4))})
    return (
        bytes(atm.to_netcdf(engine="h5netcdf")),
        bytes(surface.to_netcdf(engine="h5netcdf")),
        bytes(control.to_netcdf(engine="h5netcdf")),
    )


@pytest.fixture
def fake_remote(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    timestamp = pd.Timestamp("2024-01-01")
    for member, offset in (("c00", 0.0), ("p01", 10.0)):
        atm_bytes, surface_bytes, control_bytes = _netcdf_bytes(offset)
        control, atmospheric, surface = _member_file_paths(timestamp, member)
        files[control] = control_bytes
        for path in atmospheric:
            files[path] = atm_bytes
        for path in surface:
            files[path] = surface_bytes

    store = _FakeStore(files)
    monkeypatch.setattr(obstore.store, "GCSStore", lambda **kwargs: store)
    monkeypatch.setattr(obstore, "open_reader", lambda current_store, path: _FakeReader(current_store.files[path]))

    def list_with_delimiter(current_store, prefix=None):
        prefix = prefix or ""
        if prefix.endswith("/init/"):
            members = {path[len(prefix) :].split("/", 1)[0] for path in current_store.files if path.startswith(prefix)}
            return {"common_prefixes": [f"{prefix}{member}" for member in sorted(members)], "objects": []}
        return {
            "common_prefixes": [],
            "objects": [{"path": path} for path in current_store.files if path.startswith(prefix)],
        }

    monkeypatch.setattr(obstore, "list_with_delimiter", list_with_delimiter)
    return files


def _config(
    *,
    members: list[str] | None = None,
    mode: str = "remote",
    base_path: str | None = None,
    variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "dataset_type": "gefs",
        "mode": mode,
        "levels": [1, 3],
        "variables": variables
        or {
            "prognostic": {"vars_3D": ["t", "u_a", "v_a", "zh"], "vars_2D": ["ps", "t2m"]},
            "static": {"vars_2D": ["slmsk"]},
        },
    }
    if members is not None:
        source["members"] = members
    if base_path is not None:
        source["base_path"] = base_path
    return {
        "source": {"GEFS": source},
        "start_datetime": "2024-01-01",
        "end_datetime": "2024-01-01",
        "timestep": "6h",
        "forecast_len": 0,
    }


def _assert_finite(sample: dict[str, Any]) -> None:
    for data_type in ("input", "target"):
        for tensor in sample.get(data_type, {}).values():
            assert torch.isfinite(tensor).all()


def test_default_control_member_and_unstaggered_shapes(fake_remote):
    dataset = GEFSDataset(_config())
    sample = dataset[(dataset.datetimes[0], 0)]

    assert dataset.members == ["c00"]
    assert sample["input"]["GEFS/prognostic/3d/t"].shape == (1, 2, 1, 24)
    assert sample["input"]["GEFS/prognostic/3d/u_a"].shape == (1, 2, 1, 24)
    assert sample["input"]["GEFS/prognostic/2d/ps"].shape == (1, 1, 1, 24)
    assert sample["input"]["GEFS/static/2d/slmsk"].shape == (1, 1, 1, 24)
    assert dataset.static_metadata["grid"]["grid_type"] == "unstructured"
    assert dataset.static_metadata["grid"]["lat"].shape == (24,)
    _assert_finite(sample)


def test_all_members_and_raw_staggered_winds(fake_remote):
    variables = {"prognostic": {"vars_3D": ["u_s", "v_w"]}}
    dataset = GEFSDataset(_config(members=[], variables=variables))
    sample = dataset[(dataset.datetimes[0], 0)]

    assert dataset.members == ["c00", "p01"]
    assert sample["input"]["GEFS/prognostic/3d/u_s"].shape == (2, 2, 1, 36)
    assert sample["input"]["GEFS/prognostic/3d/v_w"].shape == (2, 2, 1, 36)
    _assert_finite(sample)


def test_zh_is_converted_from_interfaces_to_selected_midlevels(fake_remote):
    config = _config(variables={"prognostic": {"vars_3D": ["zh"]}})
    dataset = GEFSDataset(config)
    sample = dataset[(dataset.datetimes[0], 0)]
    values = sample["input"]["GEFS/prognostic/3d/zh"]

    assert values.shape == (1, 2, 1, 24)
    assert torch.equal(values[0, :, 0, 0], torch.tensor([2.0, 10.0]))


def test_missing_selected_member_fails_initialization(fake_remote):
    fake_remote.pop(next(path for path in fake_remote if "/p01/" in path and path.endswith("sfc_data.tile6.nc")))
    with pytest.raises(FileNotFoundError, match="p01.*sfc_data.tile6.nc"):
        GEFSDataset(_config(members=["c00", "p01"]))


def test_download_and_local_read(fake_remote, tmp_path: Path):
    config = _config(members=["c00", "p01"], mode="local", base_path=str(tmp_path))
    download_gefs(config, num_workers=1)

    dataset = GEFSDataset(config)
    sample = dataset[(dataset.datetimes[0], 0)]
    assert len(list(tmp_path.rglob("*.nc"))) == 26
    assert sample["input"]["GEFS/prognostic/3d/t"].shape == (2, 2, 1, 24)
    _assert_finite(sample)


def test_forecast_hour_is_rejected(fake_remote):
    config = _config()
    config["source"]["GEFS"]["forecast_hour"] = 3
    with pytest.raises(ValueError, match="initialization-time"):
        GEFSDataset(config)
