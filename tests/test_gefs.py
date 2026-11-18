"""Fast mocked tests for the Gen2 GEFS dataset and downloader."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cftime
import numpy as np
import obstore
import pandas as pd
import pytest
import torch
import xarray as xr
from torch.utils.data import DataLoader
from credit.datasets.gen_2.gefs import _MICROPHYSICS_VARIABLES, GEFSDataset, _member_file_paths
from credit.datasets.gen_2.gefs_download import download_gefs
from credit.datasets.gen_2.multi_source import MultiSourceDataset
from credit.preblock.concat import ConcatToTensor
from credit.samplers import MultiStepBatchSamplerSubset


class _FakeBytes:
    def __init__(self, value: bytes) -> None:
        self.value = value

    def to_bytes(self) -> bytes:
        return self.value

    def bytes(self) -> bytes:
        return self.value

    def stream(self, min_chunk_size: int = 10 * 1024 * 1024) -> Iterator[bytes]:
        for start in range(0, len(self.value), min_chunk_size):
            yield self.value[start : start + min_chunk_size]


class _TruncatedBytes(_FakeBytes):
    """Fail partway through the body, as a GCS read timeout does."""

    def stream(self, min_chunk_size: int = 10 * 1024 * 1024) -> Iterator[bytes]:
        yield self.value[:8]
        raise RuntimeError("Generic GCS error: HTTP error: request or response body error")


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
            # Qtot species: sphum varies per level (0, 1, 2), the five condensates
            # are distinct powers of two summing to 62, so Qtot == level + 62.
            "sphum": (("lev", "lat", "lon"), np.tile(np.arange(3, dtype=np.float32)[:, None, None], (1, 2, 2))),
            "liq_wat": (("lev", "lat", "lon"), np.full((3, 2, 2), 2, dtype=np.float32)),
            "ice_wat": (("lev", "lat", "lon"), np.full((3, 2, 2), 4, dtype=np.float32)),
            "rainwat": (("lev", "lat", "lon"), np.full((3, 2, 2), 8, dtype=np.float32)),
            "snowwat": (("lev", "lat", "lon"), np.full((3, 2, 2), 16, dtype=np.float32)),
            "graupel": (("lev", "lat", "lon"), np.full((3, 2, 2), 32, dtype=np.float32)),
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
    assert sample["input"]["GEFS/prognostic/3d/t"].shape == (2, 1, 24)
    assert sample["input"]["GEFS/prognostic/3d/u_a"].shape == (2, 1, 24)
    assert sample["input"]["GEFS/prognostic/2d/ps"].shape == (1, 1, 24)
    assert sample["input"]["GEFS/static/2d/slmsk"].shape == (1, 1, 24)
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

    assert values.shape == (2, 1, 24)
    assert torch.equal(values[:, 0, 0], torch.tensor([2.0, 10.0]))


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


def test_interrupted_download_leaves_no_partial_or_truncated_file(fake_remote, tmp_path: Path, monkeypatch):
    config = _config(members=["c00"], mode="local", base_path=str(tmp_path))
    store = obstore.store.GCSStore()
    original_get = store.get
    target = "gfs_data.tile3.nc"

    def failing_get(path: str) -> _FakeBytes:
        if path.endswith(target):
            return _TruncatedBytes(original_get(path).value)
        return original_get(path)

    monkeypatch.setattr(store, "get", failing_get)
    download_gefs(config, num_workers=1)

    assert list(tmp_path.rglob("*.part")) == []
    assert target not in {path.name for path in tmp_path.rglob("*.nc")}

    monkeypatch.setattr(store, "get", original_get)
    download_gefs(config, num_workers=1)

    recovered = next(path for path in tmp_path.rglob("*.nc") if path.name == target)
    assert recovered.stat().st_size == len(fake_remote[f"gefs.20240101/00/atmos/init/c00/{target}"])


def test_single_member_batch_concatenates_on_the_channel_axis(fake_remote, tmp_path: Path):
    """A single-member sample must collate to the rank ConcatToTensor expects.

    Other Gen2 sources emit (levels, time, lat, lon) and let the DataLoader add
    the batch dim. GEFS drops its member dim at one member so the collated
    tensor is (batch, channels, time, spatial) rather than one rank too high,
    where concat would read the member axis as channels.
    """
    config = _config(members=["c00"], variables={"prognostic": {"vars_3D": ["t"], "vars_2D": ["ps"]}})
    dataset = MultiSourceDataset(config, return_target=False)
    sampler = MultiStepBatchSamplerSubset(dataset=dataset, batch_size=1, index_subset=[0], num_forecast_steps=1)
    batch = next(iter(DataLoader(dataset, batch_sampler=sampler, num_workers=0)))

    assert batch["input"]["GEFS"]["GEFS/prognostic/3d/t"].shape == (1, 2, 1, 24)
    assert batch["input"]["GEFS"]["GEFS/prognostic/2d/ps"].shape == (1, 1, 1, 24)

    x = ConcatToTensor(to_device=False)(batch)[0]
    assert x.shape == (1, 3, 1, 24)  # 2 levels of t + 1 of ps, concatenated on the channel axis


def test_multi_member_sample_keeps_the_member_dim(fake_remote):
    """Multi-member samples are untouched; folding members into batch is separate work."""
    dataset = GEFSDataset(_config(members=["c00", "p01"]))
    sample = dataset[(dataset.datetimes[0], 0)]
    assert sample["input"]["GEFS/prognostic/3d/t"].shape == (2, 2, 1, 24)


def test_qtot_sums_microphysics_species_per_level(fake_remote):
    """Qtot is the elementwise sum of the six species, taken per grid point.

    The fixture sets sphum to the level index and the five condensates to powers
    of two summing to 62, so Qtot == level + 62. Config levels [1, 3] are
    one-based, selecting level indices 0 and 2 -> 62 and 64. A sum that collapsed
    the vertical axis, or dropped a species, would not produce these.
    """
    dataset = GEFSDataset(_config(variables={"prognostic": {"vars_3D": ["Qtot"]}}))
    values = dataset[(dataset.datetimes[0], 0)]["input"]["GEFS/prognostic/3d/Qtot"]

    assert values.shape == (2, 1, 24)
    assert torch.equal(values[:, 0, 0], torch.tensor([62.0, 64.0]))
    assert torch.equal(values[0], torch.full((1, 24), 62.0))
    assert torch.equal(values[1], torch.full((1, 24), 64.0))


def test_qtot_does_not_consume_the_raw_species(fake_remote):
    """Requesting Qtot alongside a species returns both, independently."""
    dataset = GEFSDataset(_config(variables={"prognostic": {"vars_3D": ["Qtot", "sphum", "graupel"]}}))
    sample = dataset[(dataset.datetimes[0], 0)]["input"]

    assert sample["GEFS/prognostic/3d/Qtot"][:, 0, 0].tolist() == [62.0, 64.0]
    assert sample["GEFS/prognostic/3d/sphum"][:, 0, 0].tolist() == [0.0, 2.0]
    assert sample["GEFS/prognostic/3d/graupel"][:, 0, 0].tolist() == [32.0, 32.0]


def test_raw_species_alone_never_produce_qtot(fake_remote):
    """Not asking for Qtot leaves the channel set untouched."""
    dataset = GEFSDataset(_config(variables={"prognostic": {"vars_3D": ["sphum", "liq_wat"]}}))
    sample = dataset[(dataset.datetimes[0], 0)]["input"]

    assert set(sample) == {"GEFS/prognostic/3d/sphum", "GEFS/prognostic/3d/liq_wat"}


def test_qtot_names_the_missing_species(fake_remote):
    """A species absent from the file names Qtot and the culprit, not a bare KeyError."""
    dataset = GEFSDataset(_config(variables={"prognostic": {"vars_3D": ["Qtot"]}}))
    partial = xr.Dataset(
        {name: (("lev", "lat", "lon"), np.zeros((3, 2, 2), dtype=np.float32)) for name in _MICROPHYSICS_VARIABLES[:-1]},
        coords={"lev": [1, 2, 3], "lat": [0, 1], "lon": [0, 1]},
    )
    with pytest.raises(KeyError, match="graupel"):
        dataset._read_atmospheric_variable(partial, "Qtot")  # pyright: ignore[reportPrivateUsage]


def test_forecast_hour_is_rejected(fake_remote):
    config = _config()
    config["source"]["GEFS"]["forecast_hour"] = 3
    with pytest.raises(ValueError, match="initialization-time"):
        GEFSDataset(config)


# --------------------------------------------------------------------------- #
# Non-standard calendars: the shared master clock may hand over cftime
# --------------------------------------------------------------------------- #
def test_member_file_paths_accepts_either_calendar():
    """Path construction only strftime-formats the timestamp, so both types work."""
    pandas_paths = _member_file_paths(pd.Timestamp("2024-01-01"), "c00", "/base")
    cftime_paths = _member_file_paths(cftime.DatetimeNoLeap(2024, 1, 1), "c00", "/base")
    assert pandas_paths == cftime_paths
    assert pandas_paths[0] == "/base/gefs.20240101/00/atmos/init/c00/gfs_ctrl.nc"


def test_reading_a_sample_accepts_a_cftime_timestamp(fake_remote):
    """MultiSourceDataset resolves ONE calendar across all its sources.

    A noleap source -- CESM forcing or statics, say -- promotes the shared clock
    from pandas to cftime via most_restrictive_calendar, and every sub-dataset is
    then handed cftime timestamps. GEFS has to read from one. Wrapping it in
    pd.Timestamp() raised  TypeError: Cannot convert input ... DatetimeNoLeap.
    """
    dataset = GEFSDataset(_config())
    from_pandas = dataset[(dataset.datetimes[0], 0)]["input"]["GEFS/prognostic/3d/t"]
    from_cftime = dataset[(cftime.DatetimeNoLeap(2024, 1, 1), 0)]["input"]["GEFS/prognostic/3d/t"]

    assert from_cftime.shape == (2, 1, 24)
    assert torch.equal(from_pandas, from_cftime)
