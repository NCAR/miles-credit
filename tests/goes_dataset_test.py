"""
goes_dataset_test.py
--------------------
Tests for GOESDataset (credit.datasets.gen_2.goes).

"""

import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from torch.utils.data import DataLoader

from credit.datasets.gen_2.goes import (
    GOESDataset,
    _FILE_NOT_FOUND,
    _FILE_QC_MASKED,
    _FILE_SKIP,
    _build_spatial_slices,
    _extent_covers,
    _extent_to_bbox,
    _find_nearest_latlon,
    _validate_catalog_row_path,
    _variables_covers,
)
from credit.samplers import DistributedMultiStepBatchSampler

CMI_C04 = "CMI_C04"
CMI_C07 = "CMI_C07"

# Fake grid dimensions (must match the lat/lon fixture)
NY = 40  # rows  (latitude  axis)
NX = 80  # cols  (longitude axis)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def goes_xr_dataset():
    """Single xr.Dataset on a fake curvilinear grid."""

    return xr.Dataset(
        data_vars={
            CMI_C04: (
                ("y", "x"),
                np.random.rand(NY, NX).astype(np.float32),
            ),
            CMI_C07: (
                ("y", "x"),
                np.random.rand(NY, NX).astype(np.float32),
            ),
        },
        coords={
            "y": np.arange(NY),
            "x": np.arange(NX),
        },
    )


@pytest.fixture
def latlon_xr_dataset():
    """Fake 2-D lat/lon grid matching goes_xr_dataset dimensions."""
    lat2d = np.linspace(20.0, 55.0, NY * NX).reshape(NY, NX).astype(np.float32)
    lon2d = np.linspace(-130.0, -60.0, NY * NX).reshape(NY, NX).astype(np.float32)

    return xr.Dataset(
        data_vars={
            "latitude": (("y", "x"), lat2d),
            "longitude": (("y", "x"), lon2d),
        },
    )


@pytest.fixture
def patch_goes_io(monkeypatch, goes_xr_dataset, latlon_xr_dataset):
    """Patch file-system and xr.open_dataset so GOESDataset uses fake data."""

    # --- patch _collect_GOES_file_path to return a fixed file map ---
    def fake_collect(self, base_dir="", verbose=False):
        extended = pd.date_range(
            self.datetimes[0],
            self.datetimes[-1] + self.dt,
            freq=self.dt,
        )
        result = []
        for t in extended:
            start = t
            end = t + self.dt - pd.Timedelta("1ns")
            result.append((start, end, "/fake/goes16_fake.nc"))
        return result

    monkeypatch.setattr(GOESDataset, "_collect_GOES_file_path", fake_collect)

    # --- patch xr.open_dataset to return fake GOES or latlon dataset ---
    original_open = xr.open_dataset

    def fake_open_dataset(path_or_obj, **kwargs):
        if isinstance(path_or_obj, str) and "lat_lon" in path_or_obj:
            return latlon_xr_dataset
        return goes_xr_dataset

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    return goes_xr_dataset


@pytest.fixture
def minimal_config():
    """Config with only prognostic field type."""
    return {
        "timestep": "6h",
        "forecast_len": 1,
        "start_datetime": "2021-06-01",
        "end_datetime": "2021-06-02",
        "source": {
            "TEST_GOES": {
                "dataset_type": "goes",
                "goes_position": "east",
                "mode": "local",
                "product": "ABI-L2-MCMIPC",
                "latlon2d_dir": "/fake/latlon/",
                "variables": {
                    "prognostic": {
                        "vars_2D": [CMI_C04, CMI_C07],
                        "path": "/fake/goes/",
                    },
                    "diagnostic": None,
                    "dynamic_forcing": None,
                },
            }
        },
    }


# ---------------------------------------------------------------------------
# Tests — basic construction
# ---------------------------------------------------------------------------


def test_goes_dataset_len(minimal_config, patch_goes_io):
    """Constructing the dataset yields at least one sample."""
    ds = GOESDataset(minimal_config)
    assert len(ds) > 0


def test_goes_dataset_datetimes_type(minimal_config, patch_goes_io):
    """self.datetimes is a pandas DatetimeIndex."""
    ds = GOESDataset(minimal_config)
    assert isinstance(ds.datetimes, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# Tests — key format
# ---------------------------------------------------------------------------


def test_goes_key_format(minimal_config, patch_goes_io):
    """Both CMI variables should appear under goes16/prognostic/2d/."""
    ds = GOESDataset(minimal_config)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    inp = sample["input"]
    assert f"TEST_GOES/prognostic/2d/{CMI_C04}" in inp
    assert f"TEST_GOES/prognostic/2d/{CMI_C07}" in inp
    assert "metadata" in sample


# ---------------------------------------------------------------------------
# Tests — step semantics
# ---------------------------------------------------------------------------


def test_goes_prognostic_loaded_at_step0(minimal_config, patch_goes_io):
    """Prognostic variables should appear in input at step i=0."""
    ds = GOESDataset(minimal_config)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    assert f"TEST_GOES/prognostic/2d/{CMI_C04}" in sample["input"]
    assert f"TEST_GOES/prognostic/2d/{CMI_C07}" in sample["input"]


def test_goes_prognostic_absent_at_step1(minimal_config, patch_goes_io):
    """Prognostic variables should NOT appear in input at step i > 0."""
    ds = GOESDataset(minimal_config)
    t = ds.datetimes[0]
    sample = ds[(t, 1)]

    # No dynamic_forcing configured here — input should be empty at i > 0
    assert len(sample["input"]) == 0


# ---------------------------------------------------------------------------
# Tests — tensor shape and dtype
# ---------------------------------------------------------------------------


def test_goes_tensor_shape(minimal_config, patch_goes_io):
    """All input tensors should have shape (1, 1, NY, NX)."""
    ds = GOESDataset(minimal_config)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    for key, tensor in sample["input"].items():
        assert tensor.shape == (1, 1, NY, NX), f"{key}: expected (1, 1, {NY}, {NX}), got {tensor.shape}"
        assert tensor.dtype == torch.float32


def test_goes_all_tensors_ndim4(minimal_config, patch_goes_io):
    """All input tensors must have exactly 4 dimensions."""
    ds = GOESDataset(minimal_config)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    for key, tensor in sample["input"].items():
        assert tensor.ndim == 4, f"{key} has {tensor.ndim} dims, expected 4"


# ---------------------------------------------------------------------------
# Tests — target
# ---------------------------------------------------------------------------


def test_goes_target_is_dict(minimal_config, patch_goes_io):
    """Target should be a dict, not a tensor."""
    ds = GOESDataset(minimal_config, return_target=True)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    assert "target" in sample
    assert isinstance(sample["target"], dict)


def test_goes_target_keys(minimal_config, patch_goes_io):
    """Target should contain prognostic variable keys."""
    ds = GOESDataset(minimal_config, return_target=True)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    assert f"TEST_GOES/prognostic/2d/{CMI_C04}" in sample["target"]
    assert f"TEST_GOES/prognostic/2d/{CMI_C07}" in sample["target"]


def test_goes_target_tensor_shapes(minimal_config, patch_goes_io):
    """Target tensors should have the same shape as the corresponding input tensors."""
    ds = GOESDataset(minimal_config, return_target=True)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    for key in sample["target"]:
        inp_key = key  # prognostic keys are shared between input and target
        if inp_key in sample["input"]:
            assert sample["target"][key].shape == sample["input"][inp_key].shape


def test_goes_no_target_without_flag(minimal_config, patch_goes_io):
    """'target' key should be absent when return_target=False (default)."""
    ds = GOESDataset(minimal_config, return_target=False)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    assert "target" not in sample


# ---------------------------------------------------------------------------
# Tests — metadata
# ---------------------------------------------------------------------------


def test_goes_metadata_input_datetime(minimal_config, patch_goes_io):
    """metadata['input_datetime'] should match the sampled timestamp (nanoseconds)."""
    ds = GOESDataset(minimal_config)
    for t in ds.datetimes:
        sample = ds[(t, 0)]
        assert sample["metadata"]["input_datetime"] == int(t.value)


def test_goes_metadata_target_datetime(minimal_config, patch_goes_io):
    """metadata['target_datetime'] should equal input_datetime + dt."""
    ds = GOESDataset(minimal_config, return_target=True)
    for t in ds.datetimes:
        sample = ds[(t, 0)]
        expected = int((t + ds.dt).value)
        assert sample["metadata"]["target_datetime"] == expected


def test_goes_metadata_datetimes_all_samples(minimal_config, patch_goes_io):
    """input and target datetimes should be consistent across the full dataset."""
    ds = GOESDataset(minimal_config, return_target=True)
    x_times, y_times = [], []
    for t in ds.datetimes:
        sample = ds[(t, 0)]
        x_times.append(sample["metadata"]["input_datetime"])
        y_times.append(sample["metadata"]["target_datetime"])

    assert (pd.to_datetime(x_times) == pd.to_datetime(ds.datetimes)).all()
    assert (pd.to_datetime(y_times) == (pd.to_datetime(ds.datetimes) + ds.dt)).all()


# ---------------------------------------------------------------------------
# Tests — spatial extent
# ---------------------------------------------------------------------------


def test_goes_extent_applied(minimal_config, patch_goes_io, monkeypatch, latlon_xr_dataset):
    """With extent set, spatial dims should be smaller than the full grid."""
    cfg = {
        **minimal_config,
        "source": {
            "TEST_GOES": {
                **minimal_config["source"]["TEST_GOES"],
                "extent": [-130, -95, 20, 55],  # roughly half the fake lon range
            }
        },
    }

    ds = GOESDataset(cfg)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    key = f"TEST_GOES/prognostic/2d/{CMI_C04}"
    tensor = sample["input"][key]

    # At least one spatial dim should be strictly smaller than the full grid
    assert tensor.shape[-1] < NX or tensor.shape[-2] < NY, (
        f"Expected cropped spatial dims, got full shape {tensor.shape}"
    )


# ---------------------------------------------------------------------------
# Tests — DataLoader integration
# ---------------------------------------------------------------------------


def test_goes_dataloader_default_collate(minimal_config, patch_goes_io):
    """DataLoader + DistributedMultiStepBatchSampler should collate correctly."""
    ds = GOESDataset(minimal_config, return_target=True)
    sampler = DistributedMultiStepBatchSampler(
        ds,
        batch_size=2,
        num_forecast_steps=minimal_config["forecast_len"],
        shuffle=False,
        num_replicas=1,
        rank=0,
    )
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=None,
    )

    batch = next(iter(loader))

    key = f"TEST_GOES/prognostic/2d/{CMI_C04}"
    assert batch["input"][key].shape == (2, 1, 1, NY, NX)
    assert batch["target"][key].shape == (2, 1, 1, NY, NX)


# ---------------------------------------------------------------------------
# Tests — module-level helper functions
# ---------------------------------------------------------------------------


def test_extent_to_bbox_none():
    """A None extent converts to a None bbox (the full grid)."""
    assert _extent_to_bbox(None) is None


def test_extent_to_bbox_list():
    """A [lon_min, lon_max, lat_min, lat_max] list converts to a (lat_min, lat_max, lon_min, lon_max) bbox."""
    assert _extent_to_bbox([-130, -60, 20, 55]) == (20, 55, -130, -60)


def test_extent_to_bbox_dict():
    """An {nw, se} dict converts to the same bbox ordering as the list form."""
    assert _extent_to_bbox({"nw": [55, -130], "se": [20, -60]}) == (20, 55, -130, -60)


def test_extent_to_bbox_dict_missing_keys_raises():
    """A dict extent missing 'nw' or 'se' raises ValueError."""
    with pytest.raises(ValueError, match="nw"):
        _extent_to_bbox({"nw": [55, -130]})


def test_extent_to_bbox_bad_type_raises():
    """A non-list/dict/None extent raises TypeError."""
    with pytest.raises(TypeError):
        _extent_to_bbox("not-an-extent")


def test_extent_covers_catalog_full_grid_covers_anything():
    """A catalog with no extent (the full grid) covers any request."""
    assert _extent_covers(None, [-130, -60, 20, 55]) is True
    assert _extent_covers(None, None) is True


def test_extent_covers_request_full_grid_not_covered_by_partial_catalog():
    """A full-grid request is not covered by a catalog built for a partial extent."""
    assert _extent_covers([-130, -60, 20, 55], None) is False


def test_extent_covers_identical_extent_no_margin_needed():
    """An identical extent is covered with no safety margin required."""
    extent = [-130, -60, 20, 55]
    assert _extent_covers(extent, extent) is True


def test_extent_covers_subdomain_with_margin():
    """A request extent inset well within the catalog's extent is covered."""
    catalog = [-130, -60, 20, 55]
    request = [-125, -65, 25, 50]  # inset by 5 deg on every side
    assert _extent_covers(catalog, request) is True


def test_extent_covers_too_close_to_boundary_rejected():
    """A request extent inset by less than the safety margin is rejected."""
    catalog = [-130, -60, 20, 55]
    request = [-129.9, -65, 25, 50]  # inset by only 0.1 deg on one side
    assert _extent_covers(catalog, request) is False


def test_extent_covers_superset_rejected():
    """A request extent larger than the catalog's extent is rejected."""
    catalog = [-125, -65, 25, 50]
    request = [-130, -60, 20, 55]  # larger than the catalog
    assert _extent_covers(catalog, request) is False


def test_variables_covers_none_catalog_rejected():
    """A None catalog (no variables metadata) never covers a request."""
    assert _variables_covers(None, {"prognostic": {"vars_2D": [CMI_C04]}}) is False


def test_variables_covers_subset_accepted():
    """A request for a subset of the catalog's checked channels is covered."""
    catalog = {"prognostic": {"vars_2D": [CMI_C04, CMI_C07, "CMI_C08"], "vars_3D": []}}
    request = {"prognostic": {"vars_2D": [CMI_C04], "vars_3D": []}}
    assert _variables_covers(catalog, request) is True


def test_variables_covers_missing_field_type_rejected():
    """A request for a field type the catalog never checked is rejected."""
    catalog = {"prognostic": {"vars_2D": [CMI_C04], "vars_3D": []}}
    request = {"diagnostic": {"vars_2D": [CMI_C04], "vars_3D": []}}
    assert _variables_covers(catalog, request) is False


def test_variables_covers_uncovered_channel_rejected():
    """A request for a channel outside the catalog's checked channels is rejected."""
    catalog = {"prognostic": {"vars_2D": [CMI_C04], "vars_3D": []}}
    request = {"prognostic": {"vars_2D": [CMI_C04, CMI_C07], "vars_3D": []}}
    assert _variables_covers(catalog, request) is False


def test_validate_catalog_row_path_accepts_real_paths():
    """Real file paths (local, s3, or bare .nc filenames) pass validation."""
    _validate_catalog_row_path("/data/goes/file.nc", pd.Timestamp("2021-06-01"))
    _validate_catalog_row_path("s3://bucket/file.nc", pd.Timestamp("2021-06-01"))
    _validate_catalog_row_path("bare_filename.nc", pd.Timestamp("2021-06-01"))


def test_validate_catalog_row_path_accepts_known_flags():
    """Each recognized availability flag passes validation."""
    for flag in (_FILE_NOT_FOUND, _FILE_QC_MASKED, _FILE_SKIP):
        _validate_catalog_row_path(flag, pd.Timestamp("2021-06-01"))


def test_validate_catalog_row_path_rejects_typo():
    """An unrecognized flag-like value raises ValueError."""
    with pytest.raises(ValueError, match="Unrecognized flag"):
        _validate_catalog_row_path("SKIPP", pd.Timestamp("2021-06-01"))


def test_find_nearest_latlon():
    """Finds the grid index nearest to a target lat/lon via Haversine distance."""
    lat2d, lon2d = np.meshgrid(np.arange(20.0, 25.0), np.arange(-100.0, -95.0), indexing="ij")
    i, j = _find_nearest_latlon(lat2d, lon2d, lat_target=22.9, lon_target=-97.1)
    assert lat2d[i, j] == pytest.approx(23.0)
    assert lon2d[i, j] == pytest.approx(-97.0)


def test_build_spatial_slices_none_extent():
    """A None extent returns full-grid slices."""
    y_slice, x_slice = _build_spatial_slices(None)
    assert y_slice == slice(None)
    assert x_slice == slice(None)


def test_build_spatial_slices_requires_grid_for_extent():
    """A non-None extent without lat2d/lon2d raises ValueError."""
    with pytest.raises(ValueError, match="lat2d and lon2d"):
        _build_spatial_slices([-100, -95, 20, 25])
    with pytest.raises(ValueError, match="lat2d and lon2d"):
        _build_spatial_slices({"nw": [25, -100], "se": [20, -95]})


def test_build_spatial_slices_bad_type_raises():
    """A non-list/dict/None extent raises TypeError."""
    with pytest.raises(TypeError):
        _build_spatial_slices("bad-extent", np.zeros((5, 5)), np.zeros((5, 5)))


def test_build_spatial_slices_dict_extent():
    """An {nw, se} dict extent crops to the expected lat/lon sub-grid."""
    lat2d, lon2d = np.meshgrid(np.arange(20.0, 26.0), np.arange(-100.0, -94.0), indexing="ij")
    y_slice, x_slice = _build_spatial_slices({"nw": [24.0, -99.0], "se": [21.0, -96.0]}, lat2d, lon2d)
    cropped_lat = lat2d[y_slice, x_slice]
    cropped_lon = lon2d[y_slice, x_slice]
    assert cropped_lat.min() >= 21.0 and cropped_lat.max() <= 24.0
    assert cropped_lon.min() >= -99.0 and cropped_lon.max() <= -96.0


def test_build_spatial_slices_dict_extent_missing_keys_raises():
    """A dict extent missing 'nw' or 'se' raises ValueError."""
    with pytest.raises(ValueError, match="nw"):
        _build_spatial_slices({"nw": [24.0, -99.0]}, np.zeros((5, 5)), np.zeros((5, 5)))


# ---------------------------------------------------------------------------
# Tests — config validation & construction edge cases
# ---------------------------------------------------------------------------


def test_goes_dataset_type_mismatch_raises(minimal_config):
    """A config with the wrong dataset_type raises ValueError."""
    cfg = {**minimal_config}
    cfg["source"] = {"TEST_GOES": {**minimal_config["source"]["TEST_GOES"], "dataset_type": "mrms"}}
    with pytest.raises(ValueError, match="Expected dataset_type 'goes'"):
        GOESDataset(cfg)


def test_goes_invalid_mode_raises(minimal_config):
    """An unrecognized mode value raises ValueError."""
    cfg = {**minimal_config}
    cfg["source"] = {"TEST_GOES": {**minimal_config["source"]["TEST_GOES"], "mode": "bogus"}}
    with pytest.raises(ValueError, match="Unknown mode"):
        GOESDataset(cfg)


def test_goes_invalid_goes_position_raises(minimal_config):
    """An unrecognized goes_position value raises ValueError."""
    cfg = {**minimal_config}
    cfg["source"] = {"TEST_GOES": {**minimal_config["source"]["TEST_GOES"], "goes_position": "north"}}
    with pytest.raises(ValueError, match="goes_position must be"):
        GOESDataset(cfg)


def test_goes_invalid_product_raises(minimal_config):
    """A product not ending in 'C' or 'F' raises ValueError."""
    cfg = {**minimal_config}
    cfg["source"] = {"TEST_GOES": {**minimal_config["source"]["TEST_GOES"], "product": "ABI-L2-MCMIPX"}}
    with pytest.raises(ValueError, match="Unrecognized product"):
        GOESDataset(cfg)


def test_goes_missing_latlon_grid_raises(minimal_config, monkeypatch, goes_xr_dataset):
    """A missing lat/lon grid file raises FileNotFoundError at construction time."""

    def fake_collect(self, base_dir="", verbose=False):
        extended = pd.date_range(self.datetimes[0], self.datetimes[-1] + self.dt, freq=self.dt)
        return [(t, t + self.dt - pd.Timedelta("1ns"), "/fake/goes16_fake.nc") for t in extended]

    monkeypatch.setattr(GOESDataset, "_collect_GOES_file_path", fake_collect)

    def fake_open_dataset(path_or_obj, **kwargs):
        if isinstance(path_or_obj, str) and "lat_lon" in path_or_obj:
            raise FileNotFoundError("no such file")
        return goes_xr_dataset

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    with pytest.raises(FileNotFoundError, match="Latitude/longitude grid file not found"):
        GOESDataset(minimal_config)


def test_goes_west_full_disk_config(minimal_config, patch_goes_io):
    """west/full-disk should resolve the goes18_..._full_disk lat/lon grid filename and
    construct normally (exercising the branches the CONUS/east tests never touch)."""
    cfg = {
        **minimal_config,
        "source": {
            "TEST_GOES": {
                **minimal_config["source"]["TEST_GOES"],
                "goes_position": "west",
                "product": "ABI-L2-MCMIPF",
            }
        },
    }
    ds = GOESDataset(cfg)
    assert ds.goes_position == "west"
    assert len(ds.datetimes) > 0


# ---------------------------------------------------------------------------
# Tests — availability-flag filtering (_filter_unavailable_timestamps)
# ---------------------------------------------------------------------------


@pytest.fixture
def make_patch_goes_io(monkeypatch, goes_xr_dataset, latlon_xr_dataset):
    """Factory fixture: patch GOESDataset IO with a customizable per-timestamp file map.

    Unlike ``patch_goes_io``, callers control exactly which timestamp gets which
    file path (a real path vs. an availability flag), to exercise
    ``_filter_unavailable_timestamps``.
    """

    def _apply(overrides=None):
        overrides = overrides or {}

        def fake_collect(self, base_dir="", verbose=False):
            extended = pd.date_range(
                self.datetimes[0],
                self.datetimes[-1] + self.num_forecast_steps * self.dt,
                freq=self.dt,
            )
            result = []
            for t in extended:
                path = overrides.get(t, "/fake/goes16_fake.nc")
                result.append((t, t + self.dt - pd.Timedelta("1ns"), path))
            return result

        monkeypatch.setattr(GOESDataset, "_collect_GOES_file_path", fake_collect)

        def fake_open_dataset(path_or_obj, **kwargs):
            if isinstance(path_or_obj, str) and "lat_lon" in path_or_obj:
                return latlon_xr_dataset
            return goes_xr_dataset

        monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    return _apply


def test_goes_filter_drops_qc_masked_timestamp(minimal_config, make_patch_goes_io):
    """A QC_MASKED timestamp is dropped from self.datetimes."""
    make_patch_goes_io()
    baseline = GOESDataset(minimal_config)
    # datetimes[0] has no earlier sibling that depends on it as a forecast target, so
    # flagging it drops exactly one timestamp instead of cascading to its predecessor too.
    bad_t = baseline.datetimes[0]

    make_patch_goes_io({bad_t: _FILE_QC_MASKED})
    ds = GOESDataset(minimal_config)

    assert bad_t not in ds.datetimes
    assert len(ds.datetimes) == len(baseline.datetimes) - 1


def test_goes_filter_drops_missing_timestamp(minimal_config, make_patch_goes_io):
    """A MISSING timestamp is dropped from self.datetimes."""
    make_patch_goes_io()
    baseline = GOESDataset(minimal_config)
    # datetimes[0] has no earlier sibling that depends on it as a forecast target, so
    # flagging it drops exactly one timestamp instead of cascading to its predecessor too.
    bad_t = baseline.datetimes[0]

    make_patch_goes_io({bad_t: _FILE_NOT_FOUND})
    ds = GOESDataset(minimal_config)

    assert bad_t not in ds.datetimes
    assert len(ds.datetimes) == len(baseline.datetimes) - 1


def test_goes_filter_drops_skip_timestamp(minimal_config, make_patch_goes_io):
    """A SKIP timestamp is dropped from self.datetimes."""
    make_patch_goes_io()
    baseline = GOESDataset(minimal_config)
    # datetimes[0] has no earlier sibling that depends on it as a forecast target, so
    # flagging it drops exactly one timestamp instead of cascading to its predecessor too.
    bad_t = baseline.datetimes[0]

    make_patch_goes_io({bad_t: _FILE_SKIP})
    ds = GOESDataset(minimal_config)

    assert bad_t not in ds.datetimes
    assert len(ds.datetimes) == len(baseline.datetimes) - 1


def test_goes_filter_drops_input_when_future_forecast_step_missing(minimal_config, make_patch_goes_io):
    """An input timestamp must be dropped if any of its forecast target steps is unavailable."""
    cfg = {**minimal_config, "forecast_len": 2}
    make_patch_goes_io()
    baseline = GOESDataset(cfg)

    t0 = baseline.datetimes[1]
    target_bad = t0 + 2 * baseline.dt  # t0's 2nd forecast step, beyond the input datetime range
    assert target_bad not in baseline.datetimes  # isolates the dependency check from a direct flag

    make_patch_goes_io({target_bad: _FILE_NOT_FOUND})
    ds = GOESDataset(cfg)

    assert t0 not in ds.datetimes
    assert baseline.datetimes[0] in ds.datetimes  # unaffected sibling timestamp still present


# ---------------------------------------------------------------------------
# Tests — pre-built file catalogs (_load_file_catalog)
# ---------------------------------------------------------------------------


DEFAULT_CATALOG_VARIABLES = {"prognostic": {"vars_2D": [CMI_C04, CMI_C07], "vars_3D": []}}


def _catalog_rows_for_range(start, end, freq, path="/fake/s3/goes_fake.nc"):
    """Build [start_iso, end_iso, path] rows covering every timestamp in [start, end]."""
    times = pd.date_range(start, end, freq=freq)
    rows = []
    for t in times:
        period_end = t + pd.Timedelta(freq) - pd.Timedelta("1ns")
        rows.append([t.isoformat(), period_end.isoformat(), path])
    return rows


def _write_catalog(
    tmp_path,
    name,
    *,
    mode,
    timestep,
    product,
    goes_position,
    extent,
    variables,
    rows,
    start_datetime,
    end_datetime,
):
    catalog = {
        "metadata": {
            "mode": mode,
            "timestep": timestep,
            "product": product,
            "goes_position": goes_position,
            "extent": extent,
            "variables": variables,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
        },
        "catalog": rows,
    }
    path = tmp_path / name
    path.write_text(json.dumps(catalog))
    return str(path)


def _catalog_config(minimal_config, catalog_path, **source_overrides):
    cfg = {**minimal_config}
    cfg["source"] = {
        "TEST_GOES": {
            **minimal_config["source"]["TEST_GOES"],
            "file_catalog_path": catalog_path,
            **source_overrides,
        }
    }
    return cfg


@pytest.fixture
def patch_goes_xr_open(monkeypatch, goes_xr_dataset, latlon_xr_dataset):
    """Patch only xr.open_dataset; real file-catalog JSON I/O is left untouched."""

    def fake_open_dataset(path_or_obj, **kwargs):
        if isinstance(path_or_obj, str) and "lat_lon" in path_or_obj:
            return latlon_xr_dataset
        return goes_xr_dataset

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)


def test_goes_catalog_load_success(tmp_path, minimal_config, patch_goes_xr_open):
    """A catalog matching the current config exactly loads and serves a sample successfully."""
    dt_str = str(pd.Timedelta(minimal_config["timestep"]))
    rows = _catalog_rows_for_range(
        minimal_config["start_datetime"], minimal_config["end_datetime"], minimal_config["timestep"]
    )
    catalog_path = _write_catalog(
        tmp_path,
        "catalog.json",
        mode="local",
        timestep=dt_str,
        product="ABI-L2-MCMIPC",
        goes_position="east",
        extent=None,
        variables=DEFAULT_CATALOG_VARIABLES,
        rows=rows,
        start_datetime=minimal_config["start_datetime"],
        end_datetime=minimal_config["end_datetime"],
    )

    ds = GOESDataset(_catalog_config(minimal_config, catalog_path))
    assert len(ds.datetimes) > 0
    sample = ds[(ds.datetimes[0], 0)]
    assert f"TEST_GOES/prognostic/2d/{CMI_C04}" in sample["input"]


def test_goes_catalog_mismatched_product_raises(tmp_path, minimal_config, patch_goes_xr_open):
    """A catalog built for a different product is rejected, not silently reused."""
    dt_str = str(pd.Timedelta(minimal_config["timestep"]))
    rows = _catalog_rows_for_range(
        minimal_config["start_datetime"], minimal_config["end_datetime"], minimal_config["timestep"]
    )
    catalog_path = _write_catalog(
        tmp_path,
        "catalog.json",
        mode="local",
        timestep=dt_str,
        product="ABI-L2-MCMIPF",
        goes_position="east",
        extent=None,
        variables=DEFAULT_CATALOG_VARIABLES,
        rows=rows,
        start_datetime=minimal_config["start_datetime"],
        end_datetime=minimal_config["end_datetime"],
    )

    with pytest.raises(ValueError, match="File catalog mismatch"):
        GOESDataset(_catalog_config(minimal_config, catalog_path))


def test_goes_catalog_extent_subdomain_accepted(tmp_path, minimal_config, patch_goes_xr_open):
    """A request extent safely inset within the catalog's extent is accepted."""
    dt_str = str(pd.Timedelta(minimal_config["timestep"]))
    rows = _catalog_rows_for_range(
        minimal_config["start_datetime"], minimal_config["end_datetime"], minimal_config["timestep"]
    )
    catalog_path = _write_catalog(
        tmp_path,
        "catalog.json",
        mode="local",
        timestep=dt_str,
        product="ABI-L2-MCMIPC",
        goes_position="east",
        extent=[-130, -60, 20, 55],
        variables=DEFAULT_CATALOG_VARIABLES,
        rows=rows,
        start_datetime=minimal_config["start_datetime"],
        end_datetime=minimal_config["end_datetime"],
    )

    request_extent = [-125, -65, 25, 50]  # inset by 5 deg on every side
    ds = GOESDataset(_catalog_config(minimal_config, catalog_path, extent=request_extent))
    assert len(ds.datetimes) > 0


def test_goes_catalog_extent_too_close_to_boundary_raises(tmp_path, minimal_config, patch_goes_xr_open):
    """A request extent inset by less than the safety margin is rejected."""
    dt_str = str(pd.Timedelta(minimal_config["timestep"]))
    rows = _catalog_rows_for_range(
        minimal_config["start_datetime"], minimal_config["end_datetime"], minimal_config["timestep"]
    )
    catalog_path = _write_catalog(
        tmp_path,
        "catalog.json",
        mode="local",
        timestep=dt_str,
        product="ABI-L2-MCMIPC",
        goes_position="east",
        extent=[-130, -60, 20, 55],
        variables=DEFAULT_CATALOG_VARIABLES,
        rows=rows,
        start_datetime=minimal_config["start_datetime"],
        end_datetime=minimal_config["end_datetime"],
    )

    request_extent = [-129.9, -65, 25, 50]  # inset by only 0.1 deg on one side
    with pytest.raises(ValueError, match="extent"):
        GOESDataset(_catalog_config(minimal_config, catalog_path, extent=request_extent))


def test_goes_catalog_overlapping_files_raises(tmp_path, minimal_config, patch_goes_xr_open):
    """Two matched catalog files covering overlapping time ranges cannot be merged."""
    dt_str = str(pd.Timedelta(minimal_config["timestep"]))
    rows = _catalog_rows_for_range(
        minimal_config["start_datetime"], minimal_config["end_datetime"], minimal_config["timestep"]
    )
    common_kwargs = dict(
        mode="local",
        timestep=dt_str,
        product="ABI-L2-MCMIPC",
        goes_position="east",
        extent=None,
        variables=DEFAULT_CATALOG_VARIABLES,
        rows=rows,
        start_datetime=minimal_config["start_datetime"],
        end_datetime=minimal_config["end_datetime"],
    )
    _write_catalog(tmp_path, "catalog_a.json", **common_kwargs)
    _write_catalog(tmp_path, "catalog_b.json", **common_kwargs)  # identical range -> overlaps catalog_a

    with pytest.raises(ValueError, match="Overlapping catalog files"):
        GOESDataset(_catalog_config(minimal_config, str(tmp_path / "catalog_*.json")))


def test_goes_catalog_missing_timestamp_raises(tmp_path, minimal_config, patch_goes_xr_open):
    """A catalog missing a row for a required timestamp is rejected."""
    dt_str = str(pd.Timedelta(minimal_config["timestep"]))
    rows = _catalog_rows_for_range(
        minimal_config["start_datetime"], minimal_config["end_datetime"], minimal_config["timestep"]
    )
    del rows[1]  # drop one required timestamp
    catalog_path = _write_catalog(
        tmp_path,
        "catalog.json",
        mode="local",
        timestep=dt_str,
        product="ABI-L2-MCMIPC",
        goes_position="east",
        extent=None,
        variables=DEFAULT_CATALOG_VARIABLES,
        rows=rows,
        start_datetime=minimal_config["start_datetime"],
        end_datetime=minimal_config["end_datetime"],
    )

    with pytest.raises(ValueError, match="Catalog is missing"):
        GOESDataset(_catalog_config(minimal_config, catalog_path))


def test_goes_catalog_bad_flag_typo_raises(tmp_path, minimal_config, patch_goes_xr_open):
    """A catalog row with an unrecognized (typo'd) flag value is rejected."""
    dt_str = str(pd.Timedelta(minimal_config["timestep"]))
    rows = _catalog_rows_for_range(
        minimal_config["start_datetime"], minimal_config["end_datetime"], minimal_config["timestep"]
    )
    rows[1][2] = "SKIPP"  # typo of "SKIP" -- not a recognized flag and not a path
    catalog_path = _write_catalog(
        tmp_path,
        "catalog.json",
        mode="local",
        timestep=dt_str,
        product="ABI-L2-MCMIPC",
        goes_position="east",
        extent=None,
        variables=DEFAULT_CATALOG_VARIABLES,
        rows=rows,
        start_datetime=minimal_config["start_datetime"],
        end_datetime=minimal_config["end_datetime"],
    )

    with pytest.raises(ValueError, match="Unrecognized flag"):
        GOESDataset(_catalog_config(minimal_config, catalog_path))


def test_goes_catalog_variables_disagreement_between_files_raises(tmp_path, minimal_config, patch_goes_xr_open):
    """Two matched catalog files that disagree on 'variables' can't be safely merged."""
    dt_str = str(pd.Timedelta(minimal_config["timestep"]))
    rows_a = _catalog_rows_for_range("2021-06-01", "2021-06-01 18:00", minimal_config["timestep"])
    rows_b = _catalog_rows_for_range("2021-06-02", "2021-06-02 18:00", minimal_config["timestep"])
    common = dict(mode="local", timestep=dt_str, product="ABI-L2-MCMIPC", goes_position="east", extent=None)

    _write_catalog(
        tmp_path,
        "catalog_a.json",
        **common,
        variables={"prognostic": {"vars_2D": [CMI_C04, CMI_C07], "vars_3D": []}},
        rows=rows_a,
        start_datetime="2021-06-01",
        end_datetime="2021-06-01 18:00",
    )
    _write_catalog(
        tmp_path,
        "catalog_b.json",
        **common,
        variables={"prognostic": {"vars_2D": [CMI_C04], "vars_3D": []}},
        rows=rows_b,
        start_datetime="2021-06-02",
        end_datetime="2021-06-02 18:00",
    )

    cfg = _catalog_config(minimal_config, str(tmp_path / "catalog_*.json"))
    with pytest.raises(ValueError, match="File catalog mismatch.*'variables'"):
        GOESDataset(cfg)


def test_goes_catalog_variables_undercoverage_raises(tmp_path, minimal_config, patch_goes_xr_open):
    """A catalog that only QC'd CMI_C04 cannot safely serve a config that also requests CMI_C07."""
    dt_str = str(pd.Timedelta(minimal_config["timestep"]))
    rows = _catalog_rows_for_range(
        minimal_config["start_datetime"], minimal_config["end_datetime"], minimal_config["timestep"]
    )
    catalog_path = _write_catalog(
        tmp_path,
        "catalog.json",
        mode="local",
        timestep=dt_str,
        product="ABI-L2-MCMIPC",
        goes_position="east",
        extent=None,
        variables={"prognostic": {"vars_2D": [CMI_C04], "vars_3D": []}},
        rows=rows,
        start_datetime=minimal_config["start_datetime"],
        end_datetime=minimal_config["end_datetime"],
    )

    with pytest.raises(ValueError, match="File catalog variables mismatch"):
        GOESDataset(_catalog_config(minimal_config, catalog_path))


def test_goes_catalog_reused_across_field_types(tmp_path, minimal_config, patch_goes_xr_open):
    """The catalog JSON should be parsed once and reused for every active field type."""
    dt_str = str(pd.Timedelta(minimal_config["timestep"]))
    rows = _catalog_rows_for_range(
        minimal_config["start_datetime"], minimal_config["end_datetime"], minimal_config["timestep"]
    )
    variables_meta = {
        "prognostic": {"vars_2D": [CMI_C04, CMI_C07], "vars_3D": []},
        "diagnostic": {"vars_2D": [CMI_C04], "vars_3D": []},
    }
    catalog_path = _write_catalog(
        tmp_path,
        "catalog.json",
        mode="local",
        timestep=dt_str,
        product="ABI-L2-MCMIPC",
        goes_position="east",
        extent=None,
        variables=variables_meta,
        rows=rows,
        start_datetime=minimal_config["start_datetime"],
        end_datetime=minimal_config["end_datetime"],
    )

    cfg = _catalog_config(minimal_config, catalog_path)
    cfg["source"]["TEST_GOES"]["variables"] = {
        "prognostic": {"vars_2D": [CMI_C04, CMI_C07]},
        "diagnostic": {"vars_2D": [CMI_C04]},
        "dynamic_forcing": None,
    }

    ds = GOESDataset(cfg, return_target=True)
    sample = ds[(ds.datetimes[0], 0)]
    assert f"TEST_GOES/prognostic/2d/{CMI_C04}" in sample["input"]
    assert f"TEST_GOES/diagnostic/2d/{CMI_C04}" in sample["target"]


def test_load_file_catalog_no_matching_files_raises():
    """A file_catalog_path glob matching no files raises FileNotFoundError."""
    ds = GOESDataset.__new__(GOESDataset)
    ds.file_catalog_path = "/nonexistent/dir/catalog_*.json"
    ds._catalog_cache = None

    with pytest.raises(FileNotFoundError, match="No catalog files found matching"):
        ds._load_file_catalog()


# ---------------------------------------------------------------------------
# Tests — remote mode (s3fs connection reuse)
# ---------------------------------------------------------------------------


class _FakeS3FileSystem:
    """Stand-in for s3fs.S3FileSystem: records calls instead of touching the network."""

    def __init__(self, ls_result=None):
        self.ls_result = [] if ls_result is None else ls_result
        self.ls_calls = []
        self.open_calls = []

    def ls(self, path):
        self.ls_calls.append(path)
        return self.ls_result

    def open(self, path, mode):
        self.open_calls.append(path)
        return object()  # opaque byte-stream stand-in


def test_goes_remote_scan_reuses_cached_fs(monkeypatch):
    """_collect_GOES_file_path must reuse self._fs (lazily built via _start_s3_fs) rather
    than opening a fresh S3FileSystem connection on every scan."""
    fake_fs = _FakeS3FileSystem()
    start_fs_calls = []

    def fake_start_s3_fs():
        start_fs_calls.append(1)
        return fake_fs

    monkeypatch.setattr("credit.datasets.gen_2.goes._start_s3_fs", fake_start_s3_fs)

    ds = GOESDataset.__new__(GOESDataset)
    ds._fs = None
    ds.mode = "remote"
    ds.goes_position = "east"
    ds.product = "ABI-L2-MCMIPC"
    ds.tolerance = pd.Timedelta("3 minutes")
    ds.datetimes = pd.date_range("2021-06-01", "2021-06-01 06:00", freq="6h")
    ds.num_forecast_steps = 1
    ds.file_catalog_path = None

    with pytest.raises(FileNotFoundError):
        ds._collect_GOES_file_path()

    assert len(start_fs_calls) == 1
    assert ds._fs is fake_fs

    with pytest.raises(FileNotFoundError):
        ds._collect_GOES_file_path()

    assert len(start_fs_calls) == 1  # cached fs reused, no second connection
    assert ds._fs is fake_fs


def test_goes_remote_mode_full_getitem(minimal_config, monkeypatch, goes_xr_dataset, latlon_xr_dataset):
    """End-to-end remote-mode sample extraction, through _get_file_source and
    _load_remote_var, using a shared cached self._fs (see _start_s3_fs usage)."""
    cfg = {
        **minimal_config,
        "source": {"TEST_GOES": {**minimal_config["source"]["TEST_GOES"], "mode": "remote"}},
    }

    def fake_collect(self, base_dir=""):
        extended = pd.date_range(self.datetimes[0], self.datetimes[-1] + self.dt, freq=self.dt)
        return [(t, t + self.dt - pd.Timedelta("1ns"), "s3://fake/goes16_fake.nc") for t in extended]

    monkeypatch.setattr(GOESDataset, "_collect_GOES_file_path", fake_collect)

    fake_fs = _FakeS3FileSystem()
    monkeypatch.setattr("credit.datasets.gen_2.goes._start_s3_fs", lambda: fake_fs)

    def fake_open_dataset(path_or_obj, **kwargs):
        if isinstance(path_or_obj, str) and "lat_lon" in path_or_obj:
            return latlon_xr_dataset
        return goes_xr_dataset

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    ds = GOESDataset(cfg)
    sample = ds[(ds.datetimes[0], 0)]

    assert f"TEST_GOES/prognostic/2d/{CMI_C04}" in sample["input"]
    assert fake_fs.open_calls  # confirms _load_remote_var actually used the cached fs


# ---------------------------------------------------------------------------
# Tests — _collect_GOES_file_path direct unit tests (satellite transitions & scan edge cases)
# ---------------------------------------------------------------------------


def _goes_filename(product, sat_id, start_ts, mode="M6"):
    def ts_str(prefix, ts):
        return f"{prefix}{ts.strftime('%Y%j%H%M%S')}0"

    start_str = ts_str("s", start_ts)
    end_str = ts_str("e", start_ts + pd.Timedelta(minutes=9))
    creation_str = ts_str("c", start_ts + pd.Timedelta(minutes=9, seconds=30))
    return f"OR_{product}-{mode}_{sat_id}_{start_str}_{end_str}_{creation_str}.nc"


def _bare_goes_dataset(**overrides):
    """A minimally-populated GOESDataset for unit-testing a single method in isolation."""
    ds = GOESDataset.__new__(GOESDataset)
    ds.mode = "local"
    ds.goes_position = "east"
    ds.product = "ABI-L2-MCMIPC"
    ds.tolerance = pd.Timedelta("3 minutes")
    ds.datetimes = pd.date_range("2021-06-01 00:00", "2021-06-01 02:00", freq="1h")
    ds.num_forecast_steps = 0
    ds.file_catalog_path = None
    ds._fs = None
    for key, value in overrides.items():
        setattr(ds, key, value)
    return ds


def test_goes_satellite_transition_hour_skipped(monkeypatch, caplog):
    """The ambiguous GOES-16/19 transition hour (2025-04-07 15:00-16:00 UTC) must never be
    scanned against either satellite; that hour's timestamp ends up flagged as missing instead."""
    ds = _bare_goes_dataset(datetimes=pd.date_range("2025-04-07 14:00", "2025-04-07 16:00", freq="1h"))

    def fake_listdir(hourly_dir):
        if hourly_dir.endswith("/15"):
            raise AssertionError(f"transition-hour directory must never be scanned: {hourly_dir}")
        if hourly_dir.endswith("/14"):
            return [_goes_filename(ds.product, "G16", pd.Timestamp("2025-04-07 14:00:00"))]
        if hourly_dir.endswith("/16"):
            return [_goes_filename(ds.product, "G19", pd.Timestamp("2025-04-07 16:00:00"))]
        raise FileNotFoundError(hourly_dir)

    monkeypatch.setattr(os, "listdir", fake_listdir)

    with caplog.at_level("WARNING"):
        result = ds._collect_GOES_file_path(base_dir="/fake/base")

    assert any("GOES-16/19 transition" in rec.message for rec in caplog.records)

    by_start = {start: path for start, end, path in result}
    assert by_start[pd.Timestamp("2025-04-07 15:00:00")] == _FILE_NOT_FOUND
    assert by_start[pd.Timestamp("2025-04-07 14:00:00")] != _FILE_NOT_FOUND
    assert by_start[pd.Timestamp("2025-04-07 16:00:00")] != _FILE_NOT_FOUND


def test_goes_satellite_transition_hour_skipped_west(monkeypatch, caplog):
    """Same guarantee as the east test above, for the GOES-17/18 (west) transition."""
    ds = _bare_goes_dataset(
        goes_position="west",
        datetimes=pd.date_range("2023-01-04 17:00", "2023-01-04 19:00", freq="1h"),
    )

    def fake_listdir(hourly_dir):
        if hourly_dir.endswith("/18"):
            raise AssertionError(f"transition-hour directory must never be scanned: {hourly_dir}")
        if hourly_dir.endswith("/17"):
            return [_goes_filename(ds.product, "G17", pd.Timestamp("2023-01-04 17:00:00"))]
        if hourly_dir.endswith("/19"):
            return [_goes_filename(ds.product, "G18", pd.Timestamp("2023-01-04 19:00:00"))]
        raise FileNotFoundError(hourly_dir)

    monkeypatch.setattr(os, "listdir", fake_listdir)

    with caplog.at_level("WARNING"):
        result = ds._collect_GOES_file_path()

    assert any("GOES-17/18 transition" in rec.message for rec in caplog.records)

    by_start = {start: path for start, end, path in result}
    assert by_start[pd.Timestamp("2023-01-04 18:00:00")] == _FILE_NOT_FOUND
    assert by_start[pd.Timestamp("2023-01-04 17:00:00")] != _FILE_NOT_FOUND
    assert by_start[pd.Timestamp("2023-01-04 19:00:00")] != _FILE_NOT_FOUND


def test_goes_collect_file_path_falls_back_when_catalog_glob_empty(monkeypatch, caplog):
    """A file_catalog_path glob that matches nothing must fall back to a live directory scan."""
    ds = _bare_goes_dataset(file_catalog_path="/nonexistent/catalog_*.json")

    listdir_calls = []

    def fake_listdir(hourly_dir):
        listdir_calls.append(hourly_dir)
        dt = ds.datetimes[len(listdir_calls) - 1]
        return [_goes_filename(ds.product, "G16", dt)]

    monkeypatch.setattr(os, "listdir", fake_listdir)

    with caplog.at_level("WARNING"):
        result = ds._collect_GOES_file_path()

    assert any("falling back to directory scan" in rec.message for rec in caplog.records)
    assert len(result) == len(ds.datetimes)
    assert all(path != _FILE_NOT_FOUND for _, _, path in result)


def test_goes_collect_file_path_malformed_filename_raises(monkeypatch):
    """A malformed GOES filename (too few underscore-separated tokens) raises ValueError."""
    ds = _bare_goes_dataset()
    monkeypatch.setattr(os, "listdir", lambda hourly_dir: ["not_a_valid_goes_filename.nc"])

    with pytest.raises(ValueError, match="Unexpected GOES L2 filename structure"):
        ds._collect_GOES_file_path()


def test_goes_collect_file_path_logs_warning_for_missing_hour_dir(monkeypatch, caplog):
    """A hard directory-listing failure for one hour should degrade to MISSING for that
    hour rather than aborting the whole scan, as long as some other hour has data."""
    ds = _bare_goes_dataset()

    def fake_listdir(hourly_dir):
        if hourly_dir.endswith("/00"):
            return [_goes_filename(ds.product, "G16", ds.datetimes[0])]
        raise FileNotFoundError(hourly_dir)

    monkeypatch.setattr(os, "listdir", fake_listdir)

    with caplog.at_level("WARNING"):
        result = ds._collect_GOES_file_path()

    assert any("No data at" in rec.message for rec in caplog.records)
    by_start = {start: path for start, end, path in result}
    assert by_start[ds.datetimes[1]] == _FILE_NOT_FOUND


# ---------------------------------------------------------------------------
# Tests — _extract_field / _load_local_var / _load_remote_var edge cases
# ---------------------------------------------------------------------------


def test_extract_field_noop_for_disabled_field_type(minimal_config, patch_goes_io):
    """_extract_field is a no-op for a field type that was never registered."""
    ds = GOESDataset(minimal_config)
    sample = {}
    ds._extract_field("dynamic_forcing", ds.datetimes[0], sample)  # disabled in minimal_config
    assert sample == {}


def test_extract_field_noop_for_empty_vars_2d():
    """_extract_field is a no-op for a registered field type with no 2D variables."""
    ds = GOESDataset.__new__(GOESDataset)
    ds.var_dict = {"static": {"vars_2D": [], "vars_3D": ["placeholder"]}}
    sample = {}
    ds._extract_field("static", pd.Timestamp("2021-06-01"), sample)
    assert sample == {}


def test_load_local_var_missing_field_type_raises():
    """_load_local_var raises KeyError when no files are registered for the field type."""
    ds = GOESDataset.__new__(GOESDataset)
    ds.file_dict = {}
    with pytest.raises(KeyError, match="No files registered"):
        ds._load_local_var("prognostic", [CMI_C04], pd.Timestamp("2021-06-01"))


def test_load_remote_var_missing_field_type_raises():
    """_load_remote_var raises KeyError when no files are registered for the field type."""
    ds = GOESDataset.__new__(GOESDataset)
    ds.file_dict = {}
    with pytest.raises(KeyError, match="No files registered"):
        ds._load_remote_var("prognostic", [CMI_C04], pd.Timestamp("2021-06-01"))
