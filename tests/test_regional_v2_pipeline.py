"""
Tests for the V2 dispatch path of WRF/downscaling regional datasets.
These tests use synthetic zarr fixtures — no real data files needed.
They validate that:
  1. setup_boundary_data_loading builds correct file lists from conf
  2. load_dataset dispatches correctly for wrf_singlestep, wrf_multistep, dscale_singlestep
  3. A __getitem__ call returns the expected Sample_WRF / Sample_dscale keys
  4. Trainer type "standard-wrf" and "multi-step-wrf" load correctly
"""

import numpy as np
import xarray as xr


# ── helpers ──────────────────────────────────────────────────────────────────


def _write_wrf_zarr(path, times, interior_vars, surface_vars, static_vars=None, n_levels=12, ny=16, nx=16):
    """Write a minimal synthetic WRF zarr store."""
    coords = {
        "time": times,
        "level": np.arange(n_levels),
        "south_north": np.arange(ny),
        "west_east": np.arange(nx),
    }
    data_vars = {}
    for v in interior_vars:
        data_vars[v] = xr.DataArray(
            np.random.rand(len(times), n_levels, ny, nx).astype(np.float32),
            dims=["time", "level", "south_north", "west_east"],
        )
    for v in surface_vars:
        data_vars[v] = xr.DataArray(
            np.random.rand(len(times), ny, nx).astype(np.float32),
            dims=["time", "south_north", "west_east"],
        )
    if static_vars:
        for v in static_vars:
            data_vars[v] = xr.DataArray(
                np.random.rand(ny, nx).astype(np.float32),
                dims=["south_north", "west_east"],
            )
    ds = xr.Dataset(data_vars, coords={k: v for k, v in coords.items() if k in ["time"]})
    ds.to_zarr(path, mode="w")
    return path


def _write_stats_nc(path, all_varnames, n_levels=12, ny=16, nx=16):
    """Write synthetic mean/std nc file."""
    data_vars = {}
    for v in all_varnames:
        data_vars[v] = xr.DataArray(
            np.random.rand(n_levels, ny, nx).astype(np.float32),
            dims=["level", "south_north", "west_east"],
        )
    xr.Dataset(data_vars).to_netcdf(path)
    return path


def _make_wrf_conf(tmp_path, dataset_type="wrf_singlestep"):
    """Build a minimal V2 conf dict for WRF dispatch testing."""
    # Interior zarr files (two years)
    interior_vars = ["WRF_U", "WRF_V", "WRF_T"]
    surface_vars = ["WRF_SP", "WRF_T2"]
    static_vars = ["LANDMASK"]
    boundary_upper = ["U", "V", "T"]
    boundary_surface = ["MSL", "VAR_2T"]

    times_2000 = np.array(["2000-01-01", "2000-01-02", "2000-01-03"], dtype="datetime64[ns]")
    times_2001 = np.array(["2001-01-01", "2001-01-02", "2001-01-03"], dtype="datetime64[ns]")

    int_dir = tmp_path / "interior"
    int_dir.mkdir()
    bnd_dir = tmp_path / "boundary"
    bnd_dir.mkdir()

    times_2002 = np.array(["2002-01-01", "2002-01-02", "2002-01-03"], dtype="datetime64[ns]")
    for times, yr in [(times_2000, "2000"), (times_2001, "2001"), (times_2002, "2002")]:
        _write_wrf_zarr(str(int_dir / f"interior_{yr}.zarr"), times, interior_vars, surface_vars, static_vars)
        _write_wrf_zarr(str(bnd_dir / f"boundary_{yr}.zarr"), times, boundary_upper, boundary_surface, n_levels=11)

    mean_path = str(tmp_path / "mean.nc")
    std_path = str(tmp_path / "std.nc")
    all_vars = interior_vars + surface_vars
    _write_stats_nc(mean_path, all_vars)
    _write_stats_nc(std_path, all_vars)

    bnd_mean = str(tmp_path / "bnd_mean.nc")
    bnd_std = str(tmp_path / "bnd_std.nc")
    _write_stats_nc(bnd_mean, boundary_upper + boundary_surface, n_levels=11)
    _write_stats_nc(bnd_std, boundary_upper + boundary_surface, n_levels=11)

    # Write static file WITHOUT a time coordinate so expand_dims("time") works
    static_path = str(int_dir / "static.zarr")
    _ny, _nx = 16, 16
    static_data_vars = {
        v: xr.DataArray(np.random.rand(_ny, _nx).astype(np.float32), dims=["south_north", "west_east"])
        for v in static_vars
    }
    xr.Dataset(static_data_vars).to_zarr(static_path, mode="w")

    conf = {
        "seed": 42,
        "save_loc": str(tmp_path / "run"),
        "data": {
            "dataset_type": dataset_type,
            "scaler_type": "std-wrf",
            "variables": interior_vars,
            "surface_variables": surface_vars,
            "dynamic_forcing_variables": [],
            "forcing_variables": [],
            "static_variables": static_vars,
            "diagnostic_variables": [],
            "save_loc": str(int_dir / "interior_*.zarr"),
            "save_loc_surface": str(int_dir / "interior_*.zarr"),
            "save_loc_static": static_path,
            "history_len": 1,
            "valid_history_len": 1,
            "forecast_len": 0,
            "valid_forecast_len": 0,
            "max_forecast_len": None,
            "skip_periods": None,
            "one_shot": None,
            "train_years": [2000, 2001],  # exclusive end → year 2000 only
            "valid_years": [2001, 2002],  # exclusive end → year 2001 only
            "levels": 12,
            "mean_path": mean_path,
            "std_path": std_path,
            "all_varnames": interior_vars + surface_vars,
            "static_first": True,
            "lead_time_periods": 1,
            "data_clamp": None,
            "sst_forcing": {"activate": False},
            "boundary": {
                "variables": boundary_upper,
                "surface_variables": boundary_surface,
                "save_loc": str(bnd_dir / "boundary_*.zarr"),
                "save_loc_surface": str(bnd_dir / "boundary_*.zarr"),
                "history_len": 1,
                "forecast_len": 0,
                "lead_time_periods": 1,
                "levels": 11,
                "all_varnames": boundary_upper + boundary_surface,
                "mean_path": bnd_mean,
                "std_path": bnd_std,
            },
        },
        "trainer": {
            "mode": "none",
            "train_batch_size": 1,
            "valid_batch_size": 1,
            "thread_workers": 0,
            "valid_thread_workers": 0,
            "prefetch_factor": 2,
            "trainer_type": "standard-wrf",
        },
        "loss": {"training_loss": "mse"},
        "transforms": {},
    }
    return conf


# ── setup_boundary_data_loading ───────────────────────────────────────────────


class TestSetupBoundaryDataLoading:
    def test_returns_train_valid_file_lists(self, tmp_path):
        import sys

        sys.path.insert(0, "/glade/work/schreck/repos/miles-credit-main")
        from credit.datasets import setup_boundary_data_loading

        conf = _make_wrf_conf(tmp_path)
        result = setup_boundary_data_loading(conf)

        assert "train_files" in result
        assert "valid_files" in result
        assert len(result["train_files"]) == 1  # 1 file matching year 2000
        assert len(result["valid_files"]) == 1  # 1 file matching year 2001

    def test_train_valid_split_by_year(self, tmp_path):
        import sys

        sys.path.insert(0, "/glade/work/schreck/repos/miles-credit-main")
        from credit.datasets import setup_boundary_data_loading

        conf = _make_wrf_conf(tmp_path)
        result = setup_boundary_data_loading(conf)
        assert "2000" in result["train_files"][0]
        assert "2001" in result["valid_files"][0]

    def test_surface_files_present(self, tmp_path):
        import sys

        sys.path.insert(0, "/glade/work/schreck/repos/miles-credit-main")
        from credit.datasets import setup_boundary_data_loading

        conf = _make_wrf_conf(tmp_path)
        result = setup_boundary_data_loading(conf)
        assert "train_surface_files" in result
        assert len(result["train_surface_files"]) >= 1

    def test_no_surface_loc_returns_empty(self, tmp_path):
        import sys

        sys.path.insert(0, "/glade/work/schreck/repos/miles-credit-main")
        from credit.datasets import setup_boundary_data_loading

        conf = _make_wrf_conf(tmp_path)
        del conf["data"]["boundary"]["save_loc_surface"]
        result = setup_boundary_data_loading(conf)
        assert result["train_surface_files"] == []
        assert result["valid_surface_files"] == []


# ── load_dataset dispatch ─────────────────────────────────────────────────────


class TestLoadDatasetDispatchWRF:
    def test_wrf_singlestep_dispatch_returns_wrf_dataset(self, tmp_path):
        import sys

        sys.path.insert(0, "/glade/work/schreck/repos/miles-credit-main")
        from credit.datasets.load_dataset_and_dataloader import load_dataset
        from credit.datasets.wrf_singlestep import WRF_Dataset

        conf = _make_wrf_conf(tmp_path, "wrf_singlestep")
        ds = load_dataset(conf, rank=0, world_size=1, is_train=True)
        assert isinstance(ds, WRF_Dataset)

    def test_wrf_singlestep_valid_dispatch(self, tmp_path):
        import sys

        sys.path.insert(0, "/glade/work/schreck/repos/miles-credit-main")
        from credit.datasets.load_dataset_and_dataloader import load_dataset
        from credit.datasets.wrf_singlestep import WRF_Dataset

        conf = _make_wrf_conf(tmp_path, "wrf_singlestep")
        ds = load_dataset(conf, rank=0, world_size=1, is_train=False)
        assert isinstance(ds, WRF_Dataset)

    def test_wrf_multistep_dispatch_returns_wrf_multistep(self, tmp_path):
        import sys

        sys.path.insert(0, "/glade/work/schreck/repos/miles-credit-main")
        from credit.datasets.load_dataset_and_dataloader import load_dataset
        from credit.datasets.wrf_multistep import WRF_MultiStep

        conf = _make_wrf_conf(tmp_path, "wrf_multistep")
        ds = load_dataset(conf, rank=0, world_size=1, is_train=True)
        assert isinstance(ds, WRF_MultiStep)

    def test_dscale_singlestep_dispatch_returns_dscale_dataset(self, tmp_path):
        import sys

        sys.path.insert(0, "/glade/work/schreck/repos/miles-credit-main")
        from credit.datasets.load_dataset_and_dataloader import load_dataset
        from credit.datasets.dscale_singlestep import Dscale_Dataset

        conf = _make_wrf_conf(tmp_path, "dscale_singlestep")
        conf["data"]["level_pick"] = None
        ds = load_dataset(conf, rank=0, world_size=1, is_train=True)
        assert isinstance(ds, Dscale_Dataset)

    def test_dataset_has_correct_interior_varnames(self, tmp_path):
        import sys

        sys.path.insert(0, "/glade/work/schreck/repos/miles-credit-main")
        from credit.datasets.load_dataset_and_dataloader import load_dataset

        conf = _make_wrf_conf(tmp_path, "wrf_singlestep")
        ds = load_dataset(conf, rank=0, world_size=1, is_train=True)
        assert ds.varname_upper_air == ["WRF_U", "WRF_V", "WRF_T"]

    def test_dataset_has_correct_boundary_varnames(self, tmp_path):
        import sys

        sys.path.insert(0, "/glade/work/schreck/repos/miles-credit-main")
        from credit.datasets.load_dataset_and_dataloader import load_dataset

        conf = _make_wrf_conf(tmp_path, "wrf_singlestep")
        ds = load_dataset(conf, rank=0, world_size=1, is_train=True)
        assert ds.varname_upper_air_outside == ["U", "V", "T"]

    def test_dataset_history_len(self, tmp_path):
        import sys

        sys.path.insert(0, "/glade/work/schreck/repos/miles-credit-main")
        from credit.datasets.load_dataset_and_dataloader import load_dataset

        conf = _make_wrf_conf(tmp_path, "wrf_singlestep")
        ds = load_dataset(conf, rank=0, world_size=1, is_train=True)
        assert ds.history_len == 1


# ── __getitem__ produces expected keys ───────────────────────────────────────


class TestWRFDatasetGetItem:
    def test_getitem_returns_wrf_sample_keys(self, tmp_path):
        """Verify that __getitem__ returns a post-transform sample with expected tensor keys."""
        import sys

        sys.path.insert(0, "/glade/work/schreck/repos/miles-credit-main")
        from credit.datasets.load_dataset_and_dataloader import load_dataset

        conf = _make_wrf_conf(tmp_path, "wrf_singlestep")
        ds = load_dataset(conf, rank=0, world_size=1, is_train=True)

        assert len(ds) > 0, "dataset has no samples"
        sample = ds[0]

        # After ToTensorWRF transform, keys are renamed from WRF_input/boundary_input to x/x_boundary
        keys = list(sample.keys()) if hasattr(sample, "keys") else list(sample._fields)
        assert "x" in sample, f"missing 'x' (upper-air input tensor), keys: {keys}"
        assert "x_boundary" in sample, f"missing 'x_boundary' (boundary input tensor), keys: {keys}"

    def test_getitem_interior_and_boundary_have_correct_variables(self, tmp_path):
        """Check variable names via dataset attributes (pre-transform) since transforms convert xarray to tensors."""
        import sys

        sys.path.insert(0, "/glade/work/schreck/repos/miles-credit-main")
        from credit.datasets.wrf_singlestep import WRF_Dataset
        from credit.datasets import setup_boundary_data_loading, setup_data_loading

        conf = _make_wrf_conf(tmp_path, "wrf_singlestep")
        data_config = setup_data_loading(conf)
        boundary_config = setup_boundary_data_loading(conf)
        boundary = conf["data"]["boundary"]

        param_interior = {
            "varname_upper_air": conf["data"]["variables"],
            "varname_surface": conf["data"]["surface_variables"],
            "varname_dyn_forcing": conf["data"].get("dynamic_forcing_variables", []),
            "varname_forcing": conf["data"].get("forcing_variables", []),
            "varname_static": conf["data"].get("static_variables", []),
            "varname_diagnostic": conf["data"].get("diagnostic_variables", []),
            "filenames": data_config["train_files"],
            "filename_surface": data_config["train_surface_files"],
            "filename_dyn_forcing": data_config.get("train_dyn_forcing_files"),
            "filename_forcing": conf["data"].get("save_loc_forcing"),
            "filename_static": conf["data"].get("save_loc_static"),
            "filename_diagnostic": data_config.get("train_diagnostic_files"),
            "history_len": 1,
            "forecast_len": 0,
        }
        param_outside = {
            "varname_upper_air": boundary["variables"],
            "varname_surface": boundary.get("surface_variables", []),
            "filenames": boundary_config["train_files"],
            "filename_surface": boundary_config.get("train_surface_files") or None,
            "history_len": boundary.get("history_len", 1),
            "forecast_len": boundary.get("forecast_len", 0),
            "lead_time_periods": boundary.get("lead_time_periods", 1),
        }
        # No transform so sample values remain xarray datasets
        ds = WRF_Dataset(param_interior=param_interior, param_outside=param_outside, transform=None)

        assert len(ds) > 0
        sample = ds[0]

        interior = sample["WRF_input"]
        boundary_ds = sample["boundary_input"]

        for v in ["WRF_U", "WRF_V", "WRF_T"]:
            assert v in interior.data_vars, f"{v} missing from WRF_input"
        for v in ["U", "V", "T"]:
            assert v in boundary_ds.data_vars, f"{v} missing from boundary_input"


# ── trainer registration ──────────────────────────────────────────────────────


class TestWRFTrainerRegistration:
    def test_standard_wrf_trainer_loads(self):
        import sys

        sys.path.insert(0, "/glade/work/schreck/repos/miles-credit-main")
        from credit.trainers import load_trainer
        from credit.trainers.trainerWRF import Trainer as TrainerWRF

        trainer_cls = load_trainer({"trainer": {"type": "standard-wrf"}})
        assert trainer_cls is TrainerWRF

    def test_multi_step_wrf_trainer_loads(self):
        import sys

        sys.path.insert(0, "/glade/work/schreck/repos/miles-credit-main")
        from credit.trainers import load_trainer
        from credit.trainers.trainerWRF_multi import Trainer as TrainerWRFMulti

        trainer_cls = load_trainer({"trainer": {"type": "multi-step-wrf"}})
        assert trainer_cls is TrainerWRFMulti
