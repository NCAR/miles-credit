"""Raw CESM h1 -> CREDIT zarr conversion (camulator_smallscale.convert), on synthetic CAM-like files."""

import numpy as np
import pytest
import xarray as xr

from camulator_smallscale import convert

CASE = "b.e21.TEST"
T, LEV, LAT, LON = 4, 3, 5, 8
RAW_FIELDS_3D = ["U", "V", "T", "Q", "CLDLIQ", "CLDICE", "RAINQM", "SNOWQM"]
RAW_FIELDS_2D = ["PS", "TREFHT", "TS", "CLDHGH", "CLDLOW", "CLDMED", "CLDTOT", "TAUX", "TAUY", "U10", "FSUTOA", "FLUT",
                 "SOLIN", "ICEFRAC", "Z500", "PSL", "LANDFRAC", "OCNFRAC", "FLDS", "FSDS", "FLNS", "FLNT", "FSNS",
                 "FSNT", "LHFLX", "SHFLX", "QFLX", "PRECT", "SST"]  # fmt: skip


def _raw_day(day, seed=0):
    rng = np.random.default_rng(seed + day)
    time = np.arange(T) * 0.25 + day
    coords = {"time": ("time", time), "lev": np.linspace(3.6, 992.5, LEV), "ilev": np.linspace(2.0, 1000.0, LEV + 1),
              "lat": np.linspace(-90, 90, LAT), "lon": np.arange(LON) * 45.0}  # fmt: skip
    data = {name: (("time", "lev", "lat", "lon"), rng.random((T, LEV, LAT, LON))) for name in RAW_FIELDS_3D}
    data.update({name: (("time", "lat", "lon"), rng.standard_normal((T, LAT, LON))) for name in RAW_FIELDS_2D})
    # SST: zero on "land", some sea-ice values below 273, ordinary values elsewhere.
    sst = 280.0 + 20.0 * rng.random((T, LAT, LON))
    sst[:, 0, :] = 0.0
    sst[:, 1, :4] = 271.5
    data["SST"] = (("time", "lat", "lon"), sst)
    data["co2vmr"] = (("time",), 4.0e-4 + 1e-8 * np.arange(T))
    data["hyam"] = (("lev",), np.linspace(0.0, 0.5, LEV))
    data["hybm"] = (("lev",), np.linspace(0.0, 0.5, LEV))
    data["hyai"] = (("ilev",), np.linspace(0.0, 0.5, LEV + 1))
    data["hybi"] = (("ilev",), np.linspace(0.0, 0.5, LEV + 1))
    data["P0"] = ((), 100000.0)
    ds = xr.Dataset(data, coords=coords, attrs={"case": CASE})
    ds["time"].attrs = {"units": "days since 2015-01-01 00:00:00", "calendar": "noleap", "long_name": "time"}
    ds["U"].attrs = {"units": "m/s", "cell_methods": "time: mean"}
    return ds


def test_recipes():
    raw = _raw_day(0)
    ds = convert.convert_day(raw)
    s = convert.STEP_SECONDS
    np.testing.assert_allclose(ds["PRECT"], s * raw["PRECT"], rtol=1e-6)
    np.testing.assert_allclose(ds["QFLX"], -21.6 * raw["QFLX"], rtol=1e-6)
    np.testing.assert_allclose(ds["LHFLX"], -s * raw["LHFLX"], rtol=1e-6)
    np.testing.assert_allclose(ds["SHFLX"], -s * raw["SHFLX"], rtol=1e-6)
    np.testing.assert_allclose(ds["FLDS_J"], s * raw["FLDS"], rtol=1e-6)
    np.testing.assert_allclose(ds["FSDS_J"], s * raw["FSDS"], rtol=1e-6)
    np.testing.assert_allclose(ds["FLUS"], s * (raw["FLNS"] + raw["FLDS"]), rtol=1e-5, atol=1e-2)
    np.testing.assert_allclose(ds["FSUS"], s * (raw["FSDS"] - raw["FSNS"]), rtol=1e-5, atol=1e-2)
    qtot = raw["Q"] + raw["CLDLIQ"] + raw["CLDICE"] + raw["RAINQM"] + raw["SNOWQM"]
    np.testing.assert_allclose(ds["Qtot"], qtot, rtol=1e-6)
    np.testing.assert_array_equal(ds["U"], raw["U"].astype("float32"))
    # SST: land filled with 283, sea ice clipped at 273, the rest unchanged.
    sst = ds["SST"].values
    assert (sst[:, 0, :] == 283.0).all()
    assert (sst[:, 1, :4] == 273.0).all()
    np.testing.assert_allclose(sst[:, 2:, :], raw["SST"].values[:, 2:, :], rtol=1e-6)
    # co2vmr broadcast over the grid, per time step.
    assert ds["co2vmr_3d"].dims == ("time", "latitude", "longitude")
    np.testing.assert_allclose(ds["co2vmr_3d"].isel(latitude=2, longitude=3), raw["co2vmr"], rtol=1e-6)


def test_layout():
    ds = convert.convert_day(_raw_day(0))
    assert set(convert.DEFAULT_VARIABLES) <= set(ds.data_vars)
    assert ds["U"].dims == ("time", "level", "latitude", "longitude")
    assert ds["PS"].dims == ("time", "latitude", "longitude")
    assert all(ds[name].dtype == np.float32 for name in convert.DEFAULT_VARIABLES)
    assert ds["U"].attrs == {}
    assert ds["time"].attrs == {"units": "days since 2015-01-01 00:00:00", "calendar": "noleap", "long_name": "time"}
    assert "hyai" in ds and "P0" in ds
    assert convert.convert_day(_raw_day(0), dtype="float64")["U"].dtype == np.float64


def test_unknown_variable_is_an_error():
    with pytest.raises(KeyError, match="NOPE"):
        convert.convert_day(_raw_day(0), variables=["NOPE"])


@pytest.fixture
def hist_dir(tmp_path):
    hist = tmp_path / "hist"
    hist.mkdir()
    for day in range(3):
        _raw_day(day).to_netcdf(hist / f"{CASE}.cam.h1.2015-01-{day + 1:02d}-00000.nc")
    _raw_day(400).to_netcdf(hist / f"{CASE}.cam.h1.2016-01-05-00000.nc")  # another year: ignored
    return hist


def test_convert_year_appends_days_one_step_per_chunk(hist_dir, tmp_path):
    out = convert.convert_year(str(hist_dir), CASE, 2015, str(tmp_path / "2015.zarr"))
    zarr = xr.open_zarr(out)
    assert zarr.sizes["time"] == 3 * T
    assert zarr["U"].encoding["chunks"] == (1, LEV, LAT, LON)
    assert zarr["PS"].encoding["chunks"] == (1, LAT, LON)
    assert zarr["time"].values[0].calendar == "noleap"
    expected = np.concatenate([_raw_day(day)["TREFHT"].values for day in range(3)]).astype(np.float32)
    np.testing.assert_array_equal(zarr["TREFHT"].values, expected)
    assert convert.convert_year(str(hist_dir), CASE, 2015, str(tmp_path / "2.zarr"), max_days=1)
    assert xr.open_zarr(tmp_path / "2.zarr").sizes["time"] == T
    with pytest.raises(FileNotFoundError):
        convert.convert_year(str(hist_dir), CASE, 2030, str(tmp_path / "none.zarr"))


def test_verify(hist_dir, tmp_path):
    out = convert.convert_year(str(hist_dir), CASE, 2015, str(tmp_path / "a.zarr"))
    assert convert.verify(out, out) == []
    reference = xr.open_zarr(out).load()
    reference["PRECT"] = reference["PRECT"] * 1.01
    reference.to_zarr(tmp_path / "ref.zarr")
    assert convert.verify(out, str(tmp_path / "ref.zarr")) == ["PRECT"]


def test_cli_forcing_only(hist_dir, tmp_path):
    convert.main(["--hist-dir", str(hist_dir), "--case", CASE, "--years", "2015", "2015", "--out-dir", str(tmp_path),
                  "--forcing-only", "--max-days", "2"])  # fmt: skip
    zarr = xr.open_zarr(tmp_path / f"{CASE}_2015_forcing.zarr")
    assert set(zarr.data_vars) == set(convert.FORCING_VARIABLES)
    assert zarr.sizes["time"] == 2 * T
