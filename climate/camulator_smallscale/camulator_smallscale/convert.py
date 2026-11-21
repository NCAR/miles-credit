"""Convert raw CESM CAM 6-hourly history (h1, one file per day) into CREDIT per-year zarrs.

The recipes reproduce the CAMulator training data (``b.e21.CREDIT_climate_branch_1980_<year>_zmdata_ERA5scaled_
zmdata_Qtot.zarr``). They were inferred from the raw files of that run and its 1981 zarr, where every model variable
matches exactly (2026-11-21):

- copied unchanged: U, V, T, PS, TREFHT, TS, CLDHGH, CLDLOW, CLDMED, TAUX, TAUY, U10, FSUTOA, FLUT, SOLIN, ICEFRAC
  (and, for analysis, Z500, PSL, CLDTOT, LANDFRAC, OCNFRAC, FLDS, FSDS);
- 6-hour accumulations (the raw fields are 6-hour means, stamped at the end of the window): PRECT [m] = 21600
  PRECT; QFLX = -21.6 QFLX; LHFLX, SHFLX, FLNS, FLNT = -21600 x raw; FSNS, FSNT = 21600 x raw (fluxes in J m-2 per
  step, positive into the surface for the turbulent fluxes);
- FLDS_J = 21600 FLDS, FSDS_J = 21600 FSDS, FLUS = 21600 (FLNS + FLDS), FSUS = 21600 (FSDS - FSNS);
- Qtot = Q + CLDLIQ + CLDICE + RAINQM + SNOWQM;
- SST = max(SST, 273) where the raw SST is positive, 283 elsewhere (land);
- co2vmr_3d = the global co2vmr broadcast over the grid.

Coordinates follow the training zarr: ``time`` (raw values, units and calendar), ``level`` (``lev``), ``ilev``,
``latitude`` (``lat``, south to north) and ``longitude`` (``lon``). Every variable is chunked one time step per
chunk.

Pure xarray/numpy, so it runs in NCAR's ``npl`` environment without CREDIT or torch::

    python -m camulator_smallscale.convert --hist-dir <case>/atm/hist --case <case> --years 2015 2025 --out-dir <dir>
    python -m camulator_smallscale.convert verify --converted <zarr> --reference <zarr>
"""

import argparse
import glob
import os
import sys

import numpy as np
import xarray as xr

STEP_SECONDS = 21600.0

COPY = ["U", "V", "T", "PS", "TREFHT", "TS", "CLDHGH", "CLDLOW", "CLDMED", "TAUX", "TAUY", "U10", "FSUTOA", "FLUT",
        "SOLIN", "ICEFRAC"]  # fmt: skip
ANALYSIS_COPY = ["Z500", "PSL", "CLDTOT", "LANDFRAC", "OCNFRAC", "FLDS", "FSDS"]
SCALE = {
    "PRECT": STEP_SECONDS,
    "QFLX": -STEP_SECONDS / 1000.0,
    "LHFLX": -STEP_SECONDS,
    "SHFLX": -STEP_SECONDS,
    "FLNS": -STEP_SECONDS,
    "FLNT": -STEP_SECONDS,
    "FSNS": STEP_SECONDS,
    "FSNT": STEP_SECONDS,
}
DERIVED = ["FLDS_J", "FSDS_J", "FLUS", "FSUS", "Qtot", "SST", "co2vmr_3d"]

# What CAMulator reads (camulator_hf.yml): prognostic, diagnostic and dynamic forcing.
MODEL_VARIABLES = ["U", "V", "T", "Qtot", "PS", "TREFHT", "PRECT", "TS", "CLDHGH", "CLDLOW", "CLDMED", "TAUX", "TAUY",
                   "U10", "QFLX", "FSDS_J", "FLDS_J", "FSUS", "FLUS", "SHFLX", "LHFLX", "FSUTOA", "FLUT", "SOLIN", "SST",
                   "ICEFRAC", "co2vmr_3d"]  # fmt: skip
DEFAULT_VARIABLES = MODEL_VARIABLES + ["Z500", "PSL", "CLDTOT", "LANDFRAC", "OCNFRAC"]
# The dynamic forcing alone: enough to drive free runs beyond the years converted in full.
FORCING_VARIABLES = ["SOLIN", "SST", "ICEFRAC", "co2vmr_3d"]

LAND_SST = 283.0
MIN_SST = 273.0


def _derived(raw, name):
    if name == "FLDS_J":
        return STEP_SECONDS * raw["FLDS"]
    if name == "FSDS_J":
        return STEP_SECONDS * raw["FSDS"]
    if name == "FLUS":
        return STEP_SECONDS * (raw["FLNS"] + raw["FLDS"])
    if name == "FSUS":
        return STEP_SECONDS * (raw["FSDS"] - raw["FSNS"])
    if name == "Qtot":
        return raw["Q"] + raw["CLDLIQ"] + raw["CLDICE"] + raw["RAINQM"] + raw["SNOWQM"]
    if name == "SST":
        sst = raw["SST"]
        return xr.where(sst > 0, np.maximum(sst, MIN_SST), LAND_SST)
    if name == "co2vmr_3d":
        return raw["co2vmr"] * xr.ones_like(raw["PS"])
    raise KeyError(name)


def convert_day(raw, variables=None, dtype="float32"):
    """One raw h1 dataset (any number of 6-hourly steps) -> CREDIT layout, as ``dtype``."""
    variables = list(variables or DEFAULT_VARIABLES)
    out = {}
    for name in variables:
        if name in COPY or name in ANALYSIS_COPY:
            field = raw[name]
        elif name in SCALE:
            field = SCALE[name] * raw[name]
        elif name in DERIVED:
            field = _derived(raw, name)
        else:
            raise KeyError(f"No recipe for {name!r}")
        field = field.astype(dtype)
        field.attrs = {}
        out[name] = field
    ds = xr.Dataset(out)
    if any("lev" in field.dims for field in out.values()):  # vertical coefficients go with 3-D fields
        for coord in ("hyam", "hybm", "hyai", "hybi", "P0"):
            if coord in raw:
                ds[coord] = raw[coord]
    ds = ds.rename({k: v for k, v in {"lev": "level", "lat": "latitude", "lon": "longitude"}.items() if k in ds.dims})
    for name in ds.data_vars:
        ds[name].attrs = {}
    ds["time"].attrs = {k: v for k, v in raw["time"].attrs.items() if k in ("units", "calendar", "long_name")}
    ds.attrs = {"source": "camulator_smallscale.convert", "case": raw.attrs.get("case", "")}
    return ds


def _encoding(ds):
    enc = {}
    for name, var in ds.data_vars.items():
        if "time" in var.dims:
            enc[name] = {"chunks": tuple(1 if d == "time" else ds.sizes[d] for d in var.dims)}
    # time stays undecoded: its units and calendar travel as attributes.
    return enc


def day_files(hist_dir, case, year):
    """The h1 daily files of ``year``, in date order."""
    return sorted(glob.glob(os.path.join(hist_dir, f"{case}.cam.h1.{year:04d}-*.nc")))


def convert_year(hist_dir, case, year, out_path, variables=None, dtype="float32", max_days=None):
    """Append every day of ``year`` to a new zarr at ``out_path``, one time step per chunk."""
    files = day_files(hist_dir, case, year)[:max_days]
    if not files:
        raise FileNotFoundError(f"No {case}.cam.h1.{year}-*.nc files in {hist_dir}")
    for i, path in enumerate(files):
        with xr.open_dataset(path, decode_times=False) as raw:
            ds = convert_day(raw, variables, dtype).load()
        if i == 0:
            ds.attrs["case"] = ds.attrs.get("case") or case
            ds.to_zarr(out_path, mode="w", encoding=_encoding(ds), consolidated=True)
        else:
            ds.drop_vars([c for c in ("hyam", "hybm", "hyai", "hybi", "P0") if c in ds]).to_zarr(
                out_path, append_dim="time", consolidated=True
            )
    return out_path


def verify(converted, reference, variables=None, rtol=1e-6):
    """Compare a converted zarr with a reference zarr over their common time steps; return failing variables."""
    conv = xr.open_zarr(converted, decode_times=False)
    ref = xr.open_zarr(reference, decode_times=False)
    times = np.intersect1d(conv["time"].values, ref["time"].values)
    if times.size == 0:
        raise ValueError("No common time steps")
    conv, ref = conv.sel(time=times), ref.sel(time=times)
    failed = []
    for name in variables or [v for v in MODEL_VARIABLES if v in ref]:
        a = conv[name].values.astype(np.float64)
        b = ref[name].values.astype(np.float64)
        scale = np.abs(b).max() or 1.0
        err = np.abs(a - b).max()
        ok = err <= rtol * scale
        print(f"{name:10s} max|diff| {err:.3e}  max|ref| {scale:.3e}  {'OK' if ok else 'FAIL'}")
        if not ok:
            failed.append(name)
    print(f"{times.size} common time steps; {len(failed)} variable(s) failed")
    return failed


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["verify"]:
        parser = argparse.ArgumentParser(description="Compare a converted zarr with a reference zarr")
        parser.add_argument("--converted", required=True)
        parser.add_argument("--reference", required=True)
        parser.add_argument("--rtol", type=float, default=1e-6)
        args = parser.parse_args(argv[1:])
        sys.exit(1 if verify(args.converted, args.reference, rtol=args.rtol) else 0)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hist-dir", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--years", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--forcing-only", action="store_true", help="Only the dynamic forcing (FORCING_VARIABLES)")
    parser.add_argument("--max-days", type=int, default=None, help="Convert only the first N days (tests)")
    args = parser.parse_args(argv)
    variables = FORCING_VARIABLES if args.forcing_only else DEFAULT_VARIABLES
    suffix = "forcing" if args.forcing_only else "credit"
    for year in range(args.years[0], args.years[1] + 1):
        out = os.path.join(args.out_dir, f"{args.case}_{year:04d}_{suffix}.zarr")
        path = convert_year(args.hist_dir, args.case, year, out, variables, args.dtype, args.max_days)
        print(path, flush=True)


if __name__ == "__main__":
    main()
