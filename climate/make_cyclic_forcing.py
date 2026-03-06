#!/usr/bin/env python3
"""
make_cyclic_forcing.py
~~~~~~~~~~~~~~~~~~~~~~
Build a 1-year climatological forcing file from the 35-year CREDIT
climate forcing file so the coupled run can cycle indefinitely.

Transformations
---------------
SOLIN, SST, ICEFRAC  → climatological annual cycle (mean over all 35 years)
co2vmr_3d            → spatially uniform constant (mean of --co2_year, default 2000)
LANDFRAC, LANDM_COSLAT, PHIS, z_norm  → copied unchanged (time-invariant)

Output time coordinate
----------------------
1460 steps at 6-hourly resolution labelled as year 2000 (noleap calendar).
The server's cesm_to_forcing_ix detects a single-year forcing file and
wraps every model year back to this year automatically.

Usage
-----
    conda activate credit-coupling
    python make_cyclic_forcing.py                      # co2 from year 2000
    python make_cyclic_forcing.py --co2_year 1995      # co2 from year 1995
    python make_cyclic_forcing.py --out /path/to/out.nc
"""

import argparse
import numpy as np
import xarray as xr
import os

# ── Defaults ───────────────────────────────────────────────────────────────
SRC_DEFAULT = (
    "/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/"
    "b.e21.CREDIT_climate_branch_1980_2014.nc"
)
OUT_DEFAULT = (
    "/glade/derecho/scratch/wchapman/CAMULATOR_FORCING/"
    "b.e21.CREDIT_climate_cyclic_1yr.nc"
)
SRC_START_YEAR = 1980   # first year in the source file
NSTEPS_YR      = 1460   # 365 days × 4 steps/day  (noleap)
CLIM_LABEL_YR  = 2000   # year label written into the output time coordinate
CO2_YEAR_DEF   = 2000   # source year used for the CO2 constant


# ── Main ───────────────────────────────────────────────────────────────────
def main(src: str, out: str, co2_year: int) -> None:
    print(f"Source  : {src}")
    print(f"Output  : {out}")
    print(f"CO2 ref : year {co2_year}")
    print()

    # Open source file lazily (chunks keep memory manageable for 35 yr × 1° grid)
    ds = xr.open_dataset(src, chunks={"time": NSTEPS_YR})
    ntot   = ds.sizes["time"]
    nyears = ntot // NSTEPS_YR
    print(f"Source: {ntot} steps  →  {nyears} years of {NSTEPS_YR} steps each")
    assert ntot == nyears * NSTEPS_YR, (
        f"Time dimension {ntot} is not a multiple of {NSTEPS_YR}. "
        "Check calendar / step count."
    )

    lat = ds["latitude"].values   # (192,)
    lon = ds["longitude"].values  # (288,)
    nlat, nlon = len(lat), len(lon)

    # ── 1. Climatological annual cycle for SOLIN, SST, ICEFRAC ────────────
    clim_vars = ["SOLIN", "SST", "ICEFRAC"]
    clim = {}
    for var in clim_vars:
        print(f"  Computing climatology for {var} …", end="", flush=True)
        arr = ds[var].values                                   # (ntot, nlat, nlon)
        arr_r = arr.reshape(nyears, NSTEPS_YR, nlat, nlon)    # (nyears, 1460, nlat, nlon)
        clim[var] = arr_r.mean(axis=0).astype(np.float32)     # (1460, nlat, nlon)
        print(f"  range [{clim[var].min():.4g}, {clim[var].max():.4g}]")

    # ── 2. Fixed CO2 constant ─────────────────────────────────────────────
    co2_offset = (co2_year - SRC_START_YEAR) * NSTEPS_YR
    if co2_offset < 0 or co2_offset + NSTEPS_YR > ntot:
        raise ValueError(
            f"co2_year={co2_year} is outside the source file range "
            f"{SRC_START_YEAR}–{SRC_START_YEAR + nyears - 1}."
        )
    co2_slice = ds["co2vmr_3d"].isel(time=slice(co2_offset, co2_offset + NSTEPS_YR)).values
    co2_const = float(np.nanmean(co2_slice))
    print(f"  co2vmr_3d fixed to {co2_const:.6e}  (spatial+temporal mean of year {co2_year})")
    co2_field = np.full((NSTEPS_YR, nlat, nlon), co2_const, dtype=np.float32)

    # ── 3. Output time coordinate: cftime DatetimeNoLeap objects ─────────
    time_index = xr.cftime_range(
        start=f"{CLIM_LABEL_YR}-01-01",
        periods=NSTEPS_YR,
        freq="6h",
        calendar="noleap",
    )

    # ── 4. Build xarray Dataset and write via to_netcdf ──────────────────
    # Using xarray avoids raw HDF5 API calls that can fail on Lustre
    # (/glade/campaign).  NETCDF3_64BIT_OFFSET is the most portable format
    # on GLADE; use format='NETCDF4' if you prefer compression.
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"\nBuilding xarray Dataset …")

    coords = {
        "time":      time_index,
        "latitude":  lat,
        "longitude": lon,
    }

    data_vars = {}
    for var, data in [("SOLIN", clim["SOLIN"]), ("SST", clim["SST"]),
                      ("ICEFRAC", clim["ICEFRAC"]), ("co2vmr_3d", co2_field)]:
        da = xr.DataArray(
            data,
            dims=["time", "latitude", "longitude"],
            coords=coords,
            attrs=ds[var].attrs,
        )
        data_vars[var] = da

    for var in ["LANDFRAC", "LANDM_COSLAT", "PHIS", "z_norm"]:
        if var in ds:
            da = xr.DataArray(
                ds[var].values,
                dims=["latitude", "longitude"],
                coords={"latitude": lat, "longitude": lon},
                attrs=ds[var].attrs,
            )
            data_vars[var] = da

    ds_out = xr.Dataset(
        data_vars,
        attrs={
            "description": (
                f"1-year climatological forcing for indefinite cycling. "
                f"SOLIN/SST/ICEFRAC = mean of {SRC_START_YEAR}-"
                f"{SRC_START_YEAR + nyears - 1}. "
                f"co2vmr_3d = {co2_const:.4e} (mean of {co2_year}). "
                f"Time labelled as year {CLIM_LABEL_YR}. "
                f"Source: {os.path.basename(src)}"
            ),
            "source": src,
        },
    )

    print(f"Writing {out} …")
    ds_out.to_netcdf(
        out,
        format="NETCDF3_64BIT",
        encoding={v: {"dtype": "float32"} for v in ["SOLIN", "SST", "ICEFRAC",
                                                      "co2vmr_3d", "LANDFRAC",
                                                      "LANDM_COSLAT", "PHIS", "z_norm"]
                  if v in ds_out},
    )

    ds.close()
    sz_mb = os.path.getsize(out) / 1e6
    print(f"Done  →  {out}  ({sz_mb:.0f} MB)")
    print()
    print("Next steps:")
    print(f"  1. Update camulator_config.yml:")
    print(f"       forcing_file: '{out}'")
    print(f"  2. The server's cesm_to_forcing_ix will auto-detect the single-year")
    print(f"     file and wrap every model year back to {CLIM_LABEL_YR}.")


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src",      default=SRC_DEFAULT,
                        help="Source 35-year forcing NC file")
    parser.add_argument("--out",      default=OUT_DEFAULT,
                        help="Output climatological NC file")
    parser.add_argument("--co2_year", type=int, default=CO2_YEAR_DEF,
                        help=f"Year from which to take the CO2 constant (default {CO2_YEAR_DEF})")
    args = parser.parse_args()
    main(args.src, args.out, args.co2_year)
