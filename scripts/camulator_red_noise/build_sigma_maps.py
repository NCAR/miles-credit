"""Per-cell amplitude maps for the targeted_red_noise postblock, from CAMulator's one-step residual analysis.

sigma(x) = sqrt(w_s) * residual anomaly std(x): the standard deviation of the slow (weeks-long) part of the
one-step error, where w_s is the field's slow share of the error variance (w_s_corrected, pooled over the
globe) and the anomaly std is the per-cell residual spread after removing the bias and the diurnal and
annual cycles. The residual maps are on a 2x-coarsened 96 x 144 grid; they are interpolated (bilinear,
periodic in longitude, held constant over the last half cell to the poles) onto the 192 x 288 model grid in the data's orientation.

T1000 and Qtot1000 stand in for CAM's lowest model level (level 31, ~992.6 hPa).

    python scripts/camulator_red_noise/build_sigma_maps.py --out red_noise_sigma_camulator.nc
"""

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr

RESIDUALS = os.path.expanduser("~/PycharmProjects/miles-credit/research/camulator_residuals/output_lowlev")
STATICS = os.path.expanduser("~/PycharmProjects/miles-credit/climate/assets/statics_b_credit_runs_f32_02.nc")

# output variable -> residual field
FIELDS = {
    "TREFHT": "TREFHT",
    "T_lowest": "T1000",
    "Qtot_lowest": "Qtot1000",
    # diagnostics, for a later diagnostic-only block
    "TS": "TS",
    "LHFLX": "LHFLX",
    "QFLX": "QFLX",
    "SHFLX": "SHFLX",
    "FSUS": "FSUS",
}


def to_model_grid(field: xr.DataArray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    # periodic in longitude: wrap one column on each side before interpolating
    wrapped = xr.concat(
        [
            field.isel(longitude=[-1]).assign_coords(longitude=field.longitude[-1:] - 360.0),
            field,
            field.isel(longitude=[0]).assign_coords(longitude=field.longitude[:1] + 360.0),
        ],
        dim="longitude",
    )
    # the model's pole rows lie just outside the coarse grid's (+-89.53): hold the last row's value
    inside = np.clip(lat, float(field.latitude.min()), float(field.latitude.max()))
    out = wrapped.interp(latitude=inside, longitude=lon, method="linear")
    return out.fillna(0.0).values.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--residuals", default=RESIDUALS)
    p.add_argument("--statics", default=STATICS, help="supplies the model grid (latitude, longitude)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    summary = xr.open_dataset(os.path.join(args.residuals, "residual_summary.nc"))
    table = pd.read_csv(os.path.join(args.residuals, "residual_summary.csv")).set_index("field")
    grid = xr.open_dataset(args.statics)
    lat, lon = grid.latitude.values.astype(np.float64), grid.longitude.values.astype(np.float64)

    out = xr.Dataset(coords={"latitude": lat, "longitude": lon})
    for name, field in FIELDS.items():
        w_s = float(table.loc[field, "w_s_corrected"])
        tau = float(table.loc[field, "tau_days_corrected"])
        std = summary["map_anomaly_std"].sel(field=field)
        out[name] = (("latitude", "longitude"), np.sqrt(w_s) * to_model_grid(std, lat, lon))
        out[name].attrs = {
            "long_name": f"per-step std of the slow part of the one-step {field} error",
            "residual_field": field,
            "w_s_corrected": w_s,
            "tau_days_corrected": tau,
        }
        print(f"{name:12s} <- {field:9s} w_s={w_s:.3f} tau={tau:5.1f} d  sigma median={np.median(out[name]):.4g}")
    out.attrs = {
        "source": os.path.join(args.residuals, "residual_summary.nc"),
        "definition": "sigma = sqrt(w_s_corrected) * map_anomaly_std, interpolated to the model grid",
    }
    out.to_netcdf(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
