"""Build the CAMulator bridgescaler JSON from the gen1 mean/std NetCDF files.

The released CAMulator checkpoints were trained with the statistics in
``mean_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc``
and the matching ``std_...nc`` (shipped with the Hugging Face assets). PS has
per-gridpoint statistics (192 x 288); 3-D variables have one value per level;
the rest have one value. This script writes those values, unchanged, into the
JSON layout ``bridgescaler_transform`` reads, with PS as a per-gridpoint
(``spatial_variables``) entry.

It reproduces ``scaler_spatial_PS.json`` byte for byte. That file stores each
variance as ``float32(std) ** 2``, except PS, whose variance is
``float32(std ** 2)`` with the square taken in float64. The two differ by at
most one float32 unit; :data:`SPATIAL_VARS` keeps the float64 square for PS.

Usage::

    python -m camulator_gen2.make_scaler --mean MEAN.nc --std STD.nc --out scaler.json
"""

import argparse

import numpy as np
import torch
import xarray as xr

SOURCE = "CESM"

#: (field type, dimension) -> variable names, as in the CAMulator gen2 config.
VARIABLES = {
    ("prognostic", "3d"): ["U", "V", "T", "Qtot"],
    ("prognostic", "2d"): ["PS", "TREFHT"],
    ("diagnostic", "2d"): [
        "PRECT",
        "TS",
        "CLDHGH",
        "CLDLOW",
        "CLDMED",
        "TAUX",
        "TAUY",
        "U10",
        "QFLX",
        "FSDS_J",
        "FLDS_J",
        "FSUS",
        "FLUS",
        "SHFLX",
        "LHFLX",
        "FSUTOA",
        "FLUT",
    ],
    ("dynamic_forcing", "2d"): ["SOLIN", "SST", "ICEFRAC", "co2vmr_3d"],
}

#: Variables with per-gridpoint statistics (``spatial_variables`` in the config).
SPATIAL_VARS = ("PS",)

#: ``n_`` recorded in the original file (80 samples x 55296 gridpoints). It only
#: matters if the scaler is refitted; the transform does not use it.
N_SAMPLES = 4423680.0


def _scaler(mean, std, spatial, n_samples):
    from bridgescaler.distributed_tensor import DStandardScalerTensor

    mean64 = np.asarray(mean, dtype=np.float64).reshape(-1)
    std64 = np.asarray(std, dtype=np.float64).reshape(-1)
    if spatial:
        var = (std64**2).astype(np.float32)
    else:
        var = std64.astype(np.float32) ** 2
    sc = DStandardScalerTensor(channels_last=False)
    # numpy booleans, so the JSON encodes them exactly as the original file does
    sc.channels_last = np.False_
    sc.mean_x_ = torch.tensor(mean64.astype(np.float32))
    sc.var_x_ = torch.tensor(var)
    sc.x_columns_ = list(range(mean64.size))
    sc.n_ = torch.tensor(n_samples, dtype=torch.float32)
    sc._fit = np.True_
    return sc


def build_scaler_dict(mean_path, std_path, n_samples=N_SAMPLES):
    """Return ``{"input": {...}, "target": {...}}`` in ``save_scaler_dict`` layout.

    ``input`` holds everything the model reads (prognostic and dynamic forcing),
    ``target`` everything it predicts (prognostic and diagnostic). Keys are
    sorted, as in the original file.
    """
    mean_ds = xr.open_dataset(mean_path)
    std_ds = xr.open_dataset(std_path)
    sections = {"input": {}, "target": {}}
    for (field_type, dim), names in VARIABLES.items():
        for name in names:
            key = f"{SOURCE}/{field_type}/{dim}/{name}"
            sc = _scaler(mean_ds[name].values, std_ds[name].values, name in SPATIAL_VARS, n_samples)
            if field_type in ("prognostic", "dynamic_forcing"):
                sections["input"][key] = sc
            if field_type in ("prognostic", "diagnostic"):
                sections["target"][key] = sc
    mean_ds.close()
    std_ds.close()
    return {section: {SOURCE: dict(sorted(entries.items()))} for section, entries in sections.items()}


def main(argv=None):
    from bridgescaler import save_scaler_dict

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mean", required=True, help="gen1 mean NetCDF")
    parser.add_argument("--std", required=True, help="gen1 std NetCDF")
    parser.add_argument("--out", required=True, help="output scaler JSON")
    args = parser.parse_args(argv)
    save_scaler_dict(build_scaler_dict(args.mean, args.std), args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
