"""make_scaler output, run through the real bridgescaler transform."""

import filecmp
import os

import numpy as np
import pytest
import torch
import xarray as xr

from camulator_gen2.make_scaler import SOURCE, VARIABLES, build_scaler_dict, main
from credit.preblock.scaler import BridgeScalerTransform

L, H, W = 3, 4, 6
PS = f"{SOURCE}/prognostic/2d/PS"


@pytest.fixture
def stats_files(tmp_path):
    rng = np.random.default_rng(0)
    mean, std = {}, {}
    for (_, dim), names in VARIABLES.items():
        for name in names:
            if name == "PS":
                # a distinct value per gridpoint, so any reordering of the grid shows up
                mean[name] = (
                    ("latitude", "longitude"),
                    9.0e4 + np.arange(H * W, dtype=np.float64).reshape(H, W) * 100.0,
                )
                std[name] = (("latitude", "longitude"), 800.0 + np.arange(H * W, dtype=np.float64).reshape(H, W))
            elif dim == "3d":
                mean[name] = (("level",), rng.normal(size=L) * 10.0)
                std[name] = (("level",), rng.uniform(0.5, 5.0, size=L))
            else:
                mean[name] = ((), np.float64(rng.normal() * 10.0))
                std[name] = ((), np.float64(rng.uniform(0.5, 5.0)))
    mean_path, std_path = tmp_path / "mean.nc", tmp_path / "std.nc"
    xr.Dataset(mean).to_netcdf(mean_path)
    xr.Dataset(std).to_netcdf(std_path)
    return str(mean_path), str(std_path)


def _batch(mean_path, std_path):
    """Physical-unit batch in the gen2 layout: {data_type: {source: {key: (B, levels, 1, H, W)}}}."""
    g = torch.Generator().manual_seed(1)
    mean_ds, std_ds = xr.open_dataset(mean_path), xr.open_dataset(std_path)
    batch, expected = {"input": {SOURCE: {}}, "target": {SOURCE: {}}}, {"input": {SOURCE: {}}, "target": {SOURCE: {}}}
    for (field_type, dim), names in VARIABLES.items():
        for name in names:
            key = f"{SOURCE}/{field_type}/{dim}/{name}"
            n_lev = L if dim == "3d" else 1
            m = np.asarray(mean_ds[name].values, dtype=np.float64)
            s = np.asarray(std_ds[name].values, dtype=np.float64)
            m = m.reshape(1, 1, 1, H, W) if name == "PS" else m.reshape(1, -1, 1, 1, 1)
            s = s.reshape(1, 1, 1, H, W) if name == "PS" else s.reshape(1, -1, 1, 1, 1)
            x = torch.from_numpy(m + s * np.random.default_rng(len(key)).normal(size=(2, n_lev, 1, H, W))).float()
            z = (x.double().numpy() - m) / s  # the gen1 normalization, in float64
            sections = []
            if field_type in ("prognostic", "dynamic_forcing"):
                sections.append("input")
            if field_type in ("prognostic", "diagnostic"):
                sections.append("target")
            for section in sections:
                batch[section][SOURCE][key] = x.clone()
                expected[section][SOURCE][key] = z
    return batch, expected


def _transform(scaler_path, method):
    variables = [f"{SOURCE}/{f}/{d}/{n}" for (f, d), names in VARIABLES.items() for n in names]
    return BridgeScalerTransform(scaler_path, variables, method=method, spatial_variables=[PS])


def test_transform_matches_gen1_normalization(stats_files, tmp_path):
    out = tmp_path / "scaler.json"
    main(["--mean", stats_files[0], "--std", stats_files[1], "--out", str(out)])
    batch, expected = _batch(*stats_files)

    scaled = _transform(str(out), "transform")(batch)
    for section in ("input", "target"):
        for key, z in expected[section][SOURCE].items():
            got = scaled[section][SOURCE][key].double().numpy()
            np.testing.assert_allclose(got, z, rtol=1e-5, atol=1e-5, err_msg=f"{section} {key}")

    restored = _transform(str(out), "inverse_transform")(scaled)
    for key, x in batch["target"][SOURCE].items():
        torch.testing.assert_close(restored["target"][SOURCE][key], x, rtol=1e-5, atol=1e-2)


def test_ps_is_per_gridpoint(stats_files):
    scalers = build_scaler_dict(*stats_files)
    ps = scalers["input"][SOURCE][PS]
    assert tuple(ps.mean_x_.shape) == (H * W,)
    assert len(ps.x_columns_) == H * W
    assert tuple(scalers["input"][SOURCE][f"{SOURCE}/prognostic/3d/Qtot"].mean_x_.shape) == (L,)
    assert f"{SOURCE}/diagnostic/2d/PRECT" not in scalers["input"][SOURCE]
    assert f"{SOURCE}/dynamic_forcing/2d/SOLIN" not in scalers["target"][SOURCE]


_REAL = [os.environ.get(k) for k in ("CAMULATOR_MEAN_NC", "CAMULATOR_STD_NC", "CAMULATOR_SCALER_JSON")]


@pytest.mark.skipif(not all(_REAL), reason="set CAMULATOR_MEAN_NC, CAMULATOR_STD_NC and CAMULATOR_SCALER_JSON")
def test_rebuilds_the_released_scaler_byte_for_byte(tmp_path):
    mean_nc, std_nc, reference = _REAL
    out = tmp_path / "scaler.json"
    main(["--mean", mean_nc, "--std", std_nc, "--out", str(out)])
    assert filecmp.cmp(out, reference, shallow=False)
