# Performance Metrics

CREDIT includes scripts for computing and comparing forecast skill across deterministic
and ensemble model configurations.  All output files use a common NetCDF format so they
can be loaded and plotted together.

---

## Output format

Every verification script writes NetCDF files with dims `(days, time=40)`, where:

- `days` — number of initialization dates processed
- `time=40` — 40 lead times: 6 h, 12 h, …, 240 h

2D coordinate arrays `dayofyear` and `hour` are attached so individual days can be
grouped by season or init hour.  Per-100-init batch files are also written for
intermediate checkpointing.

---

## Scripts

### `wxformer_wb2_verif.py` — deterministic ACC + RMSE

Computes per-init ACC and RMSE for a single deterministic CREDIT forecast run.

```bash
python applications/wxformer_wb2_verif.py \
    --forecast /path/to/scheduler/netcdf/ \
    --out      /path/to/output/
```

**Output files**

| File | Variables |
|------|-----------|
| `ACC_006h_240h_{tag}.nc`  | Z500, T500, U500, V500, Q500, SP, t2m |
| `RMSE_006h_240h_{tag}.nc` | same + T/U/V/Q at model level 120 |

**Methodology**

- Truth: ERA5 `cesm_stage1` (192×288 Gaussian grid, 6-hourly)
- Climatology: ERA5 1990–2019 monthly-diurnal mean at same grid
- Spatial average: latitude-weighted (`w = cos(lat) / mean(cos(lat))`)
- ACC: `sp_avg(pred_anom × truth_anom) / sqrt(sp_avg(pred_anom²) × sp_avg(truth_anom²))`
- RMSE: `sqrt(sp_avg((pred - truth)²))`

---

### `ensemble_wb2_verif.py` — ensemble CRPS, spread, RMSE, ACC

Computes per-init CRPS, ensemble spread, ensemble-mean RMSE, and ensemble-mean ACC
for a 100-member CREDIT ensemble.

```bash
python applications/ensemble_wb2_verif.py \
    --forecast /glade/derecho/scratch/schreck/CREDIT_runs/ensemble/scheduler/netcdf_pressure_interp \
    --out      /glade/work/schreck/CREDIT_verif/ensemble \
    --tag      ensemble
```

**Output files**

| File | Content |
|------|---------|
| `CRPS_006h_240h_{tag}.nc`   | spatially-averaged CRPS per init/lead |
| `SPREAD_006h_240h_{tag}.nc` | spatially-averaged ensemble std |
| `RMSE_006h_240h_{tag}.nc`   | ensemble-mean RMSE |
| `ACC_006h_240h_{tag}.nc`    | ensemble-mean ACC |

Variables in all files: Z500, T500, U500, V500, Q500, SP, t2m.

**CRPS formula**

Uses the energy-decomposition identity with the sorted-ensemble trick (O(n log n)):

```
CRPS = E[|X - y|] - 0.5 × E[|X - X'|]

E[|X - X'|] = (2/n²) × Σᵢ (2i - n + 1) × x_(i)
              where x_(i) are the sorted ensemble members (0-indexed)
```

This avoids the O(n²) pairwise loop while remaining exact.

**Spread-error ratio**

Compute offline from the output files:

```python
import xarray as xr, numpy as np
spread = xr.open_dataset("SPREAD_006h_240h_ensemble.nc").mean("days")
rmse   = xr.open_dataset("RMSE_006h_240h_ensemble.nc").mean("days")
ssr    = spread / rmse  # spread-skill ratio; ideal = 1
```

---

### `hres_wb2_verif.py` — IFS HRES reference ACC + RMSE

Computes per-init ACC and RMSE for IFS HRES deterministic forecasts, evaluated at the
CREDIT 1.25° grid for direct comparison.  HRES (0.25°) is regridded to the 192×288
Gaussian grid via `xesmf` bilinear interpolation.

```bash
# All inits in HRES.zarr (2016–2023, 00Z + 12Z)
python applications/hres_wb2_verif.py \
    --out  /glade/work/schreck/CREDIT_verif/hres \
    --tag  hres

# Restrict to 2022 00Z only (matches CREDIT ensemble period)
python applications/hres_wb2_verif.py \
    --year 2022 --init-hour 0 \
    --out  /glade/work/schreck/CREDIT_verif/hres \
    --tag  hres_2022_00z
```

**Output files**

| File | Variables |
|------|-----------|
| `ACC_006h_240h_{tag}.nc`  | Z500, T500, U500, V500, Q500, SP, t2m |
| `RMSE_006h_240h_{tag}.nc` | same |

**Data sources**

- HRES forecasts: `/glade/derecho/scratch/schreck/HRES.zarr`
  (dims `time × prediction_timedelta × level × lat × lon`, 2016–2023, 0.25°, 13 pressure levels)
- Truth: ERA5 `cesm_stage1` (same as CREDIT verification)
- Regridder weight files cached in `--weights-dir` (default `/tmp`)

---

## Comparing models

All four output types share the same `(days, time)` dims and `dayofyear`/`hour`
coordinates, so comparison is straightforward:

```python
import xarray as xr, numpy as np, matplotlib.pyplot as plt

det   = xr.open_dataset("ACC_006h_240h_wxformer_v2.nc").mean("days")
ens   = xr.open_dataset("ACC_006h_240h_ensemble.nc").mean("days")
hres  = xr.open_dataset("ACC_006h_240h_hres_2022_00z.nc").mean("days")

lead_days = np.arange(1, 41) * 6 / 24

plt.plot(lead_days, det["Z500"],  label="CREDIT det")
plt.plot(lead_days, ens["Z500"],  label="CREDIT ens mean")
plt.plot(lead_days, hres["Z500"], label="HRES")
plt.xlabel("Lead time (days)")
plt.ylabel("Z500 ACC")
plt.axhline(0.6, color="k", ls="--", lw=0.8)
plt.legend()
plt.savefig("z500_acc.png", dpi=150)
```

### Seasonal subsetting

```python
# Northern hemisphere winter (DJF)
winter_mask = np.isin(det.dayofyear.isel(time=0).values % 365,
                      list(range(335, 366)) + list(range(1, 61)))
det_djf = det.isel(days=winter_mask).mean("days")
```

---

## Verification module

`credit/verification/ensemble.py` exposes `crps_spatial_avg` for use in custom scripts:

```python
from credit.verification.ensemble import crps_spatial_avg

# pred_ens: (n_members, lat, lon) float64
# truth:    (lat, lon)            float64
# w_lat:    (lat,)  cos(lat)/mean(cos(lat))
crps, spread = crps_spatial_avg(pred_ens, truth, w_lat)
```
