# WeatherBench Evaluation

CREDIT includes a WeatherBench2-style deterministic evaluation pipeline for comparing
forecast skill against ERA5 and published reference models.

---

## Quick start

### 1 — Run the forecast

Generate NetCDF forecast files using `rollout_to_netcdf.py`. Each init date produces
one file per lead time under a subdirectory named after the init timestamp (e.g.
`2022-01-01T00Z/pred_2022-01-01T00Z_006.nc`, …, `_240.nc`).

### 2 — Compute scores

```bash
python applications/eval_weatherbench.py \
    --netcdf  /path/to/forecasts/ \
    --out     wb2_scores.csv \
    --label   "My Model"
```

This writes a CSV with one row per lead time (6 h, 12 h, …, 240 h).

### 3 — Plot

```bash
python applications/plot_weatherbench.py \
    --scores  wb2_scores.csv \
    --label   "My Model" \
    --out     wb2_figures/
```

---

## eval_weatherbench.py

### Required arguments

| Argument | Description |
|---|---|
| `--netcdf PATH` | Directory of forecast NetCDF files |

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--out FILE` | `wb2_scores.csv` | Output CSV path |
| `--label STR` | `CREDIT` | Model name in output |
| `--cesm GLOB` | GLADE campaign path | cesm\_stage1 zarr glob (truth) |
| `--static FILE` | GLADE campaign path | Static fields NetCDF |
| `--workers N` | `os.cpu_count()` | Parallel init workers |
| `-v` | — | Verbose logging |

### Output columns

```
lead_time_hours, forecast_step, n_inits,
rmse_{var},        bias_{var},        acc_{var},
rmse_{var}_global, bias_{var}_global, acc_{var}_global,
rmse_{var}_tropics,
rmse_{var}_n_extratropics,
rmse_{var}_s_extratropics,
```

Variables: `T500`, `T700`, `T850`, `U500`, `U850`, `V500`, `V850`,
`Q500`, `Q850`, `Z500`, `t2m`, `SP`.

---

## plot_weatherbench.py

### Required arguments

| Argument | Description |
|---|---|
| `--scores FILE` | CSV produced by `eval_weatherbench.py` |

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--out DIR` | `wb2_figures/` | Output directory |
| `--label STR` | `CREDIT` | Model name in legend |
| `--no-refs` | — | Omit reference model lines |
| `--vars LIST` | all | Comma-separated variables to plot, e.g. `Z500,T850` |
| `--metrics LIST` | all | Comma-separated figure types: `rmse,acc,bias,scorecard,regional` |
| `--no-regional` | — | Skip regional RMSE breakdown figures |
| `--no-scorecard` | — | Skip scorecard heatmap |
| `--no-bias` | — | Skip bias figures |
| `--workers N` | `os.cpu_count()` | Parallel workers for figure generation |
| `-v` | — | Verbose logging |

### Output figures

| File | Content |
|---|---|
| `rmse_{var}.png` | RMSE vs lead time with reference model lines |
| `acc_{var}.png` | Anomaly Correlation Coefficient vs lead time |
| `regional_rmse_{var}.png` | RMSE by latitude band (global, tropics, extratropics) |
| `bias_all.png` | Mean bias (forecast − ERA5) for all variables |
| `scorecard_vs_ifs.png` | Skill score heatmap vs IFS HRES |

### Examples

Plot only Z500 and T850, RMSE and ACC only:

```bash
python applications/plot_weatherbench.py \
    --scores wb2_scores.csv \
    --vars Z500,T850 \
    --metrics rmse,acc \
    --out figures/
```

Skip regional and scorecard figures:

```bash
python applications/plot_weatherbench.py \
    --scores wb2_scores.csv \
    --no-regional --no-scorecard \
    --out figures/
```

---

## What is being computed

### RMSE

Latitude-weighted root-mean-square error:

```
MSE(t) = Σ_lat [ cos(lat) × mean_lon( (f − o)² ) ] / Σ_lat cos(lat)
RMSE   = mean_t √MSE(t)
```

Computed globally and for three latitude bands: tropics (20°S–20°N),
northern extratropics (20–90°N), and southern extratropics (90–20°S).

### ACC (Anomaly Correlation Coefficient)

True anomaly correlation against a 1990–2019 ERA5 climatology:

```
pred_anom = forecast − clim(dayofyear, hour)
true_anom = ERA5_obs  − clim(dayofyear, hour)

ACC = Σ [ cos(lat) × pred_anom × true_anom ]
      ────────────────────────────────────────────────────
      √( Σ[cos(lat)×pred_anom²] × Σ[cos(lat)×true_anom²] )
```

The climatology is indexed by `dayofyear` (1–366) and `hour` (0, 6, 12, 18).
This is **true anomaly ACC**, not Pearson correlation. ACC decreases physically
from ~1.0 at short leads to ~0.5–0.7 at day 7.

---

## Truth and climatology

**Truth**: ERA5 cesm\_stage1, 192×288 Gaussian grid (~1°), 16 hybrid model levels:

```
/glade/campaign/cisl/aiml/ksha/CREDIT_data/
ERA5_mlevel_cesm_stage1/all_in_one/ERA5_mlevel_cesm_6h_lev16_*.zarr
```

- `Z500`, `SP`, `t2m` are used directly from the zarr.
- `T`, `U`, `V`, `Q` at pressure levels are derived by running
  `credit.interp.full_state_pressure_interpolation` on the model-level data,
  identical to the `ensemble_metrics.py` pipeline.

**Climatology**: ERA5 1990–2019, 6-hourly, on the CREDIT 192×288 grid:

```
/glade/campaign/cisl/aiml/ksha/CREDIT_CESM/VERIF/ERA5_clim/
ERA5_clim_1990_2019_6h_cesm_interp.nc
```

---

## Reference model lines

Reference lines on the plots come from pre-computed verification files at:

```
/glade/campaign/cisl/aiml/ksha/CREDIT_arXiv/VERIF/verif_6h/
```

| Reference | Period | Grid | Notes |
|---|---|---|---|
| IFS HRES | 2020 | N320 (~0.28°) | RMSE only (single year, not comparable for ACC) |
| Pangu-Weather | 2020 | N320 | RMSE only |
| GraphCast | 2020 | N320 | RMSE only |
| IFS HRES (paper) | 2019–2022 | N320 | ACC + RMSE, 2192 inits |
| WXFormer v1 (paper) | 2019–2022 | N320 | ACC + RMSE, 2192 inits |
| FuXi (paper) | 2019–2022 | N320 | ACC + RMSE, 2192 inits |

The paper references (2019–2022) use the same ERA5 1990–2019 climatology and the same
true-anomaly ACC formula as CREDIT. The grid resolution differs (N320 ~0.28° vs CREDIT
192×288 ~1°), but this difference is small relative to model-to-model spread.

---

## Producing per-init ACC/RMSE files

`applications/wxformer_wb2_verif.py` produces per-initialisation NetCDF files
in the same format as the pre-computed reference files, enabling direct numerical
comparison.

```bash
python applications/wxformer_wb2_verif.py \
    --forecast /path/to/netcdf/ \
    --out      /path/to/output/
```

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--out DIR` | `$WORK/CREDIT_verif/wxformer_v2` | Output directory |
| `--cesm GLOB` | GLADE campaign path | cesm\_stage1 zarr glob |
| `--clim FILE` | GLADE campaign path | Climatology NetCDF |
| `--static FILE` | GLADE campaign path | Static fields NetCDF |
| `-v` | — | Verbose logging |

### Output files

| File | Dims | Variables |
|---|---|---|
| `ACC_006h_240h_wxformer_v2.nc` | `(days, time=40)` | Z500, T500, U500, V500, Q500, SP, t2m |
| `RMSE_006h_240h_wxformer_v2.nc` | `(days, time=40)` | same + U/V/T/Q at model level 120 |
| `combined_acc_NNNN_MMMM_*.nc` | per-100-init batches | same variables |
| `combined_rmse_NNNN_MMMM_*.nc` | per-100-init batches | same variables |

Both output files have `dayofyear` and `hour` as 2D coordinates on `(days, time)`.

### Submit on Casper

```bash
qsub scripts/casper_wb2_verif.sh
```
