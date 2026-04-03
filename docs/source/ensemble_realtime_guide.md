# CREDIT Ensemble Real-Time Rollout Guide

*For Ryan Sobash and Dave Ahijevic — severe weather ensemble prediction using the 1-degree CREDIT ensemble model.*

---

## What this gives you

A 100-member probabilistic forecast initialized from GFS analysis, running the NCAR CREDIT noisy-ensemble model (crossformer architecture with latent noise injection). Output is pressure-level interpolated NetCDF — CRPS-trained, 6-hourly, up to 15-day lead times.

---

## 1. Setup

### Branch and install

```bash
cd /glade/work/$USER/repos
git clone https://github.com/NCAR/miles-credit.git
cd miles-credit
git checkout main
pip install -e .
```

Or use the existing shared conda environment on Casper/Derecho:

```bash
# Casper
conda activate /glade/work/schreck/conda-envs/credit-main-casper
export PYTHONPATH=/glade/work/schreck/repos/miles-credit-main:$PYTHONPATH
```

### Verify install

```bash
credit_rollout_realtime --help
credit_gfs_init --help
```

---

## 2. Model

The production ensemble model lives at:

```
Config:   /glade/derecho/scratch/schreck/CREDIT_runs/ensemble/scheduler/model.yml
Weights:  /glade/derecho/scratch/schreck/CREDIT_runs/ensemble/scheduler/checkpoint.pt
```

Key model properties:
- **Architecture**: CrossFormer with latent noise injection (`noise_latent_dim: 442`)
- **Grid**: 192 × 288 (≈ 1-degree, CESM-interpolated ERA5)
- **Variables**: U, V, T, Q (16 levels) + SP, t2m, U500, V500, T500, Z500, Q500
- **Timestep**: 6-hourly
- **Ensemble size**: 100 members (inference)
- **Memory**: ~12 GB per GPU; needs 4 GPUs for inference

---

## 3. Real-time initial conditions from GFS

CREDIT needs ERA5-like model-level initial conditions. These are produced from the operational GFS analysis:

```bash
# Step 1: create a realtime config pointing to GFS
# Edit the 'realtime' block in your config (see section 4)

# Step 2: download and process GFS init
credit_gfs_init -c my_realtime_config.yml
```

This downloads the most recent GFS 0.25-degree analysis and interpolates it to the CESM 192×288 model grid.

---

## 4. Config for real-time use

Copy the production config and modify the `predict.realtime` block:

```bash
cp /glade/derecho/scratch/schreck/CREDIT_runs/ensemble/scheduler/model.yml \
   /glade/derecho/scratch/$USER/CREDIT_runs/ensemble_realtime/model.yml
```

Edit these sections in your copy:

```yaml
save_loc: '/glade/derecho/scratch/$USER/CREDIT_runs/ensemble_realtime'

predict:
    mode: none
    batch_size: 1
    ensemble_size: 20        # reduce from 100 for faster real-time use

    realtime:
        forecast_start_time: "2025-01-15 00:00"   # UTC, change to today
        forecast_end_time:   "2025-01-25 00:00"   # 10-day lead
        forecast_timestep: 6                        # hours

    initial_condition_path: '/glade/derecho/scratch/$USER/CREDIT_runs/ensemble_realtime/ic'

    metadata: '/glade/work/schreck/repos/miles-credit-main/credit/metadata/era5.yaml'
    save_forecast: '/glade/derecho/scratch/$USER/CREDIT_runs/ensemble_realtime/output'

    use_laplace_filter: False

    interp_pressure:
        interp_fields: ["U", "V", "T", "Q"]
        q_var: "Q"
        surface_pressure_var: "SP"
```

---

## 5. Run on Casper (interactive or batch)

### Interactive test (2 GPUs, short forecast)

```bash
# Request a GPU node
qsub -I -l select=1:ncpus=8:ngpus=2:mem=128GB:gpu_type=a100_80gb \
     -l walltime=02:00:00 -A NAML0001 -q casper

conda activate /glade/work/schreck/conda-envs/credit-main-casper
export PYTHONPATH=/glade/work/schreck/repos/miles-credit-main:$PYTHONPATH

# Download GFS initial conditions
credit_gfs_init -c /path/to/model.yml

# Run ensemble rollout
credit_rollout_realtime -c /path/to/model.yml
```

### Batch job (Casper, full 100-member ensemble)

```bash
cat > ensemble_realtime.sh << 'EOF'
#!/bin/bash -l
#PBS -N ensemble_realtime
#PBS -l select=1:ncpus=8:ngpus=4:mem=128GB:gpu_type=a100_80gb
#PBS -l walltime=04:00:00
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe

conda activate /glade/work/schreck/conda-envs/credit-main-casper
export PYTHONPATH=/glade/work/schreck/repos/miles-credit-main:$PYTHONPATH

# First: get GFS initial conditions
credit_gfs_init -c /glade/derecho/scratch/$USER/CREDIT_runs/ensemble_realtime/model.yml

# Then: run ensemble
credit_rollout_realtime -c /glade/derecho/scratch/$USER/CREDIT_runs/ensemble_realtime/model.yml
EOF

qsub ensemble_realtime.sh
```

---

## 6. Output

Output NetCDF files are written to `predict.save_forecast`. Each file contains all ensemble members on pressure levels. Variables match standard ERA5 names.

The output is suitable for:
- Direct comparison with GEFS/ECMWF ensemble
- CRPS verification against ERA5 analysis
- Plotting with standard xarray/cartopy tools

---

## 7. Known limitations and TODOs

| Issue | Status |
|-------|--------|
| `credit realtime` unified CLI does not route to V1 ensemble model | **TODO** — use `credit_rollout_realtime` for now |
| Grid is CESM 192×288, not ERA5 181×360 | May need regridding for comparison with other tools |
| GFS init requires network access to NCEP servers | May fail on compute nodes without outbound internet |
| 100-member ensemble on 4 GPUs takes ~90 min for 10-day forecast | Use `ensemble_size: 20` for quick tests |

---

## 8. Questions?

Contact John Schreck (schreck@ucar.edu) or open an issue at [NCAR/miles-credit](https://github.com/NCAR/miles-credit/issues).

The production model was trained with the KCRPS loss (ensemble-aware CRPS kernel) with 10 training members. It generates ensemble spread via latent noise injection at each timestep.
