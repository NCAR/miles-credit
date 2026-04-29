# Prediction Rollouts

## Prediction Ingredients
Before beginning rollouts of a CREDIT model, you will need the following ingredients/files 
available on your machine.
1. 🌎Initial conditions for upper air and surface variables in Zarr format. If running processed ERA5 
on Derecho or Casper, you can access the processed files at 
`/glade/campaign/cisl/aiml/credit/era5_zarr/`. The `y_TOTAL*.zarr` and `SixHourly_y_TOTAL*.zarr` 
are at 0.28 degree grid spacing, and `SixHourly_y_ONEdeg*.zarr` for 1 degree data.
2. 🌞 Dynamic forcing files covering the full period of prediction. In the current CREDIT models, the 
only dynamic forcing variable is top-of-atmosphere shortwave irradiance. Pre-calculated solar 
irradiance values integrated over 1 and 6 hour periods are available on Derecho/Casper at 
`/glade/campaign/cisl/aiml/credit/credit_solar_nc_6h_0.25deg` and `credit_solar_nc_1h_0.25deg`. You
can calculate top of atmosphere solar irradiance for any grid and integration time period with
`credit_calc_global_solar`. If you plan to issue regular predictions, we recommend
pre-computing solar irradiance values for a given year of inference rather than calculating on the fly.
3. ⛰️ Static forcing files with and without normalization. These forcing files include elements like
terrain height, land-sea mask, and land-use type. Static forcing files for the initial CREDIT models
are currently archived at `/glade/campaign/cisl/aiml/credit/static_scalers/`. `static_norm_old.nc` has normalized
terrain height and land sea mask, while unnormalized values are in `LSM_static_variables_ERA5_zhght.nc`.
The unnormalized values are needed for interpolation to pressure and height levels.
4. Files containing the mean and standard deviation scaling values for each variable. Currently,
CREDIT uses values stored in netCDF files. These are currently stored on Derecho in
`/glade/campaign/cisl/aiml/credit/static_scalers/`. The appropriate files to use are `mean_6h_1979_2018_16lev_0.25deg.nc`
for the mean and `std_residual_6h_1979_2018_16lev_0.25deg.nc` for the combined standard deviation of
each variable and the standard deviation of the temporal residual.

## Realtime Rollouts
The goal of realtime inference is to launch model forecasts from GFS, GEFS, or ERA5 initial conditions.
The `predict` section of your configuration file should contain the following fields:
```yaml
predict:
  mode: none
  realtime:
    forecast_start_time: "2025-04-14 12:00:00" # change to your init date
    forecast_end_time: "2025-04-24 12:00:00" # Should be sometime after init date
    forecast_timestep: "6h" # Needs to contain h for hours and should match 1 or 6 hour model.
  initial_condition_path: "/path/to/gfs_init/" # change 
  static_fields: "/Users/dgagne/data/CREDIT_data/LSM_static_variables_ERA5_zhght.nc" # Static forcing file.
  metadata: '/Users/dgagne/miles-credit/credit/metadata/era5.yaml' # Path to metadata for output
  save_forecast: '/Users/dgagne/data/wxformer_6h_test/' # path to save forecast data
```
If you want to use GFS initial conditions, run `python applications/gfs_init.py -c <config file>`.
It will download fields from a GFS initial condition on model levels, which are archived for the past 10 days
on the NOAA [NOMADS](https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/) server. GDAS Analyses and
GFS initial timesteps on model levels are also available on 
[Google Cloud](https://console.cloud.google.com/marketplace/product/noaa-public/gfs) back to 2021.
The `credit_gfs_init` program regrids the data onto the appropriate CREDIT grid and interpolates in
the vertical from the GFS to selected CREDIT ERA5 hybrid sigma-pressure levels.

:::{important}                                                                          
`credit_gfs_init` requires xesmf, which depends on the [ESMF](https://github.com/esmf-org/esmf) suite 
and cannot be installed from PyPI. The easist way to install xesmf without messing up your CREDIT
environment is to run `conda install -c conda-forge esmf esmpy` then `pip install xesmf` after building
your CREDIT environment first. 
:::

If you want to launch ensemble rollouts, you can use `credit_gefs_init` to convert raw GEFS cube sphere data
to grids for CREDIT models. 

Realtime rollouts are handled by `credit_rollout_realtime`. Update the paths in the 
data section of the config file to point to the GFS initial conditions zarr file. `credit_rollout_realtime`
only outputs one forecast at a time.

## Rollout to netCDF for ERA5 initiated forecasts
`credit rollout` generates forecasts for many initialization times using processed ERA5 data as
initial conditions. It supports deterministic and ensemble rollouts, serial and parallel modes,
single and multi-node execution.

Add the following section to your config file:

```yaml
predict:
    mode: none
    forecasts:
        type: "custom"       # keep it as "custom"
        start_year: 2020     # year of the first initialization (where rollout will start)
        start_month: 1       # month of the first initialization
        start_day: 1         # day of the first initialization
        start_hours: [0, 12] # hour-of-day for each initialization, 0 for 00Z, 12 for 12Z
        duration: 32         # number of days to initialize, starting from the (year, mon, day) above
                             # duration should be divisible by the number of GPUs
                             # (e.g., duration: 384 for 365-day rollout using 32 GPUs)
        days: 10             # forecast lead time as days (1 means 24-hour forecast)
    ensemble_size: 1         # set > 1 to save ensemble members to NetCDF
```

### Running locally

```bash
# Deterministic rollout (reads ensemble_size from config)
credit rollout -c config.yml

# Ensemble rollout — override ensemble_size from the CLI
credit rollout -c config.yml --ensemble-size 50

# Multi-GPU on a single node
credit rollout -c config.yml -m ddp
```

### Submitting PBS jobs

Use `credit submit` to submit rollout jobs to the cluster. The `--rollout` flag switches from
training submission to parallel rollout submission. `--jobs N` splits init times across N
independent PBS jobs (all start at once, no afterok chain).

```bash
# Submit 10 parallel rollout jobs on Casper (deterministic or ensemble — set by config)
credit submit --cluster casper -c config.yml --rollout --jobs 10

# Override ensemble size at submission time
credit submit --cluster casper -c config.yml --rollout --jobs 10 --gpus 1

# Dry run — inspect the PBS scripts before submitting
credit submit --cluster casper -c config.yml --rollout --jobs 10 --dry-run
```

`--jobs` controls how many PBS nodes split the init-time work. `ensemble_size` in the config
(or `--ensemble-size` at the CLI) controls how many ensemble members are run per init time.
These are independent settings.

### Multi-node rollout (MPI)

For MPI-enabled PyTorch installations:

```bash
nodes=( $( cat $PBS_NODEFILE ) )
head_node=${nodes[0]}
head_node_ip=$(ssh $head_node hostname -i | awk '{print $1}')
export NUM_RANKS=32
MASTER_ADDR=$head_node_ip
MASTER_PORT=1234
mpiexec -n $NUM_RANKS -ppn 4 --cpu-bind none python applications/rollout_to_netcdf_v2.py -c config.yml
```
## Interpolation to constant pressure and height above ground levels
Both `credit_rollout_realtime` and `credit_rollout_to_netcdf` support vertical interpolation to constant
pressure and constant height above ground level (AGL) levels from the hybrid sigma-pressure levels
used by most models in CREDIT. To enable interpolation, add the following lines to your config
file in the predict section

```yaml
data:
  level_ids: [10, 30, 40, 50, 60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 136, 137]
predict:
  interp_pressure:
    pressure_levels: [300.0, 500.0, 850.0, 925.0] # in hPa
    height_levels: [100.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0] # in meters
```
More configuration options are listed in `full_state_pressure_interpolation` in `credit/interp.py`
and can be set from the config file in the interp_pressure section. The interpolation routine
interpolates to pressure levels using approximately the same approach as ECMWF, although results
will not be exactly the same due to slight numerical and implementation differences. The routine
also calculates pressure and geopotential on all levels. Mean sea level pressure is also calculated
in this routine. 

## Saving compressed and chunked netCDF files
By default, the rollout scripts will save uncompressed netCDF files. These can grow to be quite
large if you are producing a lot of forecasts and are saving all the fields. Space can be saved
greatly by turning on netCDF compression and setting chunks that align with your preferred access
pattern. Encoding options like the ones below go into the config file. 

```yaml
model:
    # crossformer example
    type: "crossformer"
    frames: 1                         # number of input states (default: 1)
    image_height: &height 640         # number of latitude grids (default: 640)
    image_width: &width 1280          # number of longitude grids (default: 1280)
    levels: &levels 16                # number of upper-air variable levels (default: 15)
    channels: 4                       # upper-air variable channels
predict:
  ua_var_encoding:
    zlib: True # turns on zlib compression.
    complevel: 1 # ranges from 1 to 9. 1 is faster with a lower compression ratio, 9 is slower.
    shuffle: True
    chunksizes: [1, *levels, *height, *width]

  pressure_var_encoding:
    zlib: True
    complevel: 1
    shuffle: True
    chunksizes: [ 1, 4, *height, *width] # second dim should match number of interp pres. levels
    
  height_var_encoding:
    zlib: True
    complevel: 1
    shuffle: True
    chunksizes: [ 1, 8, *height, *width] # second dim should match number of interp height levels

  surface_var_encoding:
    zlib: true
    complevel: 1
    shuffle: True
    chunksizes: [1, *height, *width]
```

---

## Running Rollouts with the v2 Data Schema

If you trained with `trainer.type: era5-gen2`, use the v2 rollout commands.
The same YAML config used for training drives inference — no separate rollout config is needed.

### Batch rollout to NetCDF

`credit rollout` steps the model forward over a set of historical initial conditions
and writes one NetCDF file per forecast:

```bash
credit rollout -c config/wxformer_1dg_6hr_v2.yml
```

To run on multiple GPUs pass `--mode ddp`:

```bash
credit rollout -c config/wxformer_1dg_6hr_v2.yml --mode ddp
```

The `predict` block in your config controls which dates are run and where output goes:

```yaml
predict:
    mode: ddp           # none | ddp
    batch_size: 4       # initial conditions per GPU per batch
    ensemble_size: 1    # > 1 enables ensemble inference (requires ensemble model)
    forecasts:
        type: "custom"
        start_year: 2020
        start_month: 1
        start_day: 1
        start_hours: [0, 12]   # UTC hours to initialise each day
        duration: 1             # forecast length in days
        days: 1                 # number of days to run from start date
    metadata: '/path/to/credit/metadata/era5.yaml'
    save_forecast: '/glade/derecho/scratch/$USER/CREDIT_runs/my_run'
    use_laplace_filter: False
```

Output files land in `save_forecast/`. Filename format is
`<YYYY><MM><DD><HH>Z_<lead_hours>h.nc`.

### Realtime forecast from a single init time

`credit realtime` runs one forecast from a user-specified initialisation time,
writing output as it steps (useful for operational or near-realtime use):

```bash
credit realtime -c config/wxformer_1dg_6hr_v2.yml \
    --init-time 2024-01-15T00 \
    --steps 40
```

`--steps 40` = 40 × 6 h = 10-day forecast. Output lands in `predict.save_forecast`.

To override the output directory:

```bash
credit realtime -c config.yml --init-time 2024-06-01T12 --steps 40 \
    --save-dir /tmp/test_forecast
```

### Quick sanity-check after training

The fastest way to verify a freshly trained model produces sensible output:

```bash
# Plot 2m temperature in physical units (Kelvin) — recommended starting point
credit plot -c config/wxformer_1dg_6hr_v2.yml --field VAR_2T --denorm

# Multiple fields at once
credit plot -c config/wxformer_1dg_6hr_v2.yml --field VAR_2T SP --denorm

# 3D variable: temperature at level index 5 (pressure-level ordering)
credit plot -c config/wxformer_1dg_6hr_v2.yml --field temperature --level 5 --denorm

# Point at a specific checkpoint or date
credit plot -c config/wxformer_1dg_6hr_v2.yml --field VAR_2T \
    --checkpoint /glade/derecho/scratch/$USER/CREDIT_runs/my_run/checkpoint.pt \
    --sample-date 2020-06-15T00 --denorm
```

Each PNG is saved to `<save_loc>/plots/` and shows **truth | prediction | difference**
as a global map.

`--denorm` converts outputs from normalised (σ) units to physical units using the
mean and std files from your config — e.g. Kelvin for temperature, Pascals for surface
pressure. Without `--denorm` the colourbar is in standard-deviation units, which is
useful for diagnosing normalisation issues but harder to interpret at a glance.

**What to look for:**

| Symptom | Likely cause |
|---------|-------------|
| Loss > 100 or NaN | Normalisation broken — check mean/std paths |
| Prediction is uniform (no structure) | Too few epochs or learning rate too high |
| Tiling / grid artefacts in prediction | Normal at early epochs for window-based models; disappears with training |
| Difference panel is smooth and small | Training is going well |

### NCAR data paths

The built-in v2 configs already point to the shared ERA5 archive at
`/glade/campaign/cisl/aiml/ksha/CREDIT_data/` and the shared metadata at
`/glade/u/home/akn7/miles-credit/credit/metadata/era5.yaml`.
No path edits are needed for NCAR users.
