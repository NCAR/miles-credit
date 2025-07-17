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
`applications/calc_global_solar.py`. If you plan to issue regular predictions, we recommend
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
The goal of realtime inference is to launch model forecasts from GFS or ERA5 initial conditions.
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
The `gfs_init.py` program regrids the data onto the appropriate CREDIT grid and interpolates in
the vertical from the GFS to selected CREDIT ERA5 hybrid sigma-pressure levels. 

:::{important}                                                                          
`gfs_init.py` requires xesmf, which depends on the [ESMF](https://github.com/esmf-org/esmf) suite 
and cannot be installed from PyPI. The easist way to install xesmf without messing up your CREDIT
environment is to run `conda install -c conda-forge esmf esmpy` then `pip install xesmf` after building
your CREDIT environment first. 
:::

Realtime rollouts are handled by `applications/rollout_realtime.py`. Update the paths in the 
data section of the config file to point to the GFS initial conditions zarr file. `rollout_realtime.py`
only outputs one forecast at a time.

## Rollout to netCDF for ERA5 initiated forecasts
`rollout_to_netcdf.py` enables you to generate forecasts for many initialization times using 
processed ERA5 data as initial conditions. It can be run either in serial or parallel mode with
both single node and multi node support (MPI-enabled PyTorch required). 

To run `rollout_to_netcdf.py` include the following section in your config file.

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
```

To submit the rollout script as a PBS job, use `python rollout_to_netcdf.py -l 1 -c <config file>`.

To issue predictions on multiple GPUs on a single node:
`torchrun rollout_to_netcdf.py -c <confg file>`

For multi-node rollouts with MPI (MPI-enabled PyTorch required):
```bash
nodes=( $( cat $PBS_NODEFILE ) )
head_node=${nodes[0]}
head_node_ip=$(ssh $head_node hostname -i | awk '{print $1}')
export NUM_RANKS=32
MASTER_ADDR=$head_node_ip
MASTER_PORT=1234
mpiexec -n $NUM_RANKS -ppn 4 --cpu-bind none python rollout_to_netcdf.py -c <config file>
```
## Interpolation to constant pressure and height above ground levels
Both `rollout_realtime.py` and `rollout_to_netcdf.py` support vertical interpolation to constant
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
