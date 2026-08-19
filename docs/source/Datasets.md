# Datasets User Guide
CREDIT enables users to mix and match data sources via a modular
Dataset API that builds on the PyTorch Dataset class. All of the classes
described here derive from {py:class}`~credit.datasets.gen_2.base_dataset.BaseDataset`,
and multiple sources are combined by
{py:class}`~credit.datasets.gen_2.multi_source.MultiSourceDataset`. The goal of
a PyTorch Dataset is to load a single instance of data and provide
it as a PyTorch Tensor or collection of tensors. CREDIT Datasets 
follow additional rules around data formatting that enforce ease
of pre-processing and the application of physics constraints.

## CREDIT Data Schema

### Data Sources
CREDIT data pipelines support one or more source datasets that are 
expected to be combined in some form later in the processing pipeline.
Different sources could include the same underlying dataset (e.g., ERA5)
but with different vertical coordinates, such as model levels vs. pressure levels or soil depths.
Multisource datasets can also include Earth system model components or different
observation sources.

### Variable Types
CREDIT supports four types of variables based on whether they are
inputs and/or outputs to a given model. 

* Prognostic: is both input to and output from a model.
* Diagnostic: output only.
* Dynamic Forcing: input only and time-varying.
* Static: input only and fixed.

Each variable type can be either 3D or 2D. All variable tensors are 
assumed to have a shape of `(batch, level, time, y, x)` for data on
rectilinear or curvilinear grids and `(batch, level, time, ncol)` for
data on unstructured grids. 2D variables have a singleton dimension for level
and single time variables have a singleton dimension for time. Keeping
dimensions consistent enables variables to be stacked into one large 
tensor at the end of the preblock pipeline to fit into WXFormer and
other generic models.

### Variable Naming Scheme
Once variables are loaded into memory, they are stored in a Python dictionary
of data types (input, target, or prediction). Within each type is a dict of 
data sources, in which each data source contains a dictionary of tensors.
The variable tensor names follow the convention below with each component separated 
by forward slashes.
```
batch["<data type of input/target/pred>"]["<data source>"]["<data source>/<var type>/<n>d/<var name>"]
```
For example, an ERA5 prognostic 3D temperature field would have the name `"ERA5/prognostic/3d/temperature"`.
This naming convention allows preblocks and postblocks to be applied to 
whole classes of variables with substring matching. It also ensures that variable
names remain unique in situations where something like temperature may be defined
on pressure levels, hybrid levels, and at 2 m above ground level.

### Time Sampling
Additional `data` section configuration options control the time range over which
samples from all data sources are selected. 
* `start_datetime`: Beginning date of the training sample period in pandas-readable date string format.
* `end_datetime`: Ending date of the training sample period in pandas-readable date string format.
* `timestep`: Time spacing between the input and target in pandas time delta format (e.g., '6h' or '1d').
* `history_len`: How many integer time steps each input sample is expected to contain. 
* `forecast_len`: How many sequential autoregressive rollout steps are performed per training
sample. Each individual prediction and target always covers a single time step; with
`forecast_len > 1` the model's output is fed back as input `forecast_len` times, with the loss
applied at each step. Note that gen2 is 1-indexed: `forecast_len: 1` means a single-step
prediction (in contrast to gen1, where `forecast_len: 0` meant a single step).
* `temporal_mode`: A per-source option (set inside a `data.source.<name>` block) controlling how a
source whose native timestep or time range differs from the master clock is aligned. With
`"persist"`, a source with a slower timestep is persisted (held) forward until it aligns with the
faster timesteps. With `"cyclic"`, the source's own time range is treated as one representative
cycle (e.g., a single year of climatological forcing) and every requested timestamp is remapped
onto that cycle, so the source can answer for dates outside its own coverage; `"cyclic"` requires
a `cycle_year` in the source config. If `temporal_mode` is not set, every data source is expected
to produce output valid at exactly the requested times, or a warning is emitted and the sample is
skipped.

## Dataset Types
CREDIT supports a wide range of base Dataset types. Each source is selected by
its `dataset_type` registry key (defined in
`credit/datasets/gen_2/multi_source.py`):

| `dataset_type` | Class | Description |
|---|---|---|
| `local` | `LocalDataset` | Locally hosted netCDF/zarr file collections |
| `arco_era5` | `ARCOERA5Dataset` | Google Cloud Analysis-Ready Cloud-Optimized ERA5 |
| `weatherbench2_era5` | `WeatherBench2ERA5Dataset` | WeatherBench 2 copies of ERA5 at several resolutions |
| `tisr` | `TISRDataset` | Top-of-atmosphere solar irradiance computed on the fly |
| `goes` | `GOESDataset` | GOES geostationary satellite data streamed from AWS |
| `mrms` | `MRMSDataset` | NOAA MRMS radar mosaic from AWS |
| `hrrr`, `hrrr_nat`, `hrrr_subh` | `HRRRDataset` | HRRR from AWS GRIB2 (pressure-level `wrfprs`, native-level `wrfnat`, sub-hourly `wrfsubh`) |
| `gfs` | `GFSDataset` | GFS/GDAS analyses from the public GFS bucket (e.g., initial conditions) |
| `gefs` | `GEFSDataset` | Raw GEFS ensemble cube-sphere initialization files from Google Cloud |
| `base` | `BaseDataset` | Placeholder/testing |

Each type is detailed below with an example configuration and links
to more information about the associated dataset where appropriate.
### LocalDataset
*API reference: {py:class}`credit.datasets.gen_2.local.LocalDataset`* · `dataset_type: local`

The `LocalDataset` class covers just about any locally-hosted collection
of netCDF or zarr files. 
```yaml
data:
  source:
    ERA5:
      dataset_type: local
      level_coord: level
      levels: []
      variables:
        prognostic:
          vars_3D: [ 'u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_total_water' ]
          vars_2D: [ 'SP', 'VAR_2T', 'VAR_10U', 'VAR_10V' ]
          path: '/glade/campaign/cisl/aiml/ksha/CREDIT_data/ERA5_mlevel_1deg/all_in_one/ERA5_mlevel_1deg_6h_subset_%Y*'

        diagnostic:
          vars_2D: [ 'total_precipitation', 'evaporation', 'top_net_thermal_radiation', 'top_net_solar_radiation',
                     'surface_latent_heat_flux', 'surface_net_solar_radiation',
                     'surface_net_thermal_radiation', 'surface_sensible_heat_flux' ]
          path: '/glade/campaign/cisl/aiml/ksha/CREDIT_data/ERA5_mlevel_1deg/all_in_one/ERA5_mlevel_1deg_6h_subset_%Y*'

        dynamic_forcing:
          vars_2D: [ 'toa_incident_solar_radiation', 'land_sea_CI_mask' ]
          path: '/glade/campaign/cisl/aiml/ksha/CREDIT_data/ERA5_mlevel_1deg/all_in_one/ERA5_mlevel_1deg_6h_subset_%Y*'

        static:
          vars_2D: [ 'z_norm', 'land_sea_mask' ]
          path: '/glade/campaign/cisl/aiml/ksha/CREDIT_data/ERA5_mlevel_1deg/static/ERA5_mlevel_1deg_static_subset.zarr'
      
        
```
### ARCOERA5Dataset
*API reference: {py:class}`credit.datasets.gen_2.era5.ARCOERA5Dataset`* · `dataset_type: arco_era5`

This dataset connects to the 
[Google Cloud Analysis-Ready Cloud-Optimized ERA5](https://github.com/google-research/arco-era5). 
Tables of available analysis-ready variables are listed in the README.
```yaml
data:
  source:
    ARCOERA5:
      dataset_type: arco_era5
      level_coord: hybrid
      # List the hybrid model levels explicitly (1-137 for all of them).
      # Careful: `levels: []` selects ZERO levels, not "all levels".
      levels: [50, 70, 90, 110, 120, 130, 137]
      variables:
        prognostic: 
          vars_3D: ["temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind"]
          vars_2D: ["surface_pressure"]
```

### WeatherBench2ERA5Dataset
*API reference: {py:class}`credit.datasets.gen_2.era5.WeatherBench2ERA5Dataset`* · `dataset_type: weatherbench2_era5`

This dataset connects to the WeatherBench 2 versions of ERA5. In particular, WeatherBench2 contains copies
of ERA5 at different spatial resolutions, enabling fast testing of emulator configurations. Data are
streamed from the public Google Cloud store, so no local data are required — see
`config/gen_2/examples/weatherbench2_era5_wxformer_tiny.yml` for a complete laptop-runnable config.

```yaml
data:
  source:
    WBERA5:
      dataset_type: "weatherbench2_era5"
      resolution: "240x121"   # optional; overridden by the resolution kwarg
      level_coord: "level"
      levels: [500, 850]
      variables:
        prognostic:
          vars_3D: ["temperature", "specific_humidity"]
          vars_2D: ["surface_pressure", "2m_temperature"]
        diagnostic:
          vars_2D: ["total_precipitation_6hr"]
        dynamic_forcing:
          vars_2D: ["sea_surface_temperature"]
        static:
          vars_2D: ["land_sea_mask", "geopotential_at_surface"]
  start_datetime: "1979-01-01"
  end_datetime: "2017-12-31 18:00:00"
  timestep: "6h"
  history_len: 1
  forecast_len: 1
```

### TISRDataset
*API reference: {py:class}`credit.datasets.gen_2.tisr.TISRDataset`* · `dataset_type: tisr`

This dataset calculates total integrated top of atmosphere solar irradiance entirely in Pytorch based on lat-lon and 
time information. It follows the design patterns and calculations of the [Graphcast solar radiation module](https://github.com/google-deepmind/weathernext/blob/08cf73625c9d12bd9aaa038868bcb2fe488f2a22/graphcast/solar_radiation.py).
`num_integration_steps` sets how many trapezoidal sub-intervals span the accumulation window (the data `timestep`);
the integration is evaluated in blocks of `integration_chunk_size` sub-intervals (default 90) so peak memory stays
bounded at a few `(integration_chunk_size + 1, ny, nx)` temporaries per DataLoader worker regardless of
`num_integration_steps`, without changing the result.

### GOESDataset
*API reference: {py:class}`credit.datasets.gen_2.goes.GOESDataset`* · `dataset_type: goes`

The GOES dataset supports streaming GOES geostationary satellite data directly from the [AWS archives for GOES 16-19](https://registry.opendata.aws/noaa-goes/). 
The same dataset could be adapted to Himawari as well.

### MRMSDataset
*API reference: {py:class}`credit.datasets.gen_2.mrms.MRMSDataset`* · `dataset_type: mrms`

MRMS is the NOAA Multi-Radar Multi-Sensor radar mosaic over the conterminous US. 
The default dataset points at the [AWS archive](https://registry.opendata.aws/noaa-mrms-pds/).

### HRRRDataset
*API reference: {py:class}`credit.datasets.gen_2.hrrr.HRRRDataset`* · `dataset_type: hrrr` (pressure-level `wrfprs`), `hrrr_nat` (native model levels `wrfnat`), or `hrrr_subh` (sub-hourly `wrfsubh`)

The HRRR Dataset points at the [NOAA High Resolution Rapid Refresh model AWS archive](https://registry.opendata.aws/noaa-hrrr-pds/). HRRR is 
a 3-km WRF run over CONUS with hourly output available on either pressure or model levels. The HRRRDataset
streams directly from the GRIB files by reading the byte-ranges of individual variables.

### GFSDataset
*API reference: {py:class}`credit.datasets.gen_2.gfs.GFSDataset`* · `dataset_type: gfs`

The GFS Dataset loads GFS/GDAS analyses from the public GFS archive, discovering
available initialization times and providing native model-level 3D fields together with
2D surface fields. It is most commonly used to supply real-time or historical initial
conditions to models trained on other datasets (vertical interpolation and other
adjustments are handled by pre/postblocks).

### GEFSDataset
*API reference: {py:class}`credit.datasets.gen_2.gefs.GEFSDataset`* · `dataset_type: gefs`

The GEFS Dataset reads the raw GEFS (Global Ensemble Forecast System) initialization
files from the public `gfs-ensemble-forecast-system` Google Cloud bucket. Each selected
ensemble member contains atmospheric and surface fields on the six cube-sphere tiles,
making it useful for initializing ensemble rollouts.
## Writing Your Own Dataset

To plug in a dataset CREDIT does not ship — a new data source, file layout, or
sampling strategy — subclass
{py:class}`credit.datasets.gen_2.base_dataset.BaseDataset` in your own package
and register it through the config's `custom_objects:` block; no CREDIT source
changes are needed. The [Custom Objects](Custom.md) guide walks through the
full recipe (writing the class, making it importable, declaring it under
`custom_objects:`, and referencing its key as a `dataset_type`).
