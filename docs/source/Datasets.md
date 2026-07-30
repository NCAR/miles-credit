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
* `end_datetime`: Ending date of the training sample period in pandas-readable date string format..
* `timestep`: Time spacing between the input and target in pandas time delta format (e.g., '6h' or '1d').
* `history_len`: How many integer time steps each input sample is expected to contain. 
* `forecast_len`: How many integer time steps each prediction sample is expected to contain.
* `temporal_mode`: If included as an option and set to `"persist"`, then data sources with
slower timesteps will be persisted forward until they align with the faster timesteps. Otherwise
all data sources are expected to produce output valid at the same time or a warning will be emitted 
* and the sample will be skipped. 
## Dataset Types
CREDIT supports a wide range of base Dataset types. Each
one is detailed below with an example configuration and links
to more information about the associated dataset where appropriate.
### LocalDataset
*API reference: {py:class}`credit.datasets.gen_2.local.LocalDataset`*

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
*API reference: {py:class}`credit.datasets.gen_2.era5.ARCOERA5Dataset`*

This dataset connects to the 
[Google Cloud Analysis-Ready Cloud-Optimized ERA5](https://github.com/google-research/arco-era5). 
Tables of available analysis-ready variables are listed in the README.
```yaml
data:
  source:
    ARCOERA5:
      dataset_type: arco_era5
      level_coord: level
      levels: []
      variables:
        prognostic: 
          vars_3D: ["temperature", "specific_humitidy", "u_component_of_wind", "v_component of wind"]
          vars_2D: ["surface_pressure"]
```

### WeatherBench2ERA5Dataset
*API reference: {py:class}`credit.datasets.gen_2.era5.WeatherBench2ERA5Dataset`*

This dataset connects to the WeatherBench 2 versions of ERA5. In particular, WeatherBench2 contains copies
of ERA5 at different spatial resolutions, enabling fast testing of emulator configurations.

### TISRDataset
*API reference: {py:class}`credit.datasets.gen_2.tisr.TISRDataset`*

This dataset calculates total integrated top of atmosphere solar irradiance entirely in Pytorch based on lat-lon and 
time information. It follows the design patterns and calculations of the [Graphcast solar radiation module](https://github.com/google-deepmind/weathernext/blob/main/graphcast/solar_radiation.py).

### GOESDataset
*API reference: {py:class}`credit.datasets.gen_2.goes.GOESDataset`*

The GOES dataset supports streaming GOES geostationary satellite data directly from the [AWS archives for GOES 16-19](https://registry.opendata.aws/noaa-goes/). 
The same dataset could be adapted to Himawari as well.

### MRMSDataset
*API reference: {py:class}`credit.datasets.gen_2.mrms.MRMSDataset`*

MRMS is the NOAA Multi-Radar Multi-Sensor radar mosaic over the conterminous US. 
The default dataset points at the [AWS archive](https://registry.opendata.aws/noaa-mrms-pds/).

### HRRRDataset
*API reference: {py:class}`credit.datasets.gen_2.hrrr.HRRRDataset`*

The HRRR Dataset points at the [NOAA High Resolution Rapid Refresh model AWS archive](https://registry.opendata.aws/noaa-hrrr-pds/). HRRR is 
a 3-km WRF run over CONUS with hourly output available on either pressure or model levels. The HRRRDataset
streams directly from the GRIB files by reading the byte-ranges of individual variables.