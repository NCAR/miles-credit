# Preparing your dataset for CREDIT models

## Overall steps

Customized data can be prepared through the following steps:
* Pack your variables into yearly files using `netCDF` or `zarr` data formats.
* Prepare mean and std files for z-score normalization.
* (Optional) Prepare your static and periodic forcing variables.
  
For example, you can have:
```
train_data_2000.zarr, train_data_2001.zarr, …, train_data_2024.zarr
```

Files need to be referenced in YAML config file using regular expressions:
```
data:
  # upper-air variables
  variables: ['your_varnames',]
  save_loc: '/your_path/train_data_*.zarr'
```

## Supported variable types

The following variable types are supported:

* **Upper-air variables**: variables in **yearly files** with coordinates of `(time, level, latitude, longitude)` and as both inputs and outputs. e.g., air temperature.
* **Surface variables**: variables in **yearly files** with coordinates of `(time, latitude, longitude)` and as both inputs and outputs. e.g., surface pressure.
* **Dynamic forcing variables**: variables in **yearly files** with coordinates of `(time, latitude, longitude)` and as **input only**. e.g., solar radiation.
* **Diagnostic variables**: variables in **yearly files** with coordinates of `(time, latitude, longitude)` and as **output only**. e.g., precipitation.
* **Periodic forcing variables**: variables in **a single file** with coordinates of `(time, latitude, longitude)` with the `time` coordinate covering 366 days of a year. This variable type will be used repeatedly as annual cycles and for **inputs only**. e.g., periodic sea surface temperature.
* **Static variables**: variables **a single file** with coordinates of `(latitude, longitude)` and as input only. e.g., 'terrain elevation'.

CREDIT uses **periodic forcing variables** and **static variables** as they are, the user should normalize them. 

All variable information needs to be added in the `data` section of YAML config file:

```
data
    # upper-air variables
    variables: ['your_varnames',]
    save_loc: '/your_path/train_data_*.zarr'
    
    # surface variables
    surface_variables: ['your_varnames',]
    save_loc_surface: '/your_path/train_data_*.zarr'
  
    # dynamic forcing variables
    dynamic_forcing_variables: ['your_varnames',]
    save_loc_dynamic_forcing: '/your_path/train_data_*.zarr'
  
    # diagnostic variables
    diagnostic_variables: ['your_varnames',]
    save_loc_diagnostic: '/your_path/train_data_*.zarr'
    
    # periodic forcing variables
    forcing_variables: ['your_varnames',]
    save_loc_forcing: '/your_path/forcing_*.zarr'
    
    # static variables
    static_variables: ['your_varnames',]
    save_loc_static: '/your_path/static_*.zarr'
```
## z-score files

The mean and standard deviation of **Upper-air variables**, **Surface variables**, **Dynamic forcing variables**, **Diagnostic variables** should be prepared and listed in the YAML config file:

```
data
    mean_path: '/your_path/mean_file.nc'
    std_path: '/your_path/std_file.nc'
```

Variable types of **periodic forcing variables** and **static variables** do not have mean and std entries.

## Mandatory steps
* The customized dataset must have coordinate names and orders of `time`, `level`, `latitude`, and `longitude`.
* Z-score files must have the coordinate name `level`.
* Upper-air variables are mandatory to run CREDIT models. Other variable types are optional.
* All prepared data should not have NaN values.
* All listed variable names must exist in the file.
* The CREDIT data loader supports yearly files only.

## Recommended steps
* Chunking can speed up the data loading process.
* It is recommended to put all **Upper-air variables**, **Surface variables**, **Dynamic forcing variables**, **Diagnostic variables** in one file for each year, but CREDIT can handle separated folders, file names per variable type.

## Test the validity of your data preparation
Once the data is prepared and registered in the YAML config file. You can run the following code for a quick test:
```python
from credit.parser import credit_main_parser, training_data_check

# load your config
config_name = 'your_path/model.yml'
with open(config_name, 'r') as stream:
    conf = yaml.safe_load(stream)

conf = credit_main_parser(conf, parse_training=True, parse_predict=True, print_summary=True)

training_data_check(conf, print_summary=True)

predict_data_check(conf, print_summary=True)
```
