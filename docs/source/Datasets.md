# Datasets User Guide
CREDIT enables users to mix and match data sources via a modular
Dataset API that builds on the PyTorch Dataset class. The goal of
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


## Dataset Types
CREDIT supports a wide range of base Dataset types. Each
one is detailed below with an example configuration and links
to more information about the associated dataset where appropriate.
## LocalDataset
The `LocalDataset` class covers just about any locally-hosted collection
of netCDF or zarr files. 
```yaml
data:
  source:
    ERA5:
      dataset_type: local
      level_coord: level
      
        
```
## ARCOERA5Dataset