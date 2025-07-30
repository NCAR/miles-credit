The DownscalingDataset and associated classes provide an alternative data pipeline that is more flexible and extensible than the ERA5 data pipeline.

To explain how they differ, we first need to explain the two different kinds of modeling that they support and the nature of the data that they're modeling.


# Weather Forecasting vs Downscaling Climate Projections

A weather forecast is a near-term prediction of how the weather is likely to evolve.  A climate projection is a simulation of what the weather could look like over the long term given some assumptions about the Earth system.  Both use the same kind of numerical model, but they are used in significantly different ways.

## Weather Forecasts

For a weather forecast, you initialize the model with the best estimate of the current state of the atmosphere based on observational data.  You then run the model forward iteratively, using the prediction for one timestep as the starting point for the next timestep, and allow the state to evolve based on physics.

Because the atmosphere is a chaotic system, there are limits on how far forward you can predict.  If you run two identical simulations but initialize them with conditions that differ by the smallest possible amount (i.e., add noise at the level of floating-point precision), that different will grow exponentially until the two simulations diverge completely.  Given the uncertainty in our observational system, this means that even if we had a perfect model of the atmosphere, we can't make an accurate and detailed prediction of the weather more than about ten days out.[^limits-of-predictability]

[^limits-of-predictability]: In practice, there are some elements of the earth system, like patterns of ocean currents, that are predictable on longer timescales, so in some cases it is possible to make medium-term predictions about things like whether one summer will be warmer or wetter than average based on conditions observed in the previous winter and spring.  This is called S2S or seasonal to sub-seasonal forecasting.  But we know based on physics that it is impossible -- and always will be -- to make precise and accurate predictions about things like whether or not it's going to rain at a specific time and place six months from now.

Weather forecasts often make use of ensembles, which are a set of simulations initialized with conditions that vary by a tiny amount.  These simulations diverge over time, which gives you a measure of uncertainty associated with your forecast.


## Climate Projections

Climate is the long-term structure of weather.  As some put it, "climate is your face, weather is the expression on your face."  To study climate, you need long stretches of weather data; 30 years or more is generally regarded as an appropriate length.

As mentioned, though, you can't predict weather more than a week or two in advance.  So a very important difference between weather simulations and climate simulations is that climate simulations _are not forecasts_.  If you run a climate a climate simulation for a period in the past, there is no expectation that the sequence of weather events in the simulation will have any correspondence to what actually happened in the real world.  Instead, it's a simulation of events that could have happened under those conditions.  Modelers also sometimes generate ensembles of climate simulations (when feasible), but to get at the spread associated with the variability of the system, rather than uncertainty in initial conditions.

Scientists often refer to long-term simulations of future weather as climate _projections_ to emphasize the fact that they are not _predictions_ of what weather _will_ happen, they're merely examples of weather that _could_ happen, given some assumptions about the future state of the Earth system.

Those assumptions are important; to make a weather forecast, you can use real-world observations of important factors like sea surface temperature and the chemical composition of the atmosphere, since those will be pretty much the same a week or two into the future.  However, if you want to look at conditions decades or centuries into the future, you need to model the entire Earth system, including slower-moving elements like the oceans, cryosphere, and land surface.  Some of those (like ocean circulation) can be simulated, while others (like concentration of greenhouse gases in the atmosphere) must be given as assumptions associated with different possible scenarios for the future.

The models used to project future climate are therefore much more complex than those used to forecast future weather, because they are modeling a lot more pieces of the whole Earth system.  This makes them very computationally expensive, and means they can only run on large supercomputers designed for them.  These computational requirements also limit how much detail climate simulations can feasibly include; the spatial resolution of the global climate model (GCM) simulations in the CMIP6 project is typically a relatively coarse 100-250 km.

## Downscaling

The coarse resolution of GCM simulations omits a lot of valuable detail, particularly in coastal and mountainous regions and for those who are interested in climate impacts.  To generate higher-resolution climate projections, modelers use an approach called dynamical downscaling.

In dynamical downscaling, you use a regional climate model (RCM) to simulate weather over a limited region, using the output from a GCM simulation to provide the boundary conditions for all the parts of the earth system that are not covered by the RCM.  That allows the RCM to run at much higher resolution, both because the grid cells are spread over a smaller area, and because it can omit things like an ocean model when the simulation region is primarily over land.

Some RCMs are purpose-built, but others are weather models (e.g., WRF) that have been repurposed by forcing them with boundary conditions from a GCM. (Though it may require some adaptation to handle simulations that run for decades or centuries into the future, since the code may have assumed that they would only run forecasts a week or two out.)


# Training ML Weather and Climate Models

## Weather and Climate Data

There are several characteristics of weather / climate data that have important ramifications for training ML models to emulate traditional NWP models.

### Gridding

Atmospheric variables are time-varying spatial fields that cover the surface of the Earth.  In reality, these are continuous fields, but in computer simulations, we sample them at discrete points in time and space and represent them as large arrays of numbers.

Temporally, numerical models can have internal timesteps that vary in length (depending on the requirements of the calculations), but the input and output data are pretty much always stored at regular intervals in time.  However, different datasets and different variables may be stored at different frequencies (hourly, 6-hourly, daily, etc.).

Spatially, data is typically stored on a regular grid.[^grids]  However, the Earth is a sphere, not a flat plane, so you have to deal with that in some way when representing the data as an array of numbers.

[^grids]:  Triangular / hexagonal grids, unstructured / flexible meshes, and graph-based networks are currently beyond the scope of this framework.  Some models also internally use spectral representations, where the data is decomposed by wavelength or using spherical harmonics, but the outputs are almost always stored in a gridded array representation, and that's what we use to train ML models.


Many datasets and models use latitude-longitude grids.  The problem with lat-lon grids is that lines of longitude converge as you get closer to the poles, so the size of the gridcells is not constant.  Some model grids vary the spacing of the gridcells with latitude to compensate, but often it will be necessary to apply latitude-based adjustments to things like error metrics and conservation schemes.

For global datasets, the model also needs to deal with the fact that the data wraps around at the edges and that the grid is degenerate at the poles.

Regional models take a different approach: because they only cover a portion of the globe, they use a projected coordinate system that maps points on the surface of the sphere to points on a flat plane.  In these systems, the gridcells are all the same size, but they are not spaced regularly in latitude and longitude; if you plot lines of latitude and longitude in the projected coordinate system, they curve.  It may be necessary to provide information about how latitude varies across the grid to correctly handle Coriolis force and other such effects.

Note that there are an infinite number of possible projected coordinate systems, and they can (and do) vary depending on the dataset / simulation.

Finally, it's important to note that is common for numerical models use staggered grids, wherein there are multiple grids with the same spacing that are offset from one another, such that the cell centers of one grid are aligned with the corners or edge centers of another, and different variables are placed on different grids.  This improves accuracy, prevents unphysical oscillations, and gives a higher effective spatial resolution when it comes to computing gradients.  Usually, scalar variables are put on one grid, and vector variables on one or two others.  The Arakawa-C grid (scalar variables at grid centers, velocity components normal to cell faces) is the most popular, followed by the Arakawa-E grid.

### Dimensionality

Some weather variables are spatially 2-dimensional and some are 3-dimensional.  The 3-D variables take up the most data volume, but the 2-D variables are also important because they provide boundary conditions at the surface, and because they matter to humans, who live at the surface.

The Z-dimension is special because the density of the atmosphere varies with height, and the solid surface is not at a constant elevation.  The z-coordinate of atmospheric data can be based on pressure, height above the surface, height above the geoid, or some combination thereof ("hybrid coordinates").  As with everything else, this can vary by model and by dataset.  Note that data stored on pressure levels will have large swathes of missing data when the pressure at the surface is less than the corresponding pressure coordinate and that level would be underground.

### Static vs Dynamic

Most weather and climate variables change over time, but there are some that don't and are still very important.  In particular, elevation and whether a given gridcell is over land or over water matter a lot.

Static variables don't have a time dimension, which affects how they are handled in code.


### Usage

Finally, it's useful to distinguish between three ways that variables are used in NWP models.

* _Boundary_ variables are input-only: they are informative for predicting how the weather changes from one timestep to another, but the model does not predict future values for them.

* _Diagnostic_ variables are output-only: the model predicts them, but they don't influence the prediction for the next step.  

* _Prognostic_ variables are input-output: these are the state variables that the model is evolving forward in time.  When an ML model is running in autoregressive inference mode, it predicts these variables at time t+1 based on their value at time t+0, and then recycles that prediction as input for predicting time t+2.

When training an ML model, you provide the boundary and prognostic variables at time t+0 as inputs, and prognostic and diagnostic variables at time t+1 as targets.  In inference mode, you provide initial values for the prognostic variables at time t+0, and then you only provide boundary variables at each timestep while the model "rolls out", iteratively predicting each new timestep based on the previous one.


# Data Pipelines: ERA5_* vs DownscalingDataset

The ERA5 pipeline is designed for training a global forecast model.

This model is focused on learning the dynamics of the atmosphere from the ERA5 dataset.  ERA5 is the ECMWF Reanalysis version 5; a reanalysis is our "best estimate" of the historical state of the atmosphere, generated by combining observations with a simulation that is continuously steered to match up with those observations.

To generate a forecast, you provide the model with current conditions and run it forward.  It takes some additional boundary conditions as input (like elevation and incoming solar radiation at the top of the atmosphere), but only a couple of them; the vast majority of its training data is coming from a single dataset, and the pipeline is designed around the characteristics of the ERA5 dataset.

The downscaling pipeline is designed for training a model to emulate dynamical downscaling, in this case the CONUS404 simulation, where WRF is used to downscale the ERA5 reanalysis from 25-km resolution to 4-km resolution over the conterminous United States (CONUS).

Downscaling involves wrangling _two_ datasets: the high-res dataset generated by the RCM, and the low-res GCM dataset used to drive the RCM.  To downscale a GCM simulation, you provide the model with high-res initial conditions, and then also provide it with the low-res GCM data at each timestep.

The point of training an ML model to emulate the CONUS404 downscaling is that you can then provide it with boundary conditions from other datasets -- specifically, GCM simulations projecting future conditions -- to downscale them far more quickly and with much less computational expense than you could by running WRF.

So, the DownscalingDataset pipeline is designed to relax the assumptions of the ERA5-* pipeline that everything is coming from a single dataset.  It allows the user to define an arbitrary number of datasets that can be used as inputs and/or outputs, and that may have different characteristics with regard to temporal coverage, spatial resolution, etc.

It is designed to create training samples that are pulled from multiple datasets that may need different pre-processing by dataset and by variable, while still keeping them aligned in time, and then to apply the same preprocessing and composition to the boundary condition data for rollout, with the expectation that you'll eventually be using different datasets than for training.

# DownscalingDataset How-To

Before we talk about how to use data pipelines, we need to sort out the issue that the word "dataset" can mean a couple different things, depending on context.  The Torch and related ML libraries use the term to refer to a collection of training samples and targets, and provide the abstract class `torch.utils.data.Dataset` as a standard interface for accessing them.  In the world of climate science, the term is used to refer to a collection of data of a particular type or from a particular source, like the ERA5 reanalysis, or the CONUS404 downscaling, or the simulations performed for CMIP6.  (Since datasets in the climate sense are a categorical grouping, you can have datasets within datasets, depending on how broadly or narrowly you define the category.)  There's also the `xarray.Dataset` class, which provides a high-level interface that lets you treat a collection of netCDF or Zarr files as a single object for data analysis.

We will be using the term "dataset" to refer to a collection of data files from a single source whose contents all have the same structure.  If we mean one of the other things, we'll refer to it as a `torch.utils.data.Dataset` or `xarray.Dataset`.

## High-Level Overview of the Data Pipeline

The `DownscalingDataset` class is a `torch.utils.data.Dataset` that can pull corresponding subsets of data from multiple datasets and combine them into a training sample.  It has a collection of `DataMap` objects that provide a list-like interface to a collection of netCDF files.  (The `DataMap` class is similar to an `xarray.Dataset`, but xarray can be very slow on large datasets, so `DataMap` drops a lot of functionality and checking in order to do it faster.)

Each `DataMap` also has an associated `DataTransform` class that applies a sequence of user-defined transformations to the variables as they are read from file.  This approach makes it easy to experiment with normalizing the data in different ways, and lets you apply different transformations to different variables; for example, you could take the 4th root of precipitation before applying a min-max scaling to it, while applying a z-score normalization to temperature.

After instantiating a `DownscalingDataset`, you can ask for sample number 22, and it will figure out which timesteps that corresponds to in each of its constituent datasets, read the data off disk, apply the appropriate transformations, then combine the resulting dictionaries of `numpy.ndarray`s into a `torch.Tensor` you can feed into an ML model.  It also know how to undo all of that to convert a tensor back into dictionaries of numpy arrays, which you can then hand off to an `OutputWrangler` object that will write them out to disk using user-provided netCDF templates.

## Dataset Configuration

The CREDIT framework uses a YAML configuration file with several high-level sections: `data`, `trainer`, `model`, `loss`, `predict`, and `pbs`.  The code determines whether to use the ERA5-style data pipeline or the new downscaling-style pipeline based on whether the `data` section of the YAML has a subsection named `datasets`.

Each dataset is defined with a new sub-subsection of YAML that describes the dimensions of the samples, the start and end dates you want to use, the file paths of the data, and, most importantly, how the variables in each dataset are used in the model, as well as their associated transformations.

See `config/downscaling.yml` for detailed examples of how to define datasets.  Also note that the YAML corresponds to arguments passed to the constructors for `DownscalingDataset`, `DataMap`, and `DataTransform` objects, so see the docstrings of those classes for documentation of the available options.

The most important thing to know is that each dataset in the YAML will have a `variables` section with 4 subsections (`prognostic`, `diagnostic`, `boundary`, and `unused`) that define how the variables in the dataset will be used in training and rollout.  The `unused` section exists primarily as a holding space that makes it easier to change the configuration of the dataset around (so you can just cut-and-paste variables from one section to another), but also, if a variable that is in the template netCDF file is listed in the `unused` section, the `OutputWrangler` will skip it when writing data out, which simplifies the creation of templates.

The second most important thing to know is how to define transforms for each of the variables in the dataset.  You can define a default type of transform that can be overridden for specific variables.  If the transform takes parameters (e.g., a z-score transformation needs a mean and standard deviation), those are typically read in from a file.  Because weather and climate data has significant seasonal and geographic variation, it's important to normalize the data relative to the entire dataset, not just to the sample or the batch.  Reading the normalization parameters in from a file lets you generate the parameters for multiple variables by using a tool like the CDO operators to calculate some statistic across all the files in the dataset.  (Defining them all by hand in the configuration file would quickly become unmanageable.)

# Summary

The downscaling data pipeline is designed to make it easy to change around the datasets that you're using and how you're using the data and variables in them for a wide variety of different ML models.

The `DownscalingDataset` class manages wrangling data from multiple datasets and keeping it all aligned; it know how to read data off disk, apply normalization and other transforms, and stack it all up into a tensor, and it knows how to undo all of that to convert the results back to something you can write back out as CF-compliant netCDF.

The general philosophy for configuration is that the YAML maps directly to constructor arguments for the classes, so see the documentation of individual classes for information about the possible options.
