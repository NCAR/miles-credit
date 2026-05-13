# Datasets Selection Guide

CREDIT now supports multiple datasets including the following:

| Dataset | Local or Remote | Regional or Global | Description |
| ------- | --------------- | ------------------ | ----------- |
| [`ERA5`](credit.datasets.era5) | Local | Global | ECMWF 5th Generation Reanalysis |
| [`ARCO ERA5`](credit.datasets.era5) | Remote | Global | Analysis-Ready Cloud-Optimized ERA 5 |
| [`MRMS`](credit.datasets.mrms) | Remote | ... | ... |
| [`GOES`](credit.datasets.goes) | Remote | Regional (Western Hemisphere) | Geostationary Satellite Remote Sensing |
| [`HRRR`](credit.datasets.hrrr) | Local or Remote | Regional (Continental US) | NOAA High-Resolution Rapid Refresh |
| GFS (soon) | Local or Remote | Global | NOAA Global Forecast System (GFS) |
| GEFS (soon) | ... | ... | ... |
| WRF (soon) | Local | Regional (User Provided) | NSF NCAR Weather Research Forecast Model |

## CREDIT Datasets Inherit from Pytorch Datasets

Please see {py:class}`~credit.datasets.base_dataset.AbstractBaseDataset` for information on the inheritance from Pytorch Datasets.
A minimal implementation of single-source datasets inherit from {py:class}`~credit.datasets.base_dataset.BaseDataset`.
