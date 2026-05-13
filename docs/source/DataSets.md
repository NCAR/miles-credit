# Datasets Selection Guide

CREDIT now supports multiple datasets including the following:

| Dataset | Local or Remote | Regional or Global | Description |
| ------- | --------------- | ------------------ | ----------- |
| [ERA5](credit.datasets.era5) | Local | Global | ECMWF 5th Generation Reanalysis |
| ARCO ERA5 | Remote | Global | Analysis-Ready Cloud-Optimized ERA 5 |
| MRMS | Remote | ... | ... |
| GOES | Remote | Regional (Western Hemisphere) | Geostationary Satellite Remote Sensing |
| HRRR | Local or Remote | Regional (Continental US) | NOAA High-Resolution Rapid Refresh |
| GFS (soon) | Local or Remote | Global | NOAA Global Forecast System (GFS) |

## CREDIT Datasets Inherit from Pytorch Datasets

Please see AbstractBaseDataset
