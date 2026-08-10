# DOP+: GOES-East Full-Disk AI Nowcasting of Cloud Evolution in Observation Space

This repository contains the code for training, deploying, and evaluating the DOP+ model described in:

> Kimpara, D., Bagheri, O., Hernandez Banos, I., Jung, B.-J., & Snyder, C. (2025). GOES-East full-disk AI nowcasting of cloud evolution in observation space.

DOP+ forecasts GOES-East full-disk infrared brightness temperatures at 10-minute resolution out to 6-hour lead times by extending direct observation prediction (DOP) with conditioning on ERA5 meteorological fields via Feature-wise Linear Modulation (FiLM). The domain covers ~10^8 km^2, spanning tropical, midlatitude, and marine regimes across the Americas and adjacent ocean basins.

This software is built on the [CREDIT](https://github.com/NCAR/miles-credit) (Community Research Earth Digital Intelligence Twin) framework.

## Repository Structure

```
credit/              Core library (models, datasets, trainers, transforms, verification)
config_goes/         YAML configuration files for GOES training and evaluation runs
goes_preproc/        GOES ABI and ERA5 data preprocessing and regridding pipelines
  goes_regrid/         Regridding and local zenith angle computation
  streamlined_preproc/ Parallelized GOES I/O, Zarr store creation, and validation
  statistics/          Climatology and data statistics computation
  process_era5.py      ERA5 field subsetting and regridding
scripts/             HPC job submission scripts (PBS on Derecho/Casper)
notebooks/           Analysis, verification, and figure generation notebooks
```

## Data

- **GOES ABI Level 1b**: Channels 4, 7, 8, 9, 10, 13 coarse-grained to 0.1 degree resolution. Training: Apr 2018 -- Jul 2022; validation: Jul -- Dec 2022; evaluation: Jun -- Jul 2025. Available from the [AWS GOES archive](https://registry.opendata.aws/noaa-goes/).
- **ERA5 reanalysis**: 11 model levels plus single-level and static fields, bilinearly interpolated to 0.1 degrees. Available from the [NSF NCAR Research Data Archive](https://doi.org/10.5065/XV5R5344).
- **MPAS-JEDI simulated brightness temperatures**: Available via [Globus](https://app.globus.org/file-manager?origin_id=22275241-02e7-425a-89d6-5686435fdd46).

## Installation

```bash
git clone https://github.com/dkimpara/goes-deterministic.git
cd goes-deterministic
pip install -e .
```

## Usage

Training and rollout are launched via the CREDIT CLI entry points with a YAML config file. Example configs are in `config_goes/`.

```bash
# Training
credit_train --config config_goes/goes_era5_forcing.yml

# Rollout / inference
credit_rollout_to_netcdf --config config_goes/example_eval.yml
```

See the [CREDIT documentation](https://miles-credit.readthedocs.io/en/latest/) for detailed configuration options.

## Support

This research was supported by the United States Air Force (grant no. NA21OAR4310383) and the NSF National Center for Atmospheric Research, a major facility sponsored by the U.S. National Science Foundation under Cooperative Agreement No. 1852977.

## License

Apache-2.0. See [LICENSE](LICENSE).
