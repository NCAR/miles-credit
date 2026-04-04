# NSF NCAR MILES Community Research Earth Digital Intelligence Twin (CREDIT)

[![DOI](https://zenodo.org/badge/710968229.svg)](https://doi.org/10.5281/zenodo.14361005)

[PyPI](https://pypi.org/project/miles-credit/)

[CREDIT npj Climate and Atmospheric Science Article](nature.com/articles/s41612-025-01125-6)

## About
CREDIT is an open software platform to train and deploy AI atmospheric prediction models. CREDIT offers fast models 
that can be flexibly configured both in terms of input data and neural network architecture. The interface is designed
to be user-friendly and enable fast spin-up and iteration. CREDIT is backed by the AI and atmospheric science expertise
of the MILES group and the NSF National Center for Atmospheric Research, leading to design choices that balance advanced
AI/ML with our physical knowledge of the atmosphere.

CREDIT has reached its first stable release with a full set of models, training, and deployment options. It continues
to be under active development. Please contact [the MILES group](mailto:milescore@ucar.edu) if you have any questions about CREDIT.

MILES CREDIT also provides more detailed [documentation](https://miles-credit.readthedocs.io/en/latest/) with installation
instructions, how to get started training and deploying models, how to interpret the config files, and full API docs. 

## Citing CREDIT
If you are interested in using CREDIT as part of your research, please cite the following paper:
Schreck, J.S., Sha, Y., Chapman, W. et al. Community Research Earth Digital Intelligence Twin: a scalable framework 
for AI-driven Earth System Modeling. npj Clim Atmos Sci 8, 239 (2025). https://doi.org/10.1038/s41612-025-01125-6

# Model Weights and Data
Model weights for the CREDIT 6-hour WXFormer and FuXi models and the 1-hour WXFormer are available on huggingface.

* [6-Hour WXFormer](https://huggingface.co/djgagne2/wxformer_6h)
* [1-Hour WXFormer](https://huggingface.co/djgagne2/wxformer_1h)
* [6-Hour FuXi](https://huggingface.co/djgagne2/fuxi_6h)

Processed ERA5 Zarr Data are available for download through Globus (requires free account) through the [CREDIT ERA5 Zarr Files](https://app.globus.org/file-manager/collections/2fc90d8f-10b7-44e1-a6a5-cf844112822e/overview) collection.

Scaling/transform values for normalizing the data are available through Globus [here](https://app.globus.org/file-manager/collections/c5a23e21-1bee-4d1e-bb59-77c5dcee7c76). 

CREDIT also supports realtime runs generated from deterministic [Google Cloud GFS files](https://console.cloud.google.com/marketplace/product/noaa-public/gfs)
and raw cube sphere [GEFS files](https://console.cloud.google.com/marketplace/product/noaa-public/gfs-ensemble-forecast-system).

# Regional WRF Downscaling (feature/regional)

CREDIT supports regional training using WRF CONUS404 output as a high-resolution target,
driven by ERA5 boundary conditions. This enables training AI emulators for convection-allowing
regional weather models over the CONUS Great Plains domain.

## Data

Data for the WRF regional pipeline is maintained by Kyle Sha on Derecho scratch:

| Component | Path |
|-----------|------|
| WRF state (C404 Great Plains) | `/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/C404_new/C404_GP_*.zarr` |
| Static fields | `/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/static/C404_GP_static.zarr` |
| Mean/std (WRF vars) | `/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/mean_std/C404_mean_1980_2019_12lev.nc` |
| Residual std (WRF vars) | `/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/mean_std/C404_std_residual_1980_2019_12lev_clean.nc` |
| ERA5 boundary (3-hourly) | `/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/dscale_3h/*.zarr` |
| Mean/std (ERA5 boundary) | `/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/mean_std/ERA5_3h_mean_1980_2019.nc` |

## Quick smoke test (2 epochs, 5 batches)

```bash
conda activate credit-main-casper
python applications/train_v2_regional.py -c config/regional_smoke_test.yml
```

Expected: loss drops from ~22 → ~21, ACC rises from ~0.007 → ~0.012 in 2 epochs.

## Longer training run (20 epochs, 100 batches, dim=256 depth=4)

```bash
python applications/train_v2_regional.py -c config/regional_train.yml
```

Results save to `/glade/derecho/scratch/schreck/CREDIT_runs/regional_train/training_log.csv`.
Expected convergence: loss ~11 → ~1, ACC ~0.24 → 0.49 over 20 epochs on a single V100.

## Key implementation notes

- **No parser**: `train_v2_regional.py` bypasses `credit_main_parser`; `_populate_post_conf()`
  replicates the tracer-fixer index computation the parser normally does.
- **Boundary NaNs**: ERA5 boundary zarrs have NaN in the WRF interior (only edges are valid).
  `trainerWRF.py` applies `torch.nan_to_num(x_boundary, nan=0.0)` before the forward pass.
- **Trainer type**: `standard-wrf` → `credit/trainers/trainerWRF.py`
- **Model type**: `wrf` → `WRFTransformer` (ViT-style, Swin attention)

# Support
This software is based upon work supported by the NSF National Center for Atmospheric Research, a major facility sponsored by the 
U.S. National Science Foundation  under Cooperative Agreement No. 1852977 and managed by the University Corporation for Atmospheric Research. Any opinions, findings and conclusions or recommendations 
expressed in this material do not necessarily reflect the views of NSF. Additional support for development was provided by 
The NSF AI Institute for Research on Trustworthy AI for Weather, Climate, and Coastal Oceanography (AI2ES)  with grant
number RISE-2019758 and by Schmidt Sciences, LLC. 
