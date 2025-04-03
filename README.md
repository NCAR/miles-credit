# NSF NCAR MILES Community Research Earth Digital Intelligence Twin (CREDIT)

[![DOI](https://zenodo.org/badge/710968229.svg)](https://doi.org/10.5281/zenodo.14361005)
![PyPI - Version](https://img.shields.io/pypi/v/miles-credit)
[![arXiv](https://img.shields.io/badge/arXiv-2411.07814-b31b1b.svg)](https://arxiv.org/abs/2411.07814)

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
Schreck, J., Sha, Y., Chapman, W., Kimpara, D., Berner, J., McGinnis, S., Kazadi, A., Sobhani, N., Kirk, B., Gagne, D.J. (2024, November 9). 
Community Research Earth Digital Intelligence Twin (CREDIT). arXiv [cs.AI]. http://arxiv.org/abs/2411.07814

# Model Weights and Data
Model weights for the CREDIT 6-hour WXFormer and FuXi models and the 1-hour WXFormer are available on huggingface.

* [6-Hour WXFormer](https://huggingface.co/djgagne2/wxformer_6h)
* [1-Hour WXFormer](https://huggingface.co/djgagne2/wxformer_1h)
* [6-Hour FuXi](https://huggingface.co/djgagne2/fuxi_6h)

Processed ERA5 Zarr Data are available for download through Globus (requires free account) through the [CREDIT ERA5 Zarr Files](https://app.globus.org/file-manager/collections/2fc90d8f-10b7-44e1-a6a5-cf844112822e/overview) collection.

Scaling/transform values for normalizing the data are available through Globus [here](https://app.globus.org/file-manager/collections/c5a23e21-1bee-4d1e-bb59-77c5dcee7c76). 

# Support
This software is based upon work supported by the NSF National Center for Atmospheric Research, a major facility sponsored by the 
U.S. National Science Foundation  under Cooperative Agreement No. 1852977 and managed by the University Corporation for Atmospheric Research. Any opinions, findings and conclusions or recommendations 
expressed in this material do not necessarily reflect the views of NSF. Additional support for development was provided by 
The NSF AI Institute for Research on Trustworthy AI for Weather, Climate, and Coastal Oceanography (AI2ES)  with grant
number RISE-2019758. 
