import warnings

# Ignore DeprecationWarning for torch_geometric.distributed which was deprecated in 2.7.0
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch_geometric.distributed")

"""CREDIT is an open software platform to train and deploy AI atmospheric prediction models.

=====

How to use the documentation
----------------------------
Documentation is available via docstrings provided with the code.

Available subpackages
---------------------
datasets
    Contains PyTorch Dataset classes for common Earth system data sources.
ensemble
    Methods for generating ensembles
losses
    Contains a mix of specialized loss functions for optimizing deterministic and ensemble models.
metadata
    Contains metadata definitions for use in inference.
models
    Defines model architectures.
trainers
    Defines trainers
"""
