# Getting Started

## Installation for Single Server/Node Deployment
If you plan to use CREDIT only for running pretrained models or training on a single server/node, then
the standard Python install process will install both CREDIT and all necessary dependencies, including
the right versions of PyTorch and CUDA, for you. If you are running CREDIT on the Casper system, then
 the following instructions should work for you.

Create a minimal conda or virtual environment.
```bash
conda create -n credit python=3.11
conda activate credit
```
If you want to install the latest stable release from PyPI:
```bash
pip install miles-credit
```

If you want to install the main development branch
```bash
git clone git@github.com:NCAR/miles-credit.git
cd miles-credit
pip install -e .
```

## Installation from Scratch
See <project:installation.md> for detailed instructions on building CREDIT and its 
dependencies from scratch or for building CREDIT on the Derecho supercomputer.


