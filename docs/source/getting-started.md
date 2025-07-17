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

:::{important}
macOS users will need to ensure that the required compilers are present and properly configured before installing mile-credit for versions requiring pySTEPS (miles-credit > 2025.2.0).  See this [note in the pySTEPS documentation](https://pysteps.readthedocs.io/en/latest/user_guide/install_pysteps.html#osx-users-gcc-compiler) for details.
:::

## Installation on Derecho
If you want to build a conda environment and install a Derecho-compatible version of PyTorch, run
the `create_derecho_env.sh` script. 
```bash
git clone git@github.com:NCAR/miles-credit.git
cd miles-credit
./create_derecho_env.sh
```

:::{important}
The credit conda environment requires multiple gigabytes of space. Use the `gladequota` command
to verify that you have sufficient space in your home or work directories before installing.
You can specify where to install your conda environments in a `.condarc` file with the section
`envs_dirs`. 
:::

## Installation from source
See <project:installation.md> for detailed instructions on building CREDIT and its 
dependencies from source or for building CREDIT on the Derecho supercomputer.

## Running a pretrained model
See <project:Inference.md> for more details on how to run one of the pretrained CREDIT models.


