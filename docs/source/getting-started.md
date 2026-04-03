# Getting Started

## Installation for Single Server/Node Deployment
If you plan to use CREDIT only for running pretrained models or training on a single server/node, then
the standard Python install process will install both CREDIT and all necessary dependencies, including
the right versions of PyTorch and CUDA, for you. If you are running CREDIT on the Casper system, then
 the following instructions should work for you.

Create a minimal conda or virtual environment.
```bash
conda create -n credit python=3.12
conda activate credit
```

:::{important}
When installing PyTorch on Linux, the default option (v2.11) uses CUDA 13, which is currently not
compatible with the CUDA driver on Casper. Add `--extra-index-url https://download.pytorch.org/whl/cu126` 
to your pip install commands to install a PyTorch version compiled with CUDA 12.6, which will work
on Casper NVIDIA GPUs from the V100s to the H100s. 

If you want to install the CPU-only version of PyTorch, use `--extra-index-url https://download.pytorch.org/whl/cpu`.
:::

If you want to install the latest stable release from PyPI:
```bash
pip install miles-credit
```

If you want to install the main development branch:
```bash
git clone https://github.com/NCAR/miles-credit.git
cd miles-credit
# Linux with NVIDIA GPU support
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu126
# Mac
pip install -e .
```

If you plan to perform development work on CREDIT, install additional
dependencies for development.
```bash
pip install -e .[develop]
```

:::{important}
macOS users will need to ensure that the required compilers are present and properly configured before installing 
miles-credit for versions requiring pySTEPS (miles-credit > 2025.2.0).  
See this [note in the pySTEPS documentation](https://pysteps.readthedocs.io/en/latest/user_guide/install_pysteps.html#osx-users-gcc-compiler) for details.
:::

## Installation on Derecho
If you want to build a conda environment and install a Derecho-compatible version of PyTorch, run
the `create_derecho_env.sh` script.
```bash
git clone https://github.com/NCAR/miles-credit.git
cd miles-credit
./create_derecho_env.sh
```
This script will create a Derecho-compatible `conda` environment with a user-specified name taken from the `CREDIT_ENV_NAME` environment variable (defaults to `credit-derecho`).

:::{important}
The credit conda environment requires multiple gigabytes of space. Use the `gladequota` command
to verify that you have sufficient space in your home or work directories before installing.
You can specify where to install your conda environments in a `.condarc` file with the section
`envs_dirs`.

If you are finding your home directory to be surprisingly full with minimal installation of files,
check your `~/.cache` directory and delete everything inside it. This directory is where
python installers tend to download their files. You can shift the .cache directory to a location with
more space by adding `export XDG_CACHE_HOME="/glade/work/$USER/.cache"` to your `~/.bashrc` file.
:::



## Installation from source
See <project:installation.md> for detailed instructions on building CREDIT and its
dependencies from source or for building CREDIT on the Derecho supercomputer.

## Making sure training and inference work
To train a basic WXFormer model on Casper or Derecho, run the following command from the miles-credit directory:
```bash
credit_train -c config/example-v2026.1.0.yml
```
This script will train a simple WXFormer model on 1 degree ERA5 data for 10 batches. The model will not be good, but if it completes successfully, you can
have some confidence that CREDIT has been set up correctly in your environment.

Next, make sure inference is working by running:
```bash
credit_rollout_to_netcdf -c config/example-v2026.1.0.yml
```
This will perform a short forecast based on ERA5 and will save the outputs to a specified directory.

## Running a pretrained model
See <project:Inference.md> for more details on how to run one of the pretrained CREDIT models.
