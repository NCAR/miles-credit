# Getting Started

## Installation for Single Server/Node Deployment
If you plan to use CREDIT only for running pretrained models or training on a single server/node, then
the standard Python install process will install both CREDIT and all necessary dependencies, including
the right versions of PyTorch and CUDA, for you. If you are running CREDIT on the Casper system, then
 the following instructions should work for you.

:::{note}
**NCAR users on Casper or Derecho**: pre-built conda environments are already available —
you do not need to create a new environment from scratch.

```bash
# Casper
conda activate /glade/u/home/schreck/.conda/envs/credit-casper

# Derecho
conda activate /glade/work/benkirk/conda-envs/credit-derecho-torch28-nccl221
```

Then clone the repo and install the current branch in editable mode:

```bash
git clone https://github.com/NCAR/miles-credit.git
cd miles-credit
pip install -e . --no-deps   # dependencies already satisfied in the shared env
```
:::

For non-NCAR systems, create a minimal conda or virtual environment.
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
This script will train a simple WXFormer model on 1 degree ERA5 data for 10 batches. The model will not be good,
but if it completes successfully, you can
have some confidence that CREDIT has been set up correctly in your environment.

Next, make sure inference is working by running:
```bash
credit_rollout_to_netcdf -c config/example-v2026.1.0.yml
```
This will perform a short forecast based on ERA5 and will save the outputs to a specified directory.

If you want to run a realtime forecast, first download and regrid GFS data to your domain.
```bash
credit_gfs_init -c config/example-v2026.1.0.yml -p 1
```
The script will create a GFS initialization file in your scratch directory. It
will also create a realtime config file `config/example-v2026.1.0_realtime.yml`.
Edit the realtime config file so that `data:save_loc_dynamic_forcing` is
`/glade/campaign/cisl/aiml/credit/credit_solar_6h_1deg_era5_mlevel/*.nc`.  Then you can
run a realtime forecast with
```bash
credit_rollout_realtime -c config/example-v2026.1.0_realtime.yml -p 1
```
This script will output 6-hourly netcdf files to your scratch directory and
performs interpolation to pressure levels. From there you can view the data with your
favorite visualization tool.


## Quick start with the `credit` CLI

After installation, the `credit` command is your single entrypoint for everything:

```bash
# Generate a ready-to-use config (0.25° or 1° ERA5)
credit init --grid 0.25deg -o my_experiment.yml

# Train on the current node (single GPU)
credit train -c my_experiment.yml

# Submit a batch job to Casper or Derecho
credit submit --cluster casper  -c my_experiment.yml --gpus 1
credit submit --cluster derecho -c my_experiment.yml --gpus 4 --nodes 2

# Run a realtime forecast
credit realtime -c my_experiment.yml --init-time 2024-01-15T00 --steps 40

# See all commands
credit --help
```

Use `--dry-run` with `credit submit` to preview the PBS script before submitting.

## Running a pretrained model
See <project:Inference.md> for more details on how to run one of the pretrained CREDIT models.
