# Installing CREDIT from source

If you want to take advantage of the full power of CREDIT,
which includes scaling training across multiple nodes, you 
will need to build PyTorch from source. The instructions
for building PyTorch from source can leave out important
details if you want proper CUDA, numpy, and MPI support, 
so follow along here as we take you on the journey to 
CREDIT.

## Prerequisites
In this section, you will need to create a clean Python
virtual environment and install other system packages to
enable the building of PyTorch. You can even do this on your
Mac or Linux laptop.

### MacOS (Intel or ARM)
1. First, install either the [Homebrew](https://brew.sh/) or [MacPorts](https://www.macports.org/) package managers 
so you can easily install all the system-level dependencies and other helpful Linux programs like wget, gcc, etc.
Further instructions will assume the use of Homebrew, but should also work for MacPorts. Homebrew
should also install the XCode Command Line tools, which include git and clang.
2. Install the following programs with the `brew install <program>`
command.
    * gcc (gnu compilers for C, C++, and Fortran)
    * netcdf (for reading and writing netCDF files)
    * wget (for downloading files from the internet)
    * mpich (for MPI distributed training support)
### Linux
1. Install the following dependencies with your OS package manager or ask your sysadmins to make sure
they are installed:
   * gcc (gnu compilers for C, C++, and Fortran)
   * netcdf (for reading and writing netCDF files)
   * wget (for downloading files from the internet)
   * mpich, openmpi, or cray-mpi (for MPI distributed training support)
2. Install a recent version of the CUDA toolkit, cuDNN, and NCCL (needed for multi-GPU and distributed training).
3. If you plan to do multi-node distributed training with CREDIT, you will also need to install 
[libfabric](https://github.com/ofiwg/libfabric) and the [aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl) plugin
to interface between libfabric and NCCL. If you do not compile PyTorch with Libfabric and aws-ofi-nccl, 
we have found that inter-node communication speeds on Derecho and other HPE/Cray HPC systems are much slower.

## NSF NCAR Derecho
The NSF NCAR Derecho system has some special requirements in place to support Cray MPI and the Slingshot
interconnect. Ben Kirk has created a special [makefile](https://github.com/benkirk/derecho-pytorch-mpi) to
build PyTorch and Torchvision from source on Derecho with all appropriate environment variables and dependencies.
Please follow instructions there if you wish to build your own version of PyTorch from source. Otherwise,
built wheels of PyTorch 2.5.1 and Torchvision 0.20.1 are available on Derecho 
at `/glade/work/dgagne/credit-pytorch-envs/derecho-pytorch-mpi/wheels/`. 

## PyTorch Python Dependencies
1. Install a Python virtual environment manager. My preferred one for now is
[miniforge](https://github.com/conda-forge/miniforge), which has both conda and mamba included and does not have the 
licensing issues that may come with miniconda. [uv](https://astral.sh/blog/uv) might also work but has not been tested.
2. Create a clean virtual environment with the following command:
`mamba create -n credit python=3.12`
3. Activate your environment. `conda activate credit`
4. Clone a release branch of PyTorch and all its third party dependencies. This will take awhile.
`git clone -b v2.6.0 --recursive git@github.com:pytorch/pytorch.git`
5. Go into the PyTorch repo and install its dependencies into your `credit` environment.
```bash
cd pytorch
pip install -r requirements.txt
```
## Building PyTorch
1. Build and install PyTorch into your environment with the following command:
```bash
USE_DISTRIBUTED=1 python setup.py install
```
On Mac systems, the `USE_DISTRIBUTED` option is 0 by default and is 1 on Linux systems.

If you want to build PyTorch in a way that can be distributed to multiple users, build a wheel file.
Wheel files are binary libraries that can be pip installed into other users' environments on the same operating system.
```bash
pip install wheel build
cd pytorch # if not already in your pytorch directory
python -m build --wheel
```

With either build approach, PyTorch performs a bunch of configuration checks to know what files to compile for a given 
setup. Before letting it compile, which can take at least 1 hour, review key configuration settings in the summary.
Make sure `USE_DISTRIBUTED=1`, that numpy is included, and if on a Linux cluster, that 
CUDA and NCCL have been found and are being used.

Once a wheel file is created, you can install it into a Python environment with
`pip install <pytorch wheel file>`

## Building Torchvision
CREDIT also uses [torchvision](https://github.com/pytorch/vision), which also needs to be built and compiled to match
with your installed version of PyTorch. Instructions for building torchvision from source
can be found [here](https://github.com/pytorch/vision/blob/main/CONTRIBUTING.md#development-installation). 

## Installing CREDIT
Clone miles-credit from github and install using pip:
```bash
git clone git@github.com:NCAR/miles-credit.git
cd miles-credit
pip install -e .
```
The CREDIT installer should be able to install all dependencies from PyPI using
pip. 

## Test Your Installation
To verify that you have installed all dependencies correctly, you can conduct the following tests.
First, install and run pytest to verify all unit tests pass.
```bash
pip install pytest
cd miles-credit
pytest .
```

If all tests pass, run `applications/rollout_to_netcdf.py` on a test case.
