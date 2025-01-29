# Getting Started

## NSF NCAR Derecho Installation

Currently, the framework for running miles-credit in parallel is centered around NSF-NCAR's Derecho HPC. Derecho requires building several miles-credit dependent packages locally, including PyTorch, to enable correct MPI configuration. To begin, create a clone of the pre-built miles-credit environment, which contains compatiable versions of torch, torch-vision, numpy, and others. 

```bash
module purge 
module load ncarenv/23.09 gcc/12.2.0 ncarcompilers cray-mpich/8.1.27 cuda/12.2.1 cudnn/8.8.1.3-12 conda/latest
conda create --name credit-derecho --clone /glade/derecho/scratch/benkirk/derecho-pytorch-mpi/envs/credit-pytorch-v2.3.1-derecho-gcc-12.2.0-cray-mpich-8.1.27
```

Going forward, care must be taken when installing new packages so that PyTorch and the other relevant miles-credit dependencies are not overridden. Next, grab the most updated version of miles-credit from github (assuming no changes to the local-build dependencies):

```bash
conda activate credit-derecho
git clone git@github.com:NCAR/miles-credit.git
cd miles-credit
```

and then install without dependencies by

```bash
pip install --no-deps .
```

Henceforth, when adding new packages aim to use the no dependenices option. 

## Standard Installation 

Clone from miles-credit github page:
```bash
git clone git@github.com:NCAR/miles-credit.git
cd miles-credit
```

Install dependencies using environment_gpu.yml file (also compatible with CPU-only machines):

Note: if you are on NCAR HPC, we recommend installing to your home directory. To do this, simply append `-p /glade/u/home/$USER/[your_install_dir]/` to the `conda/mamba env create` command below:

```bash
mamba env create -f environment_gpu.yml
conda activate credit
```

CPU-only install:
```bash
mamba env create -f environment_cpu.yml
conda activate credit
```


Some metrics use WeatherBench2 for computation. Install with:
```bash
git clone git@github.com:google-research/weatherbench2.git
cd weatherbench2
pip install .
````

