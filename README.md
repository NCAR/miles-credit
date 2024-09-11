# NSF NCAR MILES Community Runnable Earth Digital Intelligence Twin (CREDIT)

## About
CREDIT is a package to train and run neural networks
that can emulate full NWP models by predicting
the next state of the atmosphere given the current state.

## NSF-NCAR Derecho Installation
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

Install dependencies using environment.yml file:
```bash
mamba env create -f environment.yml
conda activate credit
```

Some metrics use WeatherBench2 for computation. Install with:
```bash
git clone git@github.com:google-research/weatherbench2.git
cd weatherbench2
pip install .
````

To enable GPU support, install pytorch-cuda:
```bash
mamba install pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install miles-credit with the following command:
```bash
pip install .
```

## Train a Segmentation Model (like a U-Net)
```bash
python applications/train.py -c config/unet.yml
```
 ## Train a Vision Transformer
```bash
python applications/train.py -c config/vit.yml
```

Or use a fancier [variation](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/rvt.py)

```bash
python applications/train.py -c config/rvt.yml
```

## Launch with PBS on Casper or Derecho
 
Adjust the PBS settings in a configuration file for either casper or derecho. Then, submit the job via
```bash
python applications/train.py -c config/vit.yml -l 1
```
The launch script may be found in the save location that you set in the configation file. The automatic launch script generation will take care of MPI calls and other complexities if you are using more than 1 GPU.

## Inference Forecast

The predict field in the config file allows one to speficy start and end dates to roll-out a trained model. To generate a forecast,

```bash
python applications/predict.py -c config/vit.yml
```
