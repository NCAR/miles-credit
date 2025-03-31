# NSF NCAR MILES Community Research Earth Digital Intelligence Twin (CREDIT)

## About
CREDIT is a research platform to train and run neural networks that can emulate full NWP models by predicting
the next state of the atmosphere given the current state. The platform is still under very active development. 
If you are interested in using or contributing to CREDIT, please reach out to David John Gagne (dgagne@ucar.edu). 


## NSF-NCAR Derecho Installation
Currently, the framework for running miles-credit in parallel is centered around NSF NCAR's Derecho HPC. Derecho requires building several miles-credit dependent packages locally, including PyTorch, to enable correct MPI configuration. To begin, create a clone of the pre-built miles-credit environment, which contains compatiable versions of torch, torch-vision, numpy, and others. 

```bash
module --force purge 
module load ncarenv/23.09 gcc/12.2.0 ncarcompilers cray-mpich/8.1.27 cuda/12.2.1 cudnn/8.8.1.3-12 conda/latest
mamba create --name credit-derecho --clone /glade/derecho/scratch/benkirk/derecho-pytorch-mpi/envs/credit-pytorch-v2.3.1-derecho-gcc-12.2.0-cray-mpich-8.1.27
```

Going forward, care must be taken when installing new packages so that PyTorch and the other relevant miles-credit dependencies are not overridden. Next, grab the most updated version of miles-credit from github (assuming no changes to the local-build dependencies):

```bash
conda activate credit-derecho
# needed an sshkey to use this command: git clone git@github.com:NCAR/miles-credit.git
git clone https://github.com/NCAR/miles-credit.git
cd miles-credit
```

and then install without dependencies by

```bash
pip install --no-deps .
```

Henceforth, when adding new packages aim to use the no dependenices option. 

To test your Derecho installation, skip to [Launch with PBS on Casper or Derecho](#launch-with-pbs-on-casper-or-derecho).

## Standard Installation 
Clone from miles-credit github page:
```bash
# needed an sshkey to use this command: git clone git@github.com:NCAR/miles-credit.git
git clone https://github.com/NCAR/miles-credit.git
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
python applications/train.py -c config/wxformer_1dg_test.yml
```

## Launch with PBS on Casper or Derecho
 
Adjust the PBS settings in a configuration file for either casper or derecho. Then, submit the job via
```bash
python applications/train.py -c config/wxformer_1dg_test.yml -l 1
```
The launch script may be found in the save location that you set in the configation file. The automatic launch script generation will take care of MPI calls and other complexities if you are using more than 1 GPU.

## Inference Forecast

The predict field in the config file allows one to speficy start and end dates to roll-out a trained model. To generate a forecast,

```bash
python applications/rollout_to_netcdf.py -c config/wxformer_1dg_test.yml
```

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
