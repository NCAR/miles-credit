#!/bin/bash -l
#PBS -N AIWQ
#PBS -l select=1:ncpus=16:ngpus=1:mem=128GB
#PBS -l walltime=04:00:00
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe aiwq-out

module load conda
conda activate miles-credit
cd /glade/work/cbecker/miles-credit/applications/AIWQ_realtime/

DATE_FLAG=$(date -u -d "Last Thursday" +"%Y-%m-%dT0000")

python gefs_init.py \
      -d "$DATE_FLAG" \
      -p /glade/derecho/scratch/cbecker/GEFS/ \
      --out /glade/derecho/scratch/cbecker/GEFS_CAM/ \
      -w /glade/work/dgagne/camulator_gefs_weights.nc \
      -n 10 \
      -t /glade/u/home/dgagne/miles-credit/credit/metadata/cam.yaml \
      -m 30 \
      -u /glade/u/home/dgagne/miles-credit/credit/metadata/CESM_Lev_Info.nc \
      -r /glade/u/home/dgagne/miles-credit/credit/metadata/gefs_to_cam.yml


python prepare_init.py -d "$DATE_FLAG"

#python make_ensemble_aiwq.py -d "$DATE_FLAG"

torchrun realtime_wrapper.py \
      -d "$DATE_FLAG" \
      -c /glade/work/cbecker/miles-credit/applications/AIWQ_realtime/aiwq_realtime.yml