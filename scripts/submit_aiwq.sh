#!/bin/bash -l
#PBS -N AIWQ
#PBS -l select=1:ngpus=1:gpu_type=h100:mem=128GB:ncpus=16
#PBS -l walltime=12:00:00
#PBS -A NAML0001
#PBS -q casper
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -M cbecker@ucar.edu
#PBS -m abe


module load conda
conda activate /glade/work/cbecker/conda-envs/miles-credit
cd /glade/work/cbecker/miles-credit/applications/

#DATE_FLAG=$(date -u -d "Thursday" +"%Y-%m-%dT0000")
#DATE_FLAG=$"2025-11-13T0000"
DATE_FLAG=$"2025-12-04T0000"
exec > "AIWQ_log_${DATE_FLAG}.log" 2>&1

#python gefs_init.py \
#      -d "$DATE_FLAG" \
#      -p /glade/derecho/scratch/cbecker/GEFS/ \
#      --out /glade/derecho/scratch/cbecker/GEFS_CAM/ \
#      -w /glade/work/dgagne/camulator_gefs_weights.nc \
#      -n 10 \
#      -v "ps,t,sphum,liq_wat,ice_wat,rainwat,snowwat,smc,graupel,u_s,v_w,slmsk,tsea,fice,t2m,q2m,zh" \
#      -t /glade/u/home/dgagne/miles-credit/credit/metadata/cam.yaml \
#      -m 30 \
#      -u /glade/u/home/dgagne/miles-credit/credit/metadata/CESM_Lev_Info.nc \
#      -r /glade/work/cbecker/miles-credit/credit/metadata/gefs_to_cam.yml
##
#python prepare_init.py -d "$DATE_FLAG"
#
#python make_ensemble_aiwq.py -d "$DATE_FLAG"

torchrun realtime_wrapper.py \
      -d "$DATE_FLAG" \
      -c /glade/work/cbecker/miles-credit/config/subseasonal_KJM_AIWQ_v2.yml
#      -c /glade/work/cbecker/miles-credit/config/subCESMulator_CLB.yml



conda deactivate
conda activate /glade/work/kjmayer/conda-envs/AIWQ2025

python prepare_forecast.py -d $DATE_FLAG

python submit_forecast.py -d $DATE_FLAG

