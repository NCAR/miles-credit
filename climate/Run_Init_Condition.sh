#!/bin/bash
# -*- coding: utf-8 -*-
# PBS -N Run_Noise_Script
# PBS -A NAML0001 
# PBS -l walltime=12:00:00
# PBS -o RUN_Climate_RMSE.out
# PBS -e RUN_Climate_RMSE.out
# PBS -q casper
# PBS -l select=1:ncpus=32:ngpus=1:mem=250GB
# PBS -l gpu_type=a100
# PBS -m a
# PBS -M wchapman@ucar.edu

module load conda
conda activate /glade/work/wchapman/conda-envs/credit-casper-modern
# conda activate /glade/work/wchapman/conda-envs/credit-derecho
# python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel_opt.py --config /glade/derecho/scratch/wchapman/miles_branchs/CESM_Spatial_PS_WXmod/be21_coupled-v2025.2.0.yml --input_shape 1 136 1 192 288 --forcing_shape 1 6 1 192 288 --output_shape 1 145 1 192 288 --device cuda --init_noise 1 --model_name checkpoint.pt00275.pt

CONFIG=./be21_coupled-v2025.2.0_small.yml
SCRIPT=./Quick_Climate_Year_Slow_parrallel_opt_02.py
BASE_ARGS="--config $CONFIG \
  --input_shape 1 136 1 192 288 \
  --forcing_shape 1 6 1 192 288 \
  --output_shape 1 145 1 192 288 \
  --device cuda"

# #Loop over checkpoint numbers 260 → 285
# for i in {90..90}; do
#   # zero-pad to 5 digits: 00171, 00172, …
#   ckpt=$(printf "%05d" $i)
#   python $SCRIPT $BASE_ARGS --model_name checkpoint.pt${ckpt}.pt --save_append runfilt_${ckpt}
# done

# for i in {91}; do
#   # zero-pad to 5 digits: 00171, 00172, …
#   ckpt=$(printf "%05d" $i)
#   python $SCRIPT $BASE_ARGS --model_name checkpoint.pt${ckpt}.pt --save_append runfilt_${ckpt}
# done


python $SCRIPT $BASE_ARGS --model_name checkpoint.pt00091.pt --save_append runfilt_00091

# 87, 89, 90, 91

for i in {91,91}; do
  ckpt=$(printf "%05d" $i)
  python /glade/derecho/scratch/wchapman/miles_branchs/pretrain_CESM_Spatial_PS_WXmod_pxshf_LR/climate/Post_Process_Parallel.py /glade/derecho/scratch/wchapman/miles_branchs/pretrain_CESM_Spatial_PS_WXmod_pxshf_LR/be21_coupled-v2025.2.0_small.yml 1D --variables U V T Qtot PS PRECT TREFHT TS TAUX TAUY --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHTTAUXTAUY --rescale_it False --save_append runfilt_${ckpt}
done

# for i in {87..87}; do
#   ckpt=$(printf "%05d" $i)
#   python /glade/derecho/scratch/wchapman/miles_branchs/pretrain_CESM_Spatial_PS_WXmod_pxshf_LR/climate/Post_Process_Parallel.py /glade/derecho/scratch/wchapman/miles_branchs/pretrain_CESM_Spatial_PS_WXmod_pxshf_LR/be21_coupled-v2025.2.0_small.yml 1D --variables U V T Qtot PS PRECT TREFHT TS TAUX TAUY --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHTTAUXTAUY --rescale_it False --save_append runfilt_${ckpt}
# done



# python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel_opt.py --config /glade/derecho/scratch/wchapman/miles_branchs/CESM_Spatial_PS_WXmod/be21_coupled-v2025.2.0.yml --input_shape 1 136 1 192 288 --forcing_shape 1 6 1 192 288 --output_shape 1 145 1 192 288 --device cuda --init_noise 1 --model_name checkpoint.pt00275.pt

# python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS_WXmod/applications/Quick_Climate_Year_Slow_parrallel_opt.py --config /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/be21_coupled-v2025.2.0_climoCO2.yml --input_shape 1 136 1 192 288 --forcing_shape 1 6 1 192 288 --output_shape 1 145 1 192 288 --device cuda --init_noise 1 --model_name checkpoint.pt00189.pt

# conda deactivate 
# conda activate npl-2024b
# python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS_WXmod/applications/Quick_Climate_Year_Slow_parrallel_opt.py /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/be21_coupled-v2025.2.0_climoCO2.yml 1D --variables U V T Qtot PS PRECT TREFHT TS TAUX TAUY --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHTTAUXTAUY --rescale_it False

# module load conda
# conda activate npl-2024b
# python /glade/derecho/scratch/wchapman/miles_branchs/CESM_Spatial_PS_WXmod/climate/Post_Process_Parallel.py /glade/derecho/scratch/wchapman/miles_branchs/CESM_Spatial_PS_WXmod/be21_coupled-v2025.2.0.yml 1D --variables U V T Qtot PS PRECT TREFHT TS TAUX TAUY --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHTTAUXTAUY --rescale_it False


# python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/climate/Post_Process_Parallel.py /glade/derecho/scratch/wchapman/miles_branchs/CESM_Spatial_PS_WXmod/be21_coupled-v2025.2.0_climoSST_2K_newice.yml 1D --variables U V T Qtot PS PRECT TREFHT TS TAUX TAUY --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHTTAUXTAUY --rescale_it False

# module load conda
# conda activate npl-2024b
# python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/climate/Post_Process_Parallel.py /glade/derecho/scratch/wchapman/miles_branchs/CESM_Spatial_PS_WXmod/be21_coupled-v2025.2.0_climoSST_2K.yml 1D --variables U V T Qtot PS PRECT TREFHT TS TAUX TAUY --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHTTAUXTAUY --rescale_it False

# # python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel_opt.py --config /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/be21_coupled-v2025.2.0.yml --input_shape 1 136 1 192 288 --forcing_shape 1 6 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00172.pt

# # python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel_opt.py --config /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/be21_coupled-v2025.2.0.yml --input_shape 1 136 1 192 288 --forcing_shape 1 6 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00173.pt

# # python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel_opt.py --config /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/be21_coupled-v2025.2.0.yml --input_shape 1 136 1 192 288 --forcing_shape 1 6 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00174.pt

# # python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel_opt.py --config /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/be21_coupled-v2025.2.0.yml --input_shape 1 136 1 192 288 --forcing_shape 1 6 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00175.pt

# # python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel_opt.py --config /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/be21_coupled-v2025.2.0.yml --input_shape 1 136 1 192 288 --forcing_shape 1 6 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00176.pt

# # torchrun /glade/work/wchapman/miles_branchs/credit_feb15_2024/applications/Quick_Climate.py --config /glade/work/wchapman/miles_branchs/credit_feb15_2024/config/climate_rollout.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00240.pt --init_noise 1 

# # python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00283.pt


# # python ./climate/Post_Process_Parallel.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml 1D --variables PS T U V Qtot --reset_times False --dask_do False --name_string PSTUVQtot --rescale_it True --n_processes 32


# python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00289.pt

# python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00290.pt

# python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00291.pt

# python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00292.pt

# python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00293.pt

# python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00294.pt

# python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00295.pt

# python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00296.pt

# python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00297.pt

# python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00298.pt

# python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00299.pt

# python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00300.pt

# python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00301.pt
