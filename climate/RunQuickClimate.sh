#!/bin/bash
#PBS -N Run_Noise_Script
#PBS -A NAML0001 
#PBS -l walltime=12:00:00
#PBS -o RUN_Climate_RMSE.out
#PBS -e RUN_Climate_RMSE.out
#PBS -q casper
#PBS -l select=1:ncpus=32:ngpus=1:mem=250GB
#PBS -l gpu_type=a100
#PBS -m a
#PBS -M wchapman@ucar.edu

module load conda
conda activate credit-casper-modern

# torchrun /glade/work/wchapman/miles_branchs/credit_feb15_2024/applications/Quick_Climate.py --config /glade/work/wchapman/miles_branchs/credit_feb15_2024/config/climate_rollout.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00240.pt --init_noise 1 

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00283.pt


# python ./climate/Post_Process_Parallel.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml 1D --variables PS T U V Qtot --reset_times False --dask_do False --name_string PSTUVQtot --rescale_it True --n_processes 32


python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00289.pt

python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00290.pt

python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00291.pt

python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00292.pt

python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00293.pt

python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00294.pt

python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00295.pt

python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00296.pt

python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00297.pt

python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00298.pt

python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00299.pt

python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00300.pt

python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00301.pt