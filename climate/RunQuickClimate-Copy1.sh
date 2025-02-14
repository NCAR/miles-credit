#!/bin/bash
#PBS -N Collect_Climate_RMSE
#PBS -A NAML0001 
#PBS -l walltime=12:00:00
#PBS -o Collect_Climate_RMSE.out
#PBS -e Collect_Climate_RMSE.out
#PBS -q casper
#PBS -l select=1:ncpus=16:ngpus=1:mem=250GB
#PBS -l gpu_type=a100
#PBS -m a
#PBS -M wchapman@ucar.edu

module load conda
conda activate credit-dk-casper


# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00225.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00226.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00227.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00228.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00229.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00230.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00231.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00232.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00233.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00234.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00235.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00236.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00237.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00238.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00239.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00240.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00241.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00242.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00243.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00244.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00245.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00246.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00247.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00248.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00249.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00250.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00251.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00252.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00253.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00254.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00255.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00256.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00257.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00258.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00259.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00260.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00261.pt



