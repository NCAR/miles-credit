#!/bin/bash
#PBS -N Collect_Climate_RMSE
#PBS -A NAML0001 
#PBS -l walltime=12:00:00
#PBS -o JC32_full_Collect_Climate_RMSE.out
#PBS -e JC32_full_Collect_Climate_RMSE.out
#PBS -q casper
#PBS -l select=1:ncpus=16:ngpus=1:mem=250GB
#PBS -l gpu_type=a100
#PBS -m a
#PBS -M wchapman@ucar.edu

module load conda
conda activate credit-dk-casper

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/JOHN_cam32_full/model.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00000.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/JOHN_cam32_full/model.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00001.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/JOHN_cam32_full/model.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00002.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/JOHN_cam32_full/model.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00003.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/JOHN_cam32_full/model.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00004.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/JOHN_cam32_full/model.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00005.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/JOHN_cam32_full/model.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00006.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/JOHN_cam32_full/model.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00007.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/JOHN_cam32_full/model.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00008.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/JOHN_cam32_full/model.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00009.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/JOHN_cam32_full/model.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00010.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/JOHN_cam32_full/model.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00011.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/JOHN_cam32_full/model.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00012.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/JOHN_cam32_full/model.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00013.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/JOHN_cam32_full/model.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00014.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/JOHN_cam32_full/model.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00015.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/JOHN_cam32_full/model.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00016.pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/JOHN_cam32_full/model.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00017pt

python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/JOHN_cam32_full/model.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00018.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/JOHN_cam32_full/model.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt000019.pt


# torchrun /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_00223/model_multi_example_e-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt --init_noise 1 

# torchrun /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_00223/model_multi_example_f-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt --init_noise 1

# torchrun /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_00223/model_multi_example_h-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt --init_noise 1

# torchrun /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_00223/model_multi_example_g-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt --init_noise 1

# torchrun /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_00223/model_multi_example_i-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt --init_noise 1

# torchrun /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_00223/model_multi_example_j-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt --init_noise 1

# torchrun /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_00223/model_multi_example_k-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt --init_noise 1

# torchrun /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_00223/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt

# torchrun /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year_Slow.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_00223/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt

# torchrun /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year_SaveEvery.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt


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

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00240.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00241.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00242.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00243.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00244.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00245.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00246.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00247.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00248.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00249.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00250.pt

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/Quick_Climate_Year.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00251.pt

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



