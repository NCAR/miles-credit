#!/bin/bash
#PBS -N Post_Process_XXX_Clim_Run
#PBS -A NAML0001 
#PBS -l walltime=12:00:00
#PBS -o upper_pp_clim_00XXX.out
#PBS -e upper_pp_clim_00XXX.out
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=50GB
#PBS -m a
#PBS -M wchapman@ucar.edu

module load conda
conda activate npl-2023b


python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/climate/Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_00240/model_multi_example-v2025.2.0.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT --rescale_it True

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/climate/Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_00223/model_multi_example_c-v2025.2.0.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT --rescale_it False

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/climate/Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_00223/model_multi_example_d-v2025.2.0.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT --rescale_it True

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/climate/Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_00223/model_multi_example_e-v2025.2.0.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/climate/Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_00223/model_multi_example_f-v2025.2.0.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/climate/Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_00223/model_multi_example_g-v2025.2.0.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/climate/Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_00223/model_multi_example_h-v2025.2.0.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/climate/Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_00223/model_multi_example_i-v2025.2.0.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/climate/Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_00223/model_multi_example_j-v2025.2.0.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT

# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/climate/Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_00223/model_multi_example_k-v2025.2.0.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT


# python /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/climate/Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_00223/model_multi_example-v2025.2.0.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT



#/glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb/model_00223/model_multi_example-v2025.2.0.yml

# python Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00187/model_multi_WxFormer.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT

# python Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00186/model_multi_WxFormer.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT

# python Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00185/model_multi_WxFormer.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT

# python Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00184/model_multi_WxFormer.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT

# python Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00183/model_multi_WxFormer.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT

# python Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00182/model_multi_WxFormer.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT

# python Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00181/model_multi_WxFormer.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT

# python Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00188/model_multi_WxFormer.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT

# python Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00189/model_multi_WxFormer.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT

# python Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00168/model_multi_WxFormer.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT

# python Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00167/model_multi_WxFormer.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT
