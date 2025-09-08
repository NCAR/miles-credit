import os
import gc
import sys
import yaml
import time
import logging
import warnings
import copy
from glob import glob
from pathlib import Path
from argparse import ArgumentParser
import multiprocessing as mp
from collections import defaultdict
import cftime
from cftime import DatetimeNoLeap
import json
import pickle
import argparse

# ---------- #
# Numerics
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

# ---------- #
import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data import get_worker_info
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import profile, record_function, ProfilerActivity

# ---------- #
# credit
from credit.models import load_model, load_model_name
from credit.seed import seed_everything
from credit.loss import latitude_weights

from credit.data import (
    concat_and_reshape,
    reshape_only,
    drop_var_from_dataset,
    generate_datetime,
    nanoseconds_to_year,
    hour_to_nanoseconds,
    get_forward_data,
    extract_month_day_hour,
    find_common_indices,
)

from credit.transforms import load_transforms, Normalize_ERA5_and_Forcing
from credit.pbs import launch_script, launch_script_mpi
from credit.pol_lapdiff_filt import Diffusion_and_Pole_Filter
from credit.metrics import LatWeightedMetrics
from credit.forecast import load_forecasts
from credit.distributed import distributed_model_wrapper, setup
from credit.models.checkpoint import load_model_state
from credit.parser import credit_main_parser, predict_data_check
from credit.output import load_metadata, make_xarray, save_netcdf_increment
from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def save_task(data, meta_data, conf):
    """Wrapper function for saving data in parallel."""
    darray_upper_air, darray_single_level, init_datetime_str, lead_time, forecast_hour = data
    save_netcdf_increment(
        darray_upper_air,
        darray_single_level,
        init_datetime_str,
        lead_time * forecast_hour,
        meta_data,
        conf,
    )


class ForcingDataset(IterableDataset):
    """
    Streams dynamic forcing data in time-chunks, moves them to GPU asynchronously.
    """
    def __init__(self, ds, df_vars, start, num_ts, chunk_size):
        self.ds = ds
        self.df_vars = df_vars
        self.start = start
        self.num_ts = num_ts
        self.chunk = chunk_size

    def __iter__(self):
        for block_start in range(self.start, self.start + self.num_ts, self.chunk):
            block_end = min(block_start + self.chunk, self.start + self.num_ts)
            arr = self.ds.isel(time=slice(block_start, block_end)).load().values
            cpu_tensor = torch.from_numpy(arr).pin_memory()
            gpu_tensor = cpu_tensor.to(self.ds.encoding.get('device', torch.device('cuda')), non_blocking=True)
            yield gpu_tensor


def shift_input_for_next(x, y_pred, history_len, varnum_diag, static_dim):
    """
    Roll the input tensor forward by one time-step using cached dimension values.
    """
    if history_len == 1:
        if varnum_diag > 0:
            return y_pred[:, :-varnum_diag, ...].detach()
        else:
            return y_pred.detach()
    else:
        if static_dim == 0:
            x_detach = x[:, :, 1:, ...].detach()
        else:
            x_detach = x[:, :-static_dim, 1:, ...].detach()
        if varnum_diag > 0:
            return torch.cat([x_detach, y_pred[:, :-varnum_diag, ...].detach()], dim=2)
        else:
            return torch.cat([x_detach, y_pred.detach()], dim=2)


def run_year_rmse(p, config, input_shape, forcing_shape, output_shape, device, model_name=None, init_noise=None, save_append=None):
    # Load and parse configuration
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
    conf = credit_main_parser(conf, parse_training=False, parse_predict=True, print_summary=False)
    conf["predict"]["mode"] = None

    if save_append:
        base = conf["predict"].get("save_forecast")
        if not base:
            raise KeyError("'save_forecast' missing in config")
        conf["predict"]["save_forecast"] = str(Path(base).expanduser() / save_append)

    # Cache these once for speed
    history_len = conf["data"]["history_len"]
    varnum_diag = len(conf["data"]["diagnostic_variables"])
    static_dim = len(conf["data"]["static_variables"]) if not conf["data"]["static_first"] else 0

    print('...starting transform...')
    
    # Transform and model setup
    transform = load_transforms(conf)
    state_transformer = Normalize_ERA5_and_Forcing(conf) if conf["data"]["scaler_type"] == "std_new" else _not_supported()
    model = (load_model_name(conf, model_name, load_weights=True) if model_name else load_model(conf, load_weights=True)).to(device)
    truth_field = torch.load(conf['predict']['seasonal_mean_fast_climate'], map_location=torch.device(device))
    
    print('...loading model...')
    
    distributed = conf["predict"]["mode"] in ["ddp", "fsdp"]
    if distributed:
        model = distributed_model_wrapper(conf, model, device)
        if conf["predict"]["mode"] == "fsdp": model = load_model_state(conf, model, device)
    model.eval()

    x = torch.load(conf['predict']['init_cond_fast_climate'], map_location=torch.device(device)).to(device)

    if init_noise is not None:
        print('adding forecast noise')
        # Define the standard deviation for the noise (e.g., 0.01)
        noise_std = 0.05
        
        # Generate random noise tensor with the same shape as `x`
        # Use `torch.randn` for a normal distribution with mean=0 and std=1
        noise = torch.randn_like(x) * noise_std

        # Add the noise to `x`
        x = x + noise.to(device)
    
    # Post-processing flags (cached)
    post_conf = conf["model"]["post_conf"]
    flag_mass = post_conf["activate"] and post_conf["global_mass_fixer"]["activate"]
    flag_water = post_conf["activate"] and post_conf["global_water_fixer"]["activate"]
    flag_energy = post_conf["activate"] and post_conf["global_energy_fixer"]["activate"]
    if flag_mass: opt_mass = GlobalMassFixer(post_conf)
    if flag_water: opt_water = GlobalWaterFixer(post_conf)
    if flag_energy: opt_energy = GlobalEnergyFixer(post_conf)

    print('...done with fixers...')

    # Load forcing data
    df_vars = conf["data"]["dynamic_forcing_variables"]
    sf_vars = conf["data"]["static_variables"]
    num_ts = conf["predict"]["timesteps_fast_climate"]
    lead_time_periods = conf["data"]["lead_time_periods"]
    
    print('...loading forcing file...')
    chunk_size = conf["data"].get("forcing_chunk_size", 32)
    DSforc = xr.open_dataset(conf["predict"]["forcing_file"], chunks={"time": chunk_size})
    print('... start normalization ...')
    DSforc_norm = state_transformer.transform_dataset(DSforc)
    metrics = LatWeightedMetrics(conf)
    latlons = xr.open_dataset(conf["loss"]["latitude_weights"])
    print('...doing static loading...')

    # STATIC: load once
    static_arr = np.stack([DSforc[s].values for s in sf_vars], axis=0)
    static_gpu = (torch.from_numpy(static_arr).unsqueeze(0)).unsqueeze(2).to(device, non_blocking=True)

    print('shape static gpu:', static_gpu.shape)

    print('...doing dynamic loading...')

    # DYNAMIC: chunk + loader
    init_dt = conf['predict']['start_datetime']
    DSforc_norm = DSforc_norm.chunk({"time": chunk_size})
    dynamic_ds = DSforc_norm[df_vars]
    
    loc = dynamic_ds.indexes['time'].get_loc(conf['predict']['start_datetime'])
    if isinstance(loc, slice):
        start_ix = loc.start
    else:
        start_ix = loc
    # Or, if you want the nearest match, do something like:
    # import numpy as np
    # deltas = np.array([(t - init_dt).total_seconds() for t in time_vals])
    # start_ix = int(np.argmin(np.abs(deltas)))
    
    print("Resolved start_ix =", start_ix)
    # forcing_ds = ForcingDataset(dynamic_ds, df_vars, start_ix, num_ts, chunk_size)
    # loader = DataLoader(forcing_ds, batch_size=1, num_workers=0, pin_memory=True, prefetch_factor=2)

    # Trace model
    dummy_x = torch.zeros(input_shape, device=device)
    model = torch.jit.trace(model, dummy_x)

    # Run & save
    forecast_hour = 1
    print(conf['predict']['start_datetime'])

    #datetime handling:
    raw_ds = conf['predict']['start_datetime']
    if isinstance(raw_ds, str):
        # parse the “YYYY-MM-DD HH:MM:SS” string
        init_dt = datetime.strptime(raw_ds, '%Y-%m-%d %H:%M:%S')
    else:
        init_dt = raw_ds  # already datetime-like

    # put it back into conf so later code can use it
    conf['predict']['start_datetime'] = init_dt
    
    init_str = conf['predict']['start_datetime'].strftime('%Y-%m-%dT%HZ')
    test_tensor_rmse = torch.zeros(output_shape).to(device)

    kk = 0 
    loop = 0
    for block_start in range(start_ix, start_ix + num_ts, chunk_size):
        block_end = min(block_start + chunk_size, start_ix + num_ts)
        # Pure xarray slice → NumPy
        # arr = dynamic_ds.isel(time=slice(block_start, block_end)).values

        ds_slice = dynamic_ds.isel(time=slice(block_start, block_end)).load()
        ds_slice_times = ds_slice['time'].values
        # build a list of 2D arrays per time-step per var
        arr_list = [ds_slice[var].values  for var in dynamic_ds.data_vars]
        # each ds_slice[var].values has shape (t_block, H, W)
        # stack them into shape (t_block, n_dyn, H, W)
        arr = np.stack(arr_list, axis=1)

        # Torchify & transfer once per block
        cpu = torch.from_numpy(arr).unsqueeze(2).pin_memory()
        gpu = cpu.to(device, non_blocking=True)

        for t in range(gpu.shape[0]):

            cftime_obj =ds_slice_times[t]
            cftime_str = str(cftime_obj)
            utc_datetime = datetime.strptime(cftime_str, '%Y-%m-%d %H:%M:%S')
                
            if (kk+1)%20 == 0:
                print(f'model step: {kk:05}, time: {utc_datetime}')
            dyn_t = gpu[t].unsqueeze(0)
            # build input
            if kk !=0:
                if conf["data"]["static_first"]:
                    x_forc = torch.cat((static_gpu, dyn_t), dim=1)
                else:
                    x_forc = torch.cat((dyn_t, static_gpu), dim=1)
                x = torch.cat((x, x_forc), dim=1)
                # inference
            
            with torch.no_grad(): pred = model(x.float())
            kk+=1
            # fixers
            if flag_mass: pred = opt_mass({"y_pred": pred, "x": x})["y_pred"]
            if flag_water: pred = opt_water({"y_pred": pred, "x": x})["y_pred"]
            if flag_energy: pred = opt_energy({"y_pred": pred, "x": x})["y_pred"]

            
            up, sl = make_xarray(pred.cpu(), utc_datetime, latlons.latitude.values,
            latlons.longitude.values,
            conf)
            p.apply_async(save_netcdf_increment, (up, sl, init_str, lead_time_periods * forecast_hour, load_metadata(conf), conf))
            
            # shift input
            x = shift_input_for_next(x, pred, history_len, varnum_diag, static_dim)
            forecast_hour += 1
            test_tensor_rmse.add_(pred)

    if model_name is not None:
        fsout = f'{os.path.basename(config)}_{model_name}'
    else:
        fsout = os.path.basename(config)
    
    METS = metrics(test_tensor_rmse.cpu()/kk,truth_field.cpu())

    #save out: test_tensor_rmse/num_ts, MET, plots  
    with open(f'{conf["save_loc"]}/{fsout}_quick_climate_METS.pkl', 'wb') as f:
        pickle.dump(METS, f)

    torch.save(test_tensor_rmse.cpu()/kk,f'{conf["save_loc"]}/{fsout}_quick_climate_avg_tensor.pt')

    fig, axes = plt.subplots(2, 3, figsize=(25, 15))

    p1 = axes[0,0].pcolor(test_tensor_rmse.squeeze()[-16, :, :].cpu() / kk, vmin=-2, vmax=2, cmap='RdBu_r')
    fig.colorbar(p1, ax=axes[0,0])
    axes[0,0].set_title('Test Tensor RMSE')
        
    p2 = axes[0,1].pcolor(truth_field.squeeze()[-16, :, :].cpu(), vmin=-2, vmax=2, cmap='RdBu_r')
    fig.colorbar(p2, ax=axes[0,1])
    axes[0,1].set_title('Truth Field')
        
    p3 = axes[0,2].pcolor((test_tensor_rmse.squeeze()[-16, :, :].cpu() / kk) - truth_field.squeeze()[-16, :, :].cpu(), vmin=-0.1, vmax=0.1, cmap='RdBu_r')
    fig.colorbar(p3, ax=axes[0,2])
    axes[0,2].set_title('Difference (RMSE - Truth)')
        
    p1 = axes[1,0].pcolor(test_tensor_rmse.squeeze()[-15, :, :].cpu() / kk, vmin=-2, vmax=2, cmap='RdBu_r')
    fig.colorbar(p1, ax=axes[1,0])
    axes[1,0].set_title('Test Tensor RMSE')
        
    p2 = axes[1,1].pcolor(truth_field.squeeze()[-15, :, :].cpu(), vmin=-2, vmax=2, cmap='RdBu_r')
    fig.colorbar(p2, ax=axes[1,1])
    axes[1,1].set_title('Truth Field')
        
    p3 = axes[1,2].pcolor((test_tensor_rmse.squeeze()[-15, :, :].cpu() / kk) - truth_field.squeeze()[-15, :, :].cpu(), vmin=-0.5, vmax=0.5, cmap='RdBu_r')
    fig.colorbar(p3, ax=axes[1,2])
    axes[1,2].set_title('Difference (RMSE - Truth)')
    
    plt.tight_layout()
    plt.savefig(f'{conf["save_loc"]}/{fsout}_quick_climate_plot_slow.png', bbox_inches='tight')
    plt.show()

    
    return test_tensor_rmse, truth_field, metrics, conf, METS



def main():
    parser = argparse.ArgumentParser(description="Run year RMSE for CAMULATOR model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the model configuration YAML file.')
    parser.add_argument('--input_shape', type=int, nargs='+', required=True, help='Input shape as a list of integers.')
    parser.add_argument('--forcing_shape', type=int, nargs='+', required=True, help='Forcing shape as a list of integers.')
    parser.add_argument('--output_shape', type=int, nargs='+', required=True, help='Output shape as a list of integers.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (cuda or cpu).')
    parser.add_argument('--model_name', type=str, default=None, help='Optional model checkpoint name.')
    parser.add_argument('--init_noise', type=int, default=None, help='init model noise')
    parser.add_argument('--save_append', type=str, default=None, help='append a folder to the output to save to')

    args = parser.parse_args()

    start_time = time.time()

    # # Call the run_year_rmse function with parsed arguments
    # test_tensor_rmse, truth_field, inds_to_rmse, metrics, conf, METS = run_year_rmse(
    #     config=args.config,
    #     input_shape=args.input_shape,
    #     forcing_shape=args.forcing_shape,
    #     output_shape=args.output_shape,
    #     device=args.device,
    #     model_name=args.model_name
    # )
    
    num_cpus = 8
    with mp.Pool(num_cpus) as p:
        run_year_rmse(p, config=args.config, input_shape=args.input_shape,
        forcing_shape=args.forcing_shape,
        output_shape=args.output_shape,
        device=args.device,
        model_name=args.model_name,
        init_noise=args.init_noise,
        save_append=args.save_append)

    end_time = time.time()
    elapsed_time = end_time - start_time
    # How to run:
    # python Quick_Climate_Year.py --config /path/to/config.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt
    print(f"Run completed. Results saved to configured location. Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Run completed. Results saved to configured location. Elapsed time: {elapsed_time/60:.2f} minutes")


if __name__ == "__main__":
    main()
