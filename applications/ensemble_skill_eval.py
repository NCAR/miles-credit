# ensemble evaluation suite for rollouts generated with rollout_to_netcdf.py
# parallelizes across CPUs
# Calculates standard deterministic metrics on the ensemble mean, 
# AND spatial spread/error FFTs directly from the ensemble members.

from argparse import ArgumentParser
import logging
import os
from os.path import join
from pathlib import Path
import sys
import multiprocessing as mp
import tempfile
import glob

import numpy as np
import pandas as pd
import xarray as xr
from scipy import fftpack
import yaml

from credit.pbs import launch_script, launch_script_mpi, get_num_cpus
from credit.verification import load_verification
from credit.datasets.goes_load_dataset_and_dataloader import load_verification_dataset
from dateutil.parser import parse, ParserError


def is_timestamp(s):
    try:
        parse(s)
        return True
    except (ParserError, TypeError):
        return False


def compute_radial_spectrum(field2d):
    """Computes the 1D radially averaged power spectrum of a 2D spatial field."""
    f_transform = fftpack.fft2(field2d)
    f_shift = fftpack.fftshift(f_transform)
    power_spectrum = np.abs(f_shift)**2

    y, x = np.indices(power_spectrum.shape)
    center = (power_spectrum.shape[0] // 2, power_spectrum.shape[1] // 2)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)

    tbin = np.bincount(r.ravel(), power_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / nr
    return radial_profile


if __name__ == "__main__":
    description = "evaluate ensemble rollouts and calculate spread/error FFTs"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c", dest="eval_config", type=str, default=False,
        help="Path to the model configuration (yml) containing your inputs."
    )
    parser.add_argument(
        "-l", dest="launch", type=int, default=0,
        help="Submit workers to PBS."
    )
    # --- NEW: Command line flag to skip FSS ---
    parser.add_argument(
        "--no-fss", action="store_true", 
        help="Disable Fractions Skill Score (FSS) computation to speed up evaluation."
    )

    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("eval_config")
    launch = int(args_dict.pop("launch"))
    skip_fss = args_dict.pop("no_fss")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
    with open(conf["config"]) as cf:
        model_conf = yaml.load(cf, Loader=yaml.FullLoader)

    # --- NEW: Override YAML config if flag is passed ---
    if skip_fss:
        conf["compute_fss"] = False
        logging.info("FSS computation DISABLED via command line flag.")

    forecast_save_loc = conf.get("save_forecast", None)
    if not forecast_save_loc:
        logging.warning("save_forecast not specified in eval config, using forecast_save_loc in model config")
        forecast_save_loc = model_conf["predict"].get("save_forecast", None)
        if not forecast_save_loc:
            forecast_save_loc = os.path.expandvars(join(conf["save_loc"], "forecasts"))
            logging.warning("save_forecast not specified in model config, using default location")
    
    logging.info(f"evaluating forecast at {forecast_save_loc}")
    
    conf["save_filename"] = conf.get("save_filename", "verif.parquet")
    eval_save_loc = join(forecast_save_loc, conf["save_filename"])
    if not conf.get("overwrite", False):
        assert not os.path.isfile(eval_save_loc), (
                f'''{conf["save_filename"]} results already exists at {eval_save_loc}, aborting.'''
        )
    else:
        logging.warning(f"potentially overwriting existing file {conf['save_filename']}")

    dirs = [d for d in Path(forecast_save_loc).iterdir() if d.is_dir() and is_timestamp(d.name)]
    logging.info(f"in these dirs: {[d.name for d in dirs]}")

    verification = load_verification(conf)

    if launch:
        script_path = Path(__file__).absolute()
        if conf["pbs"]["queue"] == "casper":
            logging.info("Launching to PBS on Casper")
            launch_script(config, script_path)
        else:
            logging.info("Launching to PBS on Derecho")
            launch_script_mpi(config, script_path)
        sys.exit()

    num_process = conf.get("num_process", max(1, get_num_cpus() - 1))
    dataset = load_verification_dataset(model_conf)
    climo = xr.open_dataset(conf["climo_file"])["mean"]

    with mp.Pool(num_process) as p:
        df_dict = {}
        for dir in dirs:
            init_time = dir.name
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                
                # Sort files to ensure forecast_step increments properly
                nc_files = sorted(glob.glob(join(dir, "*.nc")))
                ensemble_extra_metrics = []

                # 1. Process files: Collapse ensemble & calculate new spatial FFT metrics
                for step_idx, nc_file in enumerate(nc_files):
                    forecast_step = step_idx + 1 # 1-based indexing
                    file_name = Path(nc_file).name
                    ds_ens = xr.open_dataset(nc_file)
                    
                    # Safe truth dataset extraction
                    valid_time = pd.to_datetime(ds_ens.t.values[0])
                    truth_ds = dataset.ds if hasattr(dataset, "ds") else dataset
                    
                    step_metrics = {"forecast_step": float(forecast_step)}
                    
                    if "ensemble_member_label" in ds_ens.dims:
                        # Save the mean for the deterministic verif script
                        ds_mean = ds_ens.mean(dim="ensemble_member_label")
                        ds_mean.attrs = ds_ens.attrs
                        ds_mean.to_netcdf(join(temp_dir_path, file_name))
                        
                        # Loop through available channels dynamically
                        for ch in ds_ens.channel.values:
                            ch_str = f"C{int(ch):02d}"
                            try:
                                ens_var = ds_ens["BT_or_R"].sel(channel=ch).squeeze("t")
                                spread_field = ens_var.std(dim="ensemble_member_label").values
                                mean_field = ens_var.mean(dim="ensemble_member_label").values
                                
                                # Pull matching truth slice
                                if ch_str in truth_ds.data_vars:
                                    truth_val = truth_ds[ch_str].sel(t=valid_time, method="nearest").values
                                else:
                                    truth_val = truth_ds["BT_or_R"].sel(channel=ch, t=valid_time, method="nearest").values

                                # Calculate Error (Forecast - Truth)
                                error_field = mean_field - truth_val
                                
                                # Empirical Spectral Variance (Spread)
                                deviations = ens_var.values - mean_field
                                member_spectra = []
                                for i in range(deviations.shape[0]):
                                    member_spectrum = compute_radial_spectrum(deviations[i])
                                    member_spectra.append(member_spectrum)
                                empirical_spectral_variance = np.mean(member_spectra, axis=0)

                                # Append custom metrics
                                step_metrics[f"Spread_{ch_str}"] = float(np.mean(spread_field))
                                step_metrics[f"FFT_Spread_{ch_str}"] = empirical_spectral_variance
                                step_metrics[f"FFT_Error_{ch_str}"] = compute_radial_spectrum(error_field)
                            
                            except KeyError:
                                logging.warning(f"Could not calculate ensemble metrics for {ch_str} at valid time {valid_time}")
                    else:
                        # Fallback if the file is already deterministic
                        ds_ens.to_netcdf(join(temp_dir_path, file_name))
                        
                    ensemble_extra_metrics.append(step_metrics)
                    ds_ens.close()

                # 2. Run standard verification on the temporary deterministic means
                result = verification(temp_dir_path, p, conf, model_conf, dataset, climo)
 
                # 3. Merge the custom ensemble metrics into the result dictionaries
                for det_dict in result:
                    step = det_dict.get("forecast_step")
                    match = next((x for x in ensemble_extra_metrics if x["forecast_step"] == step), None)
                    if match:
                        det_dict.update(match)

            # Reformat, sort, and save to intermediate parquet
            result.sort(key= lambda x: x["forecast_step"])
            df = pd.DataFrame(result)
            df.attrs["init_times"] = init_time
            df_dict[init_time] = df

            intermediate_eval_save_loc = join(forecast_save_loc, f"verif_{init_time}.parquet")
            df.to_parquet(intermediate_eval_save_loc) 
            logging.info(f"saved verification of {init_time} to {intermediate_eval_save_loc}")

    # 4. Take nanmean of all verifications (Handles 1D FFT arrays automatically!)
    def nanmean_cell(*values):
        arrays = [np.atleast_1d(np.array(v, dtype=float)) for v in values]
        result = np.nanmean(arrays, axis=0)
        return result[0] if result.size == 1 else result
    
    dfs = list(df_dict.values())
    df = pd.DataFrame(
        {col: [nanmean_cell(*values) for values in zip(*[df[col] for df in dfs])]
        for col in dfs[0].columns},
        index=dfs[0].index
    )
    df.attrs["init_times"] = list(df_dict.keys())

    df.to_parquet(eval_save_loc)
    logging.info(f"saved ensemble aggregate verification to {eval_save_loc}")

    p.close()
    p.join()