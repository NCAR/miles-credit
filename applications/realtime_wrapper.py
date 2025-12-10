import os
import yaml
import rollout_realtime
from argparse import ArgumentParser
import multiprocessing as mp
import pandas as pd
from os.path import join
import time

# from applications.realtime_wrapper_old import n_members
# from credit.attend import exists
from credit.parser import credit_main_parser


start_time = time.time()

parser = ArgumentParser(description="Rollout Realtime AI-NWP forecasts")
parser.add_argument("-d", "--date", help="initialization date")
parser.add_argument("-c", "--config", help="Path to config file")
args = parser.parse_args()
with open(args.config) as cf:
    conf = yaml.safe_load(cf)
conf = credit_main_parser(conf, parse_training=False, parse_predict=True, print_summary=False)
init_time = pd.to_datetime(args.date)
init = init_time.strftime("%Y-%m-%d")
conf['predict']['realtime']['forecast_start_time'] = init_time.strftime("%Y-%m-%d %H:%M:%S")
conf['predict']['realtime']['forecast_end_time'] = (init_time + pd.Timedelta(days=32)).strftime("%Y-%m-%d %H:%M:%S")
print("FORECAST START TIME TIME", conf['predict']['realtime']['forecast_start_time'])
print("FORECAST END TIME", conf['predict']['realtime']['forecast_end_time'])
n_members = 341
# n_members = 10
# members = [f"{member:03}" for member in range(1, n_members + 1)]
# conf['predict']['save_forecast'] = join(conf['predict']['save_forecast'], init)
# os.makedirs(conf['predict']['save_forecast'], exist_ok=True)
conf['file_configs']['base_path'] = join(conf['file_configs']['base_path'], init_time.strftime('%Y%m%d'))
perturb = True
if perturb:
    start = 1
else:
    start = 0
# with mp.Pool(24) as pool:
with mp.Pool(12) as pool:

    for member in range(start, n_members + 1):
        if perturb:
            member = f"{member:03}"
            file_name = f'GEFS_ICs.{member}.i.{init}-00000.nc'  ## format from "make_ensemble_aiwq.py"
        else:
            if member == 0:
                file_name = f"gefs_cam_grid_c00.nc"
            else:
                file_name = f"gefs_cam_grid_p{member:02}.nc"
        conf['file_configs']["member"] = member
        conf['data']['save_loc'] =  join(conf['file_configs']['base_path'], file_name)
        conf['data']['save_loc_surface'] = join(conf['file_configs']['base_path'], file_name)
        conf['data']['save_loc_dynamic_forcing'] = join("/glade/derecho/scratch/kjmayer/CREDIT_runs/S2Socnlndatm_MLdata/AIWQ_inits/",
                                                f'b.e21.CREDIT_climate_branch_allvars_GEFSicefracandsst_{init}.zarr')
        # conf['data']['save_loc_dynamic_forcing'] = '/glade/derecho/scratch/kjmayer/CREDIT_runs/S2Socnlndatm_MLdata/b.e21.CREDIT_climate_branch_allvars_????_SWupdate.zarr'
        conf['data']['save_loc_diagnostic'] = join(conf['file_configs']['base_path'],
                                                   file_name)
        print(conf['file_configs']['base_path'])
        print(conf['data']['save_loc'])
        print(conf['data']['save_loc_surface'])
        print(conf['data']['save_loc_dynamic_forcing'])
        print(conf['data']['save_loc_diagnostic'])
        print('MEMBER:', member)
        _ = rollout_realtime.predict(0, 1, conf, pool, member)
        print(f"Completed member {member}.")

print(f"Completed in {(time.time() - start_time) / 60} minutes.")
