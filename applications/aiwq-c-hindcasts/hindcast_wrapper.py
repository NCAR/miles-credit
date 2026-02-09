import os
import sys
import yaml
from credit.applications import rollout_realtime
from argparse import ArgumentParser
import multiprocessing as mp
import pandas as pd
from os.path import join
import time
import torch
from credit.parser import credit_main_parser

start_time = time.time()

LEAD = 46
N_MEMBERS = 11

parser = ArgumentParser(description="Rollout AI S2S Hindcast")
parser.add_argument("-d", "--date", help="initialization date")
parser.add_argument("-c", "--config", help="Path to config file")
parser.add_argument("-m", "--member", help="ensemble member, e.g. 001", default=None)
args = parser.parse_args()
with open(args.config) as cf:
    conf = yaml.safe_load(cf)
conf = credit_main_parser(conf, parse_training=False, parse_predict=True, print_summary=False)
init_time = pd.to_datetime(args.date)
init = init_time.strftime("%Y-%m-%d")

conf['predict']['forecasts']['start_year'] = int(init[:4])
conf['predict']['forecasts']['start_month'] = int(init[5:7])
conf['predict']['forecasts']['start_day'] = int(init[-2:])
conf['predict']['forecasts']['days'] = LEAD

conf['predict']['realtime']['forecast_start_time'] = init_time.strftime("%Y-%m-%d %H:%M:%S")
conf['predict']['realtime']['forecast_end_time'] = (init_time + pd.Timedelta(days=LEAD)).strftime("%Y-%m-%d %H:%M:%S")
print("FORECAST START TIME TIME", conf['predict']['realtime']['forecast_start_time'])
print("FORECAST END TIME", conf['predict']['realtime']['forecast_end_time'])
n_members = N_MEMBERS
conf['file_configs']['base_path'] = join(conf['file_configs']['base_path'], init)
os.makedirs(conf['file_configs']['base_path'], exist_ok=True)

if args.member:
    # Single-member mode (called from shell with CUDA_VISIBLE_DEVICES)
    members = [args.member]
    conf['predict']['mode'] = 'none'
else:
    # Original behavior: all members serially
    members = [f"{m:03}" for m in range(1, n_members + 1)]

with mp.Pool(4) as pool:
    for member in members:
        file_name = f'ICs.{member}.{init}-00000.nc'  ## format from "make_ensemble_hindcast.py"
        conf['file_configs']["member"] = member
        conf['data']['save_loc'] = join(conf['file_configs']['base_path'], file_name)
        conf['data']['save_loc_surface'] = join(conf['file_configs']['base_path'], file_name)
        conf['data']['save_loc_dynamic_forcing'] = join(conf['file_configs']['base_path'], 'CESM_dynamicforcing_'+init+'.nc')
        conf['data']['save_loc_diagnostic'] = join(conf['file_configs']['base_path'],file_name)

        print(conf['file_configs']['base_path'])
        print(conf['data']['save_loc'])
        print(conf['data']['save_loc_surface'])
        print(conf['data']['save_loc_dynamic_forcing'])
        print(conf['data']['save_loc_diagnostic'])
        print('MEMBER:', member)
        _ = rollout_realtime.predict(0, 1, conf, pool, member)
        torch.cuda.empty_cache()
        print(f"Completed member {member}.")

print(f"Completed in {(time.time() - start_time) / 60} minutes.")
