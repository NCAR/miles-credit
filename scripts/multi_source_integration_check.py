import time
import logging

# import_timer_start = time.perf_counter()

import yaml
import pathlib
import torch
import numpy as np

from credit.datasets.gen_2.multi_source import MultiSourceDataset
from credit.samplers import DistributedMultiStepBatchSampler
from torch.utils.data import DataLoader
from credit.preblock import build_preblocks, apply_preblocks
from credit.postblock import build_postblocks, apply_postblocks

# Configure logging: set level to logging.DEBUG to show GRIB profiling timers
logging.basicConfig(level=logging.INFO)

# To show GRIB profiling timers specifically without changing other loggers, set:
# logging.getLogger("credit.datasets.hrrr").setLevel(logging.DEBUG)

# import_timer_end = time.perf_counter()
# print(f"Imports completed in {import_timer_end - import_timer_start:.2f} seconds")

setup_timer_start = time.perf_counter()

BASE_DIR = pathlib.Path(__file__).resolve().parent
config_path = BASE_DIR.parent / "config" / "hrrr_emulator" / "sf_hrrr_emulator_test_small_workers.yml"
# config_path = BASE_DIR.parent / "config" / "gen_2" / "smoke" / "integration_test_hrrr.yml"
# config_path = BASE_DIR.parent / "config" / "multi_source_data_local.yml"

print(f"Using config: {config_path}")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Parse settings from config["trainer"]
batch_size = config["trainer"]["train_batch_size"]
num_workers = config["trainer"]["thread_workers"]
prefetch_factor = config["trainer"].get("prefetch_factor")
if isinstance(prefetch_factor, str) and prefetch_factor.lower() == "none":
    prefetch_factor = None

# Show the networking relevant parameters
print("Requesting the following resources:")
print("train_batch_size:       ", batch_size)
print("* valid_batch_size:     ", config["trainer"]["valid_batch_size"])
print("thread_workers:         ", num_workers)
print("* valid_thread_workers: ", config["trainer"]["valid_thread_workers"])
print("prefetch_factor:        ", prefetch_factor)
print("  * not used in this script\n")

nfs = config["data"]["forecast_len"]
ms_dataset = MultiSourceDataset(config["data"], return_target=True)

ms_sampler = DistributedMultiStepBatchSampler(
    ms_dataset, batch_size=batch_size, num_forecast_steps=nfs, shuffle=True, num_replicas=1, rank=0
)
ms_loader = DataLoader(
    ms_dataset, batch_sampler=ms_sampler, num_workers=num_workers, pin_memory=False, prefetch_factor=prefetch_factor
)

setup_timer_end = time.perf_counter()
print(f"Dataset, Sampler, and Loader setup completed in {setup_timer_end - setup_timer_start:.2f} seconds")

for source_name, source_config in config["data"]["source"].items():
    print(f"Source: {source_name}")
    if "extent" in source_config:
        print(f"  Extent: {source_config['extent']}")
    else:
        print("  Extent: Not specified")

for dataset_name, dataset in ms_dataset.datasets.items():
    print(f"Dataset: {dataset_name}")
    print(f"  Type: {type(dataset)}")
    if hasattr(dataset, "extent"):
        print(f"  Extent: {dataset.extent}")
    else:
        print("  Extent: Not specified")

iterator_setup_start = time.perf_counter()
ms_loader_iterator = iter(ms_loader)
iterator_setup_end = time.perf_counter()
print(
    f"DataLoader iterator initialized (workers spawned & connected) in {iterator_setup_end - iterator_setup_start:.2f} seconds"
)

print("Taking sample...")
sample_timer_start = time.perf_counter()
sample = next(ms_loader_iterator)
sample_timer_end = time.perf_counter()
print(f"First sample taken in {sample_timer_end - sample_timer_start:.2f} seconds")
sample_timer_start = time.perf_counter()
sample = next(ms_loader_iterator)
sample_timer_end = time.perf_counter()
print(f"Second sample taken in {sample_timer_end - sample_timer_start:.2f} seconds")

# Now with n samples serially
N_SAMPLES = 10
time_values = []

for _ in range(N_SAMPLES):
    sample_timer_start = time.perf_counter()
    sample = next(ms_loader_iterator)
    sample_timer_end = time.perf_counter()
    time_values.append(sample_timer_end - sample_timer_start)

time_values = np.array(time_values)
tot_samples_time = np.sum(time_values)
mean_per_sample_time = np.mean(time_values)
print(f"Ran for {N_SAMPLES}, which took {np.sum(time_values):.2f} seconds or ")
print(f"{np.mean(time_values):.2f} per sample (std of {np.std(time_values):.2f})")
print(f"Time values: \n{time_values}")

inspection_timer_start = time.perf_counter()
print(sample.keys())
input_keys = sample["input"].keys()

print("Input keys:", input_keys)
for source_name, source_data in sample["input"].items():
    print(f"Source: {source_name}, Keys: {source_data.keys()}")

preblocks = build_preblocks(config)
print(f"Preblocks of instance {type(preblocks)} of length {len(preblocks)} with items:\n{preblocks.items()}")
for k, v in preblocks.items():
    print(f"    {k:24} -> {v}")
batch = apply_preblocks(preblocks, sample)
print("BATCH KEYS: ", batch.keys())

input_key = "x" if "x" in batch else "input"
target_key = "y" if "y" in batch else "input"

# print(f"Input of key {input_key}: {batch[input_key]}")
# print(f"Target of key {target_key}: {batch[target_key]}")

print(f"Input shape ({input_key}):", batch[input_key].shape)
print(f"Target shape ({target_key}):", batch[target_key].shape)
target_shape = batch[target_key].shape

print("# Input is Nan:", torch.isnan(batch[input_key]).sum().item())
print("# Total in input:", batch[input_key].numel())
print("Input min value:", batch[input_key].min().item())
print("Input max value:", batch[input_key].max().item())
print("# Target is Nan:", torch.isnan(batch[target_key]).sum().item())
print("# Total in target:", batch[target_key].numel())
print("Target min value:", batch[target_key].min().item())
print("Target max value:", batch[target_key].max().item())

print("METADATA KEYS:", batch["metadata"].keys())
print("METADATA KEYS:", batch["metadata"]["input"].keys())

for source_name, source_metadata_input in batch["metadata"]["input"].items():
    print(f"METADATA KEYS: Source: {source_name}, Keys: {source_metadata_input.keys()}")
    if "datetime" in source_metadata_input:
        print(f"METADATA KEYS: Source: {source_name}, Datetime: {source_metadata_input['datetime']}")

for source_name, source_target_channel_map in batch["metadata"]["target"]["_channel_map"].items():
    print(f"METADATA CHANNEL_MAP KEYS: Source: {source_name}, Keys: {source_target_channel_map.keys()}")
    print(f"METADATA CHANNEL_MAP KEYS: Source: {source_name}, Slice: {source_target_channel_map['slice']}")

# prediction key for gen2 trainer
predicted_key = "y_pred"
batch[predicted_key] = torch.tensor(np.random.normal(0, 1, target_shape), dtype=torch.float32)

postblocks = build_postblocks(config["postblocks"])
postbatch = apply_postblocks(postblocks, batch)
print("POSTBATCH KEYS:", postbatch.keys())

print(f"POSTBARCH Prediction (key = {predicted_key}) of type: ", type(postbatch[predicted_key]))
print("Has shape: ", postbatch[predicted_key].shape)
# for source_name, postbatch_data in postbatch[predicted_key].items():
#     print(f"POSTBATCH KEYS: Source: {source_name}, Keys: {postbatch_data.keys()}")

inspection_timer_end = time.perf_counter()
print(f"Inspection completed in {inspection_timer_end - inspection_timer_start:.2f} seconds")
print("DONE\n")
