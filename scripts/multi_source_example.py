import yaml
import pathlib
from credit.datasets.multi_source import MultiSourceDataset
from credit.samplers import DistributedMultiStepBatchSampler
from torch.utils.data import DataLoader
from credit.preblock import build_preblocks, apply_preblocks
from credit.postblock import build_postblocks, apply_postblocks
import torch
import numpy as np

BASE_DIR = pathlib.Path(__file__).resolve().parent
config_path = BASE_DIR.parent / "config" / "gen_2" / "examples" / "multi_source_data.yml"
# config_path = BASE_DIR.parent / "config" / "multi_source_data_local.yml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

nfs = config["data"]["forecast_len"]
ms_dataset = MultiSourceDataset(config["data"], return_target=True)
ms_sampler = DistributedMultiStepBatchSampler(
    ms_dataset, batch_size=4, num_forecast_steps=nfs, shuffle=True, num_replicas=1, rank=0
)
ms_loader = DataLoader(ms_dataset, batch_sampler=ms_sampler, num_workers=0, pin_memory=False, prefetch_factor=None)
sample = next(iter(ms_loader))
print(sample.keys())
print(sample["input"].keys())
print(sample["input"]["ERA5_UpperAir"].keys())
preblocks = build_preblocks(config["preblocks"])
batch = apply_preblocks(preblocks, sample)
print("BATCH KEYS:", batch.keys())
print("Input shape:", batch["x"].shape)
print("Target shape:", batch["y"].shape)
target_shape = batch["y"].shape

print("METADATA KEYS:", batch["metadata"].keys())
print("METADATA KEYS:", batch["metadata"]["input"].keys())
print("METADATA KEYS:", batch["metadata"]["input"]["ERA5_LowerAir"].keys())
print("METADATA KEYS:", batch["metadata"]["input"]["ERA5_LowerAir"]["datetime"])

fake_prediction = torch.tensor(np.random.normal(0, 1, target_shape))
batch["y_pred"] = fake_prediction
sample["y_pred"] = fake_prediction

postblocks = build_postblocks(config["postblocks"])
postbatch = apply_postblocks(postblocks, batch)
print("POSTBATCH KEYS:", postbatch.keys())
print(postbatch["y_pred"].shape)
