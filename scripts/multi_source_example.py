import yaml
from credit.datasets.multi_source import MultiSourceDataset
from credit.samplers import DistributedMultiStepBatchSampler
from torch.utils.data import DataLoader

with open("../config/multi_source_data.yaml", "r") as f:
    config = yaml.safe_load(f)

nfs = config["data"]["forecast_len"]
ms_dataset = MultiSourceDataset(config["data"], return_target=True)
ms_sampler = DistributedMultiStepBatchSampler(
    ms_dataset, batch_size=4, num_forecast_steps=nfs, shuffle=True, num_replicas=1, rank=0
)
ms_loader = DataLoader(ms_dataset, batch_sampler=ms_sampler, num_workers=0, pin_memory=False, prefetch_factor=None)
sample = next(iter(ms_loader))

print(sample.keys())
print(sample["era5"].keys())
print(sample["mrms"].keys())
print(sample["era5"]["input"].keys())
print(sample["mrms"]["input"].keys())
print(sample["era5"]["target"].keys())
print(sample["mrms"]["target"].keys())
print(sample["era5"]["metadata"])
print(sample["mrms"]["metadata"])
