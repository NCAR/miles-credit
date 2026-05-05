import yaml
import pathlib
from credit.datasets.base_dataset import BaseDataset
from credit.samplers import DistributedMultiStepBatchSampler

from torch.utils.data import DataLoader

BASE_DIR = pathlib.Path(__file__).resolve().parent
config_path = BASE_DIR.parent / "config" / "gen_2" / "examples" / "base_dataset.yml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Build the BaseDataset instance
base_dataset = BaseDataset(config["data"], return_target=True)

# Sanity prints
print(f"Dataset length: {len(base_dataset)}")
print(f"Dataset datetimes: {base_dataset.datetimes}")
print(f"Dataset file_dict keys: {base_dataset.file_dict.keys()}")
print(f"Dataset var_dict keys: {base_dataset.var_dict.keys()}")


# Now actually try to sample from the BaseDataset
nfs = base_dataset.num_forecast_steps
ms_sampler = DistributedMultiStepBatchSampler(
    base_dataset, batch_size=4, num_forecast_steps=nfs, shuffle=True, num_replicas=1, rank=0
)
base_loader = DataLoader(base_dataset, batch_sampler=ms_sampler, num_workers=0, pin_memory=False, prefetch_factor=None)
sample = next(iter(base_loader))

# Sanity Prints
print(sample.keys())
print(sample["input"].keys())
print(sample["target"].keys())
print(sample["metadata"])
print(sample["metadata"]["input_datetime"])
print(sample["metadata"]["target_datetime"])
