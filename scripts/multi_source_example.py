import yaml
import pathlib

from typing import Any

from credit.datasets.multi_source import MultiSourceDataset
from credit.samplers import DistributedMultiStepBatchSampler
from torch.utils.data import DataLoader

BASE_DIR = pathlib.Path(__file__).resolve().parent
CONFIG_PATH_TYPICAL = BASE_DIR.parent / "config" / "gen_2" / "examples" / "multi_source_data.yaml"
CONFIG_PATH_HRRR = BASE_DIR.parent / "config" / "gen_2" / "examples" / "multi_source_hrrr_data.yaml"

# User can choose between the two configs
config_path = CONFIG_PATH_HRRR


def color_print(context_text: str, value: Any = "", color: str = "\033[96m") -> None:
    """
    Color the text in the terminal

    Args:
        context_text (str): The text to color
        value (Any): The value to print
        color (str): The color to print the text in (default is cyan = "\033[96m")

    Returns:
        None: Prints the text in the terminal
    """
    print(f"\n{color}{context_text}{'\033[0m'} {value}")


color_print("Loading config from", config_path)

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

color_print("Config loaded:", config)

datasets_in_config = list(config["data"]["source"].keys())
color_print("Datasets in config:", datasets_in_config)

nfs = config["data"]["forecast_len"]
ms_dataset = MultiSourceDataset(config["data"], return_target=True)
color_print("MultiSourceDataset initialized with sources:", list(ms_dataset.datasets.keys()))

ms_sampler = DistributedMultiStepBatchSampler(
    ms_dataset, batch_size=4, num_forecast_steps=nfs, shuffle=True, num_replicas=1, rank=0
)
color_print("Sampler initialized:", ms_sampler)

ms_loader = DataLoader(ms_dataset, batch_sampler=ms_sampler, num_workers=0, pin_memory=False, prefetch_factor=None)
color_print("DataLoader initialized:", ms_loader)
color_print("Loading samples (this may take a bit):")

sample = next(iter(ms_loader))
color_print("Sample keys:", sample.keys())

for dataset_name in datasets_in_config:
    color_print(f"{dataset_name} Input, Target, and Metadata:", sample[dataset_name].keys())
    color_print("Input keys:", sample[dataset_name]["input"].keys())
    color_print("Target keys:", sample[dataset_name]["target"].keys())
    color_print("Metadata:", sample[dataset_name]["metadata"])


color_print("Done!")
