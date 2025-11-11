import pytest
import torch
import yaml
from pathlib import Path

import os

from credit.datasets.goes_load_dataset_and_dataloader import load_dataset
from credit.transform import load_transform


TEST_FILE_DIR = "/".join(os.path.abspath(__file__).split("/")[:-1])
CONFIG_FILE_DIR = os.path.join(
    "/".join(os.path.abspath(__file__).split("/")[:-2]), "config"
)

def test_dataset():
    config = os.path.join(CONFIG_FILE_DIR, "goes_era5_forcing_test.yml")

    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    transform = load_transform(conf, "ERA5", "cpu")

    dataset = load_dataset(conf, 0, 1, era5_transform=transform, is_train=True)

    init_times = dataset.init_times
    print(init_times.values)
    
    data = dataset[(init_times[0], "init")]
    
    print(f'''seconds delta {data["era5"]["timedelta_seconds"]}''')
    print(data["datetime"])
    print(data["era5"].keys())
    print(data["era5"]["prognostic"].shape)
    print(data["era5"]["prognostic"].max())
    print(data["era5"]["static"].shape)
    print(data["era5"]["dynamic_forcing"].shape)


if __name__ == "__main__":
    test_dataset()