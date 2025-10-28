import pytest
import torch
import yaml
from pathlib import Path

import os

from credit.datasets.goes_load_dataset_and_dataloader import load_dataset


TEST_FILE_DIR = "/".join(os.path.abspath(__file__).split("/")[:-1])
CONFIG_FILE_DIR = os.path.join(
    "/".join(os.path.abspath(__file__).split("/")[:-2]), "config"
)

def test_dataset():
    config = os.path.join(CONFIG_FILE_DIR, "goes_example.yml")

    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    dataset = load_dataset(conf, 0, 1, is_train=True)

    init_times = dataset.init_times
    
    data = dataset[(init_times[0], "init")]