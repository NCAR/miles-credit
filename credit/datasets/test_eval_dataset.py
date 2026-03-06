import yaml
from credit.datasets.goes_load_dataset_and_dataloader import load_verification_dataset
import pandas as pd
import xarray as xr
import time

if __name__ == "__main__":

    with open("/glade/derecho/scratch/dkimpara/goes_10km_train/wx_big/model_continue.yml") as cf:
        model_conf = yaml.load(cf, Loader=yaml.FullLoader)

    # start = time.time()
    # ds = xr.open_dataset("/glade/derecho/scratch/dkimpara/goes-cloud-dataset/goes_10km.zarr")
    # print(time.time() - start)


    start = time.time()
    dataset = load_verification_dataset(model_conf)
    print(time.time() - start)

    batch = dataset[(pd.Timestamp("2022-06-30T23:55:06"), "y")]
    print(batch["y"])