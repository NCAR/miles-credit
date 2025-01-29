"""
Author: Will Chapman
Contact: wchapman@ucar.edu

This script takes the gathered ERA5 zarr data and creates the scaling libraries
"""

# Scientific python
import xarray as xr
import matplotlib.pyplot as plt
import bokeh

print("bokeh version: ", bokeh.__version__)

############# params ######################

ZARR = "/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/TOTAL_2010-01-01_2010-12-31_staged.zarr"
varlist = ["U", "V", "T", "Q", "t2m", "SP", "Q500", "Z500", "T500", "V500", "U500"]

###########################################

# Dask
if "client" in locals():
    client = locals()["client"]
    client.shutdown()
    print("...shutdown client...")
else:
    print("client does not exist yet")

from distributed import Client
from dask_jobqueue import PBSCluster

cluster = PBSCluster(
    account="NAML0001",
    walltime="12:00:00",
    cores=1,
    memory="100GB",
    shared_temp_directory="/glade/scratch/wchapman/tmp",
    queue="main",
)
cluster.scale(jobs=40)
client = Client(cluster)
client


if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (5, 5)
    plt.rcParams["image.interpolation"] = "none"

    def get_forward_data(filename: str = ZARR) -> xr.DataArray:
        """Lazily opens the Zarr store on gladefilesystem."""
        dataset = xr.open_zarr(filename, consolidated=True)
        return dataset

    forcing_data = get_forward_data().unify_chunks()

    # open the normalization dictionary:
    DSm = xr.open_dataset(
        "/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/All_2010_staged.mean.Lev.SLO.nc"
    )
    DSs = xr.open_dataset(
        "/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/All_2010_staged.std.Lev.SLO.nc"
    )
    # normalize the forcing:
    forcing_data_scaled = (forcing_data - DSm) / DSs

    for vardo in varlist:
        forcing_data_diff = (
            forcing_data_scaled[vardo].diff("time").to_dataset(name=vardo)
        )
        print("diff done")
        if vardo in ["U", "V", "T", "Q"]:
            Mean_latlonlev = forcing_data_diff.mean(
                ["time", "latitude", "longitude"]
            ).persist()
        else:
            Mean_latlonlev = forcing_data_diff.mean(
                ["time", "latitude", "longitude"]
            ).persist()
        print("...moving on to load...")
        Mean_latlonlev = Mean_latlonlev.load()
        print("...mean done...")
        print("...saving...")
        Mean_latlonlev.to_netcdf(
            f"/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/All_NORMtend_{vardo}_2010_staged.mean.nc"
        )
        del forcing_data_diff

    for vardo in varlist:
        forcing_data_diff = (
            forcing_data_scaled[vardo].diff("time").to_dataset(name=vardo)
        )
        print("diff done")
        if vardo in ["U", "V", "T", "Q"]:
            STD_latlonlev = forcing_data_diff.std(
                ["time", "latitude", "longitude"]
            ).persist()
        else:
            STD_latlonlev = forcing_data_diff.std(
                ["time", "latitude", "longitude"]
            ).persist()
        print("...moving on to load...")
        STD_latlonlev = STD_latlonlev.load()
        print("...STD done...")
        print("...saving...")
        STD_latlonlev.to_netcdf(
            f"/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/All_NORMtend_{vardo}_2010_staged.STD.nc"
        )
        del forcing_data_diff
