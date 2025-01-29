import os
import glob
import argparse

import xarray as xr

from distributed import Client
from dask_jobqueue import PBSCluster


def main(year):
    project_num = "NAML0001"  # Replace with your project key

    print("...setting up dask client...")
    # Shutdown the client if it exists
    if "client" in locals():
        client = locals()["client"]
        client.shutdown()
        print("...shutdown client...")
    else:
        print("client does not exist yet")

    # Set up the Dask cluster
    cluster = PBSCluster(
        account=project_num,
        walltime="12:00:00",
        cores=1,
        memory="70GB",
        shared_temp_directory="/glade/derecho/scratch/wchapman/tmp",
        queue="main",
    )
    cluster.scale(jobs=40)
    client = Client(cluster)
    print(client.dashboard_link)

    # Define paths based on the provided year
    onelev = f"/glade/derecho/scratch/wchapman/STAGING2/AllFiles_{year}-01-01_{year}-12-31_staged.zarr/"
    BigBoy = f"/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/TOTAL_{year}-01-01_{year}-12-31_staged.zarr/"

    # Load the existing datasets
    DSlev = xr.open_zarr(onelev)
    DSlev = DSlev[["U", "V", "T", "Q"]].expand_dims({"level": [137]})
    DSlev = DSlev.transpose("time", "level", "latitude", "longitude")

    DSbb = xr.open_zarr(BigBoy)

    # Scatter the large datasets across the workers
    DSlev_future = client.scatter(DSlev)
    DSbb_future = client.scatter(DSbb)

    # Get the scattered datasets
    DSlev = DSlev_future.result()
    DSbb = DSbb_future.result()

    # Separate out the variables that need to be concatenated along the 'level' dimension
    DSbb_UVTQ = DSbb[["U", "V", "T", "Q"]]

    # Rechunk DSlev to match the chunk structure of DSbb
    DSlev = DSlev.chunk({"time": 10, "level": 1, "latitude": 214, "longitude": 427})

    # Concatenate only the selected variables along the 'level' dimension
    combined_UVTQ = xr.concat([DSbb_UVTQ, DSlev], dim="level")

    # Recombine with the rest of the dataset that doesn't need to be concatenated along 'level'
    rest_of_DSbb = DSbb.drop_vars(["U", "V", "T", "Q"])
    combined_ds = xr.merge([rest_of_DSbb, combined_UVTQ])

    # Rechunk using dictionaries instead of tuples
    chunk_sizes = {
        "U": {"time": 10, "level": 1, "latitude": 214, "longitude": 427},
        "V": {"time": 10, "level": 1, "latitude": 214, "longitude": 427},
        "T": {"time": 10, "level": 1, "latitude": 214, "longitude": 427},
        "Q": {"time": 10, "level": 1, "latitude": 214, "longitude": 427},
    }

    for var, chunks in chunk_sizes.items():
        combined_ds[var] = combined_ds[var].chunk(chunks)

    # Specify encoding to ensure correct chunk sizes are applied during save
    encoding = {
        var: {"chunks": tuple(chunks.values())} for var, chunks in chunk_sizes.items()
    }

    print("...writing dataset...")
    # Save the combined dataset back to the Zarr store with the specified encoding
    combined_ds.to_zarr(
        f"/glade/derecho/scratch/wchapman/TOTAL_{year}-01-01_{year}-12-31_staged.zarr",
        mode="w",
        encoding=encoding,
    )

    print(
        f"Appended DSlev to DSbb for U, V, T, Q, rechunked, and saved the result back to the Zarr store for year {year}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Zarr datasets for a specific year."
    )
    parser.add_argument(
        "year", type=int, help="Year for which to process the datasets, e.g., 2020"
    )
    args = parser.parse_args()
    main(args.year)

    if "client" in locals():
        client.shutdown()
        print("...shutdown client...")
    else:
        print("client does not exist yet")

    remove_dask_worker_scripts = True
    if remove_dask_worker_scripts:
        print("...removing dask workers...")
        fns_rm = sorted(glob.glob("./dask-worker*"))
        print(len(fns_rm))
        for fn in fns_rm:
            os.remove(fn)
