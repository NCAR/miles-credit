import xarray as xr
import numpy as np
import gcsfs
from tqdm import tqdm
import pandas as pd
from os.path import join
from scipy.sparse import csr_matrix
import logging
import yaml
from credit.interp import create_pressure_grid, interp_hybrid_to_hybrid_levels


def download_gefs_run(init_date_str, out_path, n_pert_members=30):
    init_date = pd.Timestamp(init_date_str)
    init_date_path = init_date.strftime("gefs.%Y%m%d/%H")
    bucket = "gs://gfs-ensemble-forecast-system/"
    members = ["c00"]
    if n_pert_members > 0:
        members.extend([f"p{m:02d}" for m in range(1, n_pert_members + 1)])
    ens_path = f"{bucket}{init_date_path}/atmos/init/"
    fs = gcsfs.GCSFileSystem(token="anon")
    logging.info(f"Downloading GEFS initialization for {init_date_str}")
    for member in tqdm(members):
        member_path = join(ens_path, member)
        out_member_path = join(out_path, init_date_path, member)
        if fs.exists(member_path):
            fs.get(member_path, out_member_path, recursive=True)
    return


def load_member_tiles(path: str, init_date_str: str, member: str, variables: str):
    num_tiles = 6
    init_date = pd.Timestamp(init_date_str)
    init_date_path = init_date.strftime("gefs.%Y%m%d/%H")
    out_member_path = join(path, init_date_path, member)
    member_tiles = []
    all_ua_variables = [
        "ps",
        "w",
        "zh",
        "t",
        "delp",
        "sphum",
        "liq_wat",
        "o3mr",
        "ice_wat",
        "rainwat",
        "snowwat",
        "graupel",
        "u_w",
        "v_w",
        "u_s",
        "v_s",
    ]
    all_surface_variables = [
        "slmsk",
        "tsea",
        "sheleg",
        "tg3",
        "zorl",
        "alvsf",
        "alvwf",
        "alnsf",
        "alnwf",
        "facsf",
        "facwf",
        "vfrac",
        "canopy",
        "f10m",
        "t2m",
        "q2m",
        "vtype",
        "stype",
        "uustar",
        "ffmm",
        "ffhh",
        "hice",
        "fice",
        "tisfc",
        "tprcp",
        "srflag",
        "snwdph",
        "shdmin",
        "shdmax",
        "slope",
        "snoalb",
        "stc",
        "smc",
        "slc",
        "tref",
        "z_c",
        "c_0",
        "c_d",
        "w_0",
        "w_d",
        "xt",
        "xs",
        "xu",
        "xv",
        "xz",
        "zm",
        "xtts",
        "xzts",
        "d_conv",
        "ifd",
        "dt_cool",
        "qrain",
    ]
    select_ua_variables = np.intersect1d(all_ua_variables, variables)
    select_surface_variables = np.intersect1d(all_surface_variables, variables)
    for t in range(1, num_tiles + 1):
        tile_ua_file = join(out_member_path, f"gfs_data.tile{t:d}.nc")
        tile_sfc_file = join(out_member_path, f"sfc_data.tile{t:d}.nc")
        if len(select_ua_variables) > 0:
            with xr.open_dataset(tile_ua_file) as ua_ds:
                member_tiles.append(ua_ds[select_ua_variables].load())
            if len(select_surface_variables) > 0:
                with xr.open_dataset(tile_sfc_file) as sfc_ds:
                    for sfc_var in select_surface_variables:
                        member_tiles[-1][sfc_var] = (
                            sfc_ds[sfc_var][0]
                            .rename({"yaxis_1": "lat", "xaxis_1": "lon"})
                            .load()
                        )

        elif len(select_surface_variables) > 0:
            with xr.open_dataset(tile_sfc_file) as sfc_ds:
                member_tiles.append(
                    sfc_ds[select_surface_variables]
                    .sel(Time=1)
                    .rename({"yaxis_1": "lat", "xaxis_1": "lon"})
                    .load()
                )
                if "smc" in select_surface_variables:
                    # Only grab the topmost soil moisture value (0-10 cm) and multiply by 100 to convert to CLM kg m^2.
                    member_tiles[-1]["smc"] = member_tiles[-1]["smc"][0] * 100
        else:
            raise ValueError("You did not request any valid GEFS variables.")
    return member_tiles


def unstagger_winds(ds, u_var="u_s", v_var="v_w", out_u="u_a", out_v="v_a"):
    ds["lev"] = np.arange(ds.sizes["lev"])
    ds["lev"].attrs["axis"] = "Z"
    ds["lon"].attrs["axis"] = "X"
    ds["lat"].attrs["axis"] = "Y"
    ds[out_u] = xr.DataArray(
        0.5 * ds[u_var][:, :-1, :].values + ds[u_var][:, 1:, :].values,
        coords=dict(
            lev=ds["lev"],
            lat=ds["lat"],
            lon=ds["lon"],
            geolat=(("lat", "lon"), ds["geolat"].values),
            geolon=(("lat", "lon"), ds["geolon"].values),
        ),
        dims=("lev", "lat", "lon"),
    )
    ds[out_v] = xr.DataArray(
        0.5 * ds[v_var][:, :, :-1].values + ds[v_var][:, :, 1:].values,
        coords=dict(
            lev=ds["lev"],
            y=ds["lat"],
            x=ds["lon"],
            lat=(("lat", "lon"), ds["geolat"].values),
            lon=(("lat", "lon"), ds["geolon"].values),
        ),
        dims=("lev", "lat", "lon"),
    )
    ds = ds.drop_vars([u_var, v_var])
    return ds


def combine_tiles(
    member_tiles, flatten_dim="tile_lat_lon", coord_dims=("tile", "lat", "lon")
):
    tiles_combined = xr.concat(member_tiles, dim="tile")
    tiles_stacked = tiles_combined.stack(**{flatten_dim: coord_dims})
    return tiles_stacked


def regrid_member(member_tiles, regrid_weights_file):
    tiles_combined = combine_tiles(member_tiles)
    with xr.open_dataset(regrid_weights_file) as regrid_ds:
        # Description of weight file at https://earthsystemmodeling.org/docs/release/latest/ESMF_refdoc/node3.html#SECTION03029300000000000000
        regrid_weights = csr_matrix(
            (
                regrid_ds["S"].values,
                (regrid_ds["row"].values - 1, regrid_ds["col"].values - 1),
            ),
            shape=(regrid_ds.sizes["n_b"], regrid_ds.sizes["n_a"]),
        )
        dst_dims = regrid_ds["dst_grid_dims"][::-1].values
        lon = regrid_ds["xc_b"].values.reshape(dst_dims)[0]
        lat = regrid_ds["yc_b"].values.reshape(dst_dims)[:, 0]
        lev = tiles_combined["lev"]
        regrid_ds = xr.Dataset(coords=dict(lev=lev, lat=lat, lon=lon))
        ua_var_dim = (regrid_ds["lev"].size, regrid_ds["lat"].size, regrid_ds.lon.size)
        sfc_var_dim = (regrid_ds["lat"].size, regrid_ds["lon"].size)
        for variable in tiles_combined.data_vars:
            if "lev" in member_tiles[0][variable].dims:
                regrid_ds[variable] = xr.DataArray(
                    np.zeros(ua_var_dim, dtype=np.float32),
                    coords=dict(lev=lev, lat=lat, lon=lon),
                    name=variable,
                )
                for lev_index in np.arange(tiles_combined["lev"].size):
                    regrid_ds[variable][lev_index] = (
                        regrid_weights @ tiles_combined[variable][lev_index].values
                    ).reshape(sfc_var_dim)
            else:
                regrid_ds[variable] = xr.DataArray(
                    (regrid_weights @ tiles_combined[variable].values).reshape(
                        sfc_var_dim
                    ),
                    coords=dict(lat=lat, lon=lon),
                    name=variable,
                )
    return regrid_ds


def interpolate_vertical_levels(
    regrid_ds,
    member_path,
    init_date_str,
    member,
    vertical_level_file,
    surface_pressure_var="ps",
    a_name="hyai",
    b_name="hybi",
    vert_dim="lev",
):
    """

    Args:
        regrid_ds:
        member_path:
        vertical_level_file:
        surface_pressure_var:
        a_name:
        b_name:
        vert_dim:

    Returns:

    """
    init_date = pd.Timestamp(init_date_str)
    init_date_path = init_date.strftime("gefs.%Y%m%d/%H")
    out_member_path = join(member_path, init_date_path, member)

    gefs_vertical_level_file = join(out_member_path, "gfs_ctrl.nc")

    with xr.open_dataset(gefs_vertical_level_file) as gefs_vert_ds:
        gefs_a = gefs_vert_ds["vcoord"][0].values
        gefs_b = gefs_vert_ds["vcoord"][1].values
    with xr.open_dataset(vertical_level_file) as dest_vert_ds:
        dest_a = dest_vert_ds[a_name].values
        dest_b = dest_vert_ds[b_name].values
    gefs_pressure_grid, gefs_pressure_half_grid = create_pressure_grid(
        regrid_ds[surface_pressure_var].values, gefs_a, gefs_b
    )
    dest_pressure_grid, dest_pressure_half_grid = create_pressure_grid(
        regrid_ds[surface_pressure_var].values, dest_a, dest_b
    )
    interp_ds = xr.Dataset(
        coords=dict(
            levels=np.arange(dest_pressure_grid.shape[0]),
            lat=regrid_ds["lat"],
            lon=regrid_ds["lon"],
        )
    )
    interp_ds["P"] = xr.DataArray(
        dest_pressure_grid,
        coords=dict(
            levels=interp_ds["levels"], lat=interp_ds["lat"], lon=interp_ds["lon"]
        ),
    )
    for variable in regrid_ds.data_vars:
        if vert_dim in regrid_ds[variable].dims:
            interp_ds[variable] = xr.DataArray(
                interp_hybrid_to_hybrid_levels(
                    regrid_ds[variable].values, gefs_pressure_grid, dest_pressure_grid
                ),
                dims=("levels", "lat", "lon"),
                name=variable,
            )
        else:
            interp_ds[variable] = regrid_ds[variable]
    return interp_ds


def combine_microphysics_terms(
    regrid_ds,
    microphysics_vars=("sphum", "liq_wat", "ice_wat", "rainwat", "snowwat", "graupel"),
    total_var="Qtot",
):
    regrid_ds[total_var] = xr.DataArray(
        regrid_ds[microphysics_vars[0]].values,
        dims=regrid_ds[microphysics_vars[0]].dims,
    )
    for mpv in microphysics_vars[1:]:
        regrid_ds[total_var][:] += regrid_ds[mpv].values
    return regrid_ds


def rename_variables(ds, name_dict_file, meta_file, init_date_str):
    if name_dict_file != "":
        with open(name_dict_file, "r") as name_dict_obj:
            name_dict = yaml.safe_load(name_dict_obj)
    else:
        name_dict = {}
    new_ds = ds.rename(name_dict)
    if meta_file != "":
        with open(meta_file, "r") as meta_file_obj:
            meta_dict = yaml.safe_load(meta_file_obj)
    else:
        meta_dict = {}
    init_date = pd.Timestamp(init_date_str)
    new_ds["time"] = init_date
    new_ds = new_ds.set_coords("time").expand_dims("time")
    new_ds = new_ds.drop_vars(["Time"])
    for var in new_ds.variables:
        if var in meta_dict.keys():
            new_ds[var].attrs = meta_dict[var]
    new_ds.attrs["Conventions"] = "CF-1.7"
    return new_ds


def process_member(
    member,
    member_path=None,
    out_path=None,
    init_date_str=None,
    variables=None,
    weight_file="",
    vertical_level_file="",
    rename_dict_file="",
    meta_file="",
):
    member_tiles = load_member_tiles(member_path, init_date_str, member, variables)
    if "u_s" in variables and "v_w" in variables:
        for t in range(len(member_tiles)):
            member_tiles[t] = unstagger_winds(member_tiles[t])
    print(member + ": Regrid")
    regrid_ds = regrid_member(member_tiles, weight_file)
    regrid_ds = combine_microphysics_terms(regrid_ds)
    print(member + ": Interpolate vertical levels")
    interp_ds = interpolate_vertical_levels(
        regrid_ds, member_path, init_date_str, member, vertical_level_file
    )
    interp_ds = rename_variables(interp_ds, rename_dict_file, meta_file, init_date_str)
    out_file = f"gefs_cam_grid_{member}.nc"
    print(member + ": Save to netcdf")
    interp_ds.to_netcdf(join(out_path, out_file))
    return
