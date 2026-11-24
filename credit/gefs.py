import xarray as xr
import numpy as np
import gcsfs
from tqdm import tqdm
import pandas as pd
from os.path import join, exists, getsize
import os
from scipy.sparse import csr_matrix
import logging
import yaml
from credit.interp import create_pressure_grid, interp_hybrid_to_hybrid_levels

GRAVITY = 9.80616  # m s-2 (CAM value)
RD = 287.04  # J kg-1 K-1, dry air
STD_LAPSE_RATE = 0.0065  # K m-1


def adjust_surface_pressure(
    surface_pressure,
    source_surface_height,
    target_surface_height,
    temperature_lowest,
    specific_humidity_lowest=None,
    lapse_rate=STD_LAPSE_RATE,
):
    """Move surface pressure from the source model's terrain to the target model's terrain.

    Uses the hypsometric equation over the layer between the two surfaces. The layer temperature is the
    lowest-level (virtual) temperature, warmed or cooled at ``lapse_rate`` to the middle of the layer.

    Args:
        surface_pressure (np.ndarray): surface pressure on the source terrain (Pa).
        source_surface_height (np.ndarray): source (e.g. GEFS) surface height on the same grid (m).
        target_surface_height (np.ndarray): target (e.g. CAM) surface height, ``PHIS / g`` (m).
        temperature_lowest (np.ndarray): source temperature at its lowest model level (K).
        specific_humidity_lowest (np.ndarray, optional): source specific humidity there (kg/kg), for virtual temperature.
        lapse_rate (float): temperature lapse rate used across the layer (K/m).

    Returns:
        np.ndarray: surface pressure on the target terrain (Pa).
    """
    dz = source_surface_height - target_surface_height  # > 0 where the target surface is lower
    t_layer = temperature_lowest
    if specific_humidity_lowest is not None:
        t_layer = t_layer * (1.0 + 0.608 * specific_humidity_lowest)
    t_layer = t_layer + 0.5 * lapse_rate * dz
    return surface_pressure * np.exp(GRAVITY * dz / (RD * t_layer))


def extrapolate_temperature_below(
    temperature, target_pressure, source_lowest_pressure, source_lowest_temperature, lapse_rate=STD_LAPSE_RATE
):
    """Replace temperatures below the source model's lowest level with a constant-lapse-rate profile.

    ``interp_hybrid_to_hybrid_levels`` holds values constant outside the source column. Where the target
    surface is lower than the source surface that leaves the bottom layers too cold, so levels with
    ``target_pressure > source_lowest_pressure`` get ``T_low * (p / p_low) ** (Rd * lapse_rate / g)``.

    Args:
        temperature (np.ndarray): (levels, lat, lon) temperature on target levels (K).
        target_pressure (np.ndarray): (levels, lat, lon) target mid-level pressure (Pa).
        source_lowest_pressure (np.ndarray): (lat, lon) pressure of the source lowest level (Pa).
        source_lowest_temperature (np.ndarray): (lat, lon) temperature of the source lowest level (K).
        lapse_rate (float): lapse rate (K/m).

    Returns:
        np.ndarray: temperature with the extrapolated levels replaced.
    """
    below = target_pressure > source_lowest_pressure[None]
    profile = source_lowest_temperature[None] * (target_pressure / source_lowest_pressure[None]) ** (
        RD * lapse_rate / GRAVITY
    )
    return np.where(below, profile, temperature)


def download_gefs_run(init_date_str: str, out_path: str, n_pert_members: int = 30):
    """
    Download GEFS cube sphere netCDF files from AWS for a single initialization.

    Args:
        init_date_str: Initialization date in YYYY-MM-DD HHMM format or similar formats that pandas can handle.
        out_path: Top-level path to save GEFS data on your local machines.
        n_pert_members: Number of perturbation members to download. Max is 30. 0 only downloads the control member
    """
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
            if not exists(out_member_path):
                os.makedirs(out_member_path)
            member_files = fs.ls(member_path)
            for member_file_path in member_files:
                member_file = member_file_path.split("/")[-1]
                out_file = join(out_member_path, member_file)
                # Check if file exists before downloading
                if not exists(out_file):
                    fs.get(member_file_path, out_file)
                # If file exists but download was interrupted, delete file and try again.
                elif fs.du(member_file_path) != getsize(out_file):
                    os.remove(out_file)
                    fs.get(member_file_path, out_file)
    return


def load_member_tiles(path: str, init_date_str: str, member: str, variables: str):
    """

    Args:
        path:
        init_date_str:
        member:
        variables:

    Returns:

    """
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
    if len(select_ua_variables) == 0 and len(select_surface_variables) == 0:
        raise ValueError("You did not request any valid GEFS variables.")
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
                            sfc_ds[sfc_var][0].rename({"yaxis_1": "lat", "xaxis_1": "lon"}).load()
                        )

        elif len(select_surface_variables) > 0:
            with xr.open_dataset(tile_sfc_file) as sfc_ds:
                member_tiles.append(
                    sfc_ds[select_surface_variables].sel(Time=1).rename({"yaxis_1": "lat", "xaxis_1": "lon"}).load()
                )
        if "smc" in select_surface_variables:
            # Only grab the topmost soil moisture value (0-10 cm) and multiply by 100 to convert to CLM kg m^2.
            member_tiles[-1]["smc"] = member_tiles[-1]["smc"][0] * 100
    return member_tiles


def unstagger_winds(ds, u_var="u_s", v_var="v_w", out_u="u_a", out_v="v_a"):
    ds["lev"] = np.arange(ds.sizes["lev"])
    ds["lev"].attrs["axis"] = "Z"
    ds["lon"].attrs["axis"] = "X"
    ds["lat"].attrs["axis"] = "Y"
    ds[out_u] = xr.DataArray(
        0.5 * (ds[u_var][:, :-1, :].values + ds[u_var][:, 1:, :].values),
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
        0.5 * (ds[v_var][:, :, :-1].values + ds[v_var][:, :, 1:].values),
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


def combine_tiles(member_tiles, flatten_dim="tile_lat_lon", coord_dims=("tile", "lat", "lon")):
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
        coord_dict = dict(lev=lev, lat=lat, lon=lon)
        if "levp" in tiles_combined.dims:
            levp = tiles_combined["levp"]
            coord_dict["levp"] = levp
            zh_var_dim = (tiles_combined["levp"].size, lat.size, lon.size)
        else:
            levp = None
            zh_var_dim = None
        regrid_ds = xr.Dataset(coords=coord_dict)
        ua_var_dim = (regrid_ds["lev"].size, regrid_ds["lat"].size, regrid_ds.lon.size)
        sfc_var_dim = (regrid_ds["lat"].size, regrid_ds["lon"].size)
        for variable in tiles_combined.data_vars:
            if "lev" in member_tiles[0][variable].dims:
                regrid_ds[variable] = xr.DataArray(
                    np.zeros(ua_var_dim, dtype=np.float64),
                    coords=dict(lev=lev, lat=lat, lon=lon),
                    name=variable,
                )
                for lev_index in np.arange(tiles_combined["lev"].size):
                    regrid_ds[variable][lev_index] = (
                        regrid_weights @ tiles_combined[variable][lev_index].values
                    ).reshape(sfc_var_dim)
            elif "levp" in member_tiles[0][variable].dims and levp is not None:
                regrid_ds[variable] = xr.DataArray(
                    np.zeros(zh_var_dim, dtype=np.float64),
                    coords=dict(levp=levp, lat=lat, lon=lon),
                    name=variable,
                )
                for lev_index in np.arange(tiles_combined["levp"].size):
                    regrid_ds[variable][lev_index] = (
                        regrid_weights @ tiles_combined[variable][lev_index].values
                    ).reshape(sfc_var_dim)
            elif variable == "ps":
                regrid_ds[variable] = xr.DataArray(
                    (np.exp(regrid_weights @ np.log(tiles_combined[variable].values))).reshape(sfc_var_dim),
                    coords=dict(lat=lat, lon=lon),
                    name=variable,
                )
            else:
                regrid_ds[variable] = xr.DataArray(
                    (regrid_weights @ tiles_combined[variable].values).reshape(sfc_var_dim),
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
    target_surface_geopotential=None,
    surface_height_var="zh",
    temperature_var="t",
    humidity_var="sphum",
):
    """Interpolate regridded GEFS fields from GEFS hybrid levels to the destination model's hybrid levels.

    Args:
        regrid_ds: GEFS fields on the destination horizontal grid (output of ``regrid_member``).
        member_path: top-level path of the downloaded GEFS data (holds ``gfs_ctrl.nc``).
        init_date_str: initialization date.
        member: GEFS member name, e.g. ``c00``.
        vertical_level_file: netCDF file with the destination interface coefficients ``a_name``/``b_name``
            (``a`` in Pa).
        surface_pressure_var: GEFS surface pressure variable.
        a_name: destination interface ``a`` coefficient name.
        b_name: destination interface ``b`` coefficient name.
        vert_dim: GEFS mid-level dimension.
        target_surface_geopotential: optional (lat, lon) surface geopotential of the destination model (m2 s-2) on
            the same grid. When given, surface pressure is moved from the GEFS terrain to the destination terrain
            (``adjust_surface_pressure``), the destination levels are built from that pressure, and temperature
            below the lowest GEFS level follows a standard lapse rate. Needs ``surface_height_var`` (``zh``) and
            ``temperature_var`` in ``regrid_ds``.
        surface_height_var: GEFS interface height variable whose lowest interface is the surface height.
        temperature_var: GEFS temperature variable.
        humidity_var: GEFS specific humidity variable (optional, used for virtual temperature).

    Returns:
        xr.Dataset on the destination levels (``levels``), including the destination pressure ``P``. With
        ``target_surface_geopotential`` the surface pressure variable holds the adjusted surface pressure.
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
    gefs_surface_pressure = regrid_ds[surface_pressure_var].values
    gefs_pressure_grid, gefs_pressure_half_grid = create_pressure_grid(gefs_surface_pressure, gefs_a, gefs_b)
    dest_surface_pressure = gefs_surface_pressure
    if target_surface_geopotential is not None:
        for var in (surface_height_var, temperature_var):
            if var not in regrid_ds:
                raise ValueError(f"Adjusting surface pressure to the target terrain needs GEFS variable '{var}'.")
        # Lowest GEFS level = the level with the highest pressure (works for either level order).
        lowest = int(np.argmax(gefs_pressure_grid.mean(axis=(1, 2))))
        temperature_lowest = regrid_ds[temperature_var].values[lowest]
        humidity_lowest = regrid_ds[humidity_var].values[lowest] if humidity_var in regrid_ds else None
        gefs_surface_height = regrid_ds[surface_height_var].values.min(axis=0)
        dest_surface_pressure = adjust_surface_pressure(
            gefs_surface_pressure,
            gefs_surface_height,
            np.asarray(target_surface_geopotential) / GRAVITY,
            temperature_lowest,
            humidity_lowest,
        )
    dest_pressure_grid, dest_pressure_half_grid = create_pressure_grid(dest_surface_pressure, dest_a, dest_b)
    interp_ds = xr.Dataset(
        coords=dict(
            levels=np.arange(dest_pressure_grid.shape[0]),
            lat=regrid_ds["lat"],
            lon=regrid_ds["lon"],
        )
    )
    interp_ds["P"] = xr.DataArray(
        dest_pressure_grid,
        coords=dict(levels=interp_ds["levels"], lat=interp_ds["lat"], lon=interp_ds["lon"]),
    )
    for variable in regrid_ds.data_vars:
        if vert_dim in regrid_ds[variable].dims:
            interp_ds[variable] = xr.DataArray(
                interp_hybrid_to_hybrid_levels(regrid_ds[variable].values, gefs_pressure_grid, dest_pressure_grid),
                dims=("levels", "lat", "lon"),
                name=variable,
            )
        else:
            interp_ds[variable] = regrid_ds[variable]
    if target_surface_geopotential is not None:
        interp_ds[temperature_var].values = extrapolate_temperature_below(
            interp_ds[temperature_var].values,
            dest_pressure_grid,
            gefs_pressure_grid[lowest],
            temperature_lowest,
        )
        interp_ds[surface_pressure_var] = xr.DataArray(
            dest_surface_pressure,
            coords=regrid_ds[surface_pressure_var].coords,
            dims=regrid_ds[surface_pressure_var].dims,
        )
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
    """
    Rename variables from GEFS to target modeling system (e.g., ERA5 or CAM) variables. Uses a yaml file
    mapping GEFS to other names. Metadata can be added to the output Dataset using a metadata yaml file.

    Args:
        ds:
        name_dict_file:
        meta_file:
        init_date_str:

    Returns:

    """
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


def load_target_surface_geopotential(path, var_name, lat, lon):
    """Read the destination model's surface geopotential and check it sits on the regrid destination grid.

    Args:
        path: netCDF file holding ``var_name`` on a (lat, lon) grid (extra length-1 dims are squeezed).
        var_name: surface geopotential variable (m2 s-2), e.g. ``PHIS``.
        lat: destination latitudes from the regrid weights.
        lon: destination longitudes from the regrid weights.

    Returns:
        np.ndarray: (lat, lon) surface geopotential.
    """
    with xr.open_dataset(path) as ds:
        phis = ds[var_name].squeeze()
        if phis.shape != (lat.size, lon.size):
            raise ValueError(f"{var_name} in {path} has shape {phis.shape}, expected {(lat.size, lon.size)}.")
        for names, expected in ((("latitude", "lat"), lat), (("longitude", "lon"), lon)):
            coord = next((ds[n].values for n in names if n in ds.coords and ds[n].ndim == 1), None)
            if coord is not None and not np.allclose(coord, expected, atol=1e-3):
                raise ValueError(f"{path} {names[0]} does not match the regrid destination grid (order or values).")
        return phis.values


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
    target_surface_geopotential_file="",
    target_surface_geopotential_var="PHIS",
):
    """Turn one GEFS member's cold-start tiles into a netCDF file on the destination grid and levels.

    Args:
        member: GEFS member name, e.g. ``c00``.
        member_path: top-level path of the downloaded GEFS data.
        out_path: directory for ``gefs_cam_grid_{member}.nc``.
        init_date_str: initialization date.
        variables: GEFS variables to read.
        weight_file: ESMF regrid weight file (GEFS cube sphere -> destination grid).
        vertical_level_file: netCDF file with the destination interface coefficients.
        rename_dict_file: YAML mapping GEFS names to destination names.
        meta_file: YAML with metadata for the output variables.
        target_surface_geopotential_file: optional netCDF with the destination model's surface geopotential on the
            destination grid. When set, surface pressure is adjusted to the destination terrain
            (see ``interpolate_vertical_levels``); ``zh`` must be among ``variables``.
        target_surface_geopotential_var: surface geopotential variable name in that file.
    """
    member_tiles = load_member_tiles(member_path, init_date_str, member, variables)
    if "u_s" in variables and "v_w" in variables:
        for t in range(len(member_tiles)):
            member_tiles[t] = unstagger_winds(member_tiles[t])
    print(member + ": Regrid")
    regrid_ds = regrid_member(member_tiles, weight_file)
    regrid_ds = combine_microphysics_terms(regrid_ds)
    print(member + ": Interpolate vertical levels")
    target_phis = None
    if target_surface_geopotential_file != "":
        target_phis = load_target_surface_geopotential(
            target_surface_geopotential_file,
            target_surface_geopotential_var,
            regrid_ds["lat"].values,
            regrid_ds["lon"].values,
        )
    interp_ds = interpolate_vertical_levels(
        regrid_ds,
        member_path,
        init_date_str,
        member,
        vertical_level_file,
        target_surface_geopotential=target_phis,
    )
    interp_ds = rename_variables(interp_ds, rename_dict_file, meta_file, init_date_str)
    out_file = f"gefs_cam_grid_{member}.nc"
    print(member + ": Save to netcdf")
    interp_ds.to_netcdf(join(out_path, out_file))
    return
