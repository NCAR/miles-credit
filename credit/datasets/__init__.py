import os
import sys
import glob
import logging


logger = logging.getLogger(__name__)


def set_globals(data_config, namespace=None):
    """
    Sets global variables from the provided configuration dictionary in the specified namespace.

    This method updates the global variables in either the given `namespace` or the
    caller's namespace (if `namespace` is not provided). If the `namespace` is not specified,
    it uses the global namespace of the caller (using `sys._getframe(1).f_globals`).

    Parameters:
        data_config (dict): A dictionary where the keys are the global variable names
            and the values are the corresponding values to set.
        namespace (dict, optional): The namespace (or dictionary) where the global variables
            should be set. If not provided, the caller's global namespace is used.

    The method logs each global variable being created and its name.
    """

    target = namespace or sys._getframe(1).f_globals
    target.update(data_config)

    # Identify if this is the __main__ namespace
    name = target.get("__name__")

    for key in data_config:
        logger.info(f"Creating global variable in {name}: {key}")


def setup_data_loading(conf):
    """
    Sets up the data loading configuration by reading and processing data paths,
     surface, dynamic forcing, and diagnostic files based on the given configuration.

    The function processes the configuration dictionary (`conf`) and performs the following:
    - Globs and filters data files (ERA5, surface, dynamic forcing, diagnostic).
    - Determines the training and validation file sets based on specified years.
    - Sets up variables like historical data length, forecast length, and additional metadata.
    - Returns a dictionary containing all the paths and configuration details for further use.

    Parameters:
        conf (dict): A dictionary containing configuration details, including data paths,
            variable names, forecast details, and other settings.

    Returns:
        data_config (dict): A dictionary containing paths to various datasets and other
            configuration values used in data loading, such as:
              * all_ERA_files: All ERA5 dataset files.
              * train_files: Filtered training dataset files.
              * valid_files: Filtered validation dataset files.
              * surface_files: Surface data files, if available.
              * dyn_forcing_files: Dynamic forcing files, if available.
              * diagnostic_files: Diagnostic files, if available.
              * varname_upper_air, varname_surface, varname_dyn_forcing, etc.: Variable names for
                 each data type.
              * history_len: Length of the history data for training.
              * forecast_len: Number of steps ahead to forecast.
              * Other configuration values related to skipping periods, one-shot learning, etc.
    """

    all_ERA_files = sorted(glob.glob(conf["data"]["save_loc"]))

    # <------------------------------------------ std_new
    if conf["data"]["scaler_type"] == "std_new":
        # check and glob surface files
        if ("surface_variables" in conf["data"]) and (
            len(conf["data"]["surface_variables"]) > 0
        ):
            surface_files = sorted(glob.glob(conf["data"]["save_loc_surface"]))

        else:
            surface_files = None

        # check and glob dyn forcing files
        if ("dynamic_forcing_variables" in conf["data"]) and (
            len(conf["data"]["dynamic_forcing_variables"]) > 0
        ):
            dyn_forcing_files = sorted(
                glob.glob(conf["data"]["save_loc_dynamic_forcing"])
            )

        else:
            dyn_forcing_files = None

        # check and glob diagnostic files
        if ("diagnostic_variables" in conf["data"]) and (
            len(conf["data"]["diagnostic_variables"]) > 0
        ):
            diagnostic_files = sorted(glob.glob(conf["data"]["save_loc_diagnostic"]))

        else:
            diagnostic_files = None

    # -------------------------------------------------- #
    # import training / validation years from conf

    if "train_years" in conf["data"]:
        train_years_range = conf["data"]["train_years"]
    else:
        train_years_range = [1979, 2014]

    if "valid_years" in conf["data"]:
        valid_years_range = conf["data"]["valid_years"]
    else:
        valid_years_range = [2014, 2018]

    # convert year info to str for file name search
    train_years = [
        str(year) for year in range(train_years_range[0], train_years_range[1])
    ]
    valid_years = [
        str(year) for year in range(valid_years_range[0], valid_years_range[1])
    ]

    # Filter the files for training / validation
    train_files = [
        file for file in all_ERA_files if any(year in file for year in train_years)
    ]
    valid_files = [
        file for file in all_ERA_files if any(year in file for year in valid_years)
    ]

    # <----------------------------------- std_new
    if conf["data"]["scaler_type"] == "std_new":
        if surface_files is not None:
            train_surface_files = [
                file
                for file in surface_files
                if any(year in file for year in train_years)
            ]
            valid_surface_files = [
                file
                for file in surface_files
                if any(year in file for year in valid_years)
            ]

            # ---------------------------- #
            # check total number of files
            assert (
                len(train_surface_files) == len(train_files)
            ), "Mismatch between the total number of training set [surface files] and [upper-air files]"
            assert (
                len(valid_surface_files) == len(valid_files)
            ), "Mismatch between the total number of validation set [surface files] and [upper-air files]"

        else:
            train_surface_files = None
            valid_surface_files = None

        if dyn_forcing_files is not None:
            train_dyn_forcing_files = [
                file
                for file in dyn_forcing_files
                if any(year in file for year in train_years)
            ]
            valid_dyn_forcing_files = [
                file
                for file in dyn_forcing_files
                if any(year in file for year in valid_years)
            ]

            # ---------------------------- #
            # check total number of files
            assert (
                len(train_dyn_forcing_files) == len(train_files)
            ), "Mismatch between the total number of training set [dynamic forcing files] and [upper-air files]"
            assert (
                len(valid_dyn_forcing_files) == len(valid_files)
            ), "Mismatch between the total number of validation set [dynamic forcing files] and [upper-air files]"

        else:
            train_dyn_forcing_files = None
            valid_dyn_forcing_files = None

        if diagnostic_files is not None:
            train_diagnostic_files = [
                file
                for file in diagnostic_files
                if any(year in file for year in train_years)
            ]
            valid_diagnostic_files = [
                file
                for file in diagnostic_files
                if any(year in file for year in valid_years)
            ]

            # ---------------------------- #
            # check total number of files
            assert (
                len(train_diagnostic_files) == len(train_files)
            ), "Mismatch between the total number of training set [diagnostic files] and [upper-air files]"
            assert (
                len(valid_diagnostic_files) == len(valid_files)
            ), "Mismatch between the total number of validation set [diagnostic files] and [upper-air files]"

        else:
            train_diagnostic_files = None
            valid_diagnostic_files = None

    # convert $USER to the actual user name
    conf["save_loc"] = os.path.expandvars(conf["save_loc"])

    # ======================================================== #
    # parse inputs

    # upper air variables
    varname_upper_air = conf["data"]["variables"]

    if ("forcing_variables" in conf["data"]) and (
        len(conf["data"]["forcing_variables"]) > 0
    ):
        forcing_files = conf["data"]["save_loc_forcing"]
        varname_forcing = conf["data"]["forcing_variables"]
    else:
        forcing_files = None
        varname_forcing = None

    if ("static_variables" in conf["data"]) and (
        len(conf["data"]["static_variables"]) > 0
    ):
        static_files = conf["data"]["save_loc_static"]
        varname_static = conf["data"]["static_variables"]
    else:
        static_files = None
        varname_static = None

    # get surface variable names
    if surface_files is not None:
        varname_surface = conf["data"]["surface_variables"]
    else:
        varname_surface = None

    # get dynamic forcing variable names
    if dyn_forcing_files is not None:
        varname_dyn_forcing = conf["data"]["dynamic_forcing_variables"]
    else:
        varname_dyn_forcing = None

    # get diagnostic variable names
    if diagnostic_files is not None:
        varname_diagnostic = conf["data"]["diagnostic_variables"]
    else:
        varname_diagnostic = None

    # number of previous lead time inputs
    history_len = conf["data"]["history_len"]
    valid_history_len = conf["data"]["valid_history_len"]

    # number of lead times to forecast
    forecast_len = conf["data"]["forecast_len"]
    valid_forecast_len = conf["data"]["valid_forecast_len"]

    # max_forecast_len
    if "max_forecast_len" not in conf["data"]:
        max_forecast_len = None
    else:
        max_forecast_len = conf["data"]["max_forecast_len"]

    # skip_periods
    if "skip_periods" not in conf["data"]:
        skip_periods = None
    else:
        skip_periods = conf["data"]["skip_periods"]

    # one_shot
    if "one_shot" not in conf["data"]:
        one_shot = None
    else:
        one_shot = conf["data"]["one_shot"]

    if conf["data"]["sst_forcing"]["activate"]:
        sst_forcing = {
            "varname_skt": conf["data"]["sst_forcing"]["varname_skt"],
            "varname_ocean_mask": conf["data"]["sst_forcing"]["varname_ocean_mask"],
        }
    else:
        sst_forcing = None

    data_config = {
        "all_ERA_files": all_ERA_files,
        "train_files": train_files,
        "valid_files": valid_files,
        "surface_files": surface_files,
        "dyn_forcing_files": dyn_forcing_files,
        "diagnostic_files": diagnostic_files,
        "forcing_files": forcing_files,
        "static_files": static_files,
        "train_surface_files": train_surface_files,
        "valid_surface_files": valid_surface_files,
        "train_dyn_forcing_files": train_dyn_forcing_files,
        "valid_dyn_forcing_files": valid_dyn_forcing_files,
        "train_diagnostic_files": train_diagnostic_files,
        "valid_diagnostic_files": valid_diagnostic_files,
        "varname_upper_air": varname_upper_air,
        "varname_surface": varname_surface,
        "varname_dyn_forcing": varname_dyn_forcing,
        "varname_forcing": varname_forcing,
        "varname_static": varname_static,
        "varname_diagnostic": varname_diagnostic,
        "history_len": history_len,
        "valid_history_len": valid_history_len,
        "forecast_len": forecast_len,
        "valid_forecast_len": valid_forecast_len,
        "max_forecast_len": max_forecast_len,
        "skip_periods": skip_periods,
        "one_shot": one_shot,
        "sst_forcing": sst_forcing,
    }

    return data_config
