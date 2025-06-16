"""
parser.py
-------------------------------------------------------
Content:
    - credit_main_parser
    - training_data_check
    - predict_data_check
    - remove_string_by_pattern
"""

import os
import copy
import warnings
from glob import glob
from collections import Counter


from credit.data import get_forward_data


def remove_string_by_pattern(list_string, pattern):
    """
    Given a list of strings, remove some of them based on a given pattern.
    Usage: remove 'time'/'datetime'/'lead_time' coordinates from a list of all coordinate names.
    """
    # may need to be improved
    pattern_collection = []
    for single_string in list_string:
        if pattern in single_string:
            pattern_collection.append(single_string)

    return [single_string for single_string in list_string if single_string not in pattern_collection]


def credit_main_parser(conf, parse_training=True, parse_predict=True, print_summary=False):
    """
    Parses and validates the configuration input for the CREDIT project.

    This function examines the provided configuration dictionary (`conf`), ensures that all required fields are
    present, and assigns default values where necessary. It is designed to be used in various training and
    prediction modules within the CREDIT repository. Missing critical fields will trigger assertion errors, while
    others will receive default values. A standardized version of the input configuration will be returned, ensuring
    consistency across different applications.

    Args:
        conf (dict): Configuration dictionary containing all settings for data, model, trainer, and prediction phases.
        parse_training (bool, optional): If True, the function will check for training-specific fields. Defaults to True.
        parse_predict (bool, optional): If True, the function will check for prediction-specific fields. Defaults to True.
        print_summary (bool, optional): If True, a summary of the parsed variables will be printed. Defaults to False.

    Returns:
        dict: The standardized and validated configuration dictionary.

    Raises:
        AssertionError: If any critical fields are missing or invalid in the provided configuration.

    Notes:
        This function is used in the following scripts:
        - applications/train.py
        - applications/train_multistep.py
        - applications/rollout_to_netcdf.py

    """

    assert "save_loc" in conf, "save location of the CREDIT project ('save_loc') is missing from conf"
    assert "data" in conf, "data section ('data') is missing from conf"
    assert "model" in conf, "model section ('model') is missing from conf"
    assert "latitude_weights" in conf["loss"], "lat / lon file ('latitude_weights') is missing from conf['loss']"

    if parse_training:
        assert "trainer" in conf, "trainer section ('trainer') is missing from conf"
        assert "loss" in conf, "loss section ('loss') is missing from conf"

    if parse_predict:
        assert "predict" in conf, "predict section ('predict') is missing from conf"

    conf["save_loc"] = os.path.expandvars(conf["save_loc"])

    # --------------------------------------------------------- #
    # conf['data'] section

    # must have upper-air variables
    assert "variables" in conf["data"], "upper-air variable names ('variables') is missing from conf['data']"

    if (conf["data"]["variables"] is None) or (len(conf["data"]["variables"]) == 0):
        print(
            "Upper-air variable name conf['data']['variables']: {} cannot be processed".format(
                conf["data"]["variables"]
            )
        )
        raise

    assert "save_loc" in conf["data"], "upper-air var save locations ('save_loc') is missing from conf['data']"

    if conf["data"]["save_loc"] is None:
        print(
            "Upper-air var save locations conf['data']['save_loc']: {} cannot be processed".format(
                conf["data"]["save_loc"]
            )
        )
        raise

    if "levels" not in conf["data"]:
        if "levels" in conf["model"]:
            conf["data"]["levels"] = conf["model"]["levels"]
        else:
            print("number of upper-air levels ('levels') is missing from both conf['data'] and conf['model']")
            raise
    # ========================================================================================= #
    # Check other input / output variable types
    # if varname is provided, its corresponding save_loc should exist
    # if varname is None, missing, or [], assign flag = False
    # the default missing varname will be converted to []

    # surface inputs
    if "surface_variables" in conf["data"]:
        if conf["data"]["surface_variables"] is None:
            conf["data"]["flag_surface"] = False
        elif len(conf["data"]["surface_variables"]) > 0:
            conf["data"]["flag_surface"] = True
            assert (
                "save_loc_surface" in conf["data"]
            ), "surface var save locations ('save_loc_surface') is missing from conf['data']"
        else:
            conf["data"]["flag_surface"] = False
    else:
        conf["data"]["flag_surface"] = False

    # dyn forcing inputs
    if "dynamic_forcing_variables" in conf["data"]:
        if conf["data"]["dynamic_forcing_variables"] is None:
            conf["data"]["flag_dyn_forcing"] = False
        elif len(conf["data"]["dynamic_forcing_variables"]) > 0:
            conf["data"]["flag_dyn_forcing"] = True
            assert (
                "save_loc_dynamic_forcing" in conf["data"]
            ), "dynamic forcing var save locations ('save_loc_dynamic_forcing') is missing from conf['data']"
        else:
            conf["data"]["flag_dyn_forcing"] = False
    else:
        conf["data"]["flag_dyn_forcing"] = False

    # diagnostic outputs
    if "diagnostic_variables" in conf["data"]:
        if conf["data"]["diagnostic_variables"] is None:
            conf["data"]["flag_diagnostic"] = False
        elif len(conf["data"]["diagnostic_variables"]) > 0:
            conf["data"]["flag_diagnostic"] = True
            assert (
                "save_loc_diagnostic" in conf["data"]
            ), "diagnostic var save locations ('save_loc_diagnostic') is missing from conf['data']"
        else:
            conf["data"]["flag_diagnostic"] = False
    else:
        conf["data"]["flag_diagnostic"] = False

    # forcing inputs
    if "forcing_variables" in conf["data"]:
        if conf["data"]["forcing_variables"] is None:
            conf["data"]["flag_forcing"] = False
        elif len(conf["data"]["forcing_variables"]) > 0:
            conf["data"]["flag_forcing"] = True
            assert (
                "save_loc_forcing" in conf["data"]
            ), "forcing var save locations ('save_loc_forcing') is missing from conf['data']"
        else:
            conf["data"]["flag_forcing"] = False
    else:
        conf["data"]["flag_forcing"] = False

    # static inputs
    if "static_variables" in conf["data"]:
        if conf["data"]["static_variables"] is None:
            conf["data"]["flag_static"] = False
        elif len(conf["data"]["static_variables"]) > 0:
            conf["data"]["flag_static"] = True
            assert (
                "save_loc_static" in conf["data"]
            ), "static var save locations ('save_loc_static') is missing from conf['data']"
        else:
            conf["data"]["flag_static"] = False
    else:
        conf["data"]["flag_static"] = False

    # ===================================================== #
    # assign default values for the missing data
    # varname = [] if not needed
    # save_loc = None if not needed
    if conf["data"]["flag_surface"] is False:
        conf["data"]["save_loc_surface"] = None
        conf["data"]["surface_variables"] = []

    if conf["data"]["flag_dyn_forcing"] is False:
        conf["data"]["save_loc_dynamic_forcing"] = None
        conf["data"]["dynamic_forcing_variables"] = []

    if conf["data"]["flag_diagnostic"] is False:
        conf["data"]["save_loc_diagnostic"] = None
        conf["data"]["diagnostic_variables"] = []

    if conf["data"]["flag_forcing"] is False:
        conf["data"]["save_loc_forcing"] = None
        conf["data"]["forcing_variables"] = []

    if conf["data"]["flag_static"] is False:
        conf["data"]["save_loc_static"] = None
        conf["data"]["static_variables"] = []
    # ===================================================== #

    # duplicated variable name check
    all_varnames = (
        conf["data"]["variables"]
        + conf["data"]["surface_variables"]
        + conf["data"]["dynamic_forcing_variables"]
        + conf["data"]["diagnostic_variables"]
        + conf["data"]["forcing_variables"]
        + conf["data"]["static_variables"]
    )

    varname_counts = Counter(all_varnames)
    duplicates = [varname for varname, count in varname_counts.items() if count > 1]

    assert len(duplicates) == 0, "Duplicated variable names: [{}] found. No duplicates allowed, stop.".format(
        duplicates
    )

    conf["data"]["all_varnames"] = (
        conf["data"]["variables"]
        + conf["data"]["surface_variables"]
        + conf["data"]["dynamic_forcing_variables"]
        + conf["data"]["diagnostic_variables"]
    )

    ## I/O data sizes

    conf["data"].setdefault("data_clamp", None)

    if parse_training:
        assert "train_years" in conf["data"], "year range for training ('train_years') is missing from conf['data']"

        # 'valid_years' is required even for conf['trainer']['skip_validation']: True
        # 'valid_years' and 'train_years' can overlap
        assert "valid_years" in conf["data"], "year range for validation ('valid_years') is missing from conf['data']"

        assert (
            "forecast_len" in conf["data"]
        ), "Number of time frames for loss compute ('forecast_len') is missing from conf['data']"

        if "valid_history_len" not in conf["data"]:
            # use "history_len" for "valid_history_len"
            conf["data"]["valid_history_len"] = conf["data"]["history_len"]

        if "valid_forecast_len" not in conf["data"]:
            # use "forecast_len" for "valid_forecast_len"
            conf["data"]["valid_forecast_len"] = conf["data"]["forecast_len"]

        if "max_forecast_len" not in conf["data"]:
            conf["data"]["max_forecast_len"] = None  # conf['data']['forecast_len']

        # one_shot
        if "one_shot" not in conf["data"]:
            conf["data"]["one_shot"] = None

        if conf["data"]["one_shot"] is not True:
            conf["data"]["one_shot"] = None

        if "total_time_steps" not in conf["data"]:
            conf["data"]["total_time_steps"] = conf["data"]["forecast_len"]

    assert "history_len" in conf["data"], "Number of input time frames ('history_len') is missing from conf['data']"
    assert (
        "lead_time_periods" in conf["data"]
    ), "Number of forecast hours ('lead_time_periods') is missing from conf['data']"
    assert "scaler_type" in conf["data"], "'scaler_type' is missing from conf['data']"

    if conf["data"]["scaler_type"] == "std_new":
        assert "mean_path" in conf["data"], "The z-score mean file ('mean_path') is missing from conf['data']"
        assert "std_path" in conf["data"], "The z-score std file ('std_path') is missing from conf['data']"

    # skip_periods
    if ("skip_periods" not in conf["data"]) or (conf["data"]["skip_periods"] is None):
        conf["data"]["skip_periods"] = 1

    if "static_first" not in conf["data"]:
        conf["data"]["static_first"] = True

    if "sst_forcing" not in conf["data"]:
        conf["data"]["sst_forcing"] = {"activate": False}

    # --------------------------------------------------------- #
    # conf['model'] section

    # spectral norm default to false
    conf["model"].setdefault("use_spectral_norm", True)

    if (conf["model"]["type"] == "fuxi") and (conf["model"]["use_spectral_norm"] is False):
        warnings.warn("FuXi may not work with 'use_spectral_norm: False' in fsdp training.")

    # use interpolation
    if "interp" not in conf["model"]:
        conf["model"]["interp"] = True

    # ======================================================== #
    # padding opts
    if "padding_conf" not in conf["model"]:
        if ("pad_lon" in conf["model"]) and ("pad_lat" in conf["model"]):
            pad_lon = int(conf["model"]["pad_lon"])
            pad_lat = int(conf["model"]["pad_lat"])
            conf["model"]["padding_conf"] = {"activate": True}
            conf["model"]["padding_conf"]["mode"] = "mirror"
            conf["model"]["padding_conf"]["pad_lon"] = [pad_lon, pad_lon]
            conf["model"]["padding_conf"]["pad_lat"] = [pad_lat, pad_lat]
    else:
        if conf["model"]["padding_conf"]["activate"]:
            pad_lon = conf["model"]["padding_conf"]["pad_lon"]
            pad_lat = conf["model"]["padding_conf"]["pad_lat"]

            if isinstance(pad_lon, int):
                conf["model"]["padding_conf"]["pad_lon"] = [pad_lon, pad_lon]

            if isinstance(pad_lat, int):
                conf["model"]["padding_conf"]["pad_lat"] = [pad_lat, pad_lat]

            pad_lon = conf["model"]["padding_conf"]["pad_lon"]
            pad_lat = conf["model"]["padding_conf"]["pad_lat"]

            assert all(p >= 0 for p in pad_lon), "padding size for longitude dim must be non-negative."
            assert all(p >= 0 for p in pad_lat), "padding size for latitude dim must be non-negative."

            assert conf["model"]["padding_conf"]["mode"] in [
                "mirror",
                "earth",
            ], 'Padding options must be "mirror" or "earth". Got ""'.format()

    # ======================================================== #
    # postblock opts
    # turn-off post if post_conf does not exist
    conf["model"].setdefault("post_conf", {"activate": False})

    # set defaults for post modules
    post_list = [
        "skebs",
        "tracer_fixer",
        "global_mass_fixer",
        "global_water_fixer",
        "global_energy_fixer",
    ]

    # if activate is false, set all post modules to false
    if not conf["model"]["post_conf"]["activate"]:
        for post_module in post_list:
            conf["model"]["post_conf"][post_module] = {"activate": False}

    # set defaults for post modules
    for post_module in post_list:
        conf["model"]["post_conf"].setdefault(post_module, {"activate": False})

    # see if any of the postconfs want to be activated
    post_conf = conf["model"]["post_conf"]
    activate_any = any([post_conf[post_module]["activate"] for post_module in post_list])
    if post_conf["activate"] and not activate_any:
        raise ("post_conf is set activate, but no post modules specified")

    if conf["model"]["post_conf"]["activate"]:
        # copy only model configs to post_conf subdictionary
        conf["model"]["post_conf"]["model"] = {k: v for k, v in conf["model"].items() if k != "post_conf"}
        # copy data configs to post_conf (for de-normalize variables)
        conf["model"]["post_conf"]["data"] = {k: v for k, v in conf["data"].items()}
        conf["model"]["post_conf"].setdefault("grid", "legendre-gauss")

        # --------------------------------------------------------------------- #
        # get the full list of input / output variables for post_conf
        # the list is ordered based on the tensor channels inside credit.models

        # upper air vars on all levels
        varname_input = []
        varname_output = []
        for var in conf["data"]["variables"]:
            for i_level in range(conf["data"]["levels"]):
                varname_input.append(var)
                varname_output.append(var)

        varname_input += conf["data"]["surface_variables"]

        # handle the order of input-only variables
        if conf["data"]["static_first"]:
            varname_input += (
                conf["data"]["static_variables"]
                + conf["data"]["dynamic_forcing_variables"]
                + conf["data"]["forcing_variables"]
            )
        else:
            varname_input += (
                conf["data"]["dynamic_forcing_variables"]
                + conf["data"]["forcing_variables"]
                + conf["data"]["static_variables"]
            )

        varname_output += conf["data"]["surface_variables"] + conf["data"]["diagnostic_variables"]

        # # debug only
        conf["model"]["post_conf"]["varname_input"] = varname_input
        conf["model"]["post_conf"]["varname_output"] = varname_output

        # --------------------------------------------------------------------- #

    # SKEBS
    if conf["model"]["post_conf"]["skebs"]["activate"]:
        assert (
            "freeze_base_model_weights" in conf["model"]["post_conf"]["skebs"]
        ), "need to specify freeze_base_model_weights in skebs config"

        assert (
            conf["trainer"]["train_batch_size"] == conf["trainer"]["valid_batch_size"]
        ), "train and valid batch sizes need to be the same for skebs"

        # setup backscatter writing
        conf["model"]["post_conf"]["predict"] = {k: v for k, v in conf["predict"].items()}

        conf["model"]["post_conf"]["skebs"].setdefault("lmax", None)
        conf["model"]["post_conf"]["skebs"].setdefault("mmax", None)

        if conf["model"]["post_conf"]["skebs"]["lmax"] in ["none", "None"]:
            conf["model"]["post_conf"]["skebs"]["lmax"] = None
        if conf["model"]["post_conf"]["skebs"]["mmax"] in ["none", "None"]:
            conf["model"]["post_conf"]["skebs"]["mmax"] = None

        U_inds = [i_var for i_var, var in enumerate(varname_output) if var == "U"]

        V_inds = [i_var for i_var, var in enumerate(varname_output) if var == "V"]
        T_inds = [i_var for i_var, var in enumerate(varname_output) if var == "T"]
        Q_inds = [i_var for i_var, var in enumerate(varname_output) if var in ["Q", "Qtot"]]

        conf["model"]["post_conf"]["skebs"]["U_inds"] = U_inds
        conf["model"]["post_conf"]["skebs"]["V_inds"] = V_inds
        conf["model"]["post_conf"]["skebs"]["Q_inds"] = Q_inds
        conf["model"]["post_conf"]["skebs"]["T_inds"] = T_inds

        if "SP" in varname_output:
            conf["model"]["post_conf"]["skebs"]["SP_ind"] = varname_output.index("SP")
        else:
            conf["model"]["post_conf"]["skebs"]["SP_ind"] = varname_output.index("PS")

        static_inds = [i_var for i_var, var in enumerate(varname_input) if var in conf["data"]["static_variables"]]
        conf["model"]["post_conf"]["skebs"]["static_inds"] = static_inds

        ###### debug mode setup #######
        conf["model"]["post_conf"]["skebs"]["save_loc"] = conf["save_loc"]

    # --------------------------------------------------------------------- #
    # tracer fixer
    flag_tracer = conf["model"]["post_conf"]["activate"] and conf["model"]["post_conf"]["tracer_fixer"]["activate"]

    if flag_tracer:
        # when tracer fixer is on, get tensor indices of tracers
        # tracers must be outputs (either prognostic or output only)

        # tracer fixer runs on de-normalized variables by default
        conf["model"]["post_conf"]["tracer_fixer"].setdefault("denorm", True)
        conf["model"]["post_conf"]["tracer_fixer"].setdefault("tracer_thres", [])

        varname_tracers = conf["model"]["post_conf"]["tracer_fixer"]["tracer_name"]
        tracers_thres_input = conf["model"]["post_conf"]["tracer_fixer"]["tracer_thres"]

        # create a mapping from tracer variable names to their thresholds
        tracer_threshold_dict = dict(zip(varname_tracers, tracers_thres_input))

        # Iterate over varname_output to find tracer indices and thresholds
        tracer_inds = []
        tracer_thres = []
        for i_var, var in enumerate(varname_output):
            if var in tracer_threshold_dict:
                tracer_inds.append(i_var)
                tracer_thres.append(float(tracer_threshold_dict[var]))

        conf["model"]["post_conf"]["tracer_fixer"]["tracer_inds"] = tracer_inds
        conf["model"]["post_conf"]["tracer_fixer"]["tracer_thres"] = tracer_thres

    # --------------------------------------------------------------------- #
    # global mass fixer

    flag_mass = conf["model"]["post_conf"]["activate"] and conf["model"]["post_conf"]["global_mass_fixer"]["activate"]

    if flag_mass:
        # when global mass fixer is on, get tensor indices of q, precip, evapor
        # these variables must be outputs

        # global mass fixer defaults
        conf["model"]["post_conf"]["global_mass_fixer"].setdefault("activate_outside_model", False)
        conf["model"]["post_conf"]["global_mass_fixer"].setdefault("denorm", True)
        conf["model"]["post_conf"]["global_mass_fixer"].setdefault("simple_demo", False)
        conf["model"]["post_conf"]["global_mass_fixer"].setdefault("midpoint", False)
        conf["model"]["post_conf"]["global_mass_fixer"].setdefault("grid_type", "pressure")

        assert (
            "fix_level_num" in conf["model"]["post_conf"]["global_mass_fixer"]
        ), "Must specifiy what level to fix on specific total water"

        if conf["model"]["post_conf"]["global_mass_fixer"]["simple_demo"] is False:
            assert (
                "lon_lat_level_name" in conf["model"]["post_conf"]["global_mass_fixer"]
            ), "Must specifiy var names for lat/lon/level in physics reference file"

        if conf["model"]["post_conf"]["global_mass_fixer"]["grid_type"] == "sigma":
            assert (
                "surface_pressure_name" in conf["model"]["post_conf"]["global_mass_fixer"]
            ), "Must specifiy surface pressure var name when using hybrid sigma-pressure coordinates"

        q_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_mass_fixer"]["specific_total_water_name"]
        ]
        conf["model"]["post_conf"]["global_mass_fixer"]["q_inds"] = q_inds

        if conf["model"]["post_conf"]["global_mass_fixer"]["grid_type"] == "sigma":
            sp_inds = [
                i_var
                for i_var, var in enumerate(varname_output)
                if var in conf["model"]["post_conf"]["global_mass_fixer"]["surface_pressure_name"]
            ]
            conf["model"]["post_conf"]["global_mass_fixer"]["sp_inds"] = sp_inds[0]

    # --------------------------------------------------------------------- #
    # global water fixer
    flag_water = conf["model"]["post_conf"]["activate"] and conf["model"]["post_conf"]["global_water_fixer"]["activate"]

    if flag_water:
        # when global water fixer is on, get tensor indices of q, precip, evapor
        # these variables must be outputs

        # global water fixer defaults
        conf["model"]["post_conf"]["global_water_fixer"].setdefault("activate_outside_model", False)
        conf["model"]["post_conf"]["global_water_fixer"].setdefault("denorm", True)
        conf["model"]["post_conf"]["global_water_fixer"].setdefault("simple_demo", False)
        conf["model"]["post_conf"]["global_water_fixer"].setdefault("midpoint", False)

        conf["model"]["post_conf"]["global_water_fixer"].setdefault("grid_type", "pressure")

        if conf["model"]["post_conf"]["global_water_fixer"]["simple_demo"] is False:
            assert (
                "lon_lat_level_name" in conf["model"]["post_conf"]["global_water_fixer"]
            ), "Must specifiy var names for lat/lon/level in physics reference file"

        if conf["model"]["post_conf"]["global_water_fixer"]["grid_type"] == "sigma":
            assert (
                "surface_pressure_name" in conf["model"]["post_conf"]["global_water_fixer"]
            ), "Must specifiy surface pressure var name when using hybrid sigma-pressure coordinates"
        q_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_water_fixer"]["specific_total_water_name"]
        ]

        precip_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_water_fixer"]["precipitation_name"]
        ]

        evapor_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_water_fixer"]["evaporation_name"]
        ]

        conf["model"]["post_conf"]["global_water_fixer"]["q_inds"] = q_inds
        conf["model"]["post_conf"]["global_water_fixer"]["precip_ind"] = precip_inds[0]
        conf["model"]["post_conf"]["global_water_fixer"]["evapor_ind"] = evapor_inds[0]

        if conf["model"]["post_conf"]["global_water_fixer"]["grid_type"] == "sigma":
            sp_inds = [
                i_var
                for i_var, var in enumerate(varname_output)
                if var in conf["model"]["post_conf"]["global_water_fixer"]["surface_pressure_name"]
            ]
            conf["model"]["post_conf"]["global_water_fixer"]["sp_inds"] = sp_inds[0]

    # --------------------------------------------------------------------- #
    # global energy fixer
    flag_energy = (
        conf["model"]["post_conf"]["activate"] and conf["model"]["post_conf"]["global_energy_fixer"]["activate"]
    )

    if flag_energy:
        # when global energy fixer is on, get tensor indices of energy components
        # geopotential at surface is input, others are outputs

        # global energy fixer defaults
        conf["model"]["post_conf"]["global_energy_fixer"].setdefault("activate_outside_model", False)
        conf["model"]["post_conf"]["global_energy_fixer"].setdefault("denorm", True)
        conf["model"]["post_conf"]["global_energy_fixer"].setdefault("simple_demo", False)
        conf["model"]["post_conf"]["global_energy_fixer"].setdefault("midpoint", False)

        conf["model"]["post_conf"]["global_energy_fixer"].setdefault("grid_type", "pressure")

        if conf["model"]["post_conf"]["global_energy_fixer"]["simple_demo"] is False:
            assert (
                "lon_lat_level_name" in conf["model"]["post_conf"]["global_energy_fixer"]
            ), "Must specifiy var names for lat/lon/level in physics reference file"

        if conf["model"]["post_conf"]["global_energy_fixer"]["grid_type"] == "sigma":
            assert (
                "surface_pressure_name" in conf["model"]["post_conf"]["global_energy_fixer"]
            ), "Must specifiy surface pressure var name when using hybrid sigma-pressure coordinates"

        T_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_energy_fixer"]["air_temperature_name"]
        ]

        q_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_energy_fixer"]["specific_total_water_name"]
        ]

        U_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_energy_fixer"]["u_wind_name"]
        ]

        V_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_energy_fixer"]["v_wind_name"]
        ]

        TOA_rad_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_energy_fixer"]["TOA_net_radiation_flux_name"]
        ]

        surf_rad_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_energy_fixer"]["surface_net_radiation_flux_name"]
        ]

        surf_flux_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_energy_fixer"]["surface_energy_flux_name"]
        ]

        conf["model"]["post_conf"]["global_energy_fixer"]["T_inds"] = T_inds
        conf["model"]["post_conf"]["global_energy_fixer"]["q_inds"] = q_inds
        conf["model"]["post_conf"]["global_energy_fixer"]["U_inds"] = U_inds
        conf["model"]["post_conf"]["global_energy_fixer"]["V_inds"] = V_inds
        conf["model"]["post_conf"]["global_energy_fixer"]["TOA_rad_inds"] = TOA_rad_inds
        conf["model"]["post_conf"]["global_energy_fixer"]["surf_rad_inds"] = surf_rad_inds
        conf["model"]["post_conf"]["global_energy_fixer"]["surf_flux_inds"] = surf_flux_inds

        if conf["model"]["post_conf"]["global_energy_fixer"]["grid_type"] == "sigma":
            sp_inds = [
                i_var
                for i_var, var in enumerate(varname_output)
                if var in conf["model"]["post_conf"]["global_energy_fixer"]["surface_pressure_name"]
            ]
            conf["model"]["post_conf"]["global_energy_fixer"]["sp_inds"] = sp_inds[0]

    # --------------------------------------------------------- #
    # conf['trainer'] section

    if parse_training:
        assert "mode" in conf["trainer"], "Resource type ('mode') is missing from conf['trainer']"

        assert conf["trainer"]["mode"] in [
            "fsdp",
            "ddp",
            "none",
        ], "conf['trainer']['mode'] accepts fsdp, ddp, and none"

        assert "type" in conf["trainer"], "Training strategy ('type') is missing from conf['trainer']"

        assert "load_weights" in conf["trainer"], "must specify 'load_weights' in conf['trainer']"
        assert "learning_rate" in conf["trainer"], "must specify 'learning_rate' in conf['trainer']"

        assert (
            "batches_per_epoch" in conf["trainer"]
        ), "Number of training batches per epoch ('batches_per_epoch') is missing from onf['trainer']"

        assert (
            "train_batch_size" in conf["trainer"]
        ), "Training set batch size ('train_batch_size') is missing from onf['trainer']"

        if "ensemble_size" not in conf["trainer"]:
            conf["trainer"]["ensemble_size"] = 1  # default value of 1 means deterministic training

        if conf["trainer"]["ensemble_size"] > 1:
            assert (
                conf["loss"]["training_loss"] in ["KCRPS", "almost-fair-crps"]
            ), f"""{conf["loss"]["training_loss"]} loss incompatible with ensemble training. ensemble_size is {conf["trainer"]["ensemble_size"]}"""

        if "load_scaler" not in conf["trainer"]:
            conf["trainer"]["load_scaler"] = False

        if "load_scheduler" not in conf["trainer"]:
            conf["trainer"]["load_scheduler"] = False

        if "load_optimizer" not in conf["trainer"]:
            conf["trainer"]["load_optimizer"] = False

        if "thread_workers" not in conf["trainer"]:
            conf["trainer"]["thread_workers"] = 4

        if "valid_thread_workers" not in conf["trainer"]:
            conf["trainer"]["valid_thread_workers"] = 0

        if "save_backup_weights" not in conf["trainer"]:
            conf["trainer"]["save_backup_weights"] = False

        if "save_best_weights" not in conf["trainer"]:
            conf["trainer"]["save_best_weights"] = True

        if "skip_validation" not in conf["trainer"]:
            conf["trainer"]["skip_validation"] = False

        if conf["trainer"]["skip_validation"] is False:
            # do not skip validaiton
            assert (
                "valid_batch_size" in conf["trainer"]
            ), "Validation set batch size ('valid_batch_size') is missing from conf['trainer']"

            assert (
                "valid_batches_per_epoch" in conf["trainer"]
            ), "Number of validation batches per epoch ('valid_batches_per_epoch') is missing from conf['trainer']"

        if "save_metric_vars" not in conf["trainer"]:
            conf["trainer"]["save_metric_vars"] = []  # averaged metrics only

        if "use_scheduler" in conf["trainer"]:
            # ------------------------------------------------------------------------------ #
            # if use scheduler
            if conf["trainer"]["use_scheduler"]:
                # lr will be controlled by scheduler
                conf["trainer"]["update_learning_rate"] = False

                assert (
                    "scheduler" in conf["trainer"]
                ), "must specify 'scheduler' in conf['trainer'] when a scheduler is used"

                assert (
                    "reload_epoch" in conf["trainer"]
                ), "must specify 'reload_epoch' in conf['trainer'] when a scheduler is used"

                assert (
                    "load_optimizer" in conf["trainer"]
                ), "must specify 'load_optimizer' in conf['trainer'] when a scheduler is used"

                assert (
                    "load_scheduler" in conf["trainer"]
                ), "must specify 'load_scheduler' in conf['trainer'] when a scheduler is used"

            # ------------------------------------------------------------------------------ #
            else:
                if "load_scaler" not in conf["trainer"]:
                    conf["trainer"]["load_scaler"] = False

                if "load_scheduler" not in conf["trainer"]:
                    conf["trainer"]["load_scheduler"] = False

                if "load_optimizer" not in conf["trainer"]:
                    conf["trainer"]["load_optimizer"] = False

        if "update_learning_rate" not in conf["trainer"]:
            conf["trainer"]["update_learning_rate"] = False

        if "train_one_epoch" not in conf["trainer"]:
            conf["trainer"]["train_one_epoch"] = False

        if conf["trainer"]["train_one_epoch"] is False:
            assert "start_epoch" in conf["trainer"], "must specify 'start_epoch' in conf['trainer']"
            assert "epochs" in conf["trainer"], "must specify 'epochs' in conf['trainer']"
        else:
            conf["trainer"]["epochs"] = 999
            if "num_epoch" in conf["trainer"]:
                warnings.warn(
                    "conf['trainer']['num_epoch'] will be overridden by conf['trainer']['train_one_epoch']: True"
                )

        if "amp" not in conf["trainer"]:
            conf["trainer"]["amp"] = False

        if conf["trainer"]["amp"]:
            assert "load_scaler" in conf["trainer"], "must specify 'load_scaler' in conf['trainer'] if AMP is used"

        if "weight_decay" not in conf["trainer"]:
            conf["trainer"]["weight_decay"] = 0

        if "stopping_patience" not in conf["trainer"]:
            conf["trainer"]["stopping_patience"] = 999

        if "activation_checkpoint" not in conf["trainer"]:
            conf["trainer"]["activation_checkpoint"] = True

        if "cpu_offload" not in conf["trainer"]:
            conf["trainer"]["cpu_offload"] = False

        if "grad_accum_every" not in conf["trainer"]:
            conf["trainer"]["grad_accum_every"] = 1

        # gradient clipping
        conf["trainer"].setdefault("grad_max_norm", None)

        if conf["trainer"]["grad_max_norm"] == 0:
            conf["trainer"]["grad_max_norm"] = None

    # --------------------------------------------------------- #
    # conf['loss'] section

    if parse_training:
        assert "training_loss" in conf["loss"], "Training loss ('training_loss') is missing from conf['loss']"
        assert "use_latitude_weights" in conf["loss"], "must specify 'use_latitude_weights' in conf['loss']"
        assert "use_variable_weights" in conf["loss"], "must specify 'use_variable_weights' in conf['loss']"

        if conf["loss"]["use_variable_weights"]:
            assert (
                "variable_weights" in conf["loss"]
            ), "must specify 'variable_weights' in conf['loss'] if 'use_variable_weights': True"

            # ----------------------------------------------------------------------------------------- #
            # check and reorganize variable weights
            varname_upper_air = conf["data"]["variables"]
            varname_surface = conf["data"]["surface_variables"]
            varname_diagnostics = conf["data"]["diagnostic_variables"]
            N_levels = conf["data"]["levels"]

            weights_dict_ordered = {}

            varname_covered = list(conf["loss"]["variable_weights"].keys())

            for varname in varname_upper_air:
                assert varname in varname_covered, "missing variable weights for '{}'".format(varname)
                N_weights = len(conf["loss"]["variable_weights"][varname])
                assert N_weights == N_levels, "{} levels were defined, but weights only have {} levels".format(
                    N_levels, N_weights
                )
                weights_dict_ordered[varname] = conf["loss"]["variable_weights"][varname]

            for varname in varname_surface + varname_diagnostics:
                assert varname in varname_covered, "missing variable weights for '{}'".format(varname)
                weights_dict_ordered[varname] = conf["loss"]["variable_weights"][varname]

            conf["loss"]["variable_weights"] = weights_dict_ordered
            # ----------------------------------------------------------------------------------------- #

        if "use_power_loss" not in conf["loss"]:
            conf["loss"]["use_power_loss"] = False

        if "use_spectral_loss" not in conf["loss"]:
            conf["loss"]["use_spectral_loss"] = False

        if conf["loss"]["use_power_loss"] and conf["loss"]["use_spectral_loss"]:
            warnings.warn("'use_power_loss: True' and 'use_spectral_loss: True' are both applied")

        if conf["loss"]["use_power_loss"] or conf["loss"]["use_spectral_loss"]:
            if "spectral_lambda_reg" not in conf["loss"]:
                conf["loss"]["spectral_lambda_reg"] = 0.1

            if "spectral_wavenum_init" not in conf["loss"]:
                conf["loss"]["spectral_wavenum_init"] = 20

    # --------------------------------------------------------- #
    # conf['parse_predict'] section

    if parse_predict:
        assert "forecasts" in conf["predict"], "Rollout settings ('forecasts') is missing from conf['predict']"
        assert (
            "save_forecast" in conf["predict"]
        ), "Rollout save location ('save_forecast') is missing from conf['predict']"

        conf["predict"]["save_forecast"] = os.path.expandvars(conf["predict"]["save_forecast"])

        if "use_laplace_filter" not in conf["predict"]:
            conf["predict"]["use_laplace_filter"] = False

        if "metadata" not in conf["predict"]:
            conf["predict"]["metadata"] = False

        if "save_vars" not in conf["predict"]:
            conf["predict"]["save_vars"] = []

        if "mode" not in conf["predict"]:
            if "mode" in conf["trainer"]:
                conf["predict"]["mode"] = conf["trainer"]["mode"]
            else:
                print("Resource type ('mode') is missing from both conf['trainer'] and conf['predict']")
                raise

    # ==================================================== #
    # print summary
    if print_summary:
        print("Upper-air variables: {}".format(conf["data"]["variables"]))
        print("Surface variables: {}".format(conf["data"]["surface_variables"]))
        print("Dynamic forcing variables: {}".format(conf["data"]["dynamic_forcing_variables"]))
        print("Diagnostic variables: {}".format(conf["data"]["diagnostic_variables"]))
        print("Forcing variables: {}".format(conf["data"]["forcing_variables"]))
        print("Static variables: {}".format(conf["data"]["static_variables"]))

    return conf


def training_data_check(conf, print_summary=False):
    """
    Note: this function is designed for model training, NOT for rollout

    The following items are covered:
        - All yearly files (upper-air, surface, dynamic forcing, diagnostic)
          can support conf['data']['train_years'], conf['data']['valid_years']
        - All variables (upper-air, surface, dynamic forcing, diagnostic)
          do exist in their corresponding files
          Note: only one file of each group will be checked.
        - All files (upper-air, surface, dynamic forcing, diagnostic, forcing, static, mean, std, lat_weights)
          have the same coordinate names and coordinate values
          Note: this part checks lat, lon, level coordinates, and it ignores 'time' coordinates.

    Where is it applied?
        - applications/train.py
        - applications/train_multistep.py

    """
    # -------------------------------------------------- #
    # import training / validation years from conf

    train_years_range = conf["data"]["train_years"]
    valid_years_range = conf["data"]["valid_years"]

    # convert year info to str for file name search
    train_years = [str(year) for year in range(train_years_range[0], train_years_range[1])]
    valid_years = [str(year) for year in range(valid_years_range[0], valid_years_range[1])]

    # -------------------------------------------------- #
    # check file consistencies
    ## upper-air files
    all_ERA_files = sorted(glob(conf["data"]["save_loc"]))

    train_ERA_files = [file for file in all_ERA_files if any(year in file for year in train_years)]
    valid_ERA_files = [file for file in all_ERA_files if any(year in file for year in valid_years)]

    for i_year, year in enumerate(train_years):
        assert year in train_ERA_files[i_year], "[Year {}] is missing from [upper-air files {}]".format(
            year, conf["data"]["save_loc"]
        )

    for i_year, year in enumerate(valid_years):
        assert year in valid_ERA_files[i_year], "[Year {}] is missing from [upper-air files {}]".format(
            year, conf["data"]["save_loc"]
        )

    ## surface files
    if conf["data"]["flag_surface"]:
        surface_files = sorted(glob(conf["data"]["save_loc_surface"]))

        train_surface_files = [file for file in surface_files if any(year in file for year in train_years)]
        valid_surface_files = [file for file in surface_files if any(year in file for year in valid_years)]

        for i_year, year in enumerate(train_years):
            assert year in train_surface_files[i_year], "[Year {}] is missing from [surface files {}]".format(
                year, conf["data"]["save_loc_surface"]
            )

        for i_year, year in enumerate(valid_years):
            assert year in valid_surface_files[i_year], "[Year {}] is missing from [surface files {}]".format(
                year, conf["data"]["save_loc_surface"]
            )

    ## dynamic forcing files
    if conf["data"]["flag_dyn_forcing"]:
        dyn_forcing_files = sorted(glob(conf["data"]["save_loc_dynamic_forcing"]))

        train_dyn_forcing_files = [file for file in dyn_forcing_files if any(year in file for year in train_years)]
        valid_dyn_forcing_files = [file for file in dyn_forcing_files if any(year in file for year in valid_years)]

        for i_year, year in enumerate(train_years):
            assert (
                year in train_dyn_forcing_files[i_year]
            ), "[Year {}] is missing from [dynamic forcing files {}]".format(
                year, conf["data"]["save_loc_dynamic_forcing"]
            )

        for i_year, year in enumerate(valid_years):
            assert (
                year in valid_dyn_forcing_files[i_year]
            ), "[Year {}] is missing from [dynamic forcing files {}]".format(
                year, conf["data"]["save_loc_dynamic_forcing"]
            )

    ## diagnostic files
    if conf["data"]["flag_diagnostic"]:
        diagnostic_files = sorted(glob(conf["data"]["save_loc_diagnostic"]))

        train_diagnostic_files = [file for file in diagnostic_files if any(year in file for year in train_years)]
        valid_diagnostic_files = [file for file in diagnostic_files if any(year in file for year in valid_years)]

        for i_year, year in enumerate(train_years):
            assert year in train_diagnostic_files[i_year], "[Year {}] is missing from [diagnostic files {}]".format(
                year, conf["data"]["save_loc_diagnostic"]
            )

        for i_year, year in enumerate(valid_years):
            assert year in valid_diagnostic_files[i_year], "[Year {}] is missing from [diagnostic files {}]".format(
                year, conf["data"]["save_loc_diagnostic"]
            )

    if print_summary:
        print("Filename checking passed")
        print("All input files can cover conf['data']['train_years'] and conf['data']['valid_years']")

    # --------------------------------------------------------------------------- #
    # variable checks
    # !!! the first train file (e.g., train_ERA_files[0]) is opened and examined

    # upper-air variables
    ds_upper_air = get_forward_data(train_ERA_files[0])
    varnames_upper_air = list(ds_upper_air.keys())

    assert all(
        varname in varnames_upper_air for varname in conf["data"]["variables"]
    ), "upper-air variables [{}] are not fully covered by conf['data']['save_loc']".format(conf["data"]["variables"])

    # assign the upper_air vars in yaml if it can pass checks
    varnames_upper_air = conf["data"]["variables"]

    # collecting all variables that require zscores
    # deep copy to avoid changing conf['data'] by accident
    all_vars = copy.deepcopy(conf["data"]["variables"])

    # surface variables
    if conf["data"]["flag_surface"]:
        ds_surface = get_forward_data(train_surface_files[0])
        varnames_surface = list(ds_surface.keys())

        assert all(
            varname in varnames_surface for varname in conf["data"]["surface_variables"]
        ), "Surface variables [{}] are not fully covered by conf['data']['save_loc_surface']".format(
            conf["data"]["surface_variables"]
        )

        all_vars += conf["data"]["surface_variables"]

    # dynamic forcing variables
    if conf["data"]["flag_dyn_forcing"]:
        ds_dyn_forcing = get_forward_data(train_dyn_forcing_files[0])
        varnames_dyn_forcing = list(ds_dyn_forcing.keys())

        assert all(
            varname in varnames_dyn_forcing for varname in conf["data"]["dynamic_forcing_variables"]
        ), "Dynamic forcing variables [{}] are not fully covered by conf['data']['save_loc_dynamic_forcing']".format(
            conf["data"]["dynamic_forcing_variables"]
        )

        all_vars += conf["data"]["dynamic_forcing_variables"]

    # diagnostic variables
    if conf["data"]["flag_diagnostic"]:
        ds_diagnostic = get_forward_data(train_diagnostic_files[0])
        varnames_diagnostic = list(ds_diagnostic.keys())

        assert all(
            varname in varnames_diagnostic for varname in conf["data"]["diagnostic_variables"]
        ), "Diagnostic variables [{}] are not fully covered by conf['data']['save_loc_diagnostic']".format(
            conf["data"]["diagnostic_variables"]
        )

        all_vars += conf["data"]["diagnostic_variables"]

    # forcing variables
    if conf["data"]["flag_forcing"]:
        ds_forcing = get_forward_data(conf["data"]["save_loc_forcing"])
        varnames_forcing = list(ds_forcing.keys())

        assert all(
            varname in varnames_forcing for varname in conf["data"]["forcing_variables"]
        ), "Forcing variables [{}] are not fully covered by conf['data']['save_loc_forcing']".format(
            conf["data"]["forcing_variables"]
        )

    # static variables
    if conf["data"]["flag_static"]:
        ds_static = get_forward_data(conf["data"]["save_loc_static"])
        varnames_static = list(ds_static.keys())

        assert all(
            varname in varnames_static for varname in conf["data"]["static_variables"]
        ), "Static variables [{}] are not fully covered by conf['data']['save_loc_static']".format(
            conf["data"]["static_variables"]
        )

    # comparing all_vars against mean, std files
    ds_mean = get_forward_data(conf["data"]["mean_path"])
    varname_ds_mean = list(ds_mean.keys())

    assert all(
        varname in varname_ds_mean for varname in all_vars
    ), "Variables are not fully covered by conf['data']['mean_path']"

    ds_std = get_forward_data(conf["data"]["std_path"])
    varname_ds_std = list(ds_std.keys())

    assert all(
        varname in varname_ds_std for varname in all_vars
    ), "Variables are not fully covered by conf['data']['std_path']"

    if print_summary:
        print("Variable name checking passed")
        print("All input files and zscore files have the required variables")

    # -------------------------------------------------- #
    # xr.Dataset coordinate checks
    # !!!! assuming time-coordinate has string pattern 'time' !!!!
    # !!!! Can be improved !!!!

    coord_upper_air = list(ds_upper_air.coords.keys())
    coord_upper_air = remove_string_by_pattern(coord_upper_air, "time")

    # surface files
    if conf["data"]["flag_surface"]:
        coord_surface = list(ds_surface.coords.keys())
        coord_surface = remove_string_by_pattern(coord_surface, "time")

        assert all(
            coord_name in coord_upper_air for coord_name in coord_surface
        ), "Surface file coordinate names mismatched with upper-air files"

        for coord_name in coord_surface:
            assert ds_upper_air.coords[coord_name].equals(
                ds_surface.coords[coord_name]
            ), "coordinate {} mismatched between upper-air and surface files".format(coord_name)

    # dyn forcing files
    if conf["data"]["flag_dyn_forcing"]:
        coord_dyn_forcing = list(ds_dyn_forcing.coords.keys())
        coord_dyn_forcing = remove_string_by_pattern(coord_dyn_forcing, "time")

        assert all(
            coord_name in coord_upper_air for coord_name in coord_dyn_forcing
        ), "Dynamic forcing file coordinate names mismatched with upper-air files"

        for coord_name in coord_dyn_forcing:
            assert ds_upper_air.coords[coord_name].equals(
                ds_dyn_forcing.coords[coord_name]
            ), "coordinate {} mismatched between upper-air and dynamic forcing files".format(coord_name)

    # diagnostic files
    if conf["data"]["flag_diagnostic"]:
        coord_diagnostic = list(ds_diagnostic.coords.keys())
        coord_diagnostic = remove_string_by_pattern(coord_diagnostic, "time")

        assert all(
            coord_name in coord_upper_air for coord_name in coord_diagnostic
        ), "Diagnostic file coordinate names mismatched with upper-air files"

        for coord_name in coord_diagnostic:
            assert ds_upper_air.coords[coord_name].equals(
                ds_diagnostic.coords[coord_name]
            ), "coordinate {} mismatched between upper-air and diagnostic files".format(coord_name)

    # forcing files
    if conf["data"]["flag_forcing"]:
        coord_forcing = list(ds_forcing.coords.keys())
        coord_forcing = remove_string_by_pattern(coord_forcing, "time")

        assert all(
            coord_name in coord_upper_air for coord_name in coord_forcing
        ), "Forcing file coordinate names mismatched with upper-air files"

        for coord_name in coord_forcing:
            assert ds_upper_air.coords[coord_name].equals(
                ds_forcing.coords[coord_name]
            ), "coordinate {} mismatched between upper-air and forcing files".format(coord_name)

        # ============================================== #
        # !! assumed subdaily inputs, may need to fix !! #
        assert len(ds_forcing["time"]) % 366 == 0, "forcing file does not have 366 days"
        # ============================================== #

    # static files (no time coordinate)
    if conf["data"]["flag_static"]:
        coord_static = list(ds_static.coords.keys())
        coord_static = remove_string_by_pattern(coord_static, "time")

        assert all(
            coord_name in coord_upper_air for coord_name in coord_static
        ), "Static file coordinate names mismatched with upper-air files"

        for coord_name in coord_static:
            assert ds_upper_air.coords[coord_name].equals(
                ds_static.coords[coord_name]
            ), "coordinate {} mismatched between upper-air and static files".format(coord_name)

    # zscore mean file (no time coordinate)
    coord_mean = list(ds_mean.coords.keys())
    coord_mean = remove_string_by_pattern(coord_mean, "time")

    assert all(
        coord_name in coord_upper_air for coord_name in coord_mean
    ), "zscore mean file coordinate names mismatched with upper-air files"

    for coord_name in coord_mean:
        assert ds_upper_air.coords[coord_name].equals(
            ds_mean.coords[coord_name]
        ), "coordinate {} mismatched between upper-air and mean files".format(coord_name)

    # zscore std file (no time coordinate)
    coord_std = list(ds_std.coords.keys())
    coord_std = remove_string_by_pattern(coord_std, "time")

    assert all(
        coord_name in coord_upper_air for coord_name in coord_std
    ), "zscore std file coordinate names mismatched with upper-air files"

    for coord_name in coord_std:
        assert ds_upper_air.coords[coord_name].equals(
            ds_std.coords[coord_name]
        ), "coordinate {} mismatched between upper-air and std files".format(coord_name)

    # lat / lon file
    ds_weights = get_forward_data(conf["loss"]["latitude_weights"])
    coord_latlon = list(ds_weights.coords.keys())
    coord_latlon = remove_string_by_pattern(coord_latlon, "time")

    assert all(
        coord_name in coord_upper_air for coord_name in coord_latlon
    ), "conf['loss']['latitude_weights'] file coordinate names mismatched with upper-air files"

    # model level consistency final checks
    N_level_mean = len(ds_mean[varnames_upper_air[0]].values)
    N_level_model = conf["model"]["levels"]

    assert (
        N_level_mean == N_level_model
    ), "number of upper air levels mismatched between model config {} and input data {}".format(
        N_level_model, N_level_mean
    )

    if print_summary:
        print("Coordinate checking passed")
        print(
            "All input files, zscore files, and the lat/lon file share the same lat, lon, level coordinate name and values"
        )

    return True


def predict_data_check(conf, print_summary=False):
    """
    Note: this function is designed for model rollout.
          Diagnostic variables are checked in mean and std files only

    The following items are covered:
        - All variables (upper-air, surface, dynamic forcing)
          do exist in their corresponding files
          Note: only one file of each group will be checked.
        - All files (upper-air, surface, dynamic forcing, forcing, static, mean, std, lat_weights)
          have the same coordinate names and coordinate values
          Note: this part checks lat, lon, level coordinates, and it ignores 'time' coordinates.

    Where is it applied?
        - applications/rollout_to_netcdf_new.py

    """
    # ----------------------------------------------------------------- #
    # a rough estimate of how manys years of initializations are needed
    # !!! Can be improved !!!
    if "duration" in conf["predict"]["forecasts"]:
        assert "start_year" in conf["predict"]["forecasts"], "Must specify which year to start predict."
        N_years = conf["predict"]["forecasts"]["duration"] // 365
        N_years = N_years + 1
    else:
        N_years = 0 + 1

    years_range = [
        conf["predict"]["forecasts"]["start_year"],
        conf["predict"]["forecasts"]["start_year"] + N_years,
    ]
    pred_years = [str(year) for year in range(years_range[0], years_range[1])]
    # -------------------------------------------------- #
    # check file consistencies
    ## upper-air files
    all_ERA_files = sorted(glob(conf["data"]["save_loc"]))

    pred_ERA_files = [file for file in all_ERA_files if any(year in file for year in pred_years)]

    if len(pred_years) != len(pred_ERA_files):
        warnings.warn("Provided initializations in upper air files may not cover all forecasted dates")

    ## surface files
    if conf["data"]["flag_surface"]:
        surface_files = sorted(glob(conf["data"]["save_loc_surface"]))

        pred_surface_files = [file for file in surface_files if any(year in file for year in pred_years)]

        if len(pred_years) != len(pred_surface_files):
            warnings.warn("Provided initializations in surface files may not cover all forecasted dates")

    ## dynamic forcing files
    if conf["data"]["flag_dyn_forcing"]:
        dyn_forcing_files = sorted(glob(conf["data"]["save_loc_dynamic_forcing"]))

        pred_dyn_forcing_files = [file for file in dyn_forcing_files if any(year in file for year in pred_years)]

        if len(pred_years) != len(pred_dyn_forcing_files):
            warnings.warn("Provided initializations in surface files may not cover all forecasted dates")

    if print_summary:
        print("Filename checking passed")
        print("All input files can cover conf['predict']['forecasts']['duration']")

    # --------------------------------------------------------------------------- #
    # variable checks
    # !!! the first pred file (e.g., pred_ERA_files[0]) is opened and examined

    # upper-air variables
    ds_upper_air = get_forward_data(pred_ERA_files[0])
    varnames_upper_air = list(ds_upper_air.keys())

    assert all(
        varname in varnames_upper_air for varname in conf["data"]["variables"]
    ), "upper-air variables [{}] are not fully covered by conf['data']['save_loc']".format(conf["data"]["variables"])

    # collecting all variables that require zscores
    # deep copy to avoid changing conf['data'] by accident
    all_vars = copy.deepcopy(conf["data"]["variables"])

    # surface variables
    if conf["data"]["flag_surface"]:
        ds_surface = get_forward_data(pred_surface_files[0])
        varnames_surface = list(ds_surface.keys())

        assert all(
            varname in varnames_surface for varname in conf["data"]["surface_variables"]
        ), "Surface variables [{}] are not fully covered by conf['data']['save_loc_surface']".format(
            conf["data"]["surface_variables"]
        )

        all_vars += conf["data"]["surface_variables"]

    # dynamic forcing variables
    if conf["data"]["flag_dyn_forcing"]:
        ds_dyn_forcing = get_forward_data(pred_dyn_forcing_files[0])
        varnames_dyn_forcing = list(ds_dyn_forcing.keys())

        assert all(
            varname in varnames_dyn_forcing for varname in conf["data"]["dynamic_forcing_variables"]
        ), "Dynamic forcing variables [{}] are not fully covered by conf['data']['save_loc_dynamic_forcing']".format(
            conf["data"]["dynamic_forcing_variables"]
        )

        all_vars += conf["data"]["dynamic_forcing_variables"]

    # diagnostic variables
    if conf["data"]["flag_diagnostic"]:
        all_vars += conf["data"]["diagnostic_variables"]

    # forcing variables
    if conf["data"]["flag_forcing"]:
        ds_forcing = get_forward_data(conf["data"]["save_loc_forcing"])
        varnames_forcing = list(ds_forcing.keys())

        assert all(
            varname in varnames_forcing for varname in conf["data"]["forcing_variables"]
        ), "Forcing variables [{}] are not fully covered by conf['data']['save_loc_forcing']".format(
            conf["data"]["forcing_variables"]
        )

    # static variables
    if conf["data"]["flag_static"]:
        ds_static = get_forward_data(conf["data"]["save_loc_static"])
        varnames_static = list(ds_static.keys())

        assert all(
            varname in varnames_static for varname in conf["data"]["static_variables"]
        ), "Static variables [{}] are not fully covered by conf['data']['save_loc_static']".format(
            conf["data"]["static_variables"]
        )

    # comparing all_vars against mean, std files
    ds_mean = get_forward_data(conf["data"]["mean_path"])
    varname_ds_mean = list(ds_mean.keys())

    assert all(
        varname in varname_ds_mean for varname in all_vars
    ), "Variables are not fully covered by conf['data']['mean_path']"

    ds_std = get_forward_data(conf["data"]["std_path"])
    varname_ds_std = list(ds_std.keys())

    assert all(
        varname in varname_ds_std for varname in all_vars
    ), "Variables are not fully covered by conf['data']['std_path']"

    if print_summary:
        print("Variable name checking passed")
        print("All input files and zscore files have the required variables")

    # -------------------------------------------------- #
    # xr.Dataset coordinate checks
    # !!!! assuming time-coordinate has string pattern 'time' !!!!
    # !!!! Can be improved !!!!

    coord_upper_air = list(ds_upper_air.coords.keys())
    coord_upper_air = remove_string_by_pattern(coord_upper_air, "time")

    # surface files
    if conf["data"]["flag_surface"]:
        coord_surface = list(ds_surface.coords.keys())
        coord_surface = remove_string_by_pattern(coord_surface, "time")

        assert all(
            coord_name in coord_upper_air for coord_name in coord_surface
        ), "Surface file coordinate names mismatched with upper-air files"

        for coord_name in coord_surface:
            assert ds_upper_air.coords[coord_name].equals(
                ds_surface.coords[coord_name]
            ), "coordinate {} mismatched between upper-air and surface files".format(coord_name)

    # dyn forcing files
    if conf["data"]["flag_dyn_forcing"]:
        coord_dyn_forcing = list(ds_dyn_forcing.coords.keys())
        coord_dyn_forcing = remove_string_by_pattern(coord_dyn_forcing, "time")

        assert all(
            coord_name in coord_upper_air for coord_name in coord_dyn_forcing
        ), "Dynamic forcing file coordinate names mismatched with upper-air files"

        for coord_name in coord_dyn_forcing:
            assert ds_upper_air.coords[coord_name].equals(
                ds_dyn_forcing.coords[coord_name]
            ), "coordinate {} mismatched between upper-air and dynamic forcing files".format(coord_name)

    # forcing files
    if conf["data"]["flag_forcing"]:
        coord_forcing = list(ds_forcing.coords.keys())
        coord_forcing = remove_string_by_pattern(coord_forcing, "time")

        assert all(
            coord_name in coord_upper_air for coord_name in coord_forcing
        ), "Forcing file coordinate names mismatched with upper-air files"

        for coord_name in coord_forcing:
            assert ds_upper_air.coords[coord_name].equals(
                ds_forcing.coords[coord_name]
            ), "coordinate {} mismatched between upper-air and forcing files".format(coord_name)

        # ============================================== #
        # !! assumed subdaily inputs, may need to fix !! #
        assert len(ds_forcing["time"]) % 366 == 0, "forcing file does not have 366 days"
        # ============================================== #

    # static files (no time coordinate)
    if conf["data"]["flag_static"]:
        coord_static = list(ds_static.coords.keys())
        coord_static = remove_string_by_pattern(coord_static, "time")

        assert all(
            coord_name in coord_upper_air for coord_name in coord_static
        ), "Static file coordinate names mismatched with upper-air files"

        for coord_name in coord_static:
            assert ds_upper_air.coords[coord_name].equals(
                ds_static.coords[coord_name]
            ), "coordinate {} mismatched between upper-air and static files".format(coord_name)

    # zscore mean file (no time coordinate)
    coord_mean = list(ds_mean.coords.keys())
    coord_mean = remove_string_by_pattern(coord_mean, "time")

    assert all(
        coord_name in coord_upper_air for coord_name in coord_mean
    ), "zscore mean file coordinate names mismatched with upper-air files"

    for coord_name in coord_mean:
        assert ds_upper_air.coords[coord_name].equals(
            ds_mean.coords[coord_name]
        ), "coordinate {} mismatched between upper-air and mean files".format(coord_name)

    # zscore std file (no time coordinate)
    coord_std = list(ds_std.coords.keys())
    coord_std = remove_string_by_pattern(coord_std, "time")

    assert all(
        coord_name in coord_upper_air for coord_name in coord_std
    ), "zscore std file coordinate names mismatched with upper-air files"

    for coord_name in coord_std:
        assert ds_upper_air.coords[coord_name].equals(
            ds_std.coords[coord_name]
        ), "coordinate {} mismatched between upper-air and std files".format(coord_name)

    # lat / lon file
    ds_weights = get_forward_data(conf["loss"]["latitude_weights"])
    coord_latlon = list(ds_weights.coords.keys())
    coord_latlon = remove_string_by_pattern(coord_latlon, "time")

    assert all(
        coord_name in coord_upper_air for coord_name in coord_latlon
    ), "conf['loss']['latitude_weights'] file coordinate names mismatched with upper-air files"

    if print_summary:
        print("Coordinate checking passed")
        print(
            "All input files, zscore files, and the lat/lon file share the same lat, lon, level coordinate name and values"
        )

    return True
