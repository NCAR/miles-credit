import glob
import importlib
import logging
import os
import sys

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config-driven dispatch: maps config-file keys → class.
# Currently empty — built-in datasets are not yet registered here (see _load_dataset_entry NOTE).
# Custom datasets added via @register_dataset will populate this at runtime.
# Registry entries are either:
#   (module_path: str, class_name: str)  — built-in lazy entries
#   cls: type                             — externally registered classes
_DATASET_REGISTRY = {}

# Direct-import table: maps Python names → object for lazy module attribute access.
# Enables ``from credit.datasets import ERA5Dataset`` without eager imports; kept for backward compatibility.
# Entries are mostly classes, but also includes functions (build_channel_layout, update_x).
_CLASS_SOURCES = {
    "BaseDataset": ("credit.datasets.base_dataset", "BaseDataset"),
    "MultiSourceDataset": ("credit.datasets.multi_source", "MultiSourceDataset"),
    "ERA5Dataset": ("credit.datasets.era5", "ERA5Dataset"),
    "ARCOERA5Dataset": ("credit.datasets.era5", "ARCOERA5Dataset"),
    "WeatherBench2ERA5Dataset": ("credit.datasets.era5", "WeatherBench2ERA5Dataset"),
    "GOESDataset": ("credit.datasets.goes", "GOESDataset"),
    "MRMSDataset": ("credit.datasets.mrms", "MRMSDataset"),
    "HRRRDataset": ("credit.datasets.hrrr", "HRRRDataset"),
    "TISRDataset": ("credit.datasets.tisr", "TISRDataset"),
    "build_channel_layout": ("credit.datasets.channel_layout", "build_channel_layout"),
    "update_x": ("credit.datasets.channel_layout", "update_x"),
}


# ---------------------------------------------------------------------------
# Module __getattr__: called when a name is not found via normal attribute lookup.
# Resolves names listed in _CLASS_SOURCES lazily so submodules are only imported on first access.
# Example: ``from credit.datasets import ERA5Dataset`` triggers __getattr__("ERA5Dataset"),
#          which imports credit.datasets.era5 on the spot and returns the class.
def __getattr__(name):
    if name in _CLASS_SOURCES:
        module_path, class_name = _CLASS_SOURCES[name]
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except ImportError as exc:
            raise AttributeError(f"Cannot import {name!r}: optional dependencies missing.") from exc
    raise AttributeError(f"module 'credit.datasets' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Registration
def register_dataset(dataset_type):
    """Decorator that adds an external dataset class to the dataset registry.

    The class must inherit from :class:`credit.datasets.base_dataset.BaseDataset`
    and will be instantiated with the signature::

        dataset = MyDataset(conf, rank=rank, world_size=world_size, is_train=is_train)

    Args:
        dataset_type: Key used in the config ``data.dataset_type`` field.

    Example::

        from credit.datasets import register_dataset
        from credit.datasets.base_dataset import BaseDataset

        @register_dataset("my_dataset")
        class MyDataset(BaseDataset):
            def __init__(self, conf, rank=0, world_size=1, is_train=True):
                ...
    """

    def decorator(cls):
        from credit.datasets.base_dataset import BaseDataset  # imported here to avoid loading it at module import time

        # isinstance(cls, type) guards against passing an instance or function;
        # issubclass then confirms it inherits the required base class.
        if not (isinstance(cls, type) and issubclass(cls, BaseDataset)):
            raise TypeError(
                f"register_dataset: '{cls.__name__}' must inherit from credit.datasets.base_dataset.BaseDataset."
            )
        if dataset_type in _DATASET_REGISTRY:  # warn instead of silently overwriting
            logger.warning(f"register_dataset: overwriting existing registry entry for '{dataset_type}'")
        _DATASET_REGISTRY[dataset_type] = cls  # store the class under the given key
        return cls  # must return the class, otherwise it becomes None after decoration

    return decorator


def _load_dataset_entry(dataset_type):
    """Return the class for a registered dataset type, importing lazily if needed.

    NOTE: Not currently used — load_dataset_and_dataloader.py still uses the legacy
    if/elif dispatch and does not consult _DATASET_REGISTRY. Built-in datasets should
    be added to _DATASET_REGISTRY and load_dataset_and_dataloader.py updated to call
    this in a future cleanup PR.

    Raises:
        ValueError: If dataset_type is not in _DATASET_REGISTRY.
        ImportError: If the dataset's module cannot be imported (missing optional dependencies).
    """
    if dataset_type not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset type '{dataset_type}'. "
            f"Available types: {sorted(_DATASET_REGISTRY)}. "
            "Register a custom dataset with @register_dataset or via custom_objects in your config."
        )
    entry = _DATASET_REGISTRY[dataset_type]
    if isinstance(entry, tuple):
        module_path, class_name = entry
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except ImportError as exc:
            raise ImportError(
                f"Dataset type '{dataset_type}' requires optional dependencies that are not installed. "
                f"Original error: {exc}"
            ) from exc
    return entry  # externally registered class stored directly


# ---------------------------------------------------------------------------
# Data loading utilities
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
        if ("surface_variables" in conf["data"]) and (len(conf["data"]["surface_variables"]) > 0):
            surface_files = sorted(glob.glob(conf["data"]["save_loc_surface"]))

        else:
            surface_files = None

        # check and glob dyn forcing files
        if ("dynamic_forcing_variables" in conf["data"]) and (len(conf["data"]["dynamic_forcing_variables"]) > 0):
            dyn_forcing_files = sorted(glob.glob(conf["data"]["save_loc_dynamic_forcing"]))

        else:
            dyn_forcing_files = None

        # check and glob diagnostic files
        if ("diagnostic_variables" in conf["data"]) and (len(conf["data"]["diagnostic_variables"]) > 0):
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
    train_years = [str(year) for year in range(train_years_range[0], train_years_range[1])]
    valid_years = [str(year) for year in range(valid_years_range[0], valid_years_range[1])]

    # Filter the files for training / validation
    train_files = [file for file in all_ERA_files if any(year in file for year in train_years)]
    valid_files = [file for file in all_ERA_files if any(year in file for year in valid_years)]

    # <----------------------------------- std_new
    if conf["data"]["scaler_type"] == "std_new":
        if surface_files is not None:
            train_surface_files = [file for file in surface_files if any(year in file for year in train_years)]
            valid_surface_files = [file for file in surface_files if any(year in file for year in valid_years)]

            # ---------------------------- #
            # check total number of files
            assert len(train_surface_files) == len(train_files), (
                "Mismatch between the total number of training set [surface files] and [upper-air files]"
            )
            assert len(valid_surface_files) == len(valid_files), (
                "Mismatch between the total number of validation set [surface files] and [upper-air files]"
            )

        else:
            train_surface_files = None
            valid_surface_files = None

        if dyn_forcing_files is not None:
            train_dyn_forcing_files = [file for file in dyn_forcing_files if any(year in file for year in train_years)]
            valid_dyn_forcing_files = [file for file in dyn_forcing_files if any(year in file for year in valid_years)]

            # ---------------------------- #
            # check total number of files
            assert len(train_dyn_forcing_files) == len(train_files), (
                "Mismatch between the total number of training set [dynamic forcing files] and [upper-air files]"
            )
            assert len(valid_dyn_forcing_files) == len(valid_files), (
                "Mismatch between the total number of validation set [dynamic forcing files] and [upper-air files]"
            )

        else:
            train_dyn_forcing_files = None
            valid_dyn_forcing_files = None

        if diagnostic_files is not None:
            train_diagnostic_files = [file for file in diagnostic_files if any(year in file for year in train_years)]
            valid_diagnostic_files = [file for file in diagnostic_files if any(year in file for year in valid_years)]

            # ---------------------------- #
            # check total number of files
            assert len(train_diagnostic_files) == len(train_files), (
                "Mismatch between the total number of training set [diagnostic files] and [upper-air files]"
            )
            assert len(valid_diagnostic_files) == len(valid_files), (
                "Mismatch between the total number of validation set [diagnostic files] and [upper-air files]"
            )

        else:
            train_diagnostic_files = None
            valid_diagnostic_files = None

    # convert $USER to the actual user name
    conf["save_loc"] = os.path.expandvars(conf["save_loc"])

    # ======================================================== #
    # parse inputs

    # upper air variables
    varname_upper_air = conf["data"]["variables"]

    if ("forcing_variables" in conf["data"]) and (len(conf["data"]["forcing_variables"]) > 0):
        forcing_files = conf["data"]["save_loc_forcing"]
        varname_forcing = conf["data"]["forcing_variables"]
    else:
        forcing_files = None
        varname_forcing = None

    if ("static_variables" in conf["data"]) and (len(conf["data"]["static_variables"]) > 0):
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
