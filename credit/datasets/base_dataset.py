"""
base_dataset.py
-------------------------------------------------------
AbstractBaseDataset and BaseDataset: A PyTorch Dataset class for:
1. Type hinting and annotations throughout CREDIT
2. Scaffolding the development of future datasets
3. Provide a minimal implementation of a Dataset for testing
4. Avoid redundant code across dataset classes

"""

from glob import glob
import logging
from typing import Any, Literal, get_args

import pandas as pd

from torch.utils.data import Dataset
import torch

from credit.datasets._utils import _map_files, _path_template_to_glob, _extract_time_fmt  # pyright: ignore[reportPrivateUsage]

# Expected types of fields
# * ``prognostic``      — input at step 0 and target (autoregressive rollout)
# * ``dynamic_forcing`` — input at every step; never a target
# * ``diagnostic``      — target only
# * ``static``          — input at step 0; never a target, applies to all steps
VALID_FIELD_TYPES = Literal["prognostic", "dynamic_forcing", "static", "diagnostic"]


class AbstractBaseDataset(Dataset[Any]):
    """Abstract base dataset class based on PyTorch Dataset class for CREDIT.

    This class defines the expected methods and attributes for any dataset in CREDIT,
    but does not provide any implementation. The BaseDataset class inherits from this
    class and provides a minimal implementation. Any future dataset should inherit from
    either AbstractBaseDataset or BaseDataset depending on the level of functionality needed.

    For generality, the inheritance is from torch.utils.data.Dataset[Any], however
    there may be benefits to stricter typing than Any for consistency in the
    get item return, especially if torch supports dataset type accelerations in future
    releases.
    """

    def __init__(self, data_config: dict[str, Any], return_target: bool = False) -> None:
        # The name of this source in the config
        self.curr_source_name: str
        self.dataset_type: str

        # Setting the clock for sampling
        self.dt: pd.Timedelta
        self.num_forecast_steps: int
        self.start_datetime: pd.Timestamp
        self.end_datetime: pd.Timestamp
        self.datetimes: pd.DatetimeIndex

        # Getting the data
        self.return_target: bool
        self.mode: str
        self.file_dict: dict[str, Any]
        self.var_dict: dict[str, Any]

        # Placeholder for static metadata
        self.static_metadata: dict[str, Any]

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, args: tuple[pd.Timestamp, int]) -> dict[str, Any]:
        raise NotImplementedError

    def _build_timestamps(self) -> pd.DatetimeIndex:
        raise NotImplementedError

    def _get_field_name(self, field_type: VALID_FIELD_TYPES, dim_str: str, vname: str) -> str:
        raise NotImplementedError

    def init_register_all_fields(self) -> None:
        raise NotImplementedError

    def _register_field(self, field_type: VALID_FIELD_TYPES, field_config: dict[str, Any] | None) -> None:
        raise NotImplementedError

    def _get_file_source(
        self, field_config: dict[str, Any]
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, str]] | bool | None:
        raise NotImplementedError

    def _extract_field(self, field_type: VALID_FIELD_TYPES, t: pd.Timestamp, sample: dict[str, Any]) -> None:
        raise NotImplementedError


class BaseDataset(AbstractBaseDataset):
    """PyTorch Dataset class for CREDIT that  enables:
    1. Type hinting and annotations throughout CREDIT
    2. Scaffolding the development of future datasets
    3. Provide a minimal implementation of a Dataset for testing

    Minimal YAML config for a dataset will have the following stucture::

        data:
          source:
            Example_Base:  # User-provided name (arbitrary key)
              # PARAMETERS FOR THIS DATASET TYPE
              # Ex: levels: [10, 20, 30]
              dataset_type: "base"  # Needs to match per type of dataset!
              variables:
                prognostic: null
                #  vars_3D: ['T', 'U', 'V', 'Q'] # Your 3D variables
                #  vars_2D: ['SP', 't2m'] # Your 2D variables
                # dynamic_forcing: null
                #  vars_3D: ...
                #  vars_2D: ...
                # static: null
                #  vars_3D: ...
                #  vars_2D: ...
                # diagnostic: null
                #  vars_3D: ...
                #  vars_2D: ...
              # OPTIONAL: Override the clock bounds for this dataset
              start_datetime: "2012-04-03T00:00Z"

            # <YourName2>_<DatasetType2>: # Multiple datasets (see multi_source)

          # These parameters set the overall clock of the sampler
          start_datetime: "2000-01-01T00:00:00Z" # The earliest datetime across datasets
          end_datetime: "2020-12-31T23:00:00Z" # The latest datetime across datasets
          timestep: "12h" # The smallest time interval for the clock
          forecast_len: 1 # The number of timesteps forward that need to be rolled out per sample
    """

    def __init__(self, data_config: dict[str, Any], return_target: bool = False) -> None:
        """
        This initializes the BaseDataset class, which is designed to provide base functionality for datasets in CREDIT.
        For most datasets, you should inherit from this class and include a super().__init__(data_config, return_target)
        call in your __init__ and self.init_register_all_fields() to take advantage of the base functionality.

        _However_, you do not want to use super().__init__ if you need to handle multiple sources in your dataset, since
        the init for BaseDataset is designed to handle one source. For multisource, you should inherit from
        AbstractBaseDataset for type hinting and annotations.

        Depending on your dataset you will likely want to override or extend the following methods:
        1. _get_file_source:
                This method tells the dataset how to find the actual data, which is generally different for different datasets
                and modes (e.g., local vs. remote).
        2. _extract_field:
                This method tells the dataset how to extract the data from the files and organize them accordingly. To
                match dictionary key conventions across datasets, we suggest using the _get_field_name helper to create
                the keys for the extracted data.
        3. _build_timestamps:
                If you would like to apply Quality Control checks that limit the datetimes from which to sample from.
                You may also want to enforce time bounds automatically for your dataset here.

        Args:
            data_config: Dict containing the configuration for the dataset. See class docstring for expected structure.
            return_target: Whether to return the target (t+1) in addition to the input (t). This is used for prognostic and diagnostic fields.

        """
        super().__init__(data_config, return_target)
        # Enforce types
        if not isinstance(data_config, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"Expected data_config to be a dict, but got {type(data_config)}")
        if not isinstance(return_target, bool):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"Expected return_target to be a bool, but got {type(return_target)}")

        # The config for the data source
        if "source" not in data_config:
            raise KeyError(
                "Expected 'source' key in data_config, but it was not found. Likely you provided "
                + "a subconfig or a higher level config. Checks: "
                + f" 'data' in config: {'data' in data_config}, "
                + f" Full data_config provided: \n{data_config}"
            )
        # Ensure source is not empty
        if not data_config["source"]:
            raise ValueError(
                "Expected 'source' key in data_config to be non-empty, but it was empty. "
                + "Likely you provided a subconfig or a higher level config. Checks: "
                + f" 'data' in config: {'data' in data_config}, "
                + f" Full data_config provided: \n{data_config}\n"
                + f" Full source config: \n{data_config['source']}"
            )

        # Notice from above that the dictionary structure from the config goes:
        # data:
        #   source:
        #     <current_source_name>:
        #       # parameters for this source
        #       dataset_type: ...
        #       variables: ...
        # Unless we are parsing through a multi-source config (for which you should not inherit from BaseDataset),
        # we expect only one source to be defined in the config. To be pythonic, we can use next & iter to get this
        # source entry.
        self.curr_source_name = next(iter(data_config["source"]))
        if len(data_config["source"]) > 1:
            raise ValueError(
                f"Multiple sources found in config for class {self.__class__.__name__}, but BaseDataset is only designed to handle one source. "
                + f"Using the first source found: {self.curr_source_name}. If you would like to use multiple sources, you should reference multisource "
                + "dataset class that inherits from BaseDataset and **overrides** the __init__ method to handle multiple sources. \n"
                + f"Full data_config provided: \n{data_config}\n"
                + f"Full source config: \n{data_config['source']}\n"
                + f"Curr source config: \n{data_config['source'][self.curr_source_name]}"
            )
        self.curr_source_cfg = data_config["source"][self.curr_source_name]

        # You should definitely change this in an inherited dataset
        if type(self) is BaseDataset:
            self.dataset_type = "base"

        # Now we start loading the parameters for the dataset, starting with the clock parameters.
        self.dt: pd.Timedelta = self._load_dt(data_config, self.curr_source_cfg)
        self.num_forecast_steps: int = self._load_num_forecast_steps(data_config, self.curr_source_cfg)

        self.start_datetime: pd.Timestamp = self._load_start_datetime(data_config, self.curr_source_cfg)
        self.end_datetime: pd.Timestamp = self._load_end_datetime(data_config, self.curr_source_cfg)
        self.datetimes: pd.DatetimeIndex = self._build_timestamps()

        # Set the return target flag based on the argument passed to init.
        self.return_target: bool = return_target

        # Select if the data is being loaded from local files or remote files.
        self.mode = "local"
        if "mode" in self.curr_source_cfg:
            self.mode = self.curr_source_cfg["mode"]

        # temporal_mode: "exact" (default, exact timestamp match required) or
        # "persist" (snap to last native timestamp via asof(); used for coarser sources)
        self.temporal_mode: str = self.curr_source_cfg.get("temporal_mode", "exact")
        self._persist_cache: dict = {}

        # Placeholder for static metadata
        self.static_metadata: dict[str, Any] = {}

        # If the engine argument for xr.open_dataset needs to be specified, add the engine option to your data source config
        if "engine" in self.curr_source_cfg:
            self.engine = self.curr_source_cfg["engine"]
        else:
            self.engine = None

        # By default, we suggest that the inherited dataset use both file_dict and var_dict.
        # file_dict maps each field_type to a sorted list of (start_time, end_time, file_path)
        #   tuples produced by _map_files. _extract_field can then use the list to find the
        #   appropriate file for a given timestamp.
        # var_dict maps each field_type to {"vars_3D": [...], "vars_2D": [...]}. _extract_field
        #   can then use this to know which variables to extract for each field type from the file.
        self.file_dict: dict[str, Any] = {}
        self.var_dict: dict[str, Any] = {}

        # Only if this is __NOT__ an inherited class, we will immediately run init_register_all_fields.
        # In an inherited class, you should call super().init_register_all_fields() at the end of your
        # __init__ method after you have set up any additional parameters needed for your dataset.
        if type(self) is BaseDataset:
            self.init_register_all_fields()

    def __len__(self) -> int:
        """For a CREDIT dataset, the length is the number of unique datetimes that can be sampled from.

        Returns:
            int: Dataset length
        """
        return len(self.datetimes)

    def __getitem__(self, args: tuple[pd.Timestamp, int]) -> dict[str, Any]:
        """Return a nested input/target sample dict.

        When ``temporal_mode == "persist"``, the requested timestamp *t* is
        snapped to the last native timestamp at-or-before *t* via
        ``pd.DatetimeIndex.asof()``.  The result is cached so that multiple
        fine-resolution master-clock ticks within the same native interval
        only trigger a single file read.

        Args:
            args: ``(t, i)`` where *t* is the current timestamp (nanoseconds
                or pd.Timestamp) and *i* is the within-sequence step index
                produced by the sampler. When ``i == 0`` prognostic and static
                fields are loaded in addition to dynamic forcing.

        Returns:
            Dict with keys ``"input"``, ``"metadata"``, and optionally
            ``"target"`` (when ``return_target=True``). Both ``"input"`` and
            ``"target"`` are dicts of per-variable tensors keyed by
            ``"{source}/{field_type}/{dim}/{varname}"``.
        """
        t, i = args
        t = pd.Timestamp(t)

        if self.temporal_mode == "persist":
            t_resolved = self._resolve_persist_timestamp(t)
            cache_key = (t_resolved, i == 0)
            if cache_key not in self._persist_cache:
                # Evict entries from previous native intervals to keep cache small
                stale = [k for k in self._persist_cache if k[0] != t_resolved]
                for k in stale:
                    del self._persist_cache[k]
                self._persist_cache[cache_key] = self._load_sample(t_resolved, i)
            return self._persist_cache[cache_key]

        return self._load_sample(t, i)

    def _resolve_persist_timestamp(self, t: pd.Timestamp) -> pd.Timestamp:
        """Snap *t* to the last native timestamp at or before *t*.

        Uses ``pd.DatetimeIndex.asof()`` so non-uniform or non-zero-aligned
        cadences are handled correctly.

        Args:
            t: Master-clock timestamp to resolve.

        Returns:
            The last native timestamp ``<= t``.

        Raises:
            ValueError: If *t* is before the first native timestamp.
        """
        resolved = self.datetimes.asof(t)
        if pd.isna(resolved):
            raise ValueError(
                f"Persist source '{self.curr_source_name}': timestamp {t} is before "
                f"the first available native timestamp {self.datetimes[0]}."
            )
        return resolved

    def _load_sample(self, t: pd.Timestamp, i: int) -> dict[str, Any]:
        """Build and return the sample dict for timestamp *t* and step index *i*.

        This is the inner implementation called by ``__getitem__``, separated
        so the persist cache can call it without re-entering the dispatch logic.

        Args:
            t: Timestamp to load (already resolved for persist sources).
            i: Within-sequence step index (0 = initial step).

        Returns:
            Sample dict with ``"input"``, ``"metadata"``, and optionally ``"target"``.
        """
        t_target = t + self.dt
        input_data: dict[str, Any] = {}

        # Dynamic forcing is loaded at every step
        if "dynamic_forcing" in self.var_dict:
            self._extract_field("dynamic_forcing", t, input_data)

        # Prognostic + static are only needed at the initial step
        if i == 0:
            if "static" in self.var_dict:
                self._extract_field("static", t, input_data)
            if "prognostic" in self.var_dict:
                self._extract_field("prognostic", t, input_data)

        sample: dict[str, Any] = {
            "input": input_data,
            "metadata": {"input_datetime": int(t.value)},
        }

        # Optionally load t+1 as the supervised target
        if self.return_target:
            target_data: dict[str, Any] = {}
            for field_type in ("prognostic", "diagnostic"):
                if field_type in self.var_dict:
                    self._extract_field(field_type, t_target, target_data)

            sample["target"] = target_data
            sample["metadata"]["target_datetime"] = int(t_target.value)

        return sample

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # ---------------
    # 1. Clock Parameters: dt, num_forecast_steps, start_datetime, end_datetime
    # Likely you will not need to change these for a new dataset. You may
    # need to make a new helper if:
    # 1. You would like to apply Quality Control checks that limit the datetimes
    #    from which to sample from (see _build_timestamps below)
    # 2. You would like to enforce time bounds automatically for your dataset.
    #    You may want to do this with super() to have base functionality.
    #
    # Note we have a separate _load_... function for each variable in an effort
    # to have a more specific warning message.

    def _check_in_data_config(self, data_config: dict[str, Any], key: str) -> None:
        """Check that a key is in the data config. If not, raise an error since this is required for any dataset.

        Args:
            data_config (dict[str, Any]): Portion of the config under "data"
            key (str): The key to check (e.g., "timestep")

        Raises:
            KeyError: When the key is not found in the data config
        """
        if key not in data_config:
            raise KeyError(f"{key} must be specified in the config for any inheritance of Base Dataset.")

    def _in_source_config(self, data_config: dict[str, Any], curr_source_cfg: dict[str, Any], key: str) -> bool:
        """Helper to determine if a key is in the source config.

        Args:
            data_config (dict[str, Any]): Portion of the config under "data"
            curr_source_cfg (dict[str, Any]): Portion of the config under a specific source
            key (str): The key to check (e.g., "timestep")

        Returns:
            bool: True if the key is in the source config, False otherwise
        """
        return "source" in data_config and key in curr_source_cfg

    def _load_dt(
        self, data_config: dict[str, Any], curr_source_config: dict[str, Any], dt_key: str = "timestep"
    ) -> pd.Timedelta:
        """The timestep (dt) is a required parameter for any dataset, and is used to build the clock of the sampler.
        In general, the timestep in the data config should be the smallest timestep across all sources.
        The inherited dataset may need a coarser timestep (e.g., in the case of multi-source datasets).

        Args:
            data_config (dict[str, Any]): Portion of the config under "data"
            curr_source_config (dict[str, Any]): Portion of the config under a specific source
            dt_key (str, optional): The key for the timestep parameter. Defaults to "timestep".

        Returns:
            pd.Timedelta: The timestep for the dataset
        """
        self._check_in_data_config(data_config, dt_key)
        dt_in_data = pd.Timedelta(data_config[dt_key])

        if self._in_source_config(data_config, curr_source_config, dt_key):
            dt_in_source = pd.Timedelta(curr_source_config[dt_key])
            if dt_in_source < dt_in_data:
                logging.warning(
                    f"In loading for class {self.__class__.__name__}, "
                    + f"{dt_key} in source ({dt_in_source}) "
                    + f"is smaller than {dt_key} in data ({dt_in_data}). "
                    + f"Using {dt_in_source} as the timestep for this dataset. "
                    + "Generally, the data timestep should be the smallest timestep across all sources."
                )
            return dt_in_source

        return dt_in_data

    def _load_num_forecast_steps(
        self,
        data_config: dict[str, Any],
        curr_source_config: dict[str, Any],
        num_forecast_steps_key: str = "forecast_len",
    ) -> int:
        """The number of forecast steps (num_forecast_steps) is a required parameter for any dataset, and
        is used in the sampler. In general, the number of forecast steps in the data config should be
        the largest across all sources, since this determines how far forward the sampler needs to roll
        out. The inherited dataset may not be able to rollout further.

        Args:
            data_config (dict[str, Any]): Portion of the config under "data"
            curr_source_config (dict[str, Any]): Portion of the config under a specific source
            num_forecast_steps_key (str, optional): The key for the number of forecast steps parameter. Defaults to "forecast_len".

        Returns:
            int: The number of forecast steps (i.e., length ahead) for the dataset
        """
        self._check_in_data_config(data_config, num_forecast_steps_key)
        num_forecast_steps_in_data = data_config[num_forecast_steps_key]

        if self._in_source_config(data_config, curr_source_config, num_forecast_steps_key):
            num_forecast_steps_in_source = curr_source_config[num_forecast_steps_key]
            if num_forecast_steps_in_source > num_forecast_steps_in_data:
                logging.warning(
                    f"In loading for class {self.__class__.__name__}, "
                    + f"{num_forecast_steps_key} in source ({num_forecast_steps_in_source}) "
                    + f"is greater than {num_forecast_steps_key} in data ({num_forecast_steps_in_data}). "
                    + f"Using {num_forecast_steps_in_source} as the number of forecast steps for this dataset."
                    + "Generally, the number of forecast steps in data should be the largest across all sources."
                )
            return num_forecast_steps_in_source

        return num_forecast_steps_in_data

    def _load_start_datetime(
        self,
        data_config: dict[str, Any],
        curr_source_config: dict[str, Any],
        start_datetime_key: str = "start_datetime",
    ) -> pd.Timestamp:
        """The start_datetime is a required parameter for any dataset, and is used in the sampler. In general,
        the start_datetime in the data config should be the earliest across all sources, since this determines
        the earliest point in time that the sampler can draw from. The inherited dataset may not be able to go
        as far back.

        Args:
            data_config (dict[str, Any]): Portion of the config under "data"
            curr_source_config (dict[str, Any]): Portion of the config under a specific source
            start_datetime_key (str, optional): The key for the start datetime parameter. Defaults to "start_datetime".

        Returns:
            pd.Timestamp: The start datetime for the dataset
        """

        self._check_in_data_config(data_config, start_datetime_key)
        start_datetime_in_data = pd.Timestamp(data_config[start_datetime_key])

        if self._in_source_config(data_config, curr_source_config, start_datetime_key):
            start_datetime_in_source = pd.Timestamp(curr_source_config[start_datetime_key])
            if start_datetime_in_source < start_datetime_in_data:
                logging.warning(
                    f"In loading for class {self.__class__.__name__}, "
                    + f"{start_datetime_key} in source ({start_datetime_in_source}) "
                    + f"is earlier than {start_datetime_key} in data ({start_datetime_in_data}). "
                    + f"Using {start_datetime_in_source} as the start_datetime for this dataset."
                    + "Generally, the start_datetime in data should be the earliest across all sources."
                )
            return start_datetime_in_source

        return start_datetime_in_data

    def _load_end_datetime(
        self, data_config: dict[str, Any], curr_source_config: dict[str, Any], end_datetime_key: str = "end_datetime"
    ) -> pd.Timestamp:
        """The end_datetime is a required parameter for any dataset, and is used in the sampler. In general,
        the end_datetime in the data config should be the latest across all sources, since this determines
        the latest point in time that the sampler can draw from. The inherited dataset may not be able to go
        as far forward.

        Args:
            data_config (dict[str, Any]): Portion of the config under "data"
            curr_source_config (dict[str, Any]): Portion of the config under a specific source
            end_datetime_key (str, optional): The key for the end datetime parameter. Defaults to "end_datetime".

        Returns:
            pd.Timestamp: The end datetime for the dataset
        """

        self._check_in_data_config(data_config, end_datetime_key)
        end_datetime_in_data = pd.Timestamp(data_config[end_datetime_key])

        if self._in_source_config(data_config, curr_source_config, end_datetime_key):
            end_datetime_in_source = pd.Timestamp(curr_source_config[end_datetime_key])
            if end_datetime_in_source > end_datetime_in_data:
                logging.warning(
                    f"In loading for class {self.__class__.__name__}, "
                    + f"{end_datetime_key} in source ({end_datetime_in_source}) "
                    + f"is later than {end_datetime_key} in data ({end_datetime_in_data}). "
                    + f"Using {end_datetime_in_source} as the end_datetime for this dataset."
                    + "Generally, the end_datetime in data should be the latest across all sources."
                )
            return end_datetime_in_source

        return end_datetime_in_data

    def _build_timestamps(self) -> pd.DatetimeIndex:
        """Return timestamps for the dataset using the class parameters.
        The timestamps should ensure that there are enough future timesteps to
        rollout based on num_forecast_steps and the dt timestep length, and
        should be at the configured timestep frequency.

        Note: Please override this method if you would like to apply Quality Control
        checks that limit the datetimes from which to sample from, or if you would
        like to enforce time bounds automatically for your dataset. You can use
        super() to have base functionality in these cases.

        Returns:
            pd.DatetimeIndex: DatetimeIndex from ``start_datetime`` to ``end_datetime`` minus
                the forecast horizon, at the configured timestep frequency.
        """
        return pd.date_range(
            self.start_datetime,
            self.end_datetime - self.num_forecast_steps * self.dt,
            freq=self.dt,
        )

    # ---------------
    # 2. Registering fields

    def _get_field_name(self, field_type: VALID_FIELD_TYPES, dim_str: str, vname: str) -> str:
        """Get the field name and enforce a consistent convention across datasets.

        The convention for the field name is: ``"{user's current source name}/{field_type}/{dim_str}/{vname}"``.

        Args:
            field_type (VALID_FIELD_TYPES): The field type (e.g., "prognostic")
            dim_str (str): The dimension string (e.g., "3d" or "2d")
            vname (str): The variable name (e.g., "T" or "t2m")

        Returns:
            str: The key string that will be used to access the field variable
        """
        # Cast 3D to 3d, etc. (as needed)
        dim_str = dim_str.lower()
        # Do not change the case on everything else!
        return f"{self.curr_source_name}/{field_type}/{dim_str}/{vname}"

    def init_register_all_fields(self) -> None:
        """Initialize and register all fields for the dataset.

        Raises:
            KeyError: If the config does not include any variables.
        """

        if self.dataset_type == "base":
            logging.error(
                f"You are currently using a dataset name of {self.dataset_type} for class {self.__class__.__name__}, "
                "which is the default name for the BaseDataset class. Likely, you did not set the dataset_type attribute "
                "in the __init__ method of your inherited dataset class, which may cause issues downstream! "
            )

        # Now we load the variables for the dataset.
        if "variables" not in self.curr_source_cfg:
            raise KeyError(
                "Expected 'variables' key in source config, but it was not found. "
                + f"Full source config provided: \n{self.curr_source_cfg}"
            )
        for field_type, field_config in self.curr_source_cfg["variables"].items():
            # Notice that we are expecting to call the same _register_field method for each field type.
            # Check or override this method as needed based on the expected structure of your config for each field type.
            self._register_field(field_type, field_config)

    def _register_field(self, field_type: VALID_FIELD_TYPES, field_config: dict[str, Any] | None) -> None:
        """Validate and register one field type from the config variables block.

        Populates ``self.file_dict`` and ``self.var_dict`` for *field_type*.

        Args:
            field_type: One of VALID_FIELD_TYPES, namely: ``"prognostic"``, ``"dynamic_forcing"``,
                ``"static"``, ``"diagnostic"``.
            field_config: Field-type config dict, or ``None`` / null to disable the field.

        Raises:
            KeyError: If *field_type* is not a recognised field type.
            ValueError: If *field_config* defines neither ``vars_3D`` nor ``vars_2D``.
        """
        if field_type not in get_args(VALID_FIELD_TYPES):
            raise KeyError(
                f"Unknown field_type '{field_type}' in config['source']['ERA5']. Valid options are: {VALID_FIELD_TYPES}"
            )
        if not isinstance(field_config, dict):
            # null / disabled field
            self.file_dict[field_type] = None
            return

        if not field_config.get("vars_3D") and not field_config.get("vars_2D"):
            raise ValueError(f"Field '{field_type}' must define at least one of vars_3D or vars_2D")

        self.var_dict[field_type] = {
            "vars_3D": field_config.get("vars_3D") or [],
            "vars_2D": field_config.get("vars_2D") or [],
        }

        self.file_dict[field_type] = self._get_file_source(field_config)

    def _get_file_source(
        self, field_config: dict[str, Any]
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, str]] | bool | None:
        """Return the file source for a field. Override in subclasses for different modes/backends.

        Args:
            field_config (dict[str, Any]): Validated field-type config dict.

        Raises:
            FileNotFoundError: If ``self.mode == "local"`` and the glob pattern matches no files.
            ValueError: If ``self.mode`` is not a recognised mode.

        Returns:
            list[tuple[pd.Timestamp, pd.Timestamp, str]] | bool | None: Depending on the mode and field type,
                this method may return a list of (start_time, end_time, file_path) tuples produced by _map_files,
                a boolean indicating the presence of the field (e.g., for remote data), or None if the field is disabled.
                The expected return type should be consistent within a dataset class.
        """
        if self.mode == "local":
            path_template: str = field_config.get("path", "")
            files = sorted(glob(_path_template_to_glob(path_template)))
            return _map_files(files, _extract_time_fmt(path_template)) if files else None
        elif self.mode == "remote":
            return True
        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Expected 'local' or 'remote'.")

    def _extract_field(self, field_type: VALID_FIELD_TYPES, t: pd.Timestamp, sample: dict[str, Any]) -> None:
        """Base extract field method, which should be overridden in the inherited dataset class to extract the data for
        each field type. The method should populate data_dict with the extracted data for the given field type and
        timestamp. The keys in data_dict should follow the format in _get_field_name.

        The entries are added as tensors to the sample["input"] or sample["target"] dict in __getitem__.

        Args:
            field_type (VALID_FIELD_TYPES): One of VALID_FIELD_TYPES.
            t (pd.Timestamp): Query timestamp for which to extract the field data.
            sample (dict[str, Any]): The sample dict being built in __getitem__
        """
        logging.error(
            "You are using the default _extract_field method in BaseDataset, which does not actually extract any data. "
            "You should implement this method in your inherited dataset class to extract the data for each field type. Your inherited "
            "class is of type: " + self.__class__.__name__ + ". "
        )

        # A 3D variable should have dimensions (n_levels, 1, n_lat, n_lon).
        # A 2D variable should have dimensions (1, 1, n_lat, n_lon).
        # We select prime numbers to ensure that there is no accidental shape
        # matches when testing the dataset.
        n_lat = 17
        n_lon = 23
        n_levels = 7

        if field_type in self.var_dict:
            for var_2d in self.var_dict[field_type].get("vars_2D", []):
                key = self._get_field_name(field_type, "2d", var_2d)
                sample[key] = torch.ones(1, 1, n_lat, n_lon)
            for var_3d in self.var_dict[field_type].get("vars_3D", []):
                key = self._get_field_name(field_type, "3d", var_3d)
                sample[key] = torch.ones(n_levels, 1, n_lat, n_lon)
