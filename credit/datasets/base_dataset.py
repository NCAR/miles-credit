"""
base_dataset.py
-------------------------------------------------------
BaseDataset: A PyTorch Dataset class for:
1. Type hinting and annotations throughout CREDIT
2. Scaffolding the development of future datasets
3. Provide a minimal implementation of a Dataset for testing
4. Avoid redundant code across dataset classes

"""

from glob import glob
import logging
from typing import Any

import pandas as pd

from torch.utils.data import Dataset

from credit.datasets._file_utils import _map_files

logger = logging.getLogger(__name__)

VALID_FIELD_TYPES = {"prognostic", "dynamic_forcing", "static", "diagnostic"}


class BaseDataset(Dataset[Any]):
    """PyTorch Dataset class for CREDIT that  enables:
    1. Type hinting and annotations throughout CREDIT
    2. Scaffolding the development of future datasets
    3. Provide a minimal implementation of a Dataset for testing


    Minimal YAML config for a dataset will have the following stucture:

    ```yaml
        data:
          source:
            Example_Base: # Notice format is <YourName>_<DatasetType>:
              # PARAMETERS FOR THIS DATASET TYPE
              # Ex: levels: [10, 20, 30]
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
    ```
    """

    def __init__(self, data_config: dict[str, Any], return_target: bool = False) -> None:
        """


        Args:
            data_config: Dict containing the configuration for the dataset. See class docstring for expected structure.
            return_target: Whether to return the target (t+1) in addition to the input (t). This should be True for supervised learning and False for self-supervised learning.

        """

        # Enforce types
        if not isinstance(data_config, dict): # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"Expected data_config to be a dict, but got {type(data_config)}")
        if not isinstance(return_target, bool): # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"Expected return_target to be a bool, but got {type(return_target)}")

        # The config for the data source
        if "source" not in data_config:
            raise KeyError(
                "Expected 'source' key in data_config, but it was not found. Likely you provided "
                + "a subconfig or a higher level config. Checks: "
                + f" 'data' in config: {'data' in data_config}, "
                + f" Full data_config provided: \n{data_config}"
            )

        source_cfg = data_config["source"]

        self.dt: pd.Timedelta = self._load_dt(data_config)
        self.num_forecast_steps: int = self._load_num_forecast_steps(data_config)

        self.start_datetime: pd.Timestamp = self._load_start_datetime(data_config)
        self.end_datetime: pd.Timestamp = self._load_end_datetime(data_config)
        self.datetimes: pd.DatetimeIndex = self._build_timestamps()

        self.return_target: bool = return_target

        assert "variables" in source_cfg, (
            "Expected 'variables' key in source config, but it was not found. "
            + f"Full source config provided: \n{source_cfg}"
        )

        self.file_dict: dict[str, Any] = {}
        self.var_dict: dict[str, Any] = {}
        for field_type, d in source_cfg["variables"].items():
            self._register_field(field_type, d)

    def __len__(self) -> int:
        return len(self.datetimes)

    def __getitem__(self, args: tuple[pd.Timestamp, int]) -> dict[str, Any]:
        """Return a nested input/target sample dict.

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
                if not self.file_dict:
                    continue
                if self.file_dict.get(field_type) and field_type in self.var_dict:
                    self._extract_field(field_type, t_target, target_data)

            sample["target"] = target_data
            sample["metadata"]["target_datetime"] = int(t_target.value)

        return sample

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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

    def _check_in_data_config(self, config: dict[str, Any], key: str) -> None:
        """
        Check that a key is in the data config. If not, raise an error since this is required for any dataset.
        """
        if key not in config:
            raise KeyError(f"{key} must be specified in the config for any inheritance of Base Dataset.")

    def _in_source_config(self, config: dict[str, Any], key: str) -> bool:
        """
        Helper check that a key is in the source config.
        """
        return "source" in config and key in config["source"]

    def _load_dt(self, config: dict[str, Any], dt_key: str = "timestep") -> pd.Timedelta:
        """
        The timestep (dt) is a required parameter for any dataset, and is used to build the clock of the sampler.
        In general, the timestep in the data config should be the smallest timestep across all sources.
        The inherited dataset may need a coarser timestep (e.g., in the case of multi-source datasets).
        """
        self._check_in_data_config(config, dt_key)
        dt_in_data = pd.Timedelta(config[dt_key])

        if self._in_source_config(config, dt_key):
            dt_in_source = pd.Timedelta(config["source"][dt_key])
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

    def _load_num_forecast_steps(self, config: dict[str, Any], num_forecast_steps_key: str = "forecast_len") -> int:
        """
        The number of forecast steps (num_forecast_steps) is a required parameter for any dataset, and
        is used in the sampler. In general, the number of forecast steps in the data config should be
        the largest across all sources, since this determines how far forward the sampler needs to roll
        out. The inherited dataset may not be able to rollout further.
        """
        self._check_in_data_config(config, num_forecast_steps_key)
        num_forecast_steps_in_data = config[num_forecast_steps_key]

        if self._in_source_config(config, num_forecast_steps_key):
            num_forecast_steps_in_source = config["source"][num_forecast_steps_key]
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

    def _load_start_datetime(self, config: dict[str, Any], start_datetime_key: str = "start_datetime") -> pd.Timestamp:
        """
        The start_datetime is a required parameter for any dataset, and is used in the sampler. In general,
        the start_datetime in the data config should be the earliest across all sources, since this determines
        the earliest point in time that the sampler can draw from. The inherited dataset may not be able to go
        as far back.
        """
        self._check_in_data_config(config, start_datetime_key)
        start_datetime_in_data = pd.Timestamp(config[start_datetime_key])

        if self._in_source_config(config, start_datetime_key):
            start_datetime_in_source = pd.Timestamp(config["source"][start_datetime_key])
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

    def _load_end_datetime(self, config: dict[str, Any], end_datetime_key: str = "end_datetime") -> pd.Timestamp:
        """
        The end_datetime is a required parameter for any dataset, and is used in the sampler. In general,
        the end_datetime in the data config should be the latest across all sources, since this determines
        the latest point in time that the sampler can draw from. The inherited dataset may not be able to go
        as far forward.
        """
        self._check_in_data_config(config, end_datetime_key)
        end_datetime_in_data = pd.Timestamp(config[end_datetime_key])

        if self._in_source_config(config, end_datetime_key):
            end_datetime_in_source = pd.Timestamp(config["source"][end_datetime_key])
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

        Returns:
            DatetimeIndex from ``start_datetime`` to ``end_datetime`` minus
            the forecast horizon, at the configured timestep frequency.
        """
        return pd.date_range(
            self.start_datetime,
            self.end_datetime - self.num_forecast_steps * self.dt,
            freq=self.dt,
        )

    # 2. Registering fields

    def _register_field(self, field_type: str, d: dict[str, Any] | None) -> None:
        """Validate and register one field type from the config variables block.

        Populates ``self.file_dict`` and ``self.var_dict`` for *field_type*.

        Args:
            field_type: One of ``"prognostic"``, ``"dynamic_forcing"``,
                ``"static"``, ``"diagnostic"``.
            d: Field-type config dict, or ``None`` / null to disable the field.

        Raises:
            KeyError: If *field_type* is not a recognised field type.
            ValueError: If *d* defines neither ``vars_3D`` nor ``vars_2D``.
        """
        if field_type not in VALID_FIELD_TYPES:
            raise KeyError(
                f"Unknown field_type '{field_type}' in config['source']['ERA5']. "
                f"Valid options are: {sorted(VALID_FIELD_TYPES)}"
            )
        if not isinstance(d, dict):
            # null / disabled field
            self.file_dict[field_type] = None
            return

        if not d.get("vars_3D") and not d.get("vars_2D"):
            raise ValueError(f"Field '{field_type}' must define at least one of vars_3D or vars_2D")

        files = sorted(glob(d.get("path", "")))
        time_fmt: str = d.get("filename_time_format", "%Y")
        self.file_dict[field_type] = _map_files(files, time_fmt) if files else None
        self.var_dict[field_type] = {
            "vars_3D": d.get("vars_3D") or [],
            "vars_2D": d.get("vars_2D") or [],
        }
    

    def _extract_field(self, field_type: str, t: pd.Timestamp, data_dict: dict[str, Any]) -> None:
        logging.error("You are using the default _extract_field method in BaseDataset, which does not actually extract any data. " \
        "You should implement this method in your inherited dataset class to extract the data for each field type. Your inherited " \
        "class is of type: " + self.__class__.__name__ + ". ")
        raise NotImplementedError("To-Do")
