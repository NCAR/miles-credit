"""
base_dataset.py
-------------------------------------------------------
BaseDataset: A PyTorch Dataset class for:
1. Type hinting and annotations throughout CREDIT
2. Scaffolding the development of future datasets
3. Provide a minimal implementation of a Dataset for testing
4. Avoid redundant code across dataset classes

"""

import logging

import pandas as pd

from torch.utils.data import Dataset


class BaseDataset(Dataset):
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
            
            # <YourName2>_<DatasetType2>: # Multiple datasets (see multi_source)

          # These parameters set the overall clock of the sampler
          start_datetime: "2000-01-01T00:00:00Z"
          end_datetime: "2020-12-31T23:00:00Z"
          timestep: "12h"
          forecast_len: 1
    ```
    """

    def __init__(self, config: dict | None, return_target: bool = False) -> None:
        self.dt = pd.Timedelta(config["timestep"])
        self.num_forecast_steps: int = config["forecast_len"]

        self.start_datetime = pd.Timestamp(config["start_datetime"])
        self.end_datetime = pd.Timestamp(config["end_datetime"])
        self.datetimes: pd.DatetimeIndex = self._build_timestamps()

        for field_type, d in source_cfg["variables"].items():
            self._register_field(field_type, d)


    def __len__(self) -> int:
        raise NotImplementedError("To-Do")

    def __getitem__(self, args: tuple) -> dict:
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

        input_data: dict = {}

        # Dynamic forcing is loaded at every step
        if "dynamic_forcing" in self.var_dict:
            self._extract_field("dynamic_forcing", t, input_data)

        # Prognostic + static are only needed at the initial step
        if i == 0:
            if "static" in self.var_dict:
                self._extract_field("static", t, input_data)
            if "prognostic" in self.var_dict:
                self._extract_field("prognostic", t, input_data)

        sample: dict = {
            "input": input_data,
            "metadata": {"input_datetime": int(t.value)},
        }

        # Optionally load t+1 as the supervised target
        if self.return_target:
            target_data: dict = {}
            for field_type in ("prognostic", "diagnostic"):
                if self.file_dict.get(field_type) and field_type in self.var_dict:
                    self._extract_field(field_type, t_target, target_data)

            sample["target"] = target_data
            sample["metadata"]["target_datetime"] = int(t_target.value)

        return sample

    def _build_timestamps(self) -> pd.DatetimeIndex:
        """Return valid initialisation timestamps for the dataset.

        Returns:
            DatetimeIndex from ``start_datetime`` to ``end_datetime`` minus
            the forecast horizon, at the configured timestep frequency.
        """
        return pd.date_range(
            self.start_datetime,
            self.end_datetime - self.num_forecast_steps * self.dt,
            freq=self.dt,
        )
    
    def _register_field(self, field_type: str, d: dict | None) -> None:
        raise NotImplementedError("To-Do")
