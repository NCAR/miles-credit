import logging
from typing import Dict, Optional

import torch
import xarray as xr

# Experiment variables
# Assumption that all 3D variables are appended with depth_i_levels
# and all 2D variables do not have any digits / underscores in their names
DEPTH_LEVELS = [
    2.5,
    10.0,
    22.5,
    40.0,
    65.0,
    105.0,
    165.0,
    250.0,
    375.0,
    550.0,
    775.0,
    1050.0,
    1400.0,
    1850.0,
    2400.0,
    3100.0,
    4000.0,
    5000.0,
    6000.0,
]

DEPTH_I_LEVELS = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
]

MASK_VARS = [
    "mask_0",
    "mask_1",
    "mask_2",
    "mask_3",
    "mask_4",
    "mask_5",
    "mask_6",
    "mask_7",
    "mask_8",
    "mask_9",
    "mask_10",
    "mask_11",
    "mask_12",
    "mask_13",
    "mask_14",
    "mask_15",
    "mask_16",
    "mask_17",
    "mask_18",
]

PROG_VARS_MAP = {
    "thermo_dynamic": [
        k + str(j) for k in ["uo_", "vo_", "thetao_", "so_"] for j in DEPTH_I_LEVELS
    ]
    + ["zos"],
    "thermo": [k + str(j) for k in ["thetao_", "so_"] for j in DEPTH_I_LEVELS]
    + ["zos"],
}
BOUND_VARS_MAP = {
    "hfds_anom": ["tauuo", "tauvo", "hfds", "hfds_anomalies"],
}

default_metadata = {
    "thetao": {
        "long_name": "Sea Water Potential Temperature",
        "units": r"\degree C",
    },
    "so": {
        "long_name": "Sea Water Salinity",
        "units": "psu",
    },
    "uo": {
        "long_name": "Sea Water X Velocity",
        "units": "m/s",
    },
    "vo": {
        "long_name": "Sea Water Y Velocity",
        "units": "m/s",
    },
    "zos": {
        "long_name": "Sea surface height above geoid",
        "units": "m",
    },
    "tos": {
        "long_name": "Sea surface temperature",
        "units": r"\degree C",
    },
    "tauuo": {
        "long_name": "Surface Downward X Stress",
        "units": "N/m^2",
    },
    "tauvo": {
        "long_name": "Surface Downward Y Stress",
        "units": "N/m^2",
    },
    "hfds": {
        "long_name": "Surface ocean heat flux from "
        "SW+LW+latent+sensible+masstransfer+frazil+seaice_melt_heat",
        "units": "W/m^2",
    },
    "hfds_anomalies": {
        "long_name": "hfds anomalies",
        "units": "W/m^2",
    },
}


def construct_metadata(data: xr.Dataset) -> Dict[str, Dict[str, str]]:
    metadata = {}
    for var in data.variables:
        try:
            metadata[var] = {
                "long_name": data[var].long_name,
                "units": data[var].units,
            }
        except AttributeError:
            if var in default_metadata:
                metadata[var] = default_metadata[var]
            elif var.split("_")[0] in default_metadata:
                metadata[var] = default_metadata[var.split("_")[0]]
            else:
                logging.debug(f"{var} does not have any default metadata")
                metadata[var] = {
                    "long_name": "Unknown",
                    "units": "Unknown",
                }

    return metadata


class TensorMap:
    _instance: Optional["TensorMap"] = None

    def __new__(cls, *args, **kwargs) -> "TensorMap":
        # Prevent direct instantiation
        raise TypeError(
            "TensorMap cannot be instantiated directly. Use init_instance() instead."
        )

    @classmethod
    def get_instance(cls) -> "TensorMap":
        if cls._instance is None:
            raise ValueError("TensorMap not initialized")
        return cls._instance

    @classmethod
    def init_instance(
        cls, prognostic_vars_key: str, boundary_vars_key: str
    ) -> "TensorMap":
        if cls._instance is not None:
            raise ValueError("TensorMap already initialized")

        instance = super().__new__(cls)
        instance._initialize(prognostic_vars_key, boundary_vars_key)
        cls._instance = instance
        return cls._instance

    def _initialize(self, prognostic_vars_key: str, boundary_vars_key: str):
        """
        Maps input variables / depth levels to their indices in the input tensor.

        VAR_3D_IDX maps the input variables to their indices in the input tensor
        DP_3D_IDX maps the depth levels to their indices in the input tensor
        """
        self.prognostic_vars = PROG_VARS_MAP[prognostic_vars_key]
        self.boundary_vars = BOUND_VARS_MAP[boundary_vars_key]

        self.VAR_3D_IDX: Dict[str, torch.Tensor] = {}
        self.DP_3D_IDX: Dict[str, torch.Tensor] = {}

        self.VAR_SET_2D = []
        self.VAR_SET_3D = []
        for out in self.prognostic_vars:
            var_split = out.split("_")
            if len(var_split) == 1:
                self.VAR_SET_2D.append(var_split[0])
            else:
                self.VAR_SET_3D.append(var_split[0])

        # Consistent order of variables
        self.VAR_SET = list(
            dict.fromkeys(([out.split("_")[0] for out in self.prognostic_vars]))
        )
        self.DEPTH_SET = DEPTH_I_LEVELS

        self._populate_var_3d_idx()
        self._populate_dp_3d_idx()

    def _populate_var_3d_idx(self):
        for kt in self.VAR_SET:
            self.VAR_3D_IDX[kt] = torch.tensor([])
            for i, k in enumerate(self.prognostic_vars):
                if kt in k:
                    self.VAR_3D_IDX[kt] = torch.cat(
                        [self.VAR_3D_IDX[kt], torch.tensor([i])]
                    )
            self.VAR_3D_IDX[kt] = self.VAR_3D_IDX[kt].to(torch.int32)

    def _populate_dp_3d_idx(self):
        for d in self.DEPTH_SET:
            self.DP_3D_IDX[d] = torch.tensor([])
            for i, k in enumerate(self.prognostic_vars):
                k_split = k.split("_")
                if len(k_split) == 1:
                    continue
                elif d == k_split[-1]:
                    self.DP_3D_IDX[d] = torch.cat(
                        [self.DP_3D_IDX[d], torch.tensor([i])]
                    )
            self.DP_3D_IDX[d] = self.DP_3D_IDX[d].to(torch.int32)

        self.DP_3D_IDX[self.DEPTH_SET[0]] = torch.cat(
            [
                self.DP_3D_IDX[self.DEPTH_SET[0]],
                torch.tensor([self.VAR_3D_IDX[var_2D] for var_2D in self.VAR_SET_2D]),
            ]
        )
