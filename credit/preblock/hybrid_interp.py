from credit.postblock.hybrid_interp import (
    _HybridLevelInterpEngine,
)  # shared engine — lives in postblock but used by both pre and postblocks
from credit.preblock.base import BasePreblock


class HybridLevelInterp(BasePreblock):
    """Preblock that interpolates 3D variables between hybrid sigma-pressure level sets.

    Wraps the same engine as ``credit.postblock.hybrid_interp.HybridLevelInterp``:
    variables are interpolated column-by-column, linearly in log(pressure),
    with constant extrapolation outside the source pressure range, parallelized
    with ``torch.vmap``. See the postblock docstring for the coefficient-file
    conventions and the full argument list.

    Operates on ``batch[data_type][source][var_key]`` for each requested
    ``data_type`` (default: ``["input", "target"]``), replacing each variable
    in ``variables`` with its ``(B, n_dest_levels, n_time, H, W)`` counterpart.
    Variables absent from a data type are skipped silently.

    The primary use case is inference with a model trained on one vertical
    grid but initialized from another (e.g. ERA5-trained model driven by GFS
    initial conditions): run this in the ``ic_only`` preblock phase so the IC
    lands on the model's levels before normalization and concat.

    Config example (GFS IC → ERA5 levels)::

        type: "hybrid_level_interp"
        args:
            variables:
                - "GFS/prognostic/3d/temperature"
                - "GFS/prognostic/3d/specific_humidity"
            surface_pressure_var: "GFS/prognostic/2d/surface_pressure"
            source_level_info_file: "/path/to/gfs_ctrl.nc"
            source_a_var: "vcoord"
            source_b_var: "vcoord"
            dest_level_info_file: "ERA5_Lev_Info.nc"
            data_types: ["input"]
    """

    def __init__(self, data_types: list[str] = None, **engine_kwargs):
        super().__init__()
        self.data_types = data_types or ["input", "target"]
        invalid = set(self.data_types) - set(self.VALID_DATA_TYPES)
        if invalid:
            raise ValueError(
                f"Invalid data_types {invalid}. "
                f"Valid options are {self.VALID_DATA_TYPES}. "
                f"Preblocks never operate on 'metadata'."
            )
        self.engine = _HybridLevelInterpEngine(**engine_kwargs)

    def forward(self, batch: dict) -> dict:
        batch = self._copy_batch(batch)  # shallow copy — avoids mutating the caller's dict
        for data_type in self.data_types:
            if data_type not in batch:
                continue  # data type absent in this batch (e.g. no "target" during inference)
            self.engine.interp_nested(batch[data_type])
        return batch
