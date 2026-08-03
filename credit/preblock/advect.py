from credit.postblock.advect import (
    _SemiLagrangianAdvectionEngine,
)  # shared engine — lives in postblock but used by both pre and postblocks
from credit.preblock.base import BasePreblock


class SemiLagrangianAdvectionPre(BasePreblock):
    """Preblock that performs one semi-Lagrangian 3D tracer advection step.

    Wraps the same engine as ``credit.postblock.advect.SemiLagrangianAdvection``:
    for each requested data type, reads the winds and surface pressure, derives
    the pressure vertical velocity from mass continuity (or reads a precomputed
    ``omega_var``), traces a back-trajectory of length ``timestep_seconds``,
    and overwrites each configured tracer with its value interpolated at the
    trajectory departure point. See the postblock docstring for the full
    argument list, grid/coordinate assumptions, and numerics.

    Operates on ``batch[data_type][source][var_key]`` for each requested
    ``data_type`` (default: ``["input", "target"]``); a data type absent from
    the batch (e.g. no ``"target"`` during inference) is skipped silently.

    The primary use case is correcting or spinning up tracer initial conditions
    with an explicit advection step before they reach the model — run this in
    the ``ic_only`` preblock phase, using the same winds/tracers convention as
    the postblock so the same config args work in either phase.

    Config example::

        preblocks:
          ic_only:
            advect:
              type: semilagrangian_advection
              args:
                tracer_vars:
                  - "ERA5/prognostic/3d/specific_humidity"
                u_var: "ERA5/prognostic/3d/u_component_of_wind"
                v_var: "ERA5/prognostic/3d/v_component_of_wind"
                surface_pressure_var: "ERA5/prognostic/2d/surface_pressure"
                timestep_seconds: 21600.0
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
        self.engine = _SemiLagrangianAdvectionEngine(**engine_kwargs)

    def forward(self, batch: dict) -> dict:
        batch = self._copy_batch(batch)  # shallow copy — avoids mutating the caller's dict
        for data_type in self.data_types:
            if data_type not in batch:
                continue  # data type absent in this batch (e.g. no "target" during inference)
            self.engine.advect_nested(batch[data_type])
        return batch
