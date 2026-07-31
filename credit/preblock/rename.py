"""Rename variable keys and their source namespaces in nested batch dictionaries."""

import logging
from os.path import expandvars
from typing import Any

import yaml

from credit.preblock.base import BasePreblock

logger = logging.getLogger(__name__)


class RenameVariables(BasePreblock):
    """Rename batch dict keys -- the missing piece for pointing ``inference.data``
    at a data source whose naming conventions differ from the one the rest of
    the preblock chain (scaler, other preblocks) and ``ChannelSchema`` were built
    against (e.g. an ERA5-trained model driven by GFS initial conditions: GFS's
    raw variable names differ from ERA5's even after grid/level reconciliation).

    Renames full ``{source}/{field_type}/{dim}/{varname}`` keys, moving the
    tensor across the ``batch[data_type][source]`` sub-dict boundary too when
    the source prefix itself changes (e.g. ``"GFS/..." -> "ERA5/..."``). Keys
    not present in ``mapping`` are left untouched; a mapped key absent from a
    given ``data_type`` (e.g. a rename that only applies to ``"input"``) is
    silently skipped, matching every other preblock's convention for optional
    per-data_type variables. Mapping is resolved against the keys present at the
    start of each pass, so chained mappings do not cascade.

    Exactly one of ``mapping`` / ``mapping_file`` must be given. Prefer
    ``mapping_file`` for anything reused across configs or runs (e.g. a
    standing "GFS conventions" mapping) -- consistent with how
    ``HybridLevelInterpPre``'s ``source_level_info_file`` and
    ``ForecastWriter``'s ``metadata:`` reference an external data source's own
    conventions as a file rather than inlining them. ``mapping`` (inline) is
    for quick one-off overrides not worth a separate file.

    Config example (inline)::

        type: "rename"
        args:
          mapping:
            "GFS/prognostic/3d/TMP": "ERA5/prognostic/3d/T"
            "GFS/prognostic/2d/PS": "ERA5/prognostic/2d/SP"

    Config example (external file -- a flat YAML dict of the same shape)::

        type: "rename"
        args:
          mapping_file: "$SCRATCH/gfs_to_era5_rename.yaml"
    """

    def __init__(
        self,
        mapping: dict[str, str] | None = None,
        mapping_file: str | None = None,
        data_types: list[str] | None = None,
    ):
        super().__init__()
        if (mapping is None) == (mapping_file is None):
            raise ValueError("RenameVariables: exactly one of `mapping` or `mapping_file` must be given.")

        if mapping_file is not None:
            path = expandvars(mapping_file)
            with open(path) as f:
                loaded: Any = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ValueError(
                    f"RenameVariables: mapping_file '{path}' must contain a flat YAML dict, got {type(loaded)}."
                )
            mapping = loaded
        elif not isinstance(mapping, dict):
            raise ValueError(f"RenameVariables: `mapping` must be a dict of {{old_key: new_key}}, got {type(mapping)}.")

        self.mapping: dict[str, str] = dict(mapping)
        self.data_types = data_types or list(self.VALID_DATA_TYPES)

        invalid_dt = set(self.data_types) - set(self.VALID_DATA_TYPES)
        if invalid_dt:
            raise ValueError(f"Invalid data_types {invalid_dt}. Valid options are {self.VALID_DATA_TYPES}.")

    def forward(self, batch: dict) -> dict:
        batch = self._copy_batch(batch)  # shallow copy -- avoids mutating the caller's dict
        for data_type in self.data_types:
            if data_type not in batch:
                continue
            sources = batch[data_type]
            if not isinstance(sources, dict):
                continue

            original = {source: dict(variables) for source, variables in sources.items()}
            renamed = {source: {} for source in original}
            for source, variables in original.items():
                for old_key, tensor in variables.items():
                    new_key = self.mapping.get(old_key, old_key)
                    new_source = new_key.split("/", 1)[0]
                    dest = renamed.setdefault(new_source, {})
                    if new_key in dest:
                        raise ValueError(
                            f"RenameVariables: mapping target '{new_key}' already exists in "
                            f"batch['{data_type}'] -- two variables would collide under the same key."
                        )
                    dest[new_key] = tensor

            batch[data_type] = renamed

        return batch
