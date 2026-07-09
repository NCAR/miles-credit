"""
schema.py
---------
ChannelSchema: the explicit channel-layout contract between the data config,
the flat tensors built by ``ConcatToTensor``, and the nested dict rebuilt by
``Reconstruct``.

Why this exists
~~~~~~~~~~~~~~~
The channel order of the model's flat input/output tensors was previously
implicit, re-derived per batch from whatever tensors happened to be present.
That broke at inference: with ``return_target=False`` the batch carries no
diagnostic variables, so the fallback channel map covered prognostics only and
``Reconstruct`` silently dropped every diagnostic channel from ``y_pred``.

``ChannelSchema`` freezes the layout once, from the config, at training time:

* **input layout** — mirrors ``ConcatToTensor``'s sort: sources in config
  order; within each source, field types by ``FIELD_TYPE_RANK`` (prognostic <
  static < dynamic_forcing); 3d before 2d; config list order within.
  Diagnostics are never model inputs.
* **target layout** — mirrors the dataset's target insertion order
  (``BaseDataset.__getitem__``): sources in config order; ``prognostic`` then
  ``diagnostic``; ``vars_3D`` then ``vars_2D``; config list order within.
  This is the channel order of ``y`` and therefore of ``y_pred``.

Lifecycle
~~~~~~~~~
Training saves the schema to ``{save_loc}/channel_schema.yaml`` (rank 0) and
validates it once against the target channel map actually built from data.
Inference loads that file (falling back to config derivation with a warning)
and hands it to ``ConcatToTensor``, which uses the schema's target map whenever
the batch has no target — so ``Reconstruct`` recovers every predicted variable,
diagnostics included.
"""

import logging
import os

import yaml

from credit.datasets.channel_layout import FIELD_TYPE_RANK

logger = logging.getLogger(__name__)

DEFAULT_SCHEMA_FILENAME = "channel_schema.yaml"

_SCHEMA_VERSION = 1

# Field types that appear in the model input, in ConcatToTensor sort order.
_INPUT_FIELD_TYPES = tuple(ft for ft, _ in sorted(FIELD_TYPE_RANK.items(), key=lambda kv: kv[1]) if ft != "diagnostic")
# Field types in the training target, in BaseDataset insertion order.
_TARGET_FIELD_TYPES = ("prognostic", "diagnostic")


def _layout_to_channel_map(layout: list[dict]) -> dict:
    """Convert an ordered layout to a ConcatToTensor-style channel map.

    Returns ``{var_key: {"slice": slice(start, stop), "orig_shape": (n_levels, n_time)}}``.
    """
    channel_map = {}
    cursor = 0
    for entry in layout:
        n_ch = entry["n_levels"] * entry["n_time"]
        channel_map[entry["var_key"]] = {
            "slice": slice(cursor, cursor + n_ch),
            "orig_shape": (entry["n_levels"], entry["n_time"]),
        }
        cursor += n_ch
    return channel_map


class ChannelSchema:
    """Frozen channel layout for the flat model input and output tensors.

    Args:
        input_layout: ordered list of ``{"var_key", "n_levels", "n_time"}``
            dicts describing the concatenated input tensor.
        target_layout: same, for the target / ``y_pred`` tensor.
    """

    def __init__(self, input_layout: list[dict], target_layout: list[dict]):
        self.input_layout = input_layout
        self.target_layout = target_layout

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, conf: dict) -> "ChannelSchema":
        """Derive the schema from the full config dict.

        ``n_levels`` for 3D variables comes from the source's ``levels`` list
        when present, else from ``conf["model"]["levels"]``. Sources that define
        3D variables but resolve neither raise ``ValueError`` — for those
        configs the schema must come from a saved ``channel_schema.yaml``.

        Raises:
            ValueError: if 3D variables exist but no level count is resolvable.
        """
        data_conf = conf["data"]
        model_levels = (conf.get("model") or {}).get("levels")

        input_layout: list[dict] = []
        target_layout: list[dict] = []

        for source_name, src_conf in data_conf["source"].items():
            variables = src_conf.get("variables") or {}
            src_levels = src_conf.get("levels")
            n_levels = len(src_levels) if src_levels else model_levels

            def _entries(field_type: str) -> list[dict]:
                grp = variables.get(field_type) or {}
                entries = []
                for vname in grp.get("vars_3D") or []:
                    if not n_levels:
                        raise ValueError(
                            f"ChannelSchema.from_config: source '{source_name}' defines 3D variables "
                            f"but neither data.source.{source_name}.levels nor model.levels is set. "
                            "Provide one, or load the schema saved at training time "
                            f"({DEFAULT_SCHEMA_FILENAME} in save_loc)."
                        )
                    entries.append(
                        {
                            "var_key": f"{source_name}/{field_type}/3d/{vname}",
                            "n_levels": int(n_levels),
                            "n_time": 1,
                        }
                    )
                for vname in grp.get("vars_2D") or []:
                    entries.append(
                        {
                            "var_key": f"{source_name}/{field_type}/2d/{vname}",
                            "n_levels": 1,
                            "n_time": 1,
                        }
                    )
                return entries

            for field_type in _INPUT_FIELD_TYPES:
                input_layout.extend(_entries(field_type))
            for field_type in _TARGET_FIELD_TYPES:
                target_layout.extend(_entries(field_type))

        return cls(input_layout, target_layout)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Write the schema as YAML (atomically: temp file + rename)."""
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "input": self.input_layout,
            "target": self.target_layout,
        }
        # Ensure the save_loc exists — the trainer saves the schema at init, which
        # may run before anything else creates the directory (e.g. a fresh save_loc).
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        os.replace(tmp, path)
        logger.info("ChannelSchema saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "ChannelSchema":
        with open(path) as f:
            payload = yaml.safe_load(f)
        version = payload.get("schema_version")
        if version != _SCHEMA_VERSION:
            raise ValueError(f"ChannelSchema.load: unsupported schema_version {version!r} in {path}")
        return cls(payload["input"], payload["target"])

    @classmethod
    def load_or_from_config(cls, conf: dict, save_loc: str | None = None) -> "ChannelSchema | None":
        """Load ``channel_schema.yaml`` from ``save_loc``, else derive from config.

        Returns ``None`` (with a warning) when neither is possible, so callers
        can fall back to legacy per-batch behavior.
        """
        save_loc = save_loc or os.path.expandvars(conf.get("save_loc", ""))
        path = os.path.join(save_loc, DEFAULT_SCHEMA_FILENAME)
        if os.path.isfile(path):
            logger.info("Loading channel schema from %s", path)
            return cls.load(path)
        try:
            schema = cls.from_config(conf)
            logger.info("No %s in %s — channel schema derived from config.", DEFAULT_SCHEMA_FILENAME, save_loc)
            return schema
        except (KeyError, ValueError) as e:
            logger.warning(
                "No channel schema available (%s). Falling back to per-batch channel maps; "
                "at inference, diagnostic variables will be MISSING from reconstructed output.",
                e,
            )
            return None

    # ------------------------------------------------------------------
    # Channel maps
    # ------------------------------------------------------------------

    def input_channel_map(self) -> dict:
        """Channel map for the concatenated input tensor."""
        return _layout_to_channel_map(self.input_layout)

    def target_channel_map(self) -> dict:
        """Channel map for the target / ``y_pred`` tensor (prognostic + diagnostic)."""
        return _layout_to_channel_map(self.target_layout)

    # ------------------------------------------------------------------
    # Validation against maps built from real data
    # ------------------------------------------------------------------

    def validate_channel_map(self, actual_map: dict, which: str = "target") -> None:
        """Check a data-derived channel map against the schema.

        Compares key order, slices, and per-variable shapes. Raises with a
        side-by-side diff on mismatch — a mismatch means the schema (and any
        model trained against it) disagrees with what the dataset produces.

        Args:
            actual_map: ``{var_key: {"slice", "orig_shape"}}`` from ConcatToTensor.
            which: ``"target"`` or ``"input"`` — selects the schema layout.

        Raises:
            ValueError: on any ordering, slicing, or shape difference.
        """
        expected = self.target_channel_map() if which == "target" else self.input_channel_map()
        exp_items = [(k, v["slice"], tuple(v["orig_shape"])) for k, v in expected.items()]
        act_items = [(k, v["slice"], tuple(v["orig_shape"])) for k, v in actual_map.items()]
        if exp_items == act_items:
            return

        def _fmt(items):
            return "\n".join(f"  [{s.start}:{s.stop}] {k} shape={shape}" for k, s, shape in items)

        raise ValueError(
            f"ChannelSchema mismatch ({which} map): the channel layout built from data does not "
            f"match the schema. This would silently corrupt variable reconstruction.\n"
            f"Schema expects:\n{_fmt(exp_items)}\nData produced:\n{_fmt(act_items)}"
        )
