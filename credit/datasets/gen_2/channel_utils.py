"""
channel_utils.py
----------------
Everything about the channel/tensor layout contract for gen2: the canonical
field-type rank, per-group channel slicing for rollout, and ChannelSchema (the
frozen input/target layout saved at training time and reused at inference).

FIELD_TYPE_RANK / build_channel_layout / update_x
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Derive per-group channel slices directly from the variables config and
provide a standalone update function for multi-step rollout.

Both the input tensor ``x`` and the target tensor ``y_pred`` are laid out
**source-major** by ``ConcatToTensor`` — every channel of the first source
precedes every channel of the second.  Within a source the two differ:

* ``x`` orders field types by FIELD_TYPE_RANK (prognostic < static <
  dynamic_forcing); diagnostics are never inputs.
* ``y_pred`` is prognostic then diagnostic.

So with more than one source a field type's channels are *not* contiguous
across the whole tensor, and a source's prognostic block in ``y_pred`` is
offset by the diagnostics of every source before it.  Each group therefore
carries both of its positions: ``x_slice`` (where it lives in ``x``) and
``src_slice`` (where its replacement comes from in ``y_pred`` or
``x_dynfrc``).  See :class:`ChannelGroup`.

Usage::

    from credit.datasets.gen_2.channel_utils import build_channel_layout, update_x

    # once at trainer / rollout init
    groups, n_pred = build_channel_layout(conf)

    # at every t > 1 step — y_pred is the FULL prediction, diagnostics included
    x = update_x(x, x_dynfrc, y_pred, groups)

ChannelSchema
~~~~~~~~~~~~~
The explicit channel-layout contract between the data config, the flat
tensors built by ``ConcatToTensor``, and the nested dict rebuilt by
``Reconstruct``.

Why this exists
^^^^^^^^^^^^^^^^
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
^^^^^^^^^
Training saves the schema to ``{save_loc}/channel_schema.yaml`` (rank 0) and
validates it once against the target channel map actually built from data.
Inference loads that file (falling back to config derivation with a warning)
and hands it to ``ConcatToTensor``, which uses the schema's target map whenever
the batch has no target — so ``Reconstruct`` recovers every predicted variable,
diagnostics included.
"""

import contextlib
import logging
import os
from dataclasses import dataclass

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical field-type rank + per-group channel slicing
# ---------------------------------------------------------------------------

# Canonical cross-group concat rank — shared with credit.preblock.concat.
# Groups absent from the config are ignored; unknown groups sort to the end.
FIELD_TYPE_RANK: dict[str, int] = {
    "prognostic": 0,
    "static": 1,
    "dynamic_forcing": 2,
    "diagnostic": 3,
}

# Semantic roles — determines update behaviour at t > 1.  "diagnostic" is
# absent by construction: it is target-only and excluded from _INPUT_FIELD_TYPES.
_DATASET_DRIVEN = frozenset({"dynamic_forcing"})
_MODEL_PREDICTED = frozenset({"prognostic"})
# "static" (and anything else) falls through as fixed


@dataclass(frozen=True)
class ChannelGroup:
    """One ``(source, field_type)`` run of channels in the model input tensor.

    Attributes:
        source: Source name as it appears under ``data.source``.
        field_type: ``"prognostic"``, ``"static"`` or ``"dynamic_forcing"``.
        x_slice: Where these channels sit in the model input tensor ``x``.
        src_slice: Where their replacement comes from at the next rollout step —
            into ``y_pred`` for prognostic groups, into ``x_dynfrc`` for
            dynamic-forcing groups, and ``None`` for fixed groups (static),
            which are carried forward unchanged.
    """

    source: str
    field_type: str
    x_slice: slice
    src_slice: "slice | None"


def build_channel_layout(conf):
    """Return (groups, n_pred) derived from the variables config.

    Handles any number of sources.  ``ConcatToTensor`` lays both ``x`` and
    ``y_pred`` out source-major, so a field type's channels are contiguous only
    *within* a source — hence one entry per ``(source, field_type)`` pair rather
    than one per field type.

    Each source resolves its own level count (``data.source.<name>.levels``,
    falling back to ``model.levels``), so sources may differ in vertical extent.

    Parameters
    ----------
    conf : dict
        Full CREDIT config dict.

    Returns
    -------
    groups : dict[str, ChannelGroup]
        Keyed ``"{source}/{field_type}"``, in ``x`` channel order.  Diagnostic
        groups are excluded (never inputs), but their width is still accounted
        for when computing prognostic offsets into ``y_pred``.
    n_pred : int
        Total number of predicted (prognostic) channels across all sources.

    Raises
    ------
    ValueError
        If a source declares 3D variables but no level count is resolvable, or
        if ``history_len`` is greater than 1 — the flat rollout paths overwrite
        channels in place and cannot shift a history window.
    """
    data_conf = conf["data"]
    model_levels = (conf.get("model") or {}).get("levels")

    groups: dict[str, ChannelGroup] = {}
    x_cursor = 0  # position in the model input tensor x
    pred_cursor = 0  # position in y_pred (prognostic + diagnostic, source-major)
    dyn_cursor = 0  # position in x_dynfrc (dynamic_forcing only, source-major)

    for source_name, src_conf in data_conf["source"].items():
        src_conf = src_conf or {}
        variables = src_conf.get("variables") or {}
        src_levels = src_conf.get("levels")
        n_levels = len(src_levels) if src_levels else model_levels

        history_len = src_conf.get("history_len", data_conf.get("history_len", 1))
        if history_len != 1:
            raise ValueError(
                f"build_channel_layout: source '{source_name}' has history_len={history_len}, but the "
                "flat rollout paths overwrite channels in place and cannot shift a history window. "
                "Use the gen2 rollout (credit rollout), which assembles inputs through the preblocks."
            )

        def _width(field_type, _source=source_name, _variables=variables, _n_levels=n_levels):
            grp = _variables.get(field_type) or {}
            n_3d = len(grp.get("vars_3D") or [])
            n_2d = len(grp.get("vars_2D") or [])
            if n_3d and not _n_levels:
                raise ValueError(
                    f"build_channel_layout: source '{_source}' defines 3D variables but neither "
                    f"data.source.{_source}.levels nor model.levels is set."
                )
            return n_3d * (_n_levels or 0) + n_2d

        # y_pred runs prognostic-then-diagnostic within each source, so this
        # source's prognostic block starts after every earlier source's
        # prognostics AND diagnostics.  Reserve both before moving on.
        prognostic_src = slice(pred_cursor, pred_cursor + _width("prognostic"))
        pred_cursor += _width("prognostic") + _width("diagnostic")

        for field_type in _INPUT_FIELD_TYPES:
            width = _width(field_type)
            if width == 0:
                continue  # absent or null group contributes no channels
            if field_type in _MODEL_PREDICTED:
                src_slice = prognostic_src
            elif field_type in _DATASET_DRIVEN:
                src_slice = slice(dyn_cursor, dyn_cursor + width)
                dyn_cursor += width
            else:
                src_slice = None  # static and friends: carried forward
            groups[f"{source_name}/{field_type}"] = ChannelGroup(
                source=source_name,
                field_type=field_type,
                x_slice=slice(x_cursor, x_cursor + width),
                src_slice=src_slice,
            )
            x_cursor += width

    n_pred = sum(g.x_slice.stop - g.x_slice.start for g in groups.values() if g.field_type in _MODEL_PREDICTED)
    return groups, n_pred


def update_x(x_prev, x_dynfrc, y_pred, groups):
    """Build the next-step input tensor for autoregressive rollout.

    Replaces dataset-driven channels (dynamic_forcing) and model-predicted
    channels (prognostic) in x_prev.  Fixed channels (static, etc.) are
    carried forward unchanged via clone.

    Each group carries an explicit source slice, so this is correct however the
    sources interleave — it never assumes that a field type's channels are
    contiguous or that they arrive in the same order as the destination.

    Parameters
    ----------
    x_prev : Tensor  [B, C, ...]
        Full input tensor from the previous step.
    x_dynfrc : Tensor  [B, C_dyn, ...]
        New dynamic-forcing channels from the dataset, source-major.
    y_pred : Tensor  [B, C_out, ...]
        The **full** model output — prognostic and diagnostic channels, exactly
        as the model emits them.  Do not pre-slice it down to the prognostics:
        with several sources those are not a leading contiguous block, and the
        group's ``src_slice`` already indexes them.
    groups : dict[str, ChannelGroup]
        From build_channel_layout().

    Returns
    -------
    Tensor  [B, C, ...]
        Updated input tensor ready for the next forward pass.
    """
    x_new = x_prev.clone()

    for group in groups.values():
        if group.src_slice is None:
            continue  # fixed group: already present via clone()
        source = x_dynfrc if group.field_type in _DATASET_DRIVEN else y_pred
        x_new[:, group.x_slice, ...] = source[:, group.src_slice, ...]

    return x_new


# ---------------------------------------------------------------------------
# ChannelSchema
# ---------------------------------------------------------------------------

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

            # Input tensors carry a time dim of length history_len (the dataset
            # loads a history_len-step window for prognostic / dynamic_forcing /
            # static fields at i == 0). The target is always a single step at
            # t+dt, so its n_time stays 1. history_len resolution mirrors
            # BaseDataset._load_history_len: source-level first, then data-level,
            # default 1 — so history_len == 1 configs are unaffected.
            history_len = src_conf.get("history_len", data_conf.get("history_len", 1))
            if not isinstance(history_len, int) or history_len < 1:
                raise ValueError(f"history_len must be a positive int, got {history_len!r}")

            def _entries(field_type: str, n_time: int) -> list[dict]:
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
                            "n_time": n_time,
                        }
                    )
                for vname in grp.get("vars_2D") or []:
                    entries.append(
                        {
                            "var_key": f"{source_name}/{field_type}/2d/{vname}",
                            "n_levels": 1,
                            "n_time": n_time,
                        }
                    )
                return entries

            for field_type in _INPUT_FIELD_TYPES:
                input_layout.extend(_entries(field_type, history_len))
            for field_type in _TARGET_FIELD_TYPES:
                target_layout.extend(_entries(field_type, 1))

        return cls(input_layout, target_layout)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Write the schema as YAML (atomically: temp file + rename).

        The staging file is per-process: concurrent writers (multiple ranks
        saving the schema at init) would otherwise interleave their dumps into
        one shared ``<path>.tmp`` and rename a corrupted file into place.
        """
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
        tmp = f"{path}.tmp.{os.getpid()}"
        try:
            with open(tmp, "w") as f:
                yaml.safe_dump(payload, f, sort_keys=False)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(tmp)
            raise
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
