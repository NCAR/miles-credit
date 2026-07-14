"""
channel_layout.py
-----------------
Derive per-group channel slices directly from the variables config and
provide a standalone update function for multi-step rollout.

The concat order matches the canonical rank defined by FIELD_TYPE_RANK
(prognostic < dynamic_forcing < static < diagnostic), with vars_3D before
vars_2D within each group.  This is the same rank used by
credit.preblock.concat.ConcatToTensor, so slices returned here always
correspond to the actual channel positions in the model input tensor.

Usage
-----
    from credit.datasets.gen_2.channel_layout import build_channel_layout, update_x

    # once at trainer / rollout init
    slices, n_pred = build_channel_layout(conf)

    # at every t > 1 step
    x = update_x(x, x_dynfrc, y_pred, slices)
"""

# Canonical cross-group concat rank — shared with credit.preblock.concat.
# Groups absent from the config are ignored; unknown groups sort to the end.
FIELD_TYPE_RANK: dict[str, int] = {
    "prognostic": 0,
    "static": 1,
    "dynamic_forcing": 2,
    "diagnostic": 3,
}

# Semantic roles — determines update behaviour at t > 1.
_DIAGNOSTIC = frozenset({"diagnostic"})  # target-only, never in x
_DATASET_DRIVEN = frozenset({"dynamic_forcing"})
_MODEL_PREDICTED = frozenset({"prognostic"})
# "static" (and anything else) falls through as fixed


def build_channel_layout(conf):
    """Return (slices, n_pred) derived from the variables config.

    Slices are ordered by FIELD_TYPE_RANK (matching ConcatToTensor), so
    slice offsets correspond directly to channel positions in the tensor.

    Parameters
    ----------
    conf : dict
        Full CREDIT config dict.  Reads the first entry under
        conf["data"]["source"] for levels and variable groups.

    Returns
    -------
    slices : dict[str, slice]
        Mapping from group name to its slice in the channel dimension of x.
        Groups in _DIAGNOSTIC are excluded (they are never in x).
        Order matches FIELD_TYPE_RANK, not config key order.
    n_pred : int
        Total number of predicted (prognostic) channels.
    """
    src_conf = next(iter(conf["data"]["source"].values()))
    n_levels = len(src_conf["levels"])
    variables = src_conf["variables"]

    # Sort groups by canonical rank so slice offsets match concat.py output.
    sorted_groups = sorted(
        ((name, grp) for name, grp in variables.items() if grp is not None and name not in _DIAGNOSTIC),
        key=lambda pair: FIELD_TYPE_RANK.get(pair[0], len(FIELD_TYPE_RANK)),
    )

    slices = {}
    offset = 0
    for name, grp in sorted_groups:
        n = len(grp.get("vars_3D", [])) * n_levels + len(grp.get("vars_2D", []))
        slices[name] = slice(offset, offset + n)
        offset += n

    n_pred = sum(sl.stop - sl.start for name, sl in slices.items() if name in _MODEL_PREDICTED)
    return slices, n_pred


def update_x(x_prev, x_dynfrc, y_pred, slices):
    """Build the next-step input tensor for autoregressive rollout.

    Replaces dataset-driven channels (dynamic_forcing) and model-predicted
    channels (prognostic) in x_prev.  Fixed channels (static, etc.) are
    carried forward unchanged via clone.

    Parameters
    ----------
    x_prev : Tensor  [B, C, ...]
        Full input tensor from the previous step.
    x_dynfrc : Tensor  [B, C_dyn, ...]
        New dynamic-forcing channels from the dataset, in the same relative
        order as the dataset-driven groups appear in slices.
    y_pred : Tensor  [B, C_pred, ...]
        Model output, in the same relative order as the predicted groups
        appear in slices.
    slices : dict[str, slice]
        From build_channel_layout().

    Returns
    -------
    Tensor  [B, C, ...]
        Updated input tensor ready for the next forward pass.
    """
    x_new = x_prev.clone()

    dyn_offset = 0
    pred_offset = 0
    for name, sl in slices.items():
        n = sl.stop - sl.start
        if name in _DATASET_DRIVEN:
            x_new[:, sl, ...] = x_dynfrc[:, dyn_offset : dyn_offset + n, ...]
            dyn_offset += n
        elif name in _MODEL_PREDICTED:
            x_new[:, sl, ...] = y_pred[:, pred_offset : pred_offset + n, ...]
            pred_offset += n
        # fixed groups: already present via clone()

    return x_new
