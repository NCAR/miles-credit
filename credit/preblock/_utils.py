import torch


def _parse_variable_selection(variable_list: list, state_dict: dict, data_types: list = None) -> list:
    """
    The goal of this function is to expand partial variables in `variable_list` based on the keys in `state_dict`.
    For example, if `variable_list` contains "ERA5/prognostic/3d", then this function should expand that partial name
    to include every variable in `state_dict` that has the matching partial name.

    Args:
        variable_list (list): A list of partial or full variable names. An empty list means that all variables should be used.
        state_dict (dict): A dictionary of the data state following the CREDIT convention, nested as
        `state_dict[data_type][source][var_name]`. The data types (input, target, prediction) are the
        highest level key, then the source (e.g. "era5"), then the full variable names
        (e.g. `"era5/prognostic/3d/temperature"`).
        data_types (list): A list of data types to include. If None, all data types will be included.

    Returns:
        list: The full variable names from `state_dict` matched by `variable_list`, with duplicates
        removed and order preserved.
    """
    # Default to every data type present in the state.
    if data_types is None:
        data_types = list(state_dict.keys())

    # Collect every full variable name across the selected data types and sources,
    # preserving first-seen order and dropping duplicates (the same variable may
    # appear under more than one data type).
    all_variables = []
    for data_type in data_types:
        for source in state_dict.get(data_type, {}).values():
            for var_name in source:
                if var_name not in all_variables:
                    all_variables.append(var_name)

    # An empty selection means "use everything".
    if not variable_list:
        return all_variables

    # Expand each (possibly partial) name: a variable matches if it equals the
    # entry or sits beneath it in the "/"-delimited hierarchy.
    selected = []
    for partial in variable_list:
        for var_name in all_variables:
            if var_name == partial or var_name.startswith(partial + "/"):
                if var_name not in selected:
                    selected.append(var_name)
    return selected


def _flatten_spatial_tensors(state_dict: dict, spatial_variables: list) -> tuple[dict, dict]:
    """Return a copy of `state_dict` with tensors for `spatial_variables` flattened to (batch, -1).

    Grid-wise ("spatial") scalers store one statistic per gridpoint rather than
    per vertical level, so their channel dimension has to be the flattened
    spatial axis instead of the level axis. Folding every non-batch dimension
    (level, time, and the spatial dims) into a single trailing axis produces a
    rank-2 tensor, where the `channels_first` (dim=1) and `channels_last`
    (dim=-1) conventions point at the same axis — so the flattened tensor
    transforms correctly under either scaler setting with no extra branching.

    Requires the level and time dims (indices 1 and 2) to be singleton, since
    these blocks always operate on one forecast step at a time; flattening a
    >1 level or time dim would blend distinct slices into the same per-gridpoint
    statistic.

    Args:
        state_dict (dict): Nested batch dict (any depth) whose leaves are tensors.
        spatial_variables (list): Full variable keys (not partial paths — callers
            must resolve prefixes via `_parse_variable_selection` first) to flatten.
            An empty list is a no-op.

    Returns:
        tuple[dict, dict]: A copy of `state_dict` with the matched tensors
        flattened, and a `{var_key: original_shape}` map to pass to
        `_unflatten_spatial_tensors` afterward.
    """
    if not spatial_variables:
        return state_dict, {}
    spatial_set = set(spatial_variables)
    original_shapes = {}

    def _walk(d):
        new_d = {}
        for key, value in d.items():
            if isinstance(value, dict):
                new_d[key] = _walk(value)
            elif key in spatial_set:
                if value.shape[1] != 1 or value.shape[2] != 1:
                    raise ValueError(
                        f"Spatial scaling for '{key}' requires singleton level and time dims "
                        f"(dims 1 and 2), got shape {tuple(value.shape)}."
                    )
                original_shapes[key] = value.shape
                new_d[key] = value.reshape(value.shape[0], -1)
            else:
                new_d[key] = value
        return new_d

    return _walk(state_dict), original_shapes


def _spatial_scaler_template(ndim: int):
    """Unfitted per-location standard scaler for an ``ndim``-D CREDIT tensor.

    CREDIT tensors are ``(batch, level, time, *grid)``: the levels are the channel
    axis, batch and time are reduced, and every grid axis (``lat, lon`` or ``ncol``)
    gets its own statistics, so 3D variables are scaled per level and per gridpoint.
    """
    try:
        from bridgescaler.distributed_tensor import DSpatialStandardScalerTensor
    except ImportError as exc:
        raise ImportError(
            "Fitting spatial_variables needs a bridgescaler with DSpatialStandardScalerTensor (newer than 0.8.6)."
        ) from exc

    return DSpatialStandardScalerTensor(channels_last=False, spatial_dims=tuple(range(3 - ndim, 0)))


def _leaf_ndims(state_dict: dict, keys: list) -> dict:
    """``{key: ndim}`` for the tensors named in *keys* (first occurrence wins)."""
    wanted, ndims = set(keys), {}

    def _walk(d):
        for key, value in d.items():
            if isinstance(value, dict):
                _walk(value)
            elif key in wanted and key not in ndims and torch.is_tensor(value):
                ndims[key] = value.ndim

    _walk(state_dict)
    return ndims


def _merge_nested(a: dict, b: dict) -> dict:
    """Recursively merge nested dict *b* into a copy of *a* (``b`` wins on leaf clashes)."""
    merged = dict(a)
    for key, value in b.items():
        both_dicts = isinstance(value, dict) and isinstance(merged.get(key), dict)
        merged[key] = _merge_nested(merged[key], value) if both_dicts else value
    return merged


def _legacy_spatial_variables(scaler_dict: dict | None, spatial_variables: list) -> list:
    """The *spatial_variables* whose fitted scaler still needs `_flatten_spatial_tensors`.

    Scaler files fit before per-location scaling store a spatial variable as a plain
    ``DStandardScalerTensor`` with one column per gridpoint, which only lines up with
    the tensor once it is flattened to ``(batch, H*W)``. A ``DSpatialStandardScalerTensor``
    takes the tensor as it is, so those variables are left out of the flatten.
    """
    try:
        from bridgescaler.distributed_tensor import DSpatialStandardScalerTensor
    except ImportError:  # a bridgescaler without it can only have written legacy scalers
        return list(spatial_variables)

    spatial_set, legacy = set(spatial_variables), set()

    def _walk(d):
        for key, value in d.items():
            if isinstance(value, dict):
                _walk(value)
            elif key in spatial_set and not isinstance(value, DSpatialStandardScalerTensor):
                legacy.add(key)

    _walk(scaler_dict or {})
    return [v for v in spatial_variables if v in legacy]


def _pass_through_empty_cells(scaler_dict: dict) -> dict:
    """Make spatial scalers the identity at cells that never had data; returns *scaler_dict*.

    A ``DSpatialStandardScalerTensor`` cell that was NaN in every fitted sample (land
    in an ocean field, say) has ``n_ == 0`` and a NaN mean and variance, so both
    ``transform`` and ``inverse_transform`` return NaN there and the loss turns NaN.
    Setting those cells to mean 0 and variance 0 (which the scaler applies with
    std 1) makes both directions the identity; observed cells keep their statistics.
    Only the loaded scalers change, in place; the file on disk keeps NaN as
    "never observed".
    """
    try:
        from bridgescaler.distributed_tensor import DSpatialStandardScalerTensor
    except ImportError:  # no per-location scalers to patch
        return scaler_dict

    def _walk(d):
        for value in d.values():
            if isinstance(value, dict):
                _walk(value)
            elif isinstance(value, DSpatialStandardScalerTensor) and value.n_ is not None:
                empty = torch.as_tensor(value.n_) == 0
                if empty.any():
                    zero = torch.zeros((), dtype=value.mean_x_.dtype, device=value.mean_x_.device)
                    value.mean_x_ = torch.where(empty, zero, value.mean_x_)
                    value.var_x_ = torch.where(empty, zero, value.var_x_)

    _walk(scaler_dict or {})
    return scaler_dict


def _unflatten_spatial_tensors(state_dict: dict, original_shapes: dict) -> dict:
    """Return a copy of `state_dict` with tensors restored to the shapes recorded by `_flatten_spatial_tensors`."""
    if not original_shapes:
        return state_dict

    def _walk(d):
        new_d = {}
        for key, value in d.items():
            if isinstance(value, dict):
                new_d[key] = _walk(value)
            elif key in original_shapes:
                new_d[key] = value.reshape(original_shapes[key])
            else:
                new_d[key] = value
        return new_d

    return _walk(state_dict)
