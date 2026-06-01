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
