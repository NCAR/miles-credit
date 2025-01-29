import torch


def parse_dtype(dtype_str: str) -> torch.dtype:
    """
    Parses a string representation of a torch dtype and returns the corresponding torch.dtype object.

    Args:
        dtype_str (str): The string representation of the dtype (e.g., "float32", "int64").

    Returns:
        torch.dtype: The corresponding torch.dtype object.
    """
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "float": torch.float32,  # Alias for float32
        "double": torch.float64,  # Alias for float64
        "half": torch.float16,  # Alias for float16
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "short": torch.int16,  # Alias for int16
        "long": torch.int64,  # Alias for int64
        "bool": torch.bool,
    }

    # Convert the dtype string to lowercase for case-insensitive matching
    dtype_str_lower = dtype_str.lower()

    if dtype_str_lower in dtype_map:
        return dtype_map[dtype_str_lower]
    else:
        raise ValueError(f"Invalid dtype string: {dtype_str}")
