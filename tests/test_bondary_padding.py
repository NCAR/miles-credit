import torch
from credit.boundary_padding import TensorPadding


def test_TensorPadding():
    """
    Test the TensorPadding class for both 'mirror' and 'earth' padding modes.

    The test verifies the following:
    1. Correct padding shapes for input tensors with specified padding configurations.
    2. Properly padded tensor elements in both 'mirror' and 'earth' modes.
    3. Correct unpadding functionality, ensuring the unpadded tensor matches the original input.

    Test steps:
    -----------
    1. Set up configurations for 'mirror' and 'earth' padding modes.
    2. Initialize TensorPadding instances with these configurations.
    3. Create an input tensor of predefined shape (batch, channels, layers, latitude, longitude).
    4. Apply the padding to the input tensor using both 'mirror' and 'earth' modes.
    5. Extract specific padded tensor elements for validation.
    6. Unpad the padded tensors and compare them with the original input.
    7. Assert:
       - The padded tensor shape matches the expected shape.
       - The padded tensor elements match expected values.
       - The unpadded tensor returns to its original shape and values.

    Assertions:
    -----------
    - Checks that the padded tensor shape matches the expected output shape.
    - Validates that padded elements on the left and right are correct for both padding modes.
    - Ensures that the unpadded tensor has the same shape and values as the original input tensor.

    Raises:
    -------
    AssertionError if any of the padding or unpadding conditions fail.
    """
    padding_conf_mirror = {"mode": "mirror", "pad_lat": [12, 34], "pad_lon": [56, 78]}

    padding_conf_earth = {"mode": "earth", "pad_lat": [12, 34], "pad_lon": [56, 78]}

    opt_mirror = TensorPadding(**padding_conf_mirror)
    opt_earth = TensorPadding(**padding_conf_earth)

    input_shape = (1, 16, 2, 181, 360)
    output_shape = (1, 16, 2, 181 + 12 + 34, 360 + 56 + 78)

    x = torch.randn(input_shape)

    # get padded tensors
    x_pad_mirror = opt_mirror.pad(x)
    x_pad_earth = opt_mirror.pad(x)

    # get padded tensor elements
    x_pad_mirror_left = x_pad_mirror[..., :56][..., 12:-34, :]
    x_pad_mirror_right = x_pad_mirror[..., -78:][..., 12:-34, :]

    x_pad_earth_left = x_pad_earth[..., :56][..., 12:-34, :]
    x_pad_earth_right = x_pad_earth[..., -78:][..., 12:-34, :]

    # get unpadded tensors
    x_unpad_mirror = opt_mirror.unpad(x_pad_mirror)
    x_unpad_earth = opt_earth.unpad(x_pad_earth)

    # padding checks (must pass all)
    assert (
        x_pad_mirror.shape == output_shape
    ), "Shape mismatch error found in mirror padding"
    assert (
        x_pad_earth.shape == output_shape
    ), "Shape mismatch error found in earth padding"
    assert (
        x[..., -56:] - x_pad_mirror_left
    ).sum() == 0, "Tensor elements mismatch found in mirror padding"
    assert (
        x[..., -56:] - x_pad_earth_left
    ).sum() == 0, "Tensor elements mismatch found in earth padding"
    assert (
        x[..., :78] - x_pad_mirror_right
    ).sum() == 0, "Tensor elements mismatch found in mirror padding"
    assert (
        x[..., :78] - x_pad_earth_right
    ).sum() == 0, "Tensor elements mismatch found in earth padding"
    assert (
        x_unpad_mirror.shape == input_shape
    ), "Shape mismatch error found in mirror unpad"
    assert (
        x_unpad_earth.shape == input_shape
    ), "Shape mismatch error found in earth unpad"
    assert (
        x_unpad_mirror - x
    ).sum() == 0, "Tensor elements mismatch found in mirror padding"
    assert (
        x_unpad_earth - x
    ).sum() == 0, "Tensor elements mismatch found in earth padding"
