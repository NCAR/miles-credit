import torch
import torch.nn.functional as F


class TensorPadding:
    def __init__(
        self,
        mode="earth",
        pad_lat=(40, 40),
        pad_lon=(40, 40),
        pole_flip_channels=None,
        pole_flip_offsets=None,
        **kwargs,
    ):
        """
        Initialize the TensorPadding class with the specified mode and padding sizes.

        Args:
            mode (str): The padding mode, either 'mirror' or 'earth'.
            pad_lat (list[int]): Padding sizes for the North-South (latitude) dimension [top, bottom].
            pad_lon (list[int]): Padding sizes for the West-East (longitude) dimension [left, right].
            pole_flip_channels (list[int], optional): 'earth' mode only. Channels (dim 1) that change sign when
                reflected across a pole: the eastward and northward components of vector fields (u, v, ...).
                The pole padding copies the rows nearest the pole with a 180-degree longitude shift; across the
                pole both components of a horizontal vector reverse, so without this the padded rows show the
                network winds pointing the wrong way. Default: none (the original behaviour).
            pole_flip_offsets (list[float], optional): one per ``pole_flip_channels`` entry. The flip is applied
                to the (usually normalized) model input, so for a standardized channel z = (x - mu) / sigma the
                physical reflection x -> -x is z -> -z - 2 mu / sigma: pass 2 mu / sigma here. Default: zeros
                (exact for channels with zero mean).
        """

        self.mode = mode
        self.pad_NS = pad_lat
        self.pad_WE = pad_lon
        self.pole_flip_channels = list(pole_flip_channels or [])
        offsets = [0.0] * len(self.pole_flip_channels) if pole_flip_offsets is None else list(pole_flip_offsets)
        if len(offsets) != len(self.pole_flip_channels):
            raise ValueError(
                f"pole_flip_offsets has {len(offsets)} entries for {len(self.pole_flip_channels)} pole_flip_channels"
            )
        if self.pole_flip_channels and mode != "earth":
            raise ValueError("pole_flip_channels needs mode='earth'")
        self.pole_flip_offsets = [float(v) for v in offsets]

    def _pole_flip(self, rows):
        """Negate the vector channels of the pole-padding rows ``rows`` (B, C, ...): z -> -z - offset."""
        if not self.pole_flip_channels:
            return rows
        sign = torch.ones(rows.shape[1], device=rows.device, dtype=rows.dtype)
        off = torch.zeros(rows.shape[1], device=rows.device, dtype=rows.dtype)
        idx = torch.as_tensor(self.pole_flip_channels, device=rows.device)
        sign[idx] = -1.0
        off[idx] = torch.as_tensor(self.pole_flip_offsets, device=rows.device, dtype=rows.dtype)
        shape = [1, rows.shape[1]] + [1] * (rows.dim() - 2)
        return rows * sign.view(shape) - off.view(shape)

    def pad(self, x):
        """
        Apply padding to the tensor based on the specified mode.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, var, time, lat, lon).

        Returns:
            torch.Tensor: The padded tensor.
        """
        if self.mode == "mirror":
            return self._mirror_padding(x)
        elif self.mode == "earth":
            return self._earth_padding(x)

    def unpad(self, x):
        """
        Remove padding from the tensor based on the specified mode.

        Args:
            x (torch.Tensor): Padded tensor of shape (batch, var, time, lat, lon).

        Returns:
            torch.Tensor: The unpadded tensor.
        """
        if self.mode == "mirror":
            return self._mirror_unpad(x)
        elif self.mode == "earth":
            return self._earth_unpad(x)

    def _earth_padding(self, x):
        """
        Apply earth padding to the tensor (poles and circular padding).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The padded tensor.
        """
        if any(p > 0 for p in self.pad_NS):
            # 180-degree shift using half the longitude size
            shift_size = int(x.shape[-1] // 2)
            xroll = torch.roll(x, shifts=shift_size, dims=-1)
            # pad poles
            xroll_flip_top = self._pole_flip(torch.flip(xroll[..., : self.pad_NS[0], :], (-2,)))
            xroll_flip_bot = self._pole_flip(torch.flip(xroll[..., -self.pad_NS[1] :, :], (-2,)))
            x = torch.cat([xroll_flip_top, x, xroll_flip_bot], dim=-2)

        if any(p > 0 for p in self.pad_WE):
            x = F.pad(x, (self.pad_WE[0], self.pad_WE[1], 0, 0, 0, 0), mode="circular")

        return x

    def _earth_unpad(self, x):
        """
        Remove earth padding to restore the original tensor size.

        Args:
            x (torch.Tensor): Padded tensor.

        Returns:
            torch.Tensor: The unpadded tensor.
        """
        # unpad along latitude (north-south)
        if any(p > 0 for p in self.pad_NS):
            start_NS = self.pad_NS[0]
            end_NS = -self.pad_NS[1] if self.pad_NS[1] > 0 else None
            x = x[..., start_NS:end_NS, :]

        # unpad along longitude (west-east)
        if any(p > 0 for p in self.pad_WE):
            start_WE = self.pad_WE[0]
            end_WE = -self.pad_WE[1] if self.pad_WE[1] > 0 else None
            x = x[..., :, start_WE:end_WE]

        return x

    def _mirror_padding(self, x):
        """
        Apply mirror padding to the tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The padded tensor.
        """
        # pad along longitude (west-east)
        if any(p > 0 for p in self.pad_WE):
            pad_lon_left, pad_lon_right = self.pad_WE
            x = F.pad(x, pad=(self.pad_WE[0], self.pad_WE[1], 0, 0, 0, 0), mode="circular")

        # pad along latitude (north-south)
        if any(p > 0 for p in self.pad_NS):
            x = F.pad(x, pad=(0, 0, self.pad_NS[0], self.pad_NS[1], 0, 0), mode="reflect")

        return x

    def _mirror_unpad(self, x):
        """
        Remove mirror padding to restore the original tensor size.

        Args:
            x (torch.Tensor): Padded tensor.

        Returns:
            torch.Tensor: The unpadded tensor.
        """
        # unpad along latitude (north-south)
        if any(p > 0 for p in self.pad_NS):
            x = x[..., self.pad_NS[0] : -self.pad_NS[1], :]

        # unpad along longitude (west-east)
        if any(p > 0 for p in self.pad_WE):
            x = x[..., :, self.pad_WE[0] : -self.pad_WE[1]]

        return x
