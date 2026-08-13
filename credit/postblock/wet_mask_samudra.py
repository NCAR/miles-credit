import xarray as xr
import torch.nn as nn
from credit.ocean.samudra_constants import PROG_VARS_MAP, TensorMap
from credit.ocean.samudra_data import validate_data, extract_wet_mask


class WetMaskBlock(nn.Module):
    """
    Post-processing layer that applies wet mask to ocean predictions.
    Zero trainable parameters, but mask influences gradients.

    Masks predictions so land points = 0, ocean points preserve values.
    This encourages the model to focus learning on ocean regions.
    """

    def __init__(self, conf, key: str = "prediction"):
        super().__init__()
        self.key = key

        data_path = conf["data"]["data_path"]
        data_means_path = conf["data"]["mean_path"]
        data_stds_path = conf["data"]["std_path"]

        data_xr = xr.open_zarr(data_path, chunks={})
        data_mean_xr = xr.open_zarr(data_means_path, chunks={})
        data_std_xr = xr.open_zarr(data_stds_path, chunks={})

        try:
            TensorMap.init_instance(conf["data"]["prognostic_vars_key"], conf["data"]["dynamic_forcing_vars_key"])
        except ValueError as e:
            if "TensorMap already initialized" in str(e):
                TensorMap.get_instance()
            else:
                raise

        _ = validate_data(data_xr, data_mean_xr, data_std_xr)

        prognostic_vars = PROG_VARS_MAP[conf["data"]["prognostic_vars_key"]]
        wet, wet_surface = extract_wet_mask(data_xr, prognostic_vars, 0)

        self.register_buffer("wet_mask", wet.float())
        self.register_buffer("wet_surface_mask", wet_surface.float())

    def forward(self, batch_dict: dict) -> dict:
        """Apply wet mask to ``batch_dict[self.key]`` (land=0, ocean preserved)."""
        predictions = batch_dict[self.key]

        if len(self.wet_mask.shape) == 4:  # (vars, levels, lat, lon)
            mask_flat = self.wet_mask.view(-1, self.wet_mask.shape[-2], self.wet_mask.shape[-1])
            mask_expanded = mask_flat.unsqueeze(0).unsqueeze(2)  # (1, vars, 1, lat, lon)
        else:  # (vars, lat, lon)
            mask_expanded = self.wet_mask.unsqueeze(0).unsqueeze(2)  # (1, vars, 1, lat, lon)

        batch_dict[self.key] = predictions * mask_expanded.to(predictions.device)
        return batch_dict
