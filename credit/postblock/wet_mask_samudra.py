import xarray as xr
import torch.nn as nn
from credit.ocean.samudra_constants import PROG_VARS_MAP, TensorMap
from credit.ocean.samudra_data import validate_data, extract_wet_mask


class WetMaskBlock(nn.Module):
    """
    Post-processing layer that applies wet mask to ocean predictions
    Zero trainable parameters, but mask influences gradients
    
    Masks predictions so land points = 0, ocean points preserve values.
    This encourages the model to focus learning on ocean regions.
    """
    def __init__(self, conf):
        super().__init__()

        data_path = conf["data"]["data_path"]
        data_means_path = conf["data"]["mean_path"]
        data_stds_path = conf["data"]["std_path"]
        
        # Load data and initialize tensor mapping
        data_xr = xr.open_zarr(data_path, chunks={})
        data_mean_xr = xr.open_zarr(data_means_path, chunks={})
        data_std_xr = xr.open_zarr(data_stds_path, chunks={})
        
        # Initialize TensorMap for variable organization
        try:
            # whatever call initializes TensorMap
            TensorMap.init_instance(
                conf["data"]["prognostic_vars_key"], conf["data"]["boundary_vars_key"]
            )
        except ValueError as e:
            if "TensorMap already initialized" in str(e):
                TensorMap.get_instance()
            else:
                raise

        # This needs to be called to add mask vars to the open dataframe
        _ = validate_data(data_xr, data_mean_xr, data_std_xr)
        
        # Get prognostic variables from config
        prognostic_vars = PROG_VARS_MAP[conf["data"]["prognostic_vars_key"]]
        
        # Extract wet masks for ocean model
        # wet: 3D mask (vars, levels, lat, lon) or (vars, lat, lon) 
        # wet_surface: 2D surface mask (lat, lon)
        wet, wet_surface = extract_wet_mask(data_xr, prognostic_vars, 0)
        
        # Register as buffers (not trainable parameters)
        # Convert to float for multiplication, keep boolean info in names
        self.register_buffer('wet_mask', wet.float())  # Main 3D ocean mask
        self.register_buffer('wet_surface_mask', wet_surface.float())  # Surface mask
        
    def forward(self, predictions):
        """
        Apply wet mask to predictions with gradient influence
        
        Args:
            predictions: tensor of shape (batch, n_vars, time, lat, lon)
            
        Returns:
            masked_predictions: same shape, with land values set to zero
        """
        # Use the 3D wet mask for full predictions
        # wet_mask shape depends on your data structure - adjust as needed
        if len(self.wet_mask.shape) == 4:  # (vars, levels, lat, lon)
            # Flatten to (total_vars, lat, lon) to match prediction channels
            mask_flat = self.wet_mask.view(-1, self.wet_mask.shape[-2], self.wet_mask.shape[-1])
            mask_expanded = mask_flat.unsqueeze(0).unsqueeze(2)  # (1, vars, 1, lat, lon)
        else:  # Already (vars, lat, lon)
            mask_expanded = self.wet_mask.unsqueeze(0).unsqueeze(2)  # (1, vars, 1, lat, lon)
    
        # Apply mask (ocean=1 preserves values, land=0 zeros out)
        masked_predictions = predictions * mask_expanded.to(predictions.device)
        
        return masked_predictions