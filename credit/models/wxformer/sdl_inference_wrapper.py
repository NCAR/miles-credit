# Core Python
import os
import sys
import yaml
import copy
import logging
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime
import xarray as xr

# Numerical and ML
import numpy as np
import torch
import torch.nn as nn

# CREDIT framework
from credit.datasets.era5_multistep_batcher import Predict_Dataset_Batcher
from credit.datasets.load_dataset_and_dataloader import BatchForecastLenDataLoader
from credit.parser import credit_main_parser
from credit.datasets import setup_data_loading
from credit.models import load_model
from credit.transforms import load_transforms, Normalize_ERA5_and_Forcing
from credit.seed import seed_everything
from credit.interp import full_state_pressure_interpolation
from credit.output import make_xarray

import argparse
import multiprocessing as mp
import warnings


logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class StylizationWrapper(nn.Module):
    """
    Specialized wrapper for hurricane track stylization with latent vector control.
    
    Capabilities:
    - Directional bias injection (steering flow modification)
    - Intensity-dependent noise scaling
    - Store and retrieve latent vectors Z for exact forecast reproduction
    - Interpolate between latent vectors for smooth ensemble exploration
    """
    
    def __init__(self, pretrained_model: nn.Module, channel_config: Dict = None):
        super().__init__()
        self.model = pretrained_model
        
        # Ensure model is in eval mode and frozen
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        self._noise_layers = self._collect_noise_layers()
        self._original_factors = self.get_noise_factors()
        
        # Storage for latent vectors
        self._stored_latents: Dict[str, Dict] = {}  # Now stores dict with 'latents' and 'timesteps'
        self._current_latents: Optional[List[torch.Tensor]] = None
        self._current_timestep_map: Optional[List[int]] = None  # Track which timestep each latent belongs to
        self._capture_enabled = False
        
        logging.info(f"Pre-trained noise factors {self._original_factors}")

    def _collect_noise_layers(self) -> List[nn.Module]:
        """Collect all PixelNoiseInjection layers from the model."""
        noise_layers = []
        
        if hasattr(self.model, 'encoder_noise_layers') and self.model.encoder_noise:
            noise_layers.extend(self.model.encoder_noise_layers)
        
        decoder_layers = [
            self.model.noise_inject1,
            self.model.noise_inject2, 
            self.model.noise_inject3
        ]
        noise_layers.extend(decoder_layers)
        
        return noise_layers
    
    def get_noise_factors(self) -> List[float]:
        """Get current noise factors from all layers."""
        return [layer.noise_factor.item() for layer in self._noise_layers]
    
    def set_noise_factors(self, factors: Union[float, List[float]]):
        """Set noise factors for all layers."""
        if isinstance(factors, (int, float)):
            for layer in self._noise_layers:
                layer.noise_factor.data.fill_(factors)
        elif isinstance(factors, (list, tuple)):
            if len(factors) != len(self._noise_layers):
                raise ValueError(f"Expected {len(self._noise_layers)} factors, got {len(factors)}")
            
            for layer, factor in zip(self._noise_layers, factors):
                layer.noise_factor.data.fill_(factor)
    
    def reset_to_original(self):
        """Reset to original pretrained noise factors."""
        self.set_noise_factors(self._original_factors)
        logging.info("Reset to original noise factors")
    
    def set_encoder_noise_factors(self, factors: Union[float, List[float]]):
        """Set encoder noise factors."""
        if not (hasattr(self.model, 'encoder_noise_layers') and self.model.encoder_noise):
            return
            
        encoder_layers = self.model.encoder_noise_layers
        
        if isinstance(factors, (int, float)):
            for layer in encoder_layers:
                layer.noise_factor.data.fill_(factors)
        else:
            for layer, factor in zip(encoder_layers, factors):
                layer.noise_factor.data.fill_(factor)

    def set_decoder_noise_factors(self, factors: Union[float, List[float]]):
        """Set decoder noise factors."""
        decoder_layers = [
            self.model.noise_inject1,
            self.model.noise_inject2, 
            self.model.noise_inject3
        ]
        
        if isinstance(factors, (int, float)):
            for layer in decoder_layers:
                layer.noise_factor.data.fill_(factors)
        else:
            for layer, factor in zip(decoder_layers, factors):
                layer.noise_factor.data.fill_(factor)

    def set_decoder_modulation(self, target_channels: List[int] = None, weight: float = 2.0):
        """Set decoder modulation weights for specific channels."""
        decoder_layers = [
            self.model.noise_inject1,
            self.model.noise_inject2, 
            self.model.noise_inject3
        ]
        
        if target_channels:
            for layer in decoder_layers:
                for ch in target_channels:
                    if ch < layer.modulation.shape[1]:
                        layer.modulation.data[0, ch, 0, 0] *= weight
    
    def set_decoder_style_vector(self, channel_weights: Dict[int, float]):
        """Modify the model's style transformation weights."""
        decoder_layers = [
            self.model.noise_inject1,
            self.model.noise_inject2, 
            self.model.noise_inject3
        ]
        
        for layer in decoder_layers:
            for channel_idx, weight in channel_weights.items():
                if channel_idx < layer.noise_transform.weight.shape[0]:
                    layer.noise_transform.weight.data[channel_idx] *= weight
    
    def set_manual_factors(self, large_scale: float, medium_scale: float, fine_scale: float):
        """Manually set decoder noise factors with optional variable and level targeting."""
        factors = []
        
        # Add encoder factors if they exist
        if hasattr(self.model, 'encoder_noise_layers') and self.model.encoder_noise:
            factors.extend([large_scale, medium_scale, fine_scale])
        
        # Add decoder factors
        factors.extend([large_scale, medium_scale, fine_scale])
        
        # Trim to available layers
        self.set_noise_factors(factors[:len(self._noise_layers)])
    
    # =========================================================================
    # LATENT VECTOR CAPTURE & STORAGE
    # =========================================================================
    
    def enable_latent_capture(self):
        """Enable capturing of latent vectors during forward pass."""
        self._capture_enabled = True
        self._current_latents = []
        self._current_timestep_map = []
        
        # Clear any existing patches first
        for layer in self._noise_layers:
            if hasattr(layer, '_original_forward'):
                layer.forward = layer._original_forward
                delattr(layer, '_original_forward')
        
        # Patch each layer's forward method to capture THE ACTUAL NOISE ADDED
        wrapper_ref = self  # Capture reference to StylizationWrapper
        
        for idx, layer in enumerate(self._noise_layers):
            # Store original forward
            layer._original_forward = layer.forward
            
            # Create wrapper that captures the noise delta (what gets added to feature_map)
            def make_forward_wrapper(original_forward, latent_list, timestep_list, wrapper_obj, layer_idx):
                def forward_wrapper(feature_map, noise):
                    # Call original forward
                    output = original_forward(feature_map, noise)
                    
                    # Capture the DIFFERENCE (the noise that was added)
                    if wrapper_obj._capture_enabled:
                        noise_delta = output - feature_map  # This is what was added
                        latent_list.append(noise_delta.detach().cpu().clone())
                        # Track timestep (will be set externally)
                        timestep_list.append(getattr(wrapper_obj, '_current_forecast_step', 0))
                    
                    return output
                return forward_wrapper
            
            # Replace forward method directly
            layer.forward = make_forward_wrapper(
                layer._original_forward, 
                self._current_latents, 
                self._current_timestep_map,
                wrapper_ref, 
                idx
            )
        
        logging.info(f"✓ Enabled latent capture on {len(self._noise_layers)} noise layers")
    
    def disable_latent_capture(self):
        """Disable latent vector capturing and restore original forward methods."""
        self._capture_enabled = False
        
        # Restore original forwards
        for layer in self._noise_layers:
            if hasattr(layer, '_original_forward'):
                layer.forward = layer._original_forward
                delattr(layer, '_original_forward')
        
        num_captured = len(self._current_latents) if self._current_latents else 0
        logging.info(f"✓ Disabled latent capture ({num_captured} deltas captured)")
    
    def store_latents(self, name: str):
        """Store the captured latent vectors with a given name."""
        if self._current_latents is None or len(self._current_latents) == 0:
            logging.info("WARNING: No latents captured!")
            logging.info(f"  _capture_enabled: {self._capture_enabled}")
            logging.info(f"  _current_latents: {self._current_latents}")
            logging.info(f"  Number of noise layers: {len(self._noise_layers)}")
            raise ValueError("No latents captured. Call enable_latent_capture() before forward pass.")
        
        # Organize latents by timestep
        timestep_to_latents = {}
        for latent, timestep in zip(self._current_latents, self._current_timestep_map):
            if timestep not in timestep_to_latents:
                timestep_to_latents[timestep] = []
            timestep_to_latents[timestep].append(latent.clone())
        
        self._stored_latents[name] = {
            'latents_by_timestep': timestep_to_latents,
            'total_latents': len(self._current_latents)
        }
        
        logging.info(f"✓ Stored {len(self._current_latents)} latent vectors as '{name}'")
        logging.info(f"  Organized into {len(timestep_to_latents)} timesteps")
        return name
    
    def get_stored_latents(self, name: str) -> Optional[Dict]:
        """Retrieve stored latent vectors by name."""
        return self._stored_latents.get(name)
    
    def list_stored_latents(self) -> List[str]:
        """List all stored latent vector names."""
        return list(self._stored_latents.keys())
    
    def clear_stored_latents(self, name: Optional[str] = None):
        """Clear stored latents (all or specific name)."""
        if name:
            if name in self._stored_latents:
                del self._stored_latents[name]
                logging.info(f"✓ Cleared latents '{name}'")
        else:
            self._stored_latents.clear()
            logging.info("✓ Cleared all stored latents")
    
    def reset_latent_capture(self):
        """Reset latent capture state - useful if something went wrong."""
        self._capture_enabled = False
        self._current_latents = []
        
        # Restore original forwards if they exist
        for layer in self._noise_layers:
            if hasattr(layer, '_original_forward'):
                layer.forward = layer._original_forward
                delattr(layer, '_original_forward')
        
        logging.info("✓ Reset latent capture state")
    
    # =========================================================================
    # LATENT INTERPOLATION
    # =========================================================================
    
    def interpolate_latents(self, name1: str, name2: str, t: float) -> Dict:
        """
        Linear interpolation between two stored latent vectors.
        Z_t = (1-t)*Z1 + t*Z2
        
        Args:
            name1: First latent vector identifier
            name2: Second latent vector identifier
            t: Interpolation parameter [0, 1]
        """
        data1 = self.get_stored_latents(name1)
        data2 = self.get_stored_latents(name2)
        
        if data1 is None:
            raise ValueError(f"Latents '{name1}' not found. Available: {self.list_stored_latents()}")
        if data2 is None:
            raise ValueError(f"Latents '{name2}' not found. Available: {self.list_stored_latents()}")
        
        latents1 = data1['latents_by_timestep']
        latents2 = data2['latents_by_timestep']
        
        # Check timesteps match
        if set(latents1.keys()) != set(latents2.keys()):
            raise ValueError(f"Timestep mismatch between {name1} and {name2}")
        
        # Interpolate for each timestep
        interpolated_by_timestep = {}
        for timestep in latents1.keys():
            z1_list = latents1[timestep]
            z2_list = latents2[timestep]
            
            if len(z1_list) != len(z2_list):
                raise ValueError(f"Latent count mismatch at timestep {timestep}")
            
            interpolated_list = []
            for z1, z2 in zip(z1_list, z2_list):
                z_t = (1 - t) * z1 + t * z2
                interpolated_list.append(z_t)
            
            interpolated_by_timestep[timestep] = interpolated_list
        
        return {
            'latents_by_timestep': interpolated_by_timestep,
            'total_latents': data1['total_latents']
        }
    
    # =========================================================================
    # MSLP CALCULATION
    # =========================================================================
    
    def calculate_mslp_and_append(self, y_arr, datetime_ref, latlons, surface_geopotential, conf, ensemble_size=1):
        """Calculate Mean Sea Level Pressure (MSLP) and append it to the input tensor."""
        
        # Create xarray datasets from the tensor
        darray_upper_air, darray_single_level = make_xarray(
            y_arr,
            datetime_ref,
            latlons.latitude.values,
            latlons.longitude.values,
            conf,
        )
        
        # Merge the datasets
        ds_merged = xr.merge([darray_upper_air.to_dataset(dim="vars"),
                              darray_single_level.to_dataset(dim="vars")])
        
        # Perform pressure interpolation to get MSLP
        pressure_interp = full_state_pressure_interpolation(
            ds_merged,
            surface_geopotential,
            **conf["predict"]["interp_pressure"]
        )
        
        # Convert pressure to tensor and add necessary dimensions
        mslp = torch.from_numpy(pressure_interp['mean_sea_level_pressure'].values).unsqueeze(1).unsqueeze(2)
        Z500 = torch.from_numpy(pressure_interp['Z_PRES'].isel(pressure=0).values).unsqueeze(1).unsqueeze(2)
        Q500 = torch.from_numpy(pressure_interp['Q_PRES'].isel(pressure=0).values).unsqueeze(1).unsqueeze(2)
        T500 = torch.from_numpy(pressure_interp['T_PRES'].isel(pressure=0).values).unsqueeze(1).unsqueeze(2)
        U500 = torch.from_numpy(pressure_interp['U_PRES'].isel(pressure=0).values).unsqueeze(1).unsqueeze(2)
        V500 = torch.from_numpy(pressure_interp['V_PRES'].isel(pressure=0).values).unsqueeze(1).unsqueeze(2)

        Z700 = torch.from_numpy(pressure_interp['Z_PRES'].isel(pressure=1).values).unsqueeze(1).unsqueeze(2)
        Q700 = torch.from_numpy(pressure_interp['Q_PRES'].isel(pressure=1).values).unsqueeze(1).unsqueeze(2)
        T700 = torch.from_numpy(pressure_interp['T_PRES'].isel(pressure=1).values).unsqueeze(1).unsqueeze(2)
        U700 = torch.from_numpy(pressure_interp['U_PRES'].isel(pressure=1).values).unsqueeze(1).unsqueeze(2)
        V700 = torch.from_numpy(pressure_interp['V_PRES'].isel(pressure=1).values).unsqueeze(1).unsqueeze(2)

        Z850 = torch.from_numpy(pressure_interp['Z_PRES'].isel(pressure=2).values).unsqueeze(1).unsqueeze(2)
        Q850 = torch.from_numpy(pressure_interp['Q_PRES'].isel(pressure=2).values).unsqueeze(1).unsqueeze(2)
        T850 = torch.from_numpy(pressure_interp['T_PRES'].isel(pressure=2).values).unsqueeze(1).unsqueeze(2)
        U850 = torch.from_numpy(pressure_interp['U_PRES'].isel(pressure=2).values).unsqueeze(1).unsqueeze(2)
        V850 = torch.from_numpy(pressure_interp['V_PRES'].isel(pressure=2).values).unsqueeze(1).unsqueeze(2)
        
        # Concatenate original tensor with MSLP
        y_arr_with_mslp = torch.cat([y_arr, mslp, Z500, Q500, T500, U500, V500, Z700, Q700, T700, U700, V700, Z850, Q850, T850, U850, V850], dim=1)
        
        return y_arr_with_mslp

    def process_pressure_interp(self, y_phys, y_pred_phys, batch, latlons, surface_geopotential, conf):
        """Process MSLP for entire batches of truth and prediction tensors."""
        # Handle y_phys (typically shape: batch, channels, time, H, W)
        batch_size = y_phys.shape[0]
        y_phys_results = []
        for b in range(batch_size):
            datetime_str = datetime.fromtimestamp(batch["datetime"][b]).strftime('%Y-%m-%d %H:%M:%S')
            y_single = y_phys[b:b+1]  # Keep batch dim: (1, channels, time, H, W)
            y_with_mslp = self.calculate_mslp_and_append(
                y_single,
                datetime_str,
                latlons,
                surface_geopotential,
                conf
            )
            y_phys_results.append(y_with_mslp)
        y_phys = torch.cat(y_phys_results, dim=0)
    
        # Handle y_pred_phys (could have ensemble dim)
        if len(y_pred_phys.shape) == 6:  # (batch, ensemble, channels, time, H, W)
            batch_size, ensemble_size = y_pred_phys.shape[:2]
            y_pred_results = []
            for b in range(batch_size):
                datetime_str = datetime.fromtimestamp(batch["datetime"][b]).strftime('%Y-%m-%d %H:%M:%S')
                ensemble_results = []
                for e in range(ensemble_size):
                    y_single = y_pred_phys[b:b+1, e:e+1].squeeze(1)  # (1, channels, time, H, W)
                    y_with_mslp = self.calculate_mslp_and_append(
                        y_single,
                        datetime_str,
                        latlons,
                        surface_geopotential,
                        conf
                    )
                    ensemble_results.append(y_with_mslp)
                batch_ensemble = torch.stack(ensemble_results, dim=1)  # (1, ensemble, channels+1, time, H, W)
                y_pred_results.append(batch_ensemble)
            y_pred_phys = torch.cat(y_pred_results, dim=0)
        else:  # (batch, channels, time, H, W)
            batch_size = y_phys.shape[0]
            y_pred_results = []
            for b in range(batch_size):
                datetime_str = datetime.fromtimestamp(batch["datetime"][b]).strftime('%Y-%m-%d %H:%M:%S')
                y_single = y_pred_phys[b:b+1]  # (1, channels, time, H, W)
                y_with_mslp = self.calculate_mslp_and_append(
                    y_single,
                    datetime_str,
                    latlons,
                    surface_geopotential,
                    conf
                )
                y_pred_results.append(y_with_mslp)
            y_pred_phys = torch.cat(y_pred_results, dim=0)
        
        return y_phys, y_pred_phys
    
    # =========================================================================
    # INTEGRATED ROLLOUT WITH LATENT CONTROL
    # =========================================================================
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.model(*args, **kwargs)
    
    def _forward_with_latent_control(self, x, forecast_step: int, 
                                     use_latents: Optional[Dict] = None):
        """
        Internal forward pass with optional latent control.
        
        Args:
            x: Input tensor
            forecast_step: Current forecast step (0-indexed)
            use_latents: Dict with 'latents_by_timestep' containing noise deltas
        """
        if use_latents is not None:
            # Get the latents for THIS specific timestep
            latents_by_timestep = use_latents['latents_by_timestep']
            
            if forecast_step not in latents_by_timestep:
                raise ValueError(f"No latents found for forecast_step {forecast_step}. Available: {list(latents_by_timestep.keys())}")
            
            timestep_latents = latents_by_timestep[forecast_step]
            
            # Use fixed noise deltas - replace forward to add the exact delta
            original_forwards = []
            device = x.device
            
            for layer, noise_delta in zip(self._noise_layers, timestep_latents):
                # Store original forward
                if hasattr(layer, '_original_forward'):
                    original_forwards.append(layer._original_forward)
                else:
                    original_forwards.append(layer.forward)
                
                noise_delta_device = noise_delta.to(device)
                
                # Create closure that adds the exact noise delta
                def make_fixed_forward(delta_fixed):
                    def fixed_forward(feature_map, noise):
                        # Ignore the stochastic noise generation, just add our fixed delta
                        return feature_map + delta_fixed
                    return fixed_forward
                
                layer.forward = make_fixed_forward(noise_delta_device)
            
            try:
                output = self.model(x, forecast_step=forecast_step)
            finally:
                # Restore original forwards
                for layer, orig_forward in zip(self._noise_layers, original_forwards):
                    layer.forward = orig_forward
        else:
            # Normal forward with random noise (or captured if enabled)
            # Track the current forecast step for capture
            self._current_forecast_step = forecast_step
            output = self.model(x, forecast_step=forecast_step)
        
        return output

    def rollout_forecast(self, data_loader, state_transformer=None, 
                        ensemble_size: int = 1, history_len: int = 1,
                        device: str = 'cuda', metrics_fn=None,
                        # NEW LATENT CONTROL PARAMETERS
                        capture_latents: bool = False,
                        store_latents_as: Optional[str] = None,
                        use_stored_latents: Optional[str] = None,
                        use_interpolated_latents: Optional[Tuple[str, str, float]] = None,
                        conf: Optional[Dict] = None) -> Dict:
        """
        Unified hurricane forecast rollout with integrated latent vector control.
        
        Args:
            data_loader: DataLoader containing hurricane data batches
            state_transformer: Transformer for normalization/denormalization
            ensemble_size: Number of ensemble members
            history_len: Length of input history
            device: Device to run inference on
            metrics_fn: Function to compute metrics per step
            conf: Configuration dictionary (required for MSLP calculation)
            
            # Latent control parameters:
            capture_latents: If True, capture latent vectors during this rollout
            store_latents_as: Name to store captured latents (implies capture_latents=True)
            use_stored_latents: Name of stored latents to use for exact reproduction
            use_interpolated_latents: Tuple of (name1, name2, t) for interpolation
            
        Returns:
            Dict with 'predictions', 'truth', 'metrics', and optional 'latent_name'
        """
        # Handle latent control setup
        if store_latents_as:
            capture_latents = True
        
        if capture_latents:
            self.enable_latent_capture()
        
        # Get latents if needed
        latents_to_use = None
        if use_stored_latents:
            latents_to_use = self.get_stored_latents(use_stored_latents)
            if latents_to_use is None:
                raise ValueError(f"Latents '{use_stored_latents}' not found. Available: {self.list_stored_latents()}")
            logging.info(f"→ Using stored latents '{use_stored_latents}' for reproduction")
        elif use_interpolated_latents:
            name1, name2, t = use_interpolated_latents
            latents_to_use = self.interpolate_latents(name1, name2, t)
            logging.info(f"→ Using interpolated latents: t={t:.3f} between '{name1}' and '{name2}'")
        
        # Setup for MSLP calculation
        if conf is not None:
            latlons = xr.open_dataset(conf["loss"]["latitude_weights"]).load()
            with xr.open_dataset('/glade/campaign/cisl/aiml/credit/static_scalers/static_whole_20250416_1deg.nc') as df:
                surface_geopotential = df["Z_GDS4_SFC"].values
        else:
            latlons = None
            surface_geopotential = None
        
        predictions = []
        truth = []
        step_metrics = []

        with torch.no_grad():
            for batch in data_loader:
                forecast_step = batch["forecast_step"].item()
                
                # Initial input processing
                if forecast_step == 1:
                    from credit.data import concat_and_reshape, reshape_only
                    if "x_surf" in batch:
                        x = concat_and_reshape(batch["x"], batch["x_surf"]).to(device).float()
                    else:
                        x = reshape_only(batch["x"]).to(device).float()
                    
                    if ensemble_size > 1:
                        x = torch.repeat_interleave(x, ensemble_size, 0)
                
                # Add forcing and static variables
                if "x_forcing_static" in batch:
                    x_forcing_batch = batch["x_forcing_static"].to(device).permute(0, 2, 1, 3, 4).float()
                    if ensemble_size > 1:
                        x_forcing_batch = torch.repeat_interleave(x_forcing_batch, ensemble_size, 0)
                    x = torch.cat((x, x_forcing_batch), dim=1)
                
                # Load ground truth
                if "y_surf" in batch:
                    y = concat_and_reshape(batch["y"], batch["y_surf"]).to(device).float()
                else:
                    y = reshape_only(batch["y"]).to(device).float()
                
                if "y_diag" in batch:
                    y_diag_batch = batch["y_diag"].to(device).permute(0, 2, 1, 3, 4)
                    y = torch.cat((y, y_diag_batch), dim=1).to(device).float()
                
                # Model prediction with latent control
                y_pred = self._forward_with_latent_control(
                    x, 
                    forecast_step=forecast_step - 1,
                    use_latents=latents_to_use
                )

                # truth.append(y)
                # predictions.append(y_pred)
                
                # Transform to physical space
                if state_transformer is not None:
                    y_pred_phys = state_transformer.inverse_transform(y_pred.cpu())
                    y_phys = state_transformer.inverse_transform(y.cpu())
                else:
                    y_pred_phys = y_pred.cpu()
                    y_phys = y.cpu()

                # Compute metrics if function provided
                if metrics_fn is not None:
                    metrics = metrics_fn(y_pred_phys, y_phys)
                    step_metrics.append(metrics)
                
                # Prepare for next step
                y_pred_norm = state_transformer.transform_array(y_pred_phys).to(device) if state_transformer else y_pred
                
                if history_len == 1:
                    if "y_diag" in batch:
                        varnum_diag = batch["y_diag"].shape[1]
                        x = y_pred_norm[:, :-varnum_diag, ...].detach()
                    else:
                        x = y_pred_norm.detach()
                else:
                    x_detach = x[:, :, 1:, ...].detach()
                    if "y_diag" in batch:
                        varnum_diag = batch["y_diag"].shape[1]
                        x = torch.cat([x_detach, y_pred_norm[:, :-varnum_diag, ...].detach()], dim=2)
                    else:
                        x = torch.cat([x_detach, y_pred_norm.detach()], dim=2)

                # Add MSLP if config provided
                if conf is not None and latlons is not None:
                    if ensemble_size > 1:
                        batch_size = y_pred_phys.shape[0] // ensemble_size
                        y_pred_phys = y_pred_phys.view(batch_size, ensemble_size, *y_pred_phys.shape[1:])

                    y_phys, y_pred_phys = self.process_pressure_interp(
                        y_phys, 
                        y_pred_phys,
                        batch,
                        latlons, 
                        surface_geopotential, 
                        conf
                    )

                # Append results
                truth.append(y_phys[:, 64:])
                predictions.append(y_pred_phys[:, 64:])

                if batch.get("stop_forecast", torch.tensor(False)).item():
                    break

        # Store latents if requested
        result = {
            'predictions': torch.cat(predictions, dim=-3),
            # 'truth': torch.cat(truth, dim=-3), 
            # 'metrics': step_metrics
        }
        
        if capture_latents:
            if store_latents_as:
                self.store_latents(store_latents_as)
                result['latent_name'] = store_latents_as
            self.disable_latent_capture()
        
        return result
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    def generate_interpolation_sequence(self, dataset, name1: str, name2: str,
                                   num_steps: int = 5, state_transformer=None,
                                   device: str = 'cuda', conf: Optional[Dict] = None,
                                   **rollout_kwargs) -> List[Dict]:
        """
        Generate sequence of forecasts along interpolation path.
        
        Args:
            dataset: Dataset (will be deep copied for each interpolation)
            name1: Start latent vector
            name2: End latent vector
            num_steps: Number of interpolation points (default: 5)
            state_transformer: Transformer for normalization
            device: Device to run on
            conf: Configuration dictionary
            **rollout_kwargs: Additional arguments for rollout_forecast
        
        Returns:
            List of forecast dicts, one per interpolation point
        """
        results = []
        t_values = torch.linspace(0, 1, num_steps)
        
        logging.info(f"\n{'='*70}")
        logging.info(f"LATENT INTERPOLATION SEQUENCE: '{name1}' → '{name2}'")
        logging.info(f"{'='*70}")
        
        for i, t in enumerate(t_values):
            logging.info(f"\n[{i+1}/{num_steps}] Generating forecast at t={t:.3f}...")
        
            # Create fresh dataloader for each interpolation point
            data_loader = BatchForecastLenDataLoader(copy.deepcopy(dataset))
            
            forecast = self.rollout_forecast(
                data_loader, 
                state_transformer=state_transformer,
                device=device,
                conf=conf,
                use_interpolated_latents=(name1, name2, t.item()),
                **rollout_kwargs
            )
            forecast['t'] = t.item()
            forecast['name1'] = name1
            forecast['name2'] = name2
            results.append(forecast)
        
        logging.info(f"\n{'='*70}")
        logging.info(f"✓ Completed {num_steps} interpolated forecasts")
        logging.info(f"{'='*70}\n")
        
        return results    


    def scale_latents(self, name: str, beta: float) -> Dict:
        """
        Scale stored latent vectors by factor beta.
        Z_scaled = beta * Z
        
        Args:
            name: Latent vector identifier
            beta: Scaling factor
        
        Returns:
            Scaled latent dict
        """
        data = self.get_stored_latents(name)
        if data is None:
            raise ValueError(f"Latents '{name}' not found. Available: {self.list_stored_latents()}")
        
        latents_by_timestep = data['latents_by_timestep']
        
        # Scale for each timestep
        scaled_by_timestep = {}
        for timestep, latent_list in latents_by_timestep.items():
            scaled_list = [beta * z for z in latent_list]
            scaled_by_timestep[timestep] = scaled_list
        
        return {
            'latents_by_timestep': scaled_by_timestep,
            'total_latents': data['total_latents']
        }
    
    def generate_scaled_ensemble(self, dataset, base_latent_name: str,
                                beta_values: List[float],
                                state_transformer=None,
                                device: str = 'cuda',
                                conf: Optional[Dict] = None,
                                **rollout_kwargs) -> List[Dict]:
        """
        Generate ensemble with different scaling factors applied to base latent.
        
        Args:
            dataset: Dataset for forecasting
            base_latent_name: Name of base latent vector to scale
            beta_values: List of scaling factors to apply
            state_transformer: Transformer for normalization
            device: Device to run on
            conf: Configuration dictionary
            **rollout_kwargs: Additional arguments for rollout_forecast
        
        Returns:
            List of forecast dicts, one per beta value
        """
        results = []
        
        logging.info(f"\n{'='*70}")
        logging.info(f"ENSEMBLE SPREAD CONTROL: Scaling '{base_latent_name}'")
        logging.info(f"Beta values: {beta_values}")
        logging.info(f"{'='*70}")
        
        for i, beta in enumerate(beta_values):
            logging.info(f"\n[{i+1}/{len(beta_values)}] Generating forecast with β={beta:.2f}...")
            
            # Scale the latents
            scaled_latents = self.scale_latents(base_latent_name, beta)
            
            # Create fresh dataloader
            data_loader = BatchForecastLenDataLoader(copy.deepcopy(dataset))
            
            # Generate forecast with scaled latents
            # We need to temporarily store these scaled latents
            temp_name = f"_temp_scaled_{beta}"
            self._stored_latents[temp_name] = scaled_latents
            
            try:
                forecast = self.rollout_forecast(
                    data_loader,
                    state_transformer=state_transformer,
                    device=device,
                    conf=conf,
                    use_stored_latents=temp_name,
                    **rollout_kwargs
                )
                forecast['beta'] = beta
                results.append(forecast)
            finally:
                # Clean up temp storage
                if temp_name in self._stored_latents:
                    del self._stored_latents[temp_name]
        
        logging.info(f"\n{'='*70}")
        logging.info(f"✓ Completed {len(beta_values)} scaled forecasts")
        logging.info(f"{'='*70}\n")
        
        return results


    def scale_latents_multilevel(self, name: str, beta_per_layer: List[float]) -> Dict:
        """
        Scale stored latent vectors with different beta for each layer.
        
        Args:
            name: Latent vector identifier
            beta_per_layer: List of scaling factors, one per noise injection layer
                           Must match the number of layers in self._noise_layers
        
        Returns:
            Scaled latent dict with 'latents_by_timestep' structure
        """
        data = self.get_stored_latents(name)
        if data is None:
            raise ValueError(f"Latents '{name}' not found. Available: {self.list_stored_latents()}")
        
        latents_by_timestep = data['latents_by_timestep']
        
        # CRITICAL: Verify beta count matches layer count
        num_layers = len(self._noise_layers)
        if len(beta_per_layer) != num_layers:
            raise ValueError(
                f"Beta count mismatch! Got {len(beta_per_layer)} betas but model has {num_layers} noise layers.\n"
                f"Expected beta_per_layer length: {num_layers}"
            )
        
        # Scale for each timestep
        scaled_by_timestep = {}
        for timestep, latent_list in latents_by_timestep.items():
            if len(latent_list) != len(beta_per_layer):
                raise ValueError(
                    f"Timestep {timestep}: Expected {len(beta_per_layer)} latents, got {len(latent_list)}"
                )
            
            # Apply scaling: each beta to corresponding layer
            scaled_list = [beta * z for beta, z in zip(beta_per_layer, latent_list)]
            scaled_by_timestep[timestep] = scaled_list
        
        logging.info(f"✓ Scaled {len(beta_per_layer)} layers across {len(latents_by_timestep)} timesteps")
        
        return {
            'latents_by_timestep': scaled_by_timestep,
            'total_latents': data['total_latents']
        }


def run_multiscale_control_experiment(experiments, 
    wrapper, dataset, state_transformer, conf, lat_centers, lon_centers,
    device='cuda', ensemble_size=1
) -> Dict[str, Dict]:
    
    logging.info("\n" + "="*70)
    logging.info("MULTI-SCALE LATENT CONTROL EXPERIMENT")
    logging.info("="*70)
    
    # DIAGNOSTIC: Check configuration
    num_layers = len(wrapper._noise_layers)
    logging.info("\nModel configuration:")
    logging.info(f"  Total noise layers: {num_layers}")
    logging.info(f"  Ensemble size: {ensemble_size}")
    
    # Step 1: Generate baseline ONCE and capture latents
    logging.info("\n[1/2] Generating baseline forecast and capturing latents...")
    baseline = wrapper.rollout_forecast(
        copy.deepcopy(dataset),
        state_transformer=state_transformer,
        ensemble_size=1,  # Single baseline
        device=device,
        conf=conf,
        store_latents_as="baseline"
    )
    
    logging.info(f"✓ Baseline complete. Shape: {baseline['predictions'].shape}")
    
    # VERIFY latent structure
    baseline_latents = wrapper.get_stored_latents("baseline")
    first_timestep = min(baseline_latents['latents_by_timestep'].keys())
    latents_per_timestep = len(baseline_latents['latents_by_timestep'][first_timestep])
    logging.info(f"  Latents captured per timestep: {latents_per_timestep}")
    
    if latents_per_timestep != num_layers:
        raise ValueError(
            f"MISMATCH: Model has {num_layers} layers but captured {latents_per_timestep} latents!"
        )
    
    # Validate all experiments have correct beta count
    for exp_name, betas in experiments.items():
        if len(betas) != num_layers:
            raise ValueError(f"Experiment '{exp_name}' has {len(betas)} betas, need {num_layers}")
    
    logging.info(f"\n[2/2] Running {len(experiments)} experiments...")
    results = {}
    
    for i, (exp_name, betas) in enumerate(experiments.items(), 1):
        logging.info(f"  [{i}/{len(experiments)}] {exp_name}: β={betas}")
        
        if exp_name == "baseline":
            results[exp_name] = baseline
            results[exp_name]['betas'] = betas
            continue
        
        # Scale the baseline latents ONCE
        scaled_latents = wrapper.scale_latents_multilevel("baseline", betas)
        scaled_name = f"scaled_{exp_name}"
        wrapper._stored_latents[scaled_name] = scaled_latents
        
        # NOW loop over ensemble members using the same scaled latents
        logging.info(f"    Generating {ensemble_size} ensemble members...")
        exp_predictions = []
        
        for ens_idx in range(ensemble_size):
            forecast = wrapper.rollout_forecast(
                copy.deepcopy(dataset),
                state_transformer=state_transformer,
                ensemble_size=1,
                device=device,
                conf=conf,
                use_stored_latents=scaled_name
            )
            
            # Keep only first 64 channels
            exp_predictions.append(forecast['predictions'])
        
        # Stack ensemble predictions
        results[exp_name] = {
            'exp_name': exp_name
        }

        # Stack ensemble predictions
        stacked_predictions = np.stack(exp_predictions, axis=0)
        
        # Save to pickle file with beta in filename
        import os
        import pickle
        os.makedirs('beta_data', exist_ok=True)
        
        beta_str = '_'.join([f"{b:.1f}" for b in betas])
        filename = f"beta_data/{exp_name}_beta_{beta_str}.pkl"
        
        save_data = {
            'predictions': stacked_predictions,
            'betas': betas,
            'exp_name': exp_name
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        
        logging.info(f"    ✓ Saved to {filename}")
        
        # Clean up
        del wrapper._stored_latents[scaled_name]
    
    logging.info("\n" + "="*70)
    logging.info("✓ All experiments complete!")
    logging.info("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    # Use forkserver to avoid CUDA fork issues and minimize startup overhead
    try:
        mp.set_start_method('forkserver', force=True)
    except RuntimeError:
        pass  # Already set
    
    parser = argparse.ArgumentParser(description='Generate parallel ensemble forecasts with stylization')
    
    # Path arguments
    parser.add_argument('--filepath', type=str, 
                        default="/glade/derecho/scratch/schreck/CREDIT_runs/ensemble/scheduler/",
                        help='Path to model configuration and checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for pickle file (default: ensemble_by_beta_{n_members}.pkl)')
    
    # Forecast configuration
    parser.add_argument('--start-time', type=str, default='2022-01-11 00:00:00',
                        help='Forecast start time')
    parser.add_argument('--end-time', type=str, default='2022-01-17 18:00:00',
                        help='Forecast end time')
    parser.add_argument('--lead-time-periods', type=int, default=6,
                        help='Number of lead time periods')
    
    # Ensemble configuration
    parser.add_argument('--n-members', type=int, default=20,
                        help='Number of ensemble members per beta value')
    
    args = parser.parse_args()
    
    # Phase 3 experiments
    experiments = {
        # ===================================================================
        # EXTREME SUPPRESSION REGIME: β = -3 (Single-parameter variation)
        # ===================================================================
        "beta1_extreme_negative": [-3, 1, 1],                   # Extreme synoptic-scale suppression
        "beta2_extreme_negative": [1, -3, 1],                   # Extreme mesoscale suppression
        "beta3_extreme_negative": [1, 1, -3],                   # Extreme convective-scale suppression
        
        # ===================================================================
        # HIGH SUPPRESSION REGIME: β = -2 (Single-parameter variation)
        # ===================================================================
        "beta1_high_negative": [-2, 1, 1],                      # High synoptic-scale suppression
        "beta2_high_negative": [1, -2, 1],                      # High mesoscale suppression
        "beta3_high_negative": [1, 1, -2],                      # High convective-scale suppression
        
        # ===================================================================
        # MODERATE SUPPRESSION REGIME: β = -1 (Single-parameter variation)
        # ===================================================================
        "beta1_moderate_negative": [-1, 1, 1],                  # Moderate synoptic-scale suppression
        "beta2_moderate_negative": [1, -1, 1],                  # Moderate mesoscale suppression
        "beta3_moderate_negative": [1, 1, -1],                  # Moderate convective-scale suppression
        
        # ===================================================================
        # NULL UNCERTAINTY REGIME: β = 0 (Single-parameter variation)
        # ===================================================================
        "beta1_elimination": [0, 1, 1],                         # Synoptic-scale uncertainty elimination
        "beta2_elimination": [1, 0, 1],                         # Mesoscale uncertainty elimination
        "beta3_elimination": [1, 1, 0],                         # Convective-scale uncertainty elimination
        
        # ===================================================================
        # BASELINE REFERENCE REGIME: β = 1 (Control conditions)
        # ===================================================================
        "baseline_reference_1": [1, 1, 1],                      # Standard baseline configuration
        "baseline_reference_2": [1, 1, 1],                      # Baseline replication (statistical validation)
        "baseline_reference_3": [1, 1, 1],                      # Baseline replication (experimental control)
        
        # ===================================================================
        # MODERATE AMPLIFICATION REGIME: β = 2 (Single-parameter variation)
        # ===================================================================
        "beta1_moderate_positive": [2, 1, 1],                   # Moderate synoptic-scale amplification
        "beta2_moderate_positive": [1, 2, 1],                   # Moderate mesoscale amplification
        "beta3_moderate_positive": [1, 1, 2],                   # Moderate convective-scale amplification
        
        # ===================================================================
        # EXTREME AMPLIFICATION REGIME: β = 3 (Single-parameter variation)
        # ===================================================================
        "beta1_extreme_positive": [3, 1, 1],                    # Extreme synoptic-scale amplification
        "beta2_extreme_positive": [1, 3, 1],                    # Extreme mesoscale amplification
        "beta3_extreme_positive": [1, 1, 3],                    # Extreme convective-scale amplification
    }
    
    filepath = args.filepath
    device = "cuda"

    with open(os.path.join(filepath, "model.yml"), "r") as f:
        conf = yaml.safe_load(f)

    seed_everything(conf["seed"])

    conf = credit_main_parser(conf, parse_training=False, parse_predict=True, print_summary=False)
    data_config = setup_data_loading(conf)

    ensemble_size = 1
    conf["trainer"]["ensemble_size"] = ensemble_size
    conf["predict"]["ensemble_size"] = ensemble_size

    transformer = Normalize_ERA5_and_Forcing(conf)

    model = load_model(conf).to(device)
    model.load_state_dict(torch.load(os.path.join(filepath, "checkpoint.pt"))['model_state_dict'])
    model = model.eval()

    wrapper = StylizationWrapper(model)

    forecast_times = [[args.start_time, args.end_time]]

    # Create dataset
    dataset = Predict_Dataset_Batcher(
        varname_upper_air=data_config["varname_upper_air"],
        varname_surface=data_config["varname_surface"],
        varname_dyn_forcing=data_config["varname_dyn_forcing"],
        varname_forcing=data_config["varname_forcing"],
        varname_static=data_config["varname_static"],
        varname_diagnostic=data_config["varname_diagnostic"],
        filenames=data_config["all_ERA_files"],
        filename_surface=data_config["surface_files"],
        filename_dyn_forcing=data_config["dyn_forcing_files"],
        filename_forcing=data_config["forcing_files"],
        filename_static=data_config["static_files"],
        filename_diagnostic=data_config["diagnostic_files"],
        fcst_datetime=forecast_times,
        lead_time_periods=args.lead_time_periods,
        history_len=data_config["history_len"],
        skip_periods=data_config["skip_periods"],
        transform=load_transforms(conf),
        sst_forcing=data_config["sst_forcing"],
        batch_size=1,
        rank=0,
        world_size=1,
    )

    latlons = xr.open_dataset(conf["loss"]["latitude_weights"]).load()
    lat_centers = latlons.latitude.values
    lon_centers = latlons.longitude.values

    idx = int(sys.argv[1])
    exp_list = list(experiments.items())
    selected = exp_list[idx]
    selected_experiments = {selected[0]: selected[1]}

    experiments = {"baseline": [1.0, 1.0, 1.0]}
    experiments.update(selected_experiments)

    logging.info(f"Selected experiment {idx}: {list(selected_experiments.keys())[0]} = {list(selected_experiments.values())[0]}")

    # Run experiments
    results = run_multiscale_control_experiment(
        experiments,
        wrapper=wrapper,
        dataset=dataset,
        state_transformer=transformer,
        conf=conf,
        lat_centers=lat_centers,
        lon_centers=lon_centers,
        device='cuda',
        ensemble_size=args.n_members
    )