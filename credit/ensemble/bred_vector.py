import os
import copy
import torch
from datetime import datetime, timedelta
from credit.data import concat_and_reshape, reshape_only
from credit.datasets.load_dataset_and_dataloader import BatchForecastLenDataLoader
from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer
from credit.ensemble.utils import hemispheric_rescale as hemi_rescale
from credit.postblock import PostBlock
from typing import Callable, Optional
from collections import OrderedDict
import numpy as np
import xarray as xr


class BredVector:
    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        noise_amplitude: float = 0.15,
        num_cycles: int = 5,
        integration_steps: int = 1,
        perturbation_method: Optional[Callable[[torch.Tensor, OrderedDict[str, np.ndarray]], torch.Tensor]] = None,
        hemispheric_rescale: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        terrain_file: str = None,
        perturb_channel_idx: int = None,
        ensemble_perturb: bool = False,
        clamp: bool = False,
        clamp_min: float = None,
        clamp_max: float = None,
        input_static_dim: int = 3,
        varnum_diag: int = 0,
        post_conf: dict = {},
    ):
        self.model = model
        self.noise_amplitude = noise_amplitude
        self.num_cycles = num_cycles
        self.integration_steps = integration_steps
        self.perturbation_method = perturbation_method  # e.g., Brown()
        self.hemispheric_rescale = hemi_rescale if hemispheric_rescale else False
        self.ensemble_perturb = ensemble_perturb
        self.perturb_channel_idx = perturb_channel_idx
        self.clamp = clamp
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.input_static_dim = input_static_dim
        self.varnum_diag = varnum_diag
        self.post_conf = post_conf

        self.flag_mass_conserve = False
        self.flag_water_conserve = False
        self.flag_energy_conserve = False

        self.use_post_block = False
        if post_conf.get("activate", False):
            if post_conf["global_mass_fixer"]["activate"]:
                if post_conf["global_mass_fixer"]["activate_outside_model"]:
                    logger.info("Activate GlobalMassFixer outside of model")
                    self.flag_mass_conserve = True
                    self.opt_mass = GlobalMassFixer(post_conf)

            if post_conf["global_water_fixer"]["activate"]:
                if post_conf["global_water_fixer"]["activate_outside_model"]:
                    logger.info("Activate GlobalWaterFixer outside of model")
                    self.flag_water_conserve = True
                    self.opt_water = GlobalWaterFixer(post_conf)

            if post_conf["global_energy_fixer"]["activate"]:
                if post_conf["global_energy_fixer"]["activate_outside_model"]:
                    logger.info("Activate GlobalEnergyFixer outside of model")
                    self.flag_energy_conserve = True
                    self.opt_energy = GlobalEnergyFixer(post_conf)
            self.use_post_block = True

        if self.use_post_block:
            # freeze base model weights before postblock init
            if "skebs" in post_conf.keys():
                if post_conf["skebs"].get("activate", False) and post_conf["skebs"].get(
                    "freeze_base_model_weights", False
                ):
                    # logger.warning("freezing all base model weights due to skebs config")
                    for param in self.parameters():
                        param.requires_grad = False

            # logger.info("using postblock")
            self.postblock = PostBlock(self.post_conf)

        if self.hemispheric_rescale is not None:
            if not os.path.exists(terrain_file) or terrain_file is None:
                raise FileNotFoundError(f"Terrain file {terrain_file} not found")
            latlons = xr.open_dataset(terrain_file).load()
            self.latitudes = torch.tensor(latlons.latitude.values)

    def perturb(self, x_input: torch.Tensor, forecast_step: int = 1) -> torch.Tensor:
        x = x_input.clone()
        device = x.device

        if hasattr(self.noise_amplitude, "__len__"):
            # Vector case: create full tensor with 1s for static channels
            static_ones = torch.ones(self.input_static_dim, device=device, dtype=x.dtype)
            dynamic_amp = torch.tensor(self.noise_amplitude, device=device, dtype=x.dtype)
            noise_amp_full = torch.cat([static_ones, dynamic_amp]).view(1, -1, 1, 1)
            noise_amp_dynamic = dynamic_amp.view(1, -1, 1, 1, 1)  # Same length as dynamic channels
        else:
            # Scalar case
            noise_amp_full = self.noise_amplitude
            noise_amp_dynamic = self.noise_amplitude

        # Add structured noise (Z500-only or full depending on step)
        if self.perturbation_method is not None:
            delta_x0 = self.perturbation_method(x)
        else:
            delta_x0 = noise_amp_full * torch.randn_like(x)

        if forecast_step == 1 and self.perturb_channel_idx is not None:
            # Only perturb Z500 (channel index 66)
            z500_idx = self.perturb_channel_idx
            mask = torch.zeros_like(delta_x0)
            mask[:, z500_idx, ...] = 1.0
            delta_x0 *= mask

        x_perturbed = x + delta_x0

        if self.clamp:
            x = torch.clamp(x, self.clamp_min, self.clamp_max)
            x_perturbed = torch.clamp(x_perturbed, self.clamp_min, self.clamp_max)

        # Initialize based on NVIDIA pattern, using your existing variables
        x_current = x  # This is 'x' in NVIDIA code (your unperturbed state with static channels)
        # Extract the dynamic perturbation from the initial delta_x0
        # Only dynamic channels get perturbed, static channels are preserved
        dx = delta_x0[:, : -self.input_static_dim, ...]  # Initial perturbation

        for k in range(self.integration_steps):
            # x1 = x + dx (apply current perturbation to dynamic channels only)
            # Dynamic channels are first, static channels are last in input
            x_perturbed_dyn = x_current[:, : -self.input_static_dim, ...] + dx
            if self.clamp:
                x_perturbed_dyn = torch.clamp(x_perturbed_dyn, self.clamp_min, self.clamp_max)
            # Reconstruct full state (dynamic first, then static channels unchanged)
            x1 = torch.cat([x_perturbed_dyn, x_current[:, -self.input_static_dim :, ...]], dim=1)

            # x2, _ = self.model(x1, coords)
            # Model outputs ONLY dynamic channels (no static channels)
            x2 = self.model(x1.to(x.dtype))

            # Calculate what NVIDIA calls 'xd' (the unperturbed prediction)
            # Model outputs ONLY dynamic channels (no static channels)
            xd = self.model(x_current.to(x.dtype))

            if self.ensemble_perturb:  # This matches NVIDIA's ensemble_perturb flag
                # dx1 = x2 - xd
                # Both x2 and xd are model outputs with only dynamic channels
                dx1 = x2 - xd

                # dx = dx1 + self.noise_amplitude * (dx - dx.mean(dim=0))
                dx = dx1 + noise_amp_dynamic * (dx - dx.mean(dim=0))
            else:
                # dx = x2 - xd
                # Both x2 and xd are model outputs with only dynamic channels
                dx = x2 - xd

            # Apply your specific modifications (channel-specific gamma scaling during loop)
            if forecast_step == 1 and self.perturb_channel_idx is not None:
                z500_idx = self.perturb_channel_idx
                # x_dyn uses the input channels (includes static), but z500_idx is absolute
                x_dyn = x_current[:, z500_idx : z500_idx + 1, ...]
                # dx_z500 uses model output indexing (only dynamic channels)
                dx_z500 = dx[:, z500_idx : z500_idx + 1, ...]
                x_perturbed_dyn_subset = x_dyn + dx_z500
                gamma = torch.norm(x_dyn) / (torch.norm(x_perturbed_dyn_subset) + 1e-8)
                # Apply gamma to the specific channel perturbation in model output space
                dx[:, z500_idx : z500_idx + 1, ...] *= gamma
            else:
                # For full perturbation, compare input dynamic vs model output + perturbation
                x_dyn = x_current[:, : -self.input_static_dim, ...]
                x_perturbed_dyn_subset = x_dyn + dx
                gamma = torch.norm(x_dyn) / (torch.norm(x_perturbed_dyn_subset) + 1e-8)
                dx = dx * gamma

            if self.hemispheric_rescale is not None:
                latitudes = self.latitudes.to(device)
                dx = self.hemispheric_rescale(dx, latitudes)

            # Update x for next iteration - but x_current needs to keep static channels
            # So we reconstruct it from the model prediction + static channels
            x_current = torch.cat([xd, x_current[:, -self.input_static_dim :, ...]], dim=1)

        # Apply NVIDIA's final gamma scaling and return the final perturbation
        if forecast_step == 1 and self.perturb_channel_idx is not None:
            z500_idx = self.perturb_channel_idx
            # Use input space for x_final (includes static channels)
            x_final = x_current[:, z500_idx : z500_idx + 1, ...]
            # Use model output space for dx_z500 (only dynamic channels)
            dx_z500 = dx[:, z500_idx : z500_idx + 1, ...]
            x_plus_dx = x_final + dx_z500
            gamma_final = torch.norm(x_final) / (torch.norm(x_plus_dx) + 1e-8)
            # Create full perturbation tensor in model output space (only dynamic channels)
            delta_x_final = torch.zeros_like(dx)

            if isinstance(noise_amp_dynamic, torch.Tensor) and noise_amp_dynamic.ndim > 0:
                scale = noise_amp_dynamic[0, z500_idx, 0, 0, 0]
            else:
                scale = noise_amp_dynamic  # assumed to be a scalar (float or 0-dim tensor)

            delta_x_final[:, z500_idx : z500_idx + 1, ...] = dx_z500 * scale * gamma_final

            return delta_x_final.detach()
        else:
            # Use input space for x_final (dynamic channels only)
            x_final = x_current[:, : -self.input_static_dim, ...]
            x_plus_dx = x_final + dx
            gamma_final = torch.norm(x_final) / (torch.norm(x_plus_dx) + 1e-8)
            return (dx * noise_amp_dynamic * gamma_final).detach()

    def __call__(self, initial_condition: torch.Tensor, dataset, return_delta_x=False) -> list[torch.Tensor]:
        device = initial_condition.device

        bred_vectors, delta_x_vectors = [], []
        with torch.no_grad():
            for _ in range(self.num_cycles):
                # Make a copy of the dataloader
                dataset_c = clone_dataset(dataset)
                data_loader = BatchForecastLenDataLoader(dataset_c)

                # model inference loop
                for _, batch in enumerate(data_loader):
                    forecast_step = batch["forecast_step"].item()

                    # Initial input processing
                    if forecast_step == 1:
                        if "x_surf" in batch:
                            x = concat_and_reshape(batch["x"], batch["x_surf"]).to(device).float()
                        else:
                            x = reshape_only(batch["x"]).to(device).float()

                        # Add forcing and static variables
                        if "x_forcing_static" in batch:
                            x_forcing_batch = batch["x_forcing_static"].to(device).permute(0, 2, 1, 3, 4).float()
                            x = torch.cat((x, x_forcing_batch), dim=1)

                        # Clamp if needed
                        if self.clamp:
                            x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)

                    else:
                        # Add current forcing and static variables
                        if "x_forcing_static" in batch:
                            x_forcing_batch = batch["x_forcing_static"].to(device).permute(0, 2, 1, 3, 4).float()
                            x = torch.cat((x, x_forcing_batch), dim=1)

                        # Clamp if needed
                        if self.clamp:
                            x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)

                    # Load y-truth
                    if "y_surf" in batch:
                        y = concat_and_reshape(batch["y"], batch["y_surf"]).to(device).float()
                    else:
                        y = reshape_only(batch["y"]).to(device).float()

                    if "y_diag" in batch:
                        y_diag_batch = batch["y_diag"].to(device).permute(0, 2, 1, 3, 4)
                        y = torch.cat((y, y_diag_batch), dim=1).to(device).float()

                    # Compute bred vector
                    delta_x = self.perturb(x, forecast_step=forecast_step)

                    # Stop here if we are done
                    if batch["stop_forecast"].item():
                        break

                    # Add the perturbation to the model input
                    x[:, : delta_x.shape[1], :, :, :] += delta_x

                    # Pass through the model to advance one time step
                    y_pred = self.model(x)

                    # backup init state
                    if self.flag_mass_conserve:
                        if forecast_step == 1:
                            x_init = x.clone()

                    # mass conserve using initialization as reference
                    if self.flag_mass_conserve:
                        input_dict = {"y_pred": y_pred, "x": x_init}
                        input_dict = self.opt_mass(input_dict)
                        y_pred = input_dict["y_pred"]

                    # water conserve use previous step output as reference
                    if self.flag_water_conserve:
                        input_dict = {"y_pred": y_pred, "x": x}
                        input_dict = self.opt_water(input_dict)
                        y_pred = input_dict["y_pred"]

                    # energy conserve use previous step output as reference
                    if self.flag_energy_conserve:
                        input_dict = {"y_pred": y_pred, "x": x}
                        input_dict = self.opt_energy(input_dict)
                        y_pred = input_dict["y_pred"]

                    if dataset.history_len == 1:
                        if "y_diag" in batch:
                            x = y_pred[:, : -self.varnum_diag, ...].detach()
                        else:
                            x = y_pred.detach()
                    else:
                        if self.static_dim_size == 0:
                            x_detach = x[:, :, 1:, ...].detach()
                        else:
                            x_detach = x[:, : -self.static_dim_size, 1:, ...].detach()

                        if "y_diag" in batch:
                            x = torch.cat([x_detach, y_pred[:, : -self.varnum_diag, ...].detach()], dim=2)
                        else:
                            x = torch.cat([x_detach, y_pred.detach()], dim=2)

                for sign in [+1, -1]:
                    x0 = initial_condition.clone()
                    x0[:, self.input_static_dim :, ...] += sign * delta_x.detach()

                    if self.use_post_block:
                        x0 = self.postblock(
                            {
                                "y_pred": x0,
                                "x": initial_condition,
                            }
                        )

                    # Apply physical conservation laws
                    if self.flag_mass_conserve:
                        input_dict = {"y_pred": x0, "x": initial_condition}
                        x0 = self.opt_mass(input_dict)["y_pred"]

                    if self.flag_water_conserve:
                        input_dict = {"y_pred": x0, "x": initial_condition}
                        x0 = self.opt_water(input_dict)["y_pred"]

                    if self.flag_energy_conserve:
                        input_dict = {"y_pred": x0, "x": initial_condition}
                        x0 = self.opt_energy(input_dict)["y_pred"]

                    bred_vectors.append(x0)

                    delta_x_vectors.append(sign * delta_x.detach())

        if return_delta_x:
            return bred_vectors, delta_x_vectors

        return bred_vectors


def generate_bred_vectors(
    x_batch,
    model,
    num_cycles=5,
    perturbation_std=0.15,
    epsilon=1.0,
    flag_clamp=False,
    clamp_min=None,
    clamp_max=None,
):
    """
    Generate bred vectors and initialize initial conditions for the given batch.

    Args:
        x_batch (torch.Tensor): The input batch.
        batch (dict): A dictionary containing additional batch data.
        model (nn.Module): The model used for predictions.
        num_cycles (int): Number of perturbation cycles.
        perturbation_std (float): Magnitude of initial perturbations.
        epsilon (float): Scaling factor for bred vectors.
        flag_clamp (bool, optional): Whether to clamp inputs. Defaults to False.
        clamp_min (float, optional): Minimum clamp value. Required if flag_clamp is True.
        clamp_max (float, optional): Maximum clamp value. Required if flag_clamp is True.

    Returns:
        list[torch.Tensor]: List of initial conditions generated using bred vectors.
    """
    bred_vectors = []
    for _ in range(num_cycles):
        # for timesteps in total_iterations:
        # Create initial perturbation for entire batch
        delta_x0 = perturbation_std * torch.randn_like(x_batch)
        x_perturbed = x_batch.clone() + delta_x0

        # Run both unperturbed and perturbed forecasts
        x_unperturbed = x_batch.clone()

        # Batch predictions
        x_unperturbed_pred = model(x_unperturbed)
        x_perturbed_pred = model(x_perturbed)

        if flag_clamp:
            x_unperturbed = torch.clamp(x_unperturbed, min=clamp_min, max=clamp_max)
            x_perturbed = torch.clamp(x_perturbed, min=clamp_min, max=clamp_max)

        # Compute bred vectors
        ## But here we need the next step forcing not the current step
        delta_x = x_perturbed_pred - x_unperturbed_pred
        # Calculate norm across time, latitude, and longitude dimensions (dim=(2, 3, 4))
        norm = torch.norm(delta_x, p=2, dim=(2, 3, 4), keepdim=True)  # Only spatial and temporal dimensions
        delta_x_rescaled = epsilon * delta_x / (1e-8 + norm)
        bred_vectors.append(delta_x_rescaled)

        # # Compute perturbation magnitude
        # perturbation_magnitude = torch.abs(delta_x_rescaled)
        # relative_perturbation = perturbation_magnitude / (torch.abs(x_batch) + 1e-8)
        # average_relative_perturbation = relative_perturbation.mean().item() * 100
        # print(f"Average relative perturbation: {average_relative_perturbation:.2f}%")

    # Initialize ensemble members for the entire batch
    # Do not add anything to the forcing and static variables (:bv.shape[1])
    initial_conditions = []
    for bv in bred_vectors:
        x_modified = x_batch.clone()
        x_modified[:, : bv.shape[1], :, :, :] += bv
        initial_conditions.append(x_modified)
        x_modified = x_batch.clone()
        x_modified[:, : bv.shape[1], :, :, :] -= bv
        initial_conditions.append(x_modified)
    return initial_conditions


def generate_bred_vectors_cycle(
    initial_condition,
    dataset,
    model,
    num_cycles=5,
    perturbation_std=0.15,
    epsilon=1.0,
    flag_clamp=False,
    clamp_min=None,
    clamp_max=None,
    device="cuda",
    history_len=1,
    varnum_diag=None,
    static_dim_size=None,
    post_conf={},
):
    """
    Generate bred vectors and initialize initial conditions for the given batch.

    Args:
        x_batch (torch.Tensor): The input batch.
        batch (dict): A dictionary containing additional batch data.
        model (nn.Module): The model used for predictions.
        num_cycles (int): Number of perturbation cycles.
        perturbation_std (float): Magnitude of initial perturbations.
        epsilon (float): Scaling factor for bred vectors.
        flag_clamp (bool, optional): Whether to clamp inputs. Defaults to False.
        clamp_min (float, optional): Minimum clamp value. Required if flag_clamp is True.
        clamp_max (float, optional): Maximum clamp value. Required if flag_clamp is True.

    Returns:
        list[torch.Tensor]: List of initial conditions generated using bred vectors.
    """

    flag_mass_conserve = False
    flag_water_conserve = False
    flag_energy_conserve = False

    if post_conf["activate"]:
        if post_conf["global_mass_fixer"]["activate"]:
            if post_conf["global_mass_fixer"]["activate_outside_model"]:
                logger.info("Activate GlobalMassFixer outside of model")
                flag_mass_conserve = True
                opt_mass = GlobalMassFixer(post_conf)

        if post_conf["global_water_fixer"]["activate"]:
            if post_conf["global_water_fixer"]["activate_outside_model"]:
                logger.info("Activate GlobalWaterFixer outside of model")
                flag_water_conserve = True
                opt_water = GlobalWaterFixer(post_conf)

        if post_conf["global_energy_fixer"]["activate"]:
            if post_conf["global_energy_fixer"]["activate_outside_model"]:
                logger.info("Activate GlobalEnergyFixer outside of model")
                flag_energy_conserve = True
                opt_energy = GlobalEnergyFixer(post_conf)

    bred_vectors = []
    for _ in range(num_cycles):
        # Make a copy of the dataloader
        dataset_c = clone_dataset(dataset)
        data_loader = BatchForecastLenDataLoader(dataset_c)

        # model inference loop
        for _, batch in enumerate(data_loader):
            forecast_step = batch["forecast_step"].item()

            # Initial input processing
            if forecast_step == 1:
                if "x_surf" in batch:
                    x = concat_and_reshape(batch["x"], batch["x_surf"]).to(device).float()
                else:
                    x = reshape_only(batch["x"]).to(device).float()

                # Add forcing and static variables
                if "x_forcing_static" in batch:
                    x_forcing_batch = batch["x_forcing_static"].to(device).permute(0, 2, 1, 3, 4).float()
                    x = torch.cat((x, x_forcing_batch), dim=1)

                # Clamp if needed
                if flag_clamp:
                    x = torch.clamp(x, min=clamp_min, max=clamp_max)

            else:
                # Add current forcing and static variables
                if "x_forcing_static" in batch:
                    x_forcing_batch = batch["x_forcing_static"].to(device).permute(0, 2, 1, 3, 4).float()
                    x = torch.cat((x, x_forcing_batch), dim=1)

                # Clamp if needed
                if flag_clamp:
                    x = torch.clamp(x, min=clamp_min, max=clamp_max)

            # Load y-truth
            if "y_surf" in batch:
                y = concat_and_reshape(batch["y"], batch["y_surf"]).to(device).float()
            else:
                y = reshape_only(batch["y"]).to(device).float()

            if "y_diag" in batch:
                y_diag_batch = batch["y_diag"].to(device).permute(0, 2, 1, 3, 4)
                y = torch.cat((y, y_diag_batch), dim=1).to(device).float()

            # Predict
            # Create initial perturbation for entire batch
            delta_x0 = perturbation_std * torch.randn_like(x)
            x_perturbed = x.clone() + delta_x0

            # Run both unperturbed and perturbed forecasts
            x_unperturbed = x.clone()

            if flag_clamp:
                x_unperturbed = torch.clamp(x_unperturbed, min=clamp_min, max=clamp_max)
                x_perturbed = torch.clamp(x_perturbed, min=clamp_min, max=clamp_max)

            # Batch predictions
            model_inputs = [x_unperturbed, x_perturbed]
            model_outputs = []

            for x_input in model_inputs:
                # Batch predictions
                y_pred = model(x_input)

                # Post-processing blocks
                if flag_mass_conserve:
                    if forecast_step == 1:
                        x_init = x_input.clone()
                    input_dict = {"y_pred": y_pred, "x": x_init}
                    input_dict = opt_mass(input_dict)
                    y_pred = input_dict["y_pred"]

                if flag_water_conserve:
                    input_dict = {"y_pred": y_pred, "x": x_input}
                    input_dict = opt_water(input_dict)
                    y_pred = input_dict["y_pred"]

                if flag_energy_conserve:
                    input_dict = {"y_pred": y_pred, "x": x_input}
                    input_dict = opt_energy(input_dict)
                    y_pred = input_dict["y_pred"]

                model_outputs.append(y_pred)

            # Unpack results
            x_unperturbed_pred, x_perturbed_pred = model_outputs

            # Compute bred vectors
            delta_x = x_perturbed_pred - x_unperturbed_pred
            # Calculate norm across time, latitude, and longitude dimensions (dim=(2, 3, 4))
            # norm = torch.norm(
            #     delta_x, p=2, dim=(2, 3, 4), keepdim=True
            # )  # Only spatial and temporal dimensions
            # delta_x_rescaled = epsilon * delta_x  # / (1e-8 + norm)

            # Rescale bred vectors
            norm_delta_x0 = torch.norm(delta_x0[:, : delta_x.shape[1]], p=2, dim=(2, 3), keepdim=True)
            norm_delta_x = torch.norm(delta_x, p=2, dim=(2, 3), keepdim=True)
            delta_x_rescaled = epsilon * (norm_delta_x0 / (norm_delta_x + 1e-8)) * delta_x

            # Perform hemispheric rescaling -- need the latlon file so we use the right grid spacing, removing for now also this function is unused.
            # latitudes = torch.linspace(90, -90, delta_x.shape[3], device=delta_x.device)
            # delta_x_rescaled = hemi_rescale(delta_x_rescaled, latitudes)

            # Add the perturbation to the model input
            x[:, : delta_x_rescaled.shape[1], :, :, :] += delta_x_rescaled

            if batch["stop_forecast"].item():
                break

            # Pass through the model to advance one time step
            y_pred = model(x)

            if history_len == 1:
                if "y_diag" in batch:
                    x = y_pred[:, :-varnum_diag, ...].detach()
                else:
                    x = y_pred.detach()
            else:
                if static_dim_size == 0:
                    x_detach = x[:, :, 1:, ...].detach()
                else:
                    x_detach = x[:, :-static_dim_size, 1:, ...].detach()

                if "y_diag" in batch:
                    x = torch.cat([x_detach, y_pred[:, :-varnum_diag, ...].detach()], dim=2)
                else:
                    x = torch.cat([x_detach, y_pred.detach()], dim=2)

        # Add the bred vector to the return list
        x0 = initial_condition.clone()
        x0[:, : delta_x_rescaled.shape[1], :, :, :] += delta_x_rescaled
        bred_vectors.append(x0)
        x0 = initial_condition.clone()
        x0[:, : delta_x_rescaled.shape[1], :, :, :] -= delta_x_rescaled
        bred_vectors.append(x0)

    return bred_vectors


def clone_dataset(dataset):
    """
    Clones a PyTorch Dataset by creating a deep copy.

    Args:
        dataset (torch.utils.data.Dataset): The original dataset.

    Returns:
        torch.utils.data.Dataset: A cloned dataset.
    """
    return copy.deepcopy(dataset)


def adjust_start_times(time_ranges, hours=24):
    """
    Adjusts the start times by subtracting 24 hours.

    Args:
        time_ranges (list of lists): Each sublist contains [start_time, end_time] as strings.

    Returns:
        list of lists: Adjusted time ranges [[start_time - 24hrs, start_time], ...]
    """
    adjusted_times = []

    for start_time, _ in time_ranges:
        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        new_start = start_dt - timedelta(hours=hours)
        adjusted_times.append([new_start.strftime("%Y-%m-%d %H:%M:%S"), start_time])

    return adjusted_times


if __name__ == "__main__":
    from credit.models import load_model
    import logging

    # Set up the logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    crossformer_config = {
        "type": "crossformer",
        "frames": 1,  # Number of input states
        "image_height": 640,  # Number of latitude grids
        "image_width": 1280,  # Number of longitude grids
        "levels": 16,  # Number of upper-air variable levels
        "channels": 4,  # Upper-air variable channels
        "surface_channels": 7,  # Surface variable channels
        "input_only_channels": 0,  # Dynamic forcing, forcing, static channels
        "output_only_channels": 0,  # Diagnostic variable channels
        "patch_width": 1,  # Number of latitude grids in each 3D patch
        "patch_height": 1,  # Number of longitude grids in each 3D patch
        "frame_patch_size": 1,  # Number of input states in each 3D patch
        "dim": [32, 64, 128, 256],  # Dimensionality of each layer
        "depth": [2, 2, 2, 2],  # Depth of each layer
        "global_window_size": [10, 5, 2, 1],  # Global window size for each layer
        "local_window_size": 10,  # Local window size
        "cross_embed_kernel_sizes": [  # Kernel sizes for cross-embedding
            [4, 8, 16, 32],
            [2, 4],
            [2, 4],
            [2, 4],
        ],
        "cross_embed_strides": [2, 2, 2, 2],  # Strides for cross-embedding
        "attn_dropout": 0.0,  # Dropout probability for attention layers
        "ff_dropout": 0.0,  # Dropout probability for feed-forward layers
        "use_spectral_norm": True,  # Whether to use spectral normalization
    }

    num_cycles = 5
    input_tensor = torch.randn(1, 71, 1, 640, 1280).to("cuda")
    model = load_model({"model": crossformer_config}).to("cuda")

    initial_conditions = generate_bred_vectors(
        input_tensor,
        model,
        num_cycles=num_cycles,
        perturbation_std=0.15,
        epsilon=10.0,
    )

    logger.info(f"Generated {num_cycles} bred-vector initial conditions.")
