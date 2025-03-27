import copy
import torch
from datetime import datetime, timedelta
from credit.data import concat_and_reshape, reshape_only
from credit.datasets.load_dataset_and_dataloader import BatchForecastLenDataLoader
from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer


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

        if flag_clamp:
            x_unperturbed = torch.clamp(x_unperturbed, min=clamp_min, max=clamp_max)
            x_perturbed = torch.clamp(x_perturbed, min=clamp_min, max=clamp_max)

        # Batch predictions
        x_unperturbed_pred = model(x_unperturbed)
        x_perturbed_pred = model(x_perturbed)

        # Compute bred vectors
        ## But here we need the next step forcing not the current step
        delta_x = x_perturbed_pred - x_unperturbed_pred
        # Calculate norm across time, latitude, and longitude dimensions (dim=(2, 3, 4))
        norm = torch.norm(
            delta_x, p=2, dim=(2, 3, 4), keepdim=True
        )  # Only spatial and temporal dimensions
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
                    x = (
                        concat_and_reshape(batch["x"], batch["x_surf"])
                        .to(device)
                        .float()
                    )
                else:
                    x = reshape_only(batch["x"]).to(device).float()

                # Add forcing and static variables
                if "x_forcing_static" in batch:
                    x_forcing_batch = (
                        batch["x_forcing_static"]
                        .to(device)
                        .permute(0, 2, 1, 3, 4)
                        .float()
                    )
                    x = torch.cat((x, x_forcing_batch), dim=1)

                # Clamp if needed
                if flag_clamp:
                    x = torch.clamp(x, min=clamp_min, max=clamp_max)

            else:
                # Add current forcing and static variables
                if "x_forcing_static" in batch:
                    x_forcing_batch = (
                        batch["x_forcing_static"]
                        .to(device)
                        .permute(0, 2, 1, 3, 4)
                        .float()
                    )
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
            norm_delta_x0 = torch.norm(
                delta_x0[:, : delta_x.shape[1]], p=2, dim=(2, 3), keepdim=True
            )
            norm_delta_x = torch.norm(delta_x, p=2, dim=(2, 3), keepdim=True)
            delta_x_rescaled = (
                epsilon * (norm_delta_x0 / (norm_delta_x + 1e-8)) * delta_x
            )

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
                    x = torch.cat(
                        [x_detach, y_pred[:, :-varnum_diag, ...].detach()], dim=2
                    )
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
