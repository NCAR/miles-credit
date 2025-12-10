import torch


def generate_bred_vectors(
    x_batch,
    model,
    x_forcing_batch=None,
    num_cycles=5,
    perturbation_std=0.01,
    epsilon=0.01,
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

        # Add forcing and static variables if present in batch
        if x_forcing_batch is not None:
            device = x_unperturbed_pred.device
            x_forcing_batch = x_forcing_batch.to(device)
            x_unperturbed_pred = torch.cat((x_unperturbed_pred, x_forcing_batch), dim=1)
            x_perturbed_pred = torch.cat((x_perturbed_pred, x_forcing_batch), dim=1)

        # Compute bred vectors
        delta_x = x_perturbed_pred - x_unperturbed_pred
        norm = torch.norm(
            delta_x, p=2, dim=2, keepdim=True
        )  # Calculate norm across channels
        delta_x_rescaled = epsilon * delta_x / (1e-8 + norm)
        bred_vectors.append(delta_x_rescaled)

    # Initialize ensemble members for the entire batch
    initial_conditions = [x_batch.clone() + bv for bv in bred_vectors]
    return initial_conditions


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
        perturbation_std=0.01,
        epsilon=0.01,
    )

    logger.info(f"Generated {num_cycles} bred-vector initial conditions.")
