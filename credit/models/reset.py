import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from typing import Callable
from credit.models import DebuggerModel
from credit.models.checkpoint import TorchFSDPModel


def reset_model(
    fsdp_model: FSDP, model: Callable, return_unwrapped=False, return_weights=False
):
    """
    Resets a model wrapped with FullyShardedDataParallel (FSDP) and optionally returns the unwrapped model or weights.

    Args:
        fsdp_model (FSDP): The model wrapped with FSDP or a custom TorchFSDPModel.
        model (Callable): A callable to instantiate the original model before FSDP wrapping.
        return_unwrapped (bool, optional): If True, returns the unwrapped model before DDP/FSDP wrapping. Defaults to False.
        return_weights (bool, optional): If True, returns the state dictionary of the model's weights. Defaults to False.

    Raises:
        TypeError: If `fsdp_model` is not of type `FSDP` or `TorchFSDPModel`.

    Returns:
        ddp_model (DDP): The model wrapped with DistributedDataParallel (DDP) for distributed training.
        nonwrapped_model (nn.Module): The original, non-wrapped model if `return_unwrapped=True`.
        state_dict (dict): The state dictionary of the model if `return_weights=True`.
    """

    # Check if fsdp_model is an instance of TorchFSDPModel
    if isinstance(fsdp_model, TorchFSDPModel):
        # If yes, access the underlying module directly
        fsdp_model = fsdp_model.unwrap()
    elif isinstance(fsdp_model, FSDP):
        pass
    else:
        raise TypeError(
            f"Expected fsdp_model to be of type FSDP or our custom TorchFSDPModel, got {type(fsdp_model)}"
        )

    # Ensure that only rank 0 has a full state dict in memory
    with FSDP.state_dict_type(
        fsdp_model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        state_dict = fsdp_model.state_dict()

    if return_weights:
        return state_dict

    # Create a non-wrapped model (this is the original model before FSDP)
    nonwrapped_model = model  # Assume it’s initialized on CPU
    if dist.get_rank() == 0:
        nonwrapped_model.load_state_dict(state_dict)

    # Return unwrapped model if specified
    if return_unwrapped:
        return nonwrapped_model

    # Move the model to the correct device and wrap it with DDP for distributed training
    rank = dist.get_rank()  # Get current rank
    ddp_model = DDP(nonwrapped_model.to(rank), device_ids=[rank])

    return ddp_model


if __name__ == "__main__":
    # Initialize the process group for distributed training (adjust as necessary)
    dist.init_process_group(backend="gloo")  # You can use "gloo" if not using GPUs
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    image_height = 192  # 640, 192
    image_width = 288  # 1280, 288
    levels = 15
    frames = 2
    channels = 4
    surface_channels = 7
    input_only_channels = 3
    frame_patch_size = 2

    wrapper_model = DebuggerModel(
        image_height=image_height,
        image_width=image_width,
        frames=frames,
        channels=channels,
        surface_channels=surface_channels,
        input_only_channels=input_only_channels,
        levels=levels,
        post_conf=None,
    ).to(rank)

    model = DebuggerModel(
        image_height=image_height,
        image_width=image_width,
        frames=frames,
        channels=channels,
        surface_channels=surface_channels,
        input_only_channels=input_only_channels,
        levels=levels,
        post_conf=None,
    ).to(rank)

    # Example model wrapped with FSDP
    fsdp_model = TorchFSDPModel(wrapper_model)

    # Reset the model using the FSDP wrapped model and the model constructor
    # To get the unwrapped model, pass return_unwrapped=True
    unwrapped_model = reset_model(fsdp_model, model, return_unwrapped=True)

    # Now you can continue with training using the unwrapped model or DDP model
    # Example DataLoader
    dummy_data = torch.randn(
        1,
        channels * levels + surface_channels + input_only_channels,
        frames,
        image_height,
        image_width,
    ).to(rank)

    output = unwrapped_model(dummy_data)

    # Clean up the process group when done
    dist.destroy_process_group()
