import torch.distributed as dist
import numpy as np
import socket
import torch
import sys
import os

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    MixedPrecision,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from credit.models.checkpoint import TorchFSDPModel
from torch.nn.parallel import DistributedDataParallel as DDP
from credit.mixed_precision import parse_dtype
import functools
import logging


def setup(rank, world_size, mode, backend="nccl"):
    """Initializes the distributed process group.

    Args:
        rank (int): The rank of the process within the distributed setup.
        world_size (int): The total number of processes in the distributed setup.
        mode (str): The mode of operation (e.g., 'fsdp', 'ddp').
        backend (str, optional): The backend to use for distributed training. Defaults to 'nccl'.
    """

    logging.info(
        f"Running {mode.upper()} on rank {rank} with world_size {world_size} using {backend}."
    )
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def get_rank_info(trainer_mode):
    """Gets rank and size information for distributed training.

    Args:
        trainer_mode (str): The mode of training (e.g., 'fsdp', 'ddp').

    Returns:
        tuple: A tuple containing LOCAL_RANK (int), WORLD_RANK (int), and WORLD_SIZE (int).
    """

    if trainer_mode in ["fsdp", "ddp"]:
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            shmem_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)

            LOCAL_RANK = shmem_comm.Get_rank()
            WORLD_SIZE = comm.Get_size()
            WORLD_RANK = comm.Get_rank()

        except Exception:
            if "LOCAL_RANK" in os.environ:
                # Environment variables set by torch.distributed.launch or torchrun
                LOCAL_RANK = int(os.environ["LOCAL_RANK"])
                WORLD_SIZE = int(os.environ["WORLD_SIZE"])
                WORLD_RANK = int(os.environ["RANK"])
            elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
                # Environment variables set by mpirun
                LOCAL_RANK = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
                WORLD_SIZE = int(os.environ["OMPI_COMM_WORLD_SIZE"])
                WORLD_RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])
            elif "PMI_RANK" in os.environ:
                # Environment variables set by cray-mpich
                LOCAL_RANK = int(os.environ["PMI_LOCAL_RANK"])
                WORLD_SIZE = int(os.environ["PMI_SIZE"])
                WORLD_RANK = int(os.environ["PMI_RANK"])
            else:
                sys.exit("Can't find the environment variables for local rank")

        # Set MASTER_ADDR and MASTER_PORT if not already set
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = socket.gethostbyname(socket.gethostname())
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(np.random.randint(1000, 8000))
    else:
        LOCAL_RANK = 0
        WORLD_RANK = 0
        WORLD_SIZE = 1

    return LOCAL_RANK, WORLD_RANK, WORLD_SIZE


def distributed_model_wrapper(conf, neural_network, device):
    """Wraps the neural network model for distributed training.

    Args:
        conf (dict): The configuration dictionary containing training settings.
        neural_network (torch.nn.Module): The neural network model to be wrapped.
        device (torch.device): The device on which the model will be trained.

    Returns:
        torch.nn.Module: The wrapped model ready for distributed training.
    """

    # convert $USER to the actual user name
    conf["save_loc"] = os.path.expandvars(conf["save_loc"])

    # FSDP polices
    if conf["trainer"]["mode"] == "fsdp":
        # Define the sharding policies
        # crossformer
        if "crossformer" in conf["model"]["type"]:
            from credit.models.crossformer import (
                Attention,
                DynamicPositionBias,
                FeedForward,
                CrossEmbedLayer,
            )

            transformer_layers_cls = {
                Attention,
                DynamicPositionBias,
                FeedForward,
                CrossEmbedLayer,
            }

        # FuXi
        # FuXi supports "spectral_nrom = True" only
        elif "fuxi" in conf["model"]["type"]:
            from timm.models.swin_transformer_v2 import SwinTransformerV2Stage

            transformer_layers_cls = {SwinTransformerV2Stage}

        # Swin by itself
        elif "swin" in conf["model"]["type"]:
            from credit.models.swin import (
                SwinTransformerV2CrBlock,
                WindowMultiHeadAttentionNoPos,
                WindowMultiHeadAttention,
            )

            transformer_layers_cls = {
                SwinTransformerV2CrBlock,
                WindowMultiHeadAttentionNoPos,
                WindowMultiHeadAttention,
            }

        # other models not supported
        else:
            raise OSError(
                "You asked for FSDP but only crossformer and fuxi are currently supported."
            )

        auto_wrap_policy1 = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=transformer_layers_cls
        )

        auto_wrap_policy2 = functools.partial(
            size_based_auto_wrap_policy, min_num_params=100_000
        )

        def combined_auto_wrap_policy(module, recurse, nonwrapped_numel):
            # Define a new policy that combines policies
            p1 = auto_wrap_policy1(module, recurse, nonwrapped_numel)
            p2 = auto_wrap_policy2(module, recurse, nonwrapped_numel)
            return p1 or p2

        # Mixed precision

        use_mixed_precision = (
            conf["trainer"]["use_mixed_precision"]
            if "use_mixed_precision" in conf["trainer"]
            else False
        )

        logging.info(f"Using mixed_precision: {use_mixed_precision}")

        if use_mixed_precision:
            for key, val in conf["trainer"]["mixed_precision"].items():
                conf["trainer"]["mixed_precision"][key] = parse_dtype(val)
            mixed_precision_policy = MixedPrecision(
                **conf["trainer"]["mixed_precision"]
            )
        else:
            mixed_precision_policy = None

        # CPU offloading

        cpu_offload = (
            conf["trainer"]["cpu_offload"]
            if "cpu_offload" in conf["trainer"]
            else False
        )

        logging.info(f"Using CPU offloading: {cpu_offload}")

        # FSDP module

        model = TorchFSDPModel(
            neural_network,
            use_orig_params=True,
            auto_wrap_policy=combined_auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            cpu_offload=CPUOffload(offload_params=cpu_offload),
        )

        # activation checkpointing on the transformer blocks

        activation_checkpoint = (
            conf["trainer"]["activation_checkpoint"]
            if "activation_checkpoint" in conf["trainer"]
            else False
        )

        logging.info(f"Activation checkpointing: {activation_checkpoint}")

        if activation_checkpoint:
            # https://pytorch.org/blog/efficient-large-scale-training-with-pytorch/

            non_reentrant_wrapper = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )

            check_fn = lambda submodule: any(
                isinstance(submodule, cls) for cls in transformer_layers_cls
            )

            apply_activation_checkpointing(
                model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
            )

        # attempting to get around the launch issue we are having
        torch.distributed.barrier()

    elif conf["trainer"]["mode"] == "ddp":
        model = DDP(neural_network, device_ids=[device])
    else:
        model = neural_network

    return model
