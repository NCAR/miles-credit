import torch.distributed as dist
import torch.nn as nn
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
from credit.models import load_fsdp_or_checkpoint_policy
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

    logging.info(f"Running {mode.upper()} on rank {rank} with world_size {world_size} using {backend}.")
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def get_rank_info(trainer_mode):
    """Gets rank and size information for distributed training.

    Args:
        trainer_mode (str): The mode of training (e.g., 'fsdp', 'ddp').

    Returns:
        tuple: A tuple containing LOCAL_RANK (int), WORLD_RANK (int), and WORLD_SIZE (int).
    """
    if trainer_mode in ["fsdp", "ddp", "domain_parallel", "fsdp+domain_parallel"]:
        try:
            if "LOCAL_RANK" in os.environ:
                # Environment variables set by torch.distributed.launch or torchrun
                LOCAL_RANK = int(os.environ["LOCAL_RANK"])
                WORLD_SIZE = int(os.environ["WORLD_SIZE"])
                WORLD_RANK = int(os.environ["RANK"])
            else:
                from mpi4py import MPI

                comm = MPI.COMM_WORLD
                shmem_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)

                LOCAL_RANK = shmem_comm.Get_rank()
                WORLD_SIZE = comm.Get_size()
                WORLD_RANK = comm.Get_rank()

                # Set MASTER_ADDR and MASTER_PORT if not already set.
                # (broadcast these from rank 0 - they must be consistent on every node)
                if "MASTER_ADDR" not in os.environ:
                    os.environ["MASTER_ADDR"] = comm.bcast(socket.gethostbyname(socket.gethostname()), root=0)
                if "MASTER_PORT" not in os.environ:
                    os.environ["MASTER_PORT"] = comm.bcast(str(np.random.randint(1000, 8000)), root=0)

                if 0 == WORLD_RANK:
                    logging.info("Using MASTER_ADDR={}".format(os.environ["MASTER_ADDR"]))
                    logging.info("Using MASTER_PORT={}".format(os.environ["MASTER_PORT"]))

        except Exception as e:
            logging.info(e)

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
                sys.exit(
                    "Can't find the environment variables for local rank. "
                    "If you are on casper you'll want to use torchrun for now."
                )

    else:
        LOCAL_RANK = 0
        WORLD_RANK = 0
        WORLD_SIZE = 1

    return LOCAL_RANK, WORLD_RANK, WORLD_SIZE


def should_not_checkpoint(module):
    exclude_types = (
        # Regularization & Normalization
        nn.Dropout,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.LayerNorm,
        # Activations (stateless, cheap to recompute)
        nn.ReLU,
        nn.GELU,
        nn.SiLU,
        nn.Sigmoid,
        nn.Tanh,
        # Pooling (lightweight, no significant memory savings)
        nn.MaxPool1d,
        nn.MaxPool2d,
        nn.MaxPool3d,
        nn.AvgPool1d,
        nn.AvgPool2d,
        nn.AvgPool3d,
        nn.AdaptiveMaxPool1d,
        nn.AdaptiveMaxPool2d,
        nn.AdaptiveMaxPool3d,
        nn.AdaptiveAvgPool1d,
        nn.AdaptiveAvgPool2d,
        nn.AdaptiveAvgPool3d,
        # Embeddings (usually large but don’t require recomputation)
        nn.Embedding,
        # Identity & Reshaping (no computation)
        nn.Identity,
        nn.Flatten,
    )
    return isinstance(module, exclude_types)


def distributed_model_wrapper(conf, neural_network, device):
    """Wraps the neural network model for distributed training.

    Supports modes: 'fsdp', 'ddp', 'domain_parallel', 'fsdp+domain_parallel'.

    For domain_parallel modes, the model's Conv2d/Conv3d/ConvTranspose2d/GroupNorm
    layers are replaced with domain-parallel equivalents that handle halo exchange
    and distributed normalization. For fsdp+domain_parallel, domain-parallel
    conversion is done first, then FSDP wrapping uses the data-parallel subgroup.

    Args:
        conf (dict): The configuration dictionary containing training settings.
        neural_network (torch.nn.Module): The neural network model to be wrapped.
        device (torch.device): The device on which the model will be trained.

    Returns:
        torch.nn.Module: The wrapped model ready for distributed training.
    """

    # convert $USER to the actual user name
    conf["save_loc"] = os.path.expandvars(conf["save_loc"])

    mode = conf["trainer"]["mode"]

    activation_checkpoint = (
        conf["trainer"]["activation_checkpoint"] if "activation_checkpoint" in conf["trainer"] else False
    )
    checkpoint_all_layers = conf["trainer"].get("checkpoint_all_layers", False)

    # ── Domain parallelism setup ──────────────────────────────────────────
    domain_parallel_size = conf["trainer"].get("domain_parallel_size", 1)

    if mode in ["domain_parallel", "fsdp+domain_parallel"] and domain_parallel_size <= 1:
        raise ValueError(f"mode='{mode}' requires trainer.domain_parallel_size > 1 in config")

    use_domain_parallel = mode in ["domain_parallel", "fsdp+domain_parallel"] or domain_parallel_size > 1
    domain_manager = None

    if use_domain_parallel and domain_parallel_size > 1:
        from credit.domain_parallel import (
            initialize_domain_parallel,
            convert_to_domain_parallel,
        )

        world_size = dist.get_world_size()
        domain_manager = initialize_domain_parallel(
            world_size=world_size,
            domain_parallel_size=domain_parallel_size,
            shard_dim=-2,  # latitude (H) in BCHW
        )

        # Convert model layers to domain-parallel versions
        neural_network = convert_to_domain_parallel(neural_network, domain_manager)

        # Store manager on the model for access by trainer
        neural_network._domain_parallel_manager = domain_manager

        logging.info(
            f"Domain parallelism enabled: {domain_parallel_size} domain shards, "
            f"{world_size // domain_parallel_size} data-parallel replicas"
        )

    # ── FSDP / DDP wrapping ───────────────────────────────────────────────

    # Configure FSDP layers for parallel policies AND/OR activation checkpointing
    fsdp_mode = mode in ["fsdp", "fsdp+domain_parallel"]
    if fsdp_mode or activation_checkpoint:
        transformer_layers_cls = load_fsdp_or_checkpoint_policy(conf)

    # logger announcement
    if activation_checkpoint:
        logging.info(f"Activation checkpointing on {mode}: {activation_checkpoint}")
        if checkpoint_all_layers:
            logging.info("Checkpointing all available layers in your model")
            logging.warning("This may cause performance degredation -- consider supplying a list to checkpoint")
        else:
            logging.info(f"Checkpointing custom layers {transformer_layers_cls}")

    # FSDP policies
    if fsdp_mode:
        # Define the sharding policies
        auto_wrap_policy1 = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=transformer_layers_cls
        )

        auto_wrap_policy2 = functools.partial(size_based_auto_wrap_policy, min_num_params=100_000)

        def combined_auto_wrap_policy(module, recurse, nonwrapped_numel):
            # Define a new policy that combines policies
            p1 = auto_wrap_policy1(module, recurse, nonwrapped_numel)
            p2 = auto_wrap_policy2(module, recurse, nonwrapped_numel)
            return p1 or p2

        # Mixed precision

        use_mixed_precision = (
            conf["trainer"]["use_mixed_precision"] if "use_mixed_precision" in conf["trainer"] else False
        )

        logging.info(f"Using mixed_precision: {use_mixed_precision}")

        if use_mixed_precision:
            for key, val in conf["trainer"]["mixed_precision"].items():
                conf["trainer"]["mixed_precision"][key] = parse_dtype(val)
            mixed_precision_policy = MixedPrecision(**conf["trainer"]["mixed_precision"])
        else:
            mixed_precision_policy = None

        # CPU offloading

        cpu_offload = conf["trainer"]["cpu_offload"] if "cpu_offload" in conf["trainer"] else False

        logging.info(f"Using CPU offloading: {cpu_offload}")

        # FSDP module — for fsdp+domain_parallel, use data-parallel process group
        fsdp_kwargs = dict(
            use_orig_params=True,
            auto_wrap_policy=combined_auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            cpu_offload=CPUOffload(offload_params=cpu_offload),
        )

        if domain_manager is not None:
            fsdp_kwargs["process_group"] = domain_manager.data_parallel_group

        model = TorchFSDPModel(neural_network, **fsdp_kwargs)

        # Preserve domain manager reference after FSDP wrapping
        if domain_manager is not None:
            model._domain_parallel_manager = domain_manager

    elif mode == "ddp":
        model = DDP(neural_network, device_ids=[device], find_unused_parameters=True)

    elif mode == "domain_parallel":
        if domain_manager is not None and domain_manager.data_parallel_size > 1:
            # Wrap with DDP using the data-parallel subgroup so gradients are
            # synced across data-parallel replicas (domain_group handles halo exchange)
            model = DDP(
                neural_network,
                device_ids=[device],
                process_group=domain_manager.data_parallel_group,
                find_unused_parameters=False,
            )
            model._domain_parallel_manager = domain_manager
        else:
            model = neural_network

    else:
        model = neural_network

    if activation_checkpoint:
        # https://pytorch.org/blog/efficient-large-scale-training-with-pytorch/

        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )

        if checkpoint_all_layers:

            def check_fn(submodule):
                return not should_not_checkpoint(submodule)
        else:

            def check_fn(submodule):
                return any(isinstance(submodule, cls) for cls in transformer_layers_cls)

        apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)

    torch.distributed.barrier()

    return model


# ===========================================================================
# V2 distributed wrapper — FSDP2 + Tensor Parallel + Domain Parallel
# ===========================================================================


def distributed_model_wrapper_v2(conf: dict, model, device):
    """Wrap a model for V2 distributed training.

    Reads trainer.parallelism (or falls back to trainer.mode for legacy configs)
    and applies, in order:
      1. Domain parallelism (spatial H-shard, Negin's layer swap)
      2. Tensor parallelism (Col/Row linear split across TP group)
      3. FSDP2 or DDP over the data-parallel group

    V1 trainer code still calls distributed_model_wrapper — this is V2 only.

    Args:
        conf: Full training config.
        model: Raw nn.Module.
        device: torch.device for this rank.

    Returns:
        Wrapped model (and stores domain_manager on it if domain > 1).
    """
    from credit.parallel.mesh import build_device_mesh, parse_parallelism_conf

    conf["save_loc"] = os.path.expandvars(conf["save_loc"])
    p = parse_parallelism_conf(conf)
    data_mode = p["data"]
    tp_size = int(p.get("tensor", 1))
    domain_size = int(p.get("domain", 1))

    # Validation
    if domain_size > 1 and data_mode == "none" and tp_size == 1:
        logging.warning(
            "domain > 1 with data=none and tensor=1: gradients will NOT sync "
            "across data-parallel replicas. Wrap with data=ddp or data=fsdp2."
        )

    # Build DeviceMesh
    mesh, submeshes = build_device_mesh(p, device="cuda")

    # ── 1. Domain parallelism ─────────────────────────────────────────────
    domain_manager = None
    if domain_size > 1:
        from credit.domain_parallel import initialize_domain_parallel, convert_to_domain_parallel

        world_size = dist.get_world_size()
        domain_manager = initialize_domain_parallel(
            world_size=world_size,
            domain_parallel_size=domain_size,
            shard_dim=-2,
        )
        model = convert_to_domain_parallel(model, domain_manager)
        model._domain_parallel_manager = domain_manager
        logging.info(
            f"[V2] Domain parallelism: {domain_size} shards, {world_size // domain_size} data-parallel replicas"
        )

    # ── 2. Tensor parallelism ─────────────────────────────────────────────
    if tp_size > 1:
        from credit.parallel import apply_tensor_parallel

        tp_mesh = submeshes.get("tp")
        if tp_mesh is None:
            raise ValueError("TP mesh not found — check tensor > 1 and world_size")
        model = apply_tensor_parallel(model, tp_mesh)
        logging.info(f"[V2] Tensor parallelism: degree={tp_size}")

    # Ensure all parameters/buffers are on the target device after domain/TP
    # transforms (which may create new modules via wrapping). Must happen before
    # FSDP2, which converts params to DTensors and can't be moved afterwards.
    model.to(device)

    # ── 3. Data parallelism ───────────────────────────────────────────────
    dp_mesh = submeshes.get("dp")

    if data_mode == "fsdp2":
        from credit.parallel import apply_fsdp2

        model = apply_fsdp2(model, dp_mesh, conf)
        if domain_manager is not None:
            model._domain_parallel_manager = domain_manager
        logging.info("[V2] FSDP2 applied over dp_mesh")

    elif data_mode == "ddp":
        dp_group = dp_mesh.get_group() if dp_mesh is not None else None
        ddp_kwargs = dict(device_ids=[device], find_unused_parameters=False)
        if dp_group is not None:
            ddp_kwargs["process_group"] = dp_group
        model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
        if domain_manager is not None:
            model._domain_parallel_manager = domain_manager
        logging.info("[V2] DDP applied")

    # Apply activation checkpointing if requested and FSDP2 didn't do it
    if conf.get("trainer", {}).get("activation_checkpoint", False) and data_mode != "fsdp2":
        _apply_activation_checkpointing_v2(model, conf)

    torch.distributed.barrier()
    return model


def _apply_activation_checkpointing_v2(model, conf):
    """Apply no-reentrant AC to transformer-like blocks (non-FSDP2 path)."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )
    import functools

    try:
        from credit.models.wxformer.wxformer_v2 import Transformer, UpBlock, UpBlockPS

        block_types = (Transformer, UpBlock, UpBlockPS)
    except ImportError:
        return

    wrapper = functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda m: isinstance(m, block_types),
    )
    logging.info("[V2] Activation checkpointing applied")
