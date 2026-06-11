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


# torch's conventional default rendezvous port; used as a deterministic
# fallback when no MPI broadcast is available to agree on a random port.
DEFAULT_MASTER_PORT = 29500


def resolve_master_addr():
    """Resolve a routable (non-loopback) IP address for the current host.

    ``socket.gethostbyname(socket.gethostname())`` frequently returns a
    loopback address (``127.0.0.1``) on HPC nodes whose hostname maps to
    localhost in ``/etc/hosts``, which makes it unusable as a rendezvous
    address. This prefers a non-loopback result from hostname resolution, and
    otherwise discovers the outbound-interface IP by opening a dummy UDP
    socket (no packets are actually sent).

    Returns:
        str: A best-effort non-loopback IPv4 address, falling back to
            ``127.0.0.1`` if none can be determined.
    """
    try:
        addr = socket.gethostbyname(socket.gethostname())
        if not addr.startswith("127."):
            return addr
    except OSError:
        pass

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connecting a UDP socket only sets the kernel's chosen source address;
        # it does not send any traffic. The destination need not be reachable.
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def resolve_socket_ifname():
    """Return the network interface name owning this host's routable address.

    Used to set ``GLOO_SOCKET_IFNAME`` so Gloo binds to a real (non-loopback)
    interface instead of falling back to ``127.0.0.1``.

    Returns:
        str | None: The matching interface name, or ``None`` if it cannot be
            determined (no routable address, ``psutil`` unavailable, or no
            matching interface), in which case the caller should leave
            ``GLOO_SOCKET_IFNAME`` unset.
    """
    addr = resolve_master_addr()
    if addr.startswith("127."):
        return None
    try:
        import psutil
    except ImportError:
        return None
    for ifname, addrs in psutil.net_if_addrs().items():
        for snic in addrs:
            if snic.family == socket.AF_INET and snic.address == addr:
                return ifname
    return None


def configure_gloo_ifname():
    """Bind Gloo to a routable interface when ``GLOO_SOCKET_IFNAME`` is unset.

    PyTorch's Gloo backend (used for CPU-side collectives even in NCCL runs)
    binds to loopback when it cannot resolve the hostname to a local address,
    which breaks those collectives across nodes. This sets
    ``GLOO_SOCKET_IFNAME`` to the interface owning this host's routable address
    as a best-effort default. It is a no-op if the variable is already set or
    no interface can be determined, and it intentionally leaves
    ``NCCL_SOCKET_IFNAME`` alone so NCCL is free to auto-select the
    high-speed interconnect.
    """
    if "GLOO_SOCKET_IFNAME" in os.environ:
        return
    ifname = resolve_socket_ifname()
    if ifname is not None:
        os.environ["GLOO_SOCKET_IFNAME"] = ifname
        logging.info("Set GLOO_SOCKET_IFNAME=%s for Gloo CPU collectives", ifname)


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
                logging.info("Using MPI for distributed training.")
                from mpi4py import MPI

                comm = MPI.COMM_WORLD
                shmem_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)

                LOCAL_RANK = shmem_comm.Get_rank()
                WORLD_SIZE = comm.Get_size()
                WORLD_RANK = comm.Get_rank()
                print("MASTER_ADDR: ", os.environ.get("MASTER_ADDR", "Not set"))
                # Set MASTER_ADDR and MASTER_PORT consistently on every rank.
                # bcast is a collective: every rank must call it, in the same order.
                # Rank 0 picks the values (honoring any pre-set env vars) and broadcasts
                # them so all workers agree, rather than each rank reading its own env.
                if WORLD_RANK == 0:
                    master_addr = os.environ.get("MASTER_ADDR", resolve_master_addr())
                    master_port = os.environ.get("MASTER_PORT", str(np.random.randint(20000, 30000)))
                else:
                    master_addr = None
                    master_port = None
                os.environ["MASTER_ADDR"] = comm.bcast(master_addr, root=0)
                os.environ["MASTER_PORT"] = comm.bcast(master_port, root=0)
                comm.barrier()
                print("MASTER_ADDR: ", os.environ.get("MASTER_ADDR", "Still Not set"))
                logging.info("Using MASTER_ADDR={}".format(os.environ["MASTER_ADDR"]))
                logging.info("Using MASTER_PORT={}".format(os.environ["MASTER_PORT"]))
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

            # The MPI broadcast path above is unavailable here, so we cannot
            # guarantee these agree across nodes. Set sane defaults only if the
            # launcher hasn't already provided them, and warn that multi-node
            # jobs must export MASTER_ADDR/MASTER_PORT explicitly to be safe.
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = resolve_master_addr()
                logging.warning(
                    "MASTER_ADDR was not set and could not be broadcast via MPI; "
                    "defaulting to %s. For multi-node jobs, export MASTER_ADDR "
                    "explicitly so all nodes agree on the rendezvous address.",
                    os.environ["MASTER_ADDR"],
                )
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = str(DEFAULT_MASTER_PORT)
                logging.warning(
                    "MASTER_PORT was not set; defaulting to %s.",
                    os.environ["MASTER_PORT"],
                )

        # Point Gloo at a routable interface so its CPU-side collectives don't
        # fall back to loopback (runs for both the MPI and fallback branches).
        configure_gloo_ifname()

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
