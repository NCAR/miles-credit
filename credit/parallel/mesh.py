"""DeviceMesh construction for CREDIT v2 parallelism.

Builds a logical process mesh from a trainer.parallelism config block.
Supports up to 3 parallel dimensions: data (FSDP2/DDP), tensor (TP),
and domain (spatial sharding).

Example configs:
    # FSDP2 only, 8 GPUs
    parallelism: {data: fsdp2, tensor: 1, domain: 1}
    # mesh: (8,) named ["dp"]

    # FSDP2 + TP=2, 8 GPUs → dp=4, tp=2
    parallelism: {data: fsdp2, tensor: 2, domain: 1}
    # mesh: (4, 2) named ["dp", "tp"]

    # FSDP2 + TP=2 + domain=2, 8 GPUs → dp=2, tp=2, domain=2
    parallelism: {data: fsdp2, tensor: 2, domain: 2}
    # mesh: (2, 2, 2) named ["dp", "tp", "domain"]
"""

import logging
import torch.distributed as dist

logger = logging.getLogger(__name__)


def dp_world_size(parallelism_conf: dict, world_size: int) -> int:
    """Number of data-parallel replicas for a given world size.

    Single source of truth for the dp-size arithmetic (build_device_mesh,
    data_parallel_coords, and the model wrapper must all agree).

    Raises:
        ValueError: if world_size is not divisible by tensor * domain.
    """
    non_dp = max(int(parallelism_conf.get("tensor", 1)) * int(parallelism_conf.get("domain", 1)), 1)
    if world_size % non_dp != 0:
        raise ValueError(f"world_size={world_size} not divisible by tensor*domain={non_dp}")
    return world_size // non_dp


def build_device_mesh(parallelism_conf: dict, device: str = "cuda"):
    """Build a DeviceMesh from a parallelism config block.

    Args:
        parallelism_conf: dict with keys:
            data   (str): "fsdp2" | "ddp" | "none"
            tensor (int): TP degree, >= 1
            domain (int): domain parallel degree, >= 1
        device: "cuda" (default) or "cpu" for tests

    Returns:
        mesh: DeviceMesh (or None if no parallelism)
        submeshes: dict mapping dim name -> submesh (or None if single-dim)
            Keys present: "dp" if dp > 1, "tp" if tp > 1, "domain" if domain > 1

    Raises:
        ValueError: if world_size is not divisible by tensor * domain.
    """
    from torch.distributed.device_mesh import init_device_mesh

    tp_size = int(parallelism_conf.get("tensor", 1))
    domain_size = int(parallelism_conf.get("domain", 1))
    data_mode = parallelism_conf.get("data", "none")

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    dp_size = dp_world_size(parallelism_conf, world_size)

    # Build ordered dim list: always dp first, then tp, then domain
    dims = []
    names = []
    if data_mode != "none" or dp_size > 1:
        dims.append(dp_size)
        names.append("dp")
    if tp_size > 1:
        dims.append(tp_size)
        names.append("tp")
    if domain_size > 1:
        dims.append(domain_size)
        names.append("domain")

    if not dims:
        logger.info("No parallelism active (all degrees == 1, mode == none)")
        return None, {}

    if len(dims) == 1 and dims[0] == 1:
        return None, {}

    mesh = init_device_mesh(device, tuple(dims), mesh_dim_names=tuple(names))

    submeshes = {}
    for name in names:
        submeshes[name] = mesh[name]

    logger.info(f"DeviceMesh: {dict(zip(names, dims))} (world={world_size}, data_mode={data_mode})")
    return mesh, submeshes


def data_parallel_coords(conf: dict):
    """Return (dp_rank, dp_size) for dataset/dataloader sharding.

    THE SAMPLER CONTRACT
    --------------------
    Dataset samples must be sharded over the **data-parallel** dimension only,
    never over the global rank. Ranks that differ only in their tensor- or
    domain-parallel coordinate MUST receive the *same* batch:

      - TP ranks compute partial outputs of the same activation; the row-parallel
        all_reduce sums them. Feeding different samples to TP peers silently sums
        partial outputs of *different* inputs — garbage activations, and the
        replicated (non-TP) parameters drift apart because nothing syncs them
        across the tp dimension.
      - Domain ranks hold different spatial shards of the same sample; the halo
        exchange passes boundary rows between them. Different samples per domain
        rank corrupt every halo.

    So a DataLoader/DistributedSampler must be built with
    ``rank=dp_rank, num_replicas=dp_size`` from this function — NOT the global
    rank/world_size — whenever tensor > 1 or domain > 1.

    RANK LAYOUT
    -----------
    ``init_device_mesh`` arranges ranks row-major over (dp, tp, domain), with
    dp outermost and domain innermost. ``DomainParallelManager`` builds the same
    layout (domain groups are consecutive ranks). Hence for global rank g:

        domain_coord = g % domain
        tp_coord     = (g // domain) % tp
        dp_rank      = g // (tp * domain)

    Returns:
        (dp_rank, dp_size): the data-parallel coordinate of this rank and the
        number of data-parallel replicas. Falls back to (0, 1) when torch
        distributed is not initialized.
    """
    p = parse_parallelism_conf(conf)

    if not dist.is_initialized():
        return 0, 1

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    non_dp = max(int(p.get("tensor", 1)) * int(p.get("domain", 1)), 1)
    return rank // non_dp, dp_world_size(p, world_size)


def parse_parallelism_conf(conf: dict) -> dict:
    """Extract and validate the parallelism block from a trainer config.

    Returns a normalized parallelism dict with keys: data, tensor, domain.
    """
    trainer = conf.get("trainer", {})
    if "parallelism" not in trainer:
        raise ValueError(
            "Gen2 configs must define trainer.parallelism with data, tensor, and "
            "domain fields. Configs from before the parallelism block (legacy "
            "trainer.mode) can be migrated with `credit convert`."
        )

    p = trainer["parallelism"].copy()
    p.setdefault("data", "none")
    p.setdefault("tensor", 1)
    p.setdefault("domain", 1)

    if p["data"] not in ("fsdp2", "ddp", "none"):
        raise ValueError(f"trainer.parallelism.data must be 'fsdp2', 'ddp', or 'none', got {p['data']!r}")
    for key in ("tensor", "domain"):
        try:
            p[key] = int(p[key])
        except (TypeError, ValueError):
            raise ValueError(f"trainer.parallelism.{key} must be an integer >= 1, got {p[key]!r}")
        if p[key] < 1:
            raise ValueError(f"trainer.parallelism.{key} must be an integer >= 1, got {p[key]}")
    unknown = set(p) - {"data", "tensor", "domain"}
    if unknown:
        raise ValueError(f"Unknown trainer.parallelism key(s): {sorted(unknown)} (expected data, tensor, domain)")
    return p
