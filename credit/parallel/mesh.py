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
    total_non_dp = tp_size * domain_size
    if world_size % total_non_dp != 0:
        raise ValueError(f"world_size={world_size} not divisible by tensor*domain={total_non_dp}")
    dp_size = world_size // total_non_dp

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


def parse_parallelism_conf(conf: dict) -> dict:
    """Extract and validate the parallelism block from a trainer config.

    Falls back to inferring from legacy trainer.mode if parallelism block
    is absent — so old V1 configs that go through the V2 path still work.

    Returns a normalized parallelism dict with keys: data, tensor, domain.
    """
    trainer = conf.get("trainer", {})

    if "parallelism" in trainer:
        p = trainer["parallelism"].copy()
        p.setdefault("data", "none")
        p.setdefault("tensor", 1)
        p.setdefault("domain", 1)
        return p

    # Legacy mode: infer from trainer.mode string
    mode = trainer.get("mode", "none")
    mapping = {
        "fsdp": {"data": "fsdp2", "tensor": 1, "domain": 1},
        "ddp": {"data": "ddp", "tensor": 1, "domain": 1},
        "none": {"data": "none", "tensor": 1, "domain": 1},
        "domain_parallel": {"data": "none", "tensor": 1, "domain": trainer.get("domain_parallel_size", 2)},
        "fsdp+domain_parallel": {"data": "fsdp2", "tensor": 1, "domain": trainer.get("domain_parallel_size", 2)},
    }
    result = mapping.get(mode, {"data": "none", "tensor": 1, "domain": 1})
    logger.debug(f"Inferred parallelism from legacy mode='{mode}': {result}")
    return result
