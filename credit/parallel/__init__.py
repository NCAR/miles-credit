"""CREDIT v2 parallelism package.

Provides FSDP2, Tensor Parallelism (TP), and integration with Negin's
domain parallelism — all composed via PyTorch DeviceMesh.

Config block (trainer.parallelism):
    data:   fsdp2 | ddp | none   — data-parallel mode
    tensor: int >= 1             — TP degree (1 = disabled)
    domain: int >= 1             — spatial domain shards (1 = disabled)

Total GPUs = dp_size × tensor × domain
  where dp_size = world_size // (tensor × domain)

Usage (called from distributed_model_wrapper_v2):
    mesh, submeshes = build_device_mesh(conf["trainer"]["parallelism"])
    if submeshes.get("tp"):
        model = apply_tensor_parallel(model, submeshes["tp"])
    if submeshes.get("domain"):
        model = apply_domain_parallel(model, submeshes["domain"])
    if conf["trainer"]["parallelism"]["data"] == "fsdp2":
        model = apply_fsdp2(model, submeshes.get("dp"), conf)
    elif conf["trainer"]["parallelism"]["data"] == "ddp":
        model = apply_ddp(model, submeshes.get("dp"))
"""

from .mesh import build_device_mesh
from .fsdp2 import apply_fsdp2
from .tensor_parallel import apply_tensor_parallel

__all__ = ["build_device_mesh", "apply_fsdp2", "apply_tensor_parallel"]
