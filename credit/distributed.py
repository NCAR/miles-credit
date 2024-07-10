
import torch
import os

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    MixedPrecision,
    CPUOffload
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
from credit.models.checkpoint import (
    TorchFSDPModel
)
from torch.nn.parallel import DistributedDataParallel as DDP
from credit.mixed_precision import parse_dtype
import functools
import logging


def distributed_model_wrapper(conf, neural_network, device):

    # convert $USER to the actual user name
    conf['save_loc'] = os.path.expandvars(conf['save_loc'])

    # FSDP polices
    if conf["trainer"]["mode"] == "fsdp":

        # Define the sharding policies
        # crossformer
        if "crossformer" in conf["model"]["type"]:
            from credit.models.crossformer import (
                Attention, DynamicPositionBias, FeedForward, CrossEmbedLayer
            )
            transformer_layers_cls = {Attention, DynamicPositionBias, FeedForward, CrossEmbedLayer}

        # FuXi
        # FuXi supports "spectral_nrom = True" only
        elif "fuxi" in conf["model"]["type"]:
            from timm.models.swin_transformer_v2 import SwinTransformerV2Stage
            transformer_layers_cls = {SwinTransformerV2Stage}

        # Swin by itself
        elif "swin" in conf["model"]["type"]:
            from credit.models.swin import SwinTransformerV2CrBlock, WindowMultiHeadAttentionNoPos, WindowMultiHeadAttention
            transformer_layers_cls = {SwinTransformerV2CrBlock, WindowMultiHeadAttentionNoPos, WindowMultiHeadAttention}

        # other models not supported
        else:
            raise OSError("You asked for FSDP but only crossformer and fuxi are currently supported.")

        auto_wrap_policy1 = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layers_cls
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

        use_mixed_precision = conf["trainer"]["use_mixed_precision"] if "use_mixed_precision" in conf["trainer"] else False

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

        # FSDP module

        model = TorchFSDPModel(
            neural_network,
            use_orig_params=True,
            auto_wrap_policy=combined_auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            cpu_offload=CPUOffload(offload_params=cpu_offload)
        )

        # activation checkpointing on the transformer blocks

        activation_checkpoint = conf["trainer"]["activation_checkpoint"] if "activation_checkpoint" in conf["trainer"] else False

        logging.info(f"Activation checkpointing: {activation_checkpoint}")

        if activation_checkpoint:

            # https://pytorch.org/blog/efficient-large-scale-training-with-pytorch/

            non_reentrant_wrapper = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )

            check_fn = lambda submodule: any(isinstance(submodule, cls) for cls in transformer_layers_cls)

            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=check_fn
            )

        # attempting to get around the launch issue we are having
        torch.distributed.barrier()

    elif conf["trainer"]["mode"] == "ddp":
        model = DDP(neural_network, device_ids=[device])
    else:
        model = neural_network

    return model
