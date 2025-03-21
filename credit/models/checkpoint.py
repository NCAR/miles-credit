import os
import shutil
import torch
import torch.nn as nn
import logging

from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from torch.optim import Optimizer
from pathlib import Path
from typing import Union

from torch import Tensor
from torch.utils._pytree import tree_map


# adapted from https://github.com/hpcaitech/ColossalAI/blob/f2e8b9ef9ff3032513732a699d766bcde1a3506e/colossalai/booster/plugin/torch_fsdp_plugin.py


# utils


def load_state_dict_error_handler(load_msg):
    if load_msg[1]:
        raise RuntimeError(str(load_msg))
    elif load_msg[0]:
        logging.warning(f"Loaded partial model {load_msg}")
    else:  # all keys matched
        logging.info(load_msg)


def get_file_extension(file_path):
    _, ext = os.path.splitext(file_path)
    return ext

def copy_checkpoint(checkpoint_file_path: str, number) -> None:
    """
    Copy every checkpoint afterit's saved.

    Args:
        state_dict (dict): state dict.
        checkpoint_file_path (str): path to the checkpoint file.
        use_safetensors (bool): whether to use safetensors to save the checkpoint.
    """
    
    exty = get_file_extension(checkpoint_file_path)
    
    checkpoint_file_path_new = f'{checkpoint_file_path}{number:05}{exty}'
    
    shutil.copy(checkpoint_file_path, checkpoint_file_path_new)


def load_model_state(conf, model, device):
    save_loc = os.path.expandvars(conf["save_loc"])
    #  Load an optimizer, gradient scaler, and learning rate scheduler, the optimizer must come after wrapping model using FSDP
    if conf["trainer"]["mode"] == "fsdp":
        logging.info(
            f"Loading FSDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}"
        )
        checkpoint_io = TorchFSDPCheckpointIO()
        checkpoint_io.load_unsharded_model(
            model, os.path.join(save_loc, "model_checkpoint.pt")
        )
    else:
        ckpt = os.path.join(save_loc, "checkpoint.pt")
        checkpoint = torch.load(ckpt, map_location=device)
        if conf["trainer"]["mode"] == "ddp":
            logging.info(
                f"Loading DDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}"
            )
            load_msg = model.module.load_state_dict(
                checkpoint["model_state_dict"], strict=False
            )
            load_state_dict_error_handler(load_msg)
        else:
            logging.info(
                f"Loading model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}"
            )
            load_msg = model.load_state_dict(
                checkpoint["model_state_dict"], strict=False
            )
            load_state_dict_error_handler(load_msg)
    return model


def save_state_dict(
    state_dict: dict, checkpoint_file_path: str, use_safetensors: bool
) -> None:
    """
    Save state dict to checkpoint.

    Args:
        state_dict (dict): state dict.
        checkpoint_file_path (str): path to the checkpoint file.
        use_safetensors (bool): whether to use safetensors to save the checkpoint.
    """
    # Move all tensors in the state_dict to CPU before saving to avoid serialization issues
    state_dict_cpu = tree_map(
        lambda x: x.cpu() if torch.is_tensor(x) else x, state_dict
    )

    if use_safetensors:
        assert is_safetensors_available(), "safetensors is not available."
        assert checkpoint_file_path.endswith(
            ".safetensors"
        ), "safetensors only supports .safetensors suffix for checkpoint file."
        from safetensors.torch import save_file as safe_save_file

        safe_save_file(state_dict_cpu, checkpoint_file_path, metadata={"format": "pt"})
    else:
        torch.save(state_dict_cpu, checkpoint_file_path)


def load_state_dict(checkpoint_file_path: Path):
    """
    Load state dict from checkpoint.

    Args:
        checkpoint_file_path (Path): path to the checkpoint file.

    Returns:
        dict: state dict.
    """

    assert not is_dtensor_checkpoint(
        checkpoint_file_path
    ), f"Cannot load state dict from dtensor checkpoint {checkpoint_file_path}, you should convert the distributed tensors to gathered tensors with our CLI offline."

    if is_safetensor_checkpoint(checkpoint_file_path):
        assert is_safetensors_available(), f"Cannot load state dict from safetensor checkpoint {checkpoint_file_path}, because safetensors is not available. Please install safetensors first with pip install safetensors."
        # load with safetensors
        from safetensors import safe_open

        state_dict = {}
        with safe_open(checkpoint_file_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
        return state_dict

    else:
        # load with torch
        return torch.load(checkpoint_file_path, map_location=torch.device("cpu"))


def is_dtensor_checkpoint(checkpoint_file_path: str) -> bool:
    """
    Check whether the checkpoint file is a dtensor checkpoint.

    Args:
        checkpoint_file_path (str): path to the checkpoint file.

    Returns:
        bool: whether the checkpoint file is a dtensor checkpoint.
    """
    if checkpoint_file_path.endswith(".*.safetensors") or checkpoint_file_path.endswith(
        ".*.bin"
    ):
        return True
    else:
        return False


def is_safetensor_checkpoint(checkpoint_file_path: str) -> bool:
    """
    Check whether the checkpoint file is a safetensor checkpoint.

    Args:
        checkpoint_file_path (str): path to the checkpoint file.

    Returns:
        bool: whether the checkpoint file is a safetensor checkpoint.
    """
    if checkpoint_file_path.endswith(".safetensors"):
        return True
    else:
        return False


def is_safetensors_available() -> bool:
    """
    Check whether safetensors is available.

    Returns:
        bool: whether safetensors is available.
    """
    try:
        return True
    except ImportError:
        return False


class TorchFSDPCheckpointIO:
    def __init__(self) -> None:
        super().__init__()

    def load_unsharded_model(self, model, checkpoint):
        model = model.unwrap()
        checkpoint = load_state_dict(checkpoint)
        load_msg = model.load_state_dict(checkpoint, strict=False)
        load_state_dict_error_handler(load_msg)

    def load_unsharded_optimizer(self, optimizer, checkpoint):
        checkpoint = load_state_dict(checkpoint)
        fsdp_model = optimizer.unwrap_model()
        # I believe using scatter causes extra memory usage by torch, which has caused OOM problems. shard should not do this -- John
        # see https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict
        sharded_osd = FSDP.shard_full_optim_state_dict(checkpoint, fsdp_model)
        # sharded_osd = FSDP.scatter_full_optim_state_dict(checkpoint, fsdp_model)
        optimizer.load_state_dict(sharded_osd)

    def save_unsharded_model(
        self, model, checkpoint, gather_dtensor, use_safetensors, rank
    ):
        """
        Save model to checkpoint but only on master process.
        """
        model = model.unwrap()
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            full_model_state = model.state_dict()
        if rank == 0:
            save_state_dict(
                full_model_state,
                checkpoint_file_path=checkpoint,
                use_safetensors=use_safetensors,
            )

    def save_unsharded_optimizer(self, optimizer, checkpoint, gather_dtensor, rank):
        """
        Save optimizer to checkpoint but only on master process.
        """
        fsdp_model = optimizer.unwrap_model()
        full_optimizer_state = FSDP.full_optim_state_dict(
            fsdp_model, optim=optimizer, rank0_only=True
        )
        if rank == 0:
            save_state_dict(
                full_optimizer_state,
                checkpoint_file_path=checkpoint,
                use_safetensors=False,
            )


class ModelWrapper(nn.Module):
    """
    A wrapper class to define the common interface used FSDP.

    Args:
        module (nn.Module): The model to be wrapped.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def unwrap(self):
        """
        Unwrap the model to return the original model for checkpoint saving/loading.
        """
        if isinstance(self.module, ModelWrapper):
            return self.module.unwrap()
        return self.module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class TorchFSDPModel(ModelWrapper):
    def __init__(self, module, *args, **kwargs) -> None:
        super().__init__(module)
        self.module = FSDP(module, *args, **kwargs)

    def unwrap(self):
        return self.module

    def concat_and_reshape(self, x1, x2):  # to be removed when data is updated
        x1 = x1.view(
            x1.shape[0],
            x1.shape[1],
            x1.shape[2] * x1.shape[3],
            x1.shape[4],
            x1.shape[5],
        )
        x_concat = torch.cat((x1, x2), dim=2)
        return x_concat.permute(0, 2, 1, 3, 4)

    def reshape_only(self, x1):
        """
        As in "concat_and_reshape", but for upper-air variables only.
        """
        x1 = x1.view(
            x1.shape[0],
            x1.shape[1],
            x1.shape[2] * x1.shape[3],
            x1.shape[4],
            x1.shape[5],
        )
        return x1.permute(0, 2, 1, 3, 4)


class OptimizerWrapper:
    """
    A standard interface for optimizers wrapped by the Booster.

    Args:
        optim (Optimizer): The optimizer to be wrapped.
    """

    def __init__(self, optim: Optimizer):
        self.optim = optim

    @property
    def parameters(self):
        params = []

        for group in self.param_groups:
            params += group["params"]
        return params

    @property
    def param_groups(self):
        return self.optim.param_groups

    @property
    def defaults(self):
        return self.optim.defaults

    def add_param_group(self, *args, **kwargs):
        return self.optim.add_param_group(*args, **kwargs)

    def step(self, *args, **kwargs):
        """
        Performs a single optimization step.
        """
        return self.optim.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        """
        Clears the gradients of all optimized `torch.Tensor`.
        """
        self.optim.zero_grad(*args, **kwargs)

    def backward(self, loss: Tensor, *args, **kwargs):
        """
        Performs a backward pass on the loss.
        """
        loss.backward(*args, **kwargs)

    def backward_by_grad(self, tensor: Tensor, grad: Tensor):
        torch.autograd.backward(tensor, grad)

    def state_dict(self):
        """
        Returns the optimizer state.
        """
        return self.optim.state_dict()

    def load_state_dict(self, *args, **kwargs):
        """
        Loads the optimizer state.
        """
        self.optim.load_state_dict(*args, **kwargs)

    def clip_grad_by_value(self, clip_value: float, *args, **kwargs) -> None:
        """
        Clips gradient of an iterable of parameters at specified min and max values.

        Args:
            clip_value (float or int): maximum allowed value of the gradients. Gradients are clipped in the range

        Note:
            In PyTorch 2.0 and above, you can pass in foreach=True as kwargs to `clip_grad_value_` to use the
            faster implementation. Please refer to the PyTorch documentation for more details.
        """
        nn.utils.clip_grad_value_(self.parameters, clip_value, *args, **kwargs)

    def clip_grad_by_norm(
        self,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
        error_if_nonfinite: bool = False,
        *args,
        **kwargs,
    ) -> Tensor:
        """
        Clips gradient norm of an iterable of parameters.

        Args:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.
            error_if_nonfinite (bool): if True, an error is raised if the total norm is non-finite. Default: False

        Note:
            In PyTorch 2.0 and above, you can pass in foreach=True as kwargs to `clip_grad_norm_` to use the
            faster implementation. Please refer to the PyTorch documentation for more details.
        """
        norm = nn.utils.clip_grad_norm_(
            self.parameters, max_norm, norm_type, error_if_nonfinite, *args, **kwargs
        )
        return norm

    def scale_loss(self, loss: Tensor):
        """
        Scales the loss for mixed precision training.

        Note: Only available for optimizers with mixed precision training.

        Args:
            loss (Tensor): The loss to be scaled.
        """
        raise NotImplementedError(
            "The method scale_loss is only available for optimizers with mixed precision training"
        )

    def unscale_grad(self):
        """
        Unscale the gradients for mixed precision training.

        Note: Only available for optimizers with mixed precision training.
        """
        raise NotImplementedError(
            "The method unscale_grad is only available for optimizers with mixed precision training"
        )

    def unwrap(self):
        """
        Unwrap the optimizer for checkpoint saving/loading.
        """
        return self.optim


class FSDPOptimizerWrapper(OptimizerWrapper):
    def __init__(self, optimizer, model):
        self.model = model
        super().__init__(optimizer)

    def unwrap_model(self) -> nn.Module:
        return self.model


if __name__ == "__main__":
    import torch.optim as optim
    import torch.nn as nn
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=0, world_size=1)

    # Define your model, optimizer, and loss function
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2)).to("cuda")

    # Boost the model and optimizer with FSDP
    fsdp_model = TorchFSDPModel(model)

    optimizer = optim.SGD(fsdp_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    fsdp_optimizer = FSDPOptimizerWrapper(optimizer, fsdp_model)

    # Initialize the checkpoint I/O handler
    checkpoint_io = TorchFSDPCheckpointIO()

    # Save model and optimizer checkpoints
    checkpoint_io.save_unsharded_model(
        fsdp_model, "model_checkpoint.pth", gather_dtensor=True, use_safetensors=False
    )
    checkpoint_io.save_unsharded_optimizer(
        fsdp_optimizer, "optimizer_checkpoint.pth", gather_dtensor=True
    )

    # Load the model and optimizer
    loaded_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2)).to(
        "cuda"
    )

    fsdp_loaded_model = TorchFSDPModel(loaded_model)

    loaded_optimizer = optim.SGD(fsdp_loaded_model.parameters(), lr=0.01)
    fsdp_loaded_optimizer = FSDPOptimizerWrapper(loaded_optimizer, fsdp_loaded_model)

    checkpoint_io.load_unsharded_model(fsdp_loaded_model, "model_checkpoint.pth")
    checkpoint_io.load_unsharded_optimizer(
        fsdp_loaded_optimizer, "optimizer_checkpoint.pth"
    )

    # Load the model outside of FSDP context

    indy_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2)).to("cuda")
    checkpoint = torch.load("model_checkpoint.pth")
    indy_model.load_state_dict(checkpoint)
