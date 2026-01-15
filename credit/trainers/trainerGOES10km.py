import gc
import logging
from collections import defaultdict
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.fft
import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import IterableDataset
from credit.scheduler import update_on_batch
from credit.trainers.utils import cycle, accum_log
from credit.trainers.base_trainer import BaseTrainer
from credit.data import concat_and_reshape, reshape_only
from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer

import optuna
import torch

logger = logging.getLogger(__name__)

class Gather(torch.autograd.Function):
    """Custom autograd function for gathering tensors from all processes while preserving gradients.

    This layer performs an all_gather operation on the provided tensor across all
    distributed processes and concatenates them along the batch dimension (dim=0).
    The backward pass correctly routes gradients back to the originating processes.

    This is useful for operations like computng ensembles where you need to compute
    the CRPS between samples across all GPUs, while still being able to backpropagate
    through the gathered tensor.
    """

    @staticmethod
    def forward(ctx, input):
        """Gather tensors from all ranks and concatenate them on the batch dimension.

        Args:
            ctx: Context object to store information for backward pass
            input: Tensor to be gathered across processes

        Returns:
            Concatenated tensor from all processes
        """
        ctx.world_size = dist.get_world_size()
        ctx.rank = dist.get_rank()

        gathered = [torch.zeros_like(input) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, input)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        """Distribute gradients back to their originating processes.

        Args:
            ctx: Context object with stored information from forward pass
            grad_output: Gradient with respect to the forward output

        Returns:
            Gradient for the input tensor
        """
        # Each rank gets its corresponding chunk of grad_output
        input_grad = grad_output.chunk(ctx.world_size, dim=0)[ctx.rank]
        return input_grad


def gather_tensor(tensor):
    """Gathers tensors from all ranks and preserves autograd graph.

    This function allows you to gather tensors from all processes in a distributed
    setting while maintaining the autograd graph for backward passes. This is critical
    for operations that need to compute losses across all samples in a distributed
    training environment.

    Args:
        tensor: The tensor to gather across processes

    Returns:
        Tensor concatenated from all processes along dimension 0

    Example:
        >>> # On each GPU
        >>> local_tensor = torch.randn(8, 128)  # local batch of embeddings
        >>> # Gather embeddings from all GPUs (total batch_size * world_size)
        >>> gathered_tensor = gather_tensor(local_tensor)
        >>> # Now you can compute a loss that depends on all samples
    """
    return Gather.apply(tensor)

class Trainer(BaseTrainer):
    def __init__(self, model: torch.nn.Module, rank: int):
        """
        Trainer class for handling the training, validation, and checkpointing of models.

        This class is responsible for executing the training loop, validating the model
        on a separate dataset, and managing checkpoints during training. It supports
        both single-GPU and distributed (FSDP, DDP) training.

        Attributes:
            model (torch.nn.Module): The model to be trained.
            rank (int): The rank of the process in distributed training.


        Methods:
            train_one_epoch(epoch, conf, trainloader, optimizer, criterion, scaler,
                            scheduler, metrics):
                Perform training for one epoch and return training metrics.

            validate(epoch, conf, valid_loader, criterion, metrics):
                Validate the model on the validation dataset and return validation metrics.

            fit_deprecated(conf, train_loader, valid_loader, optimizer, train_criterion,
                           valid_criterion, scaler, scheduler, metrics, trial=False):
                Perform the full training loop across multiple epochs, including validation
                and checkpointing.
        """
        super().__init__(model, rank)
        # Add any additional initialization if needed
        logger.info("Loading a multi-step trainer class")
        

    # Training function.
    def train_one_epoch(
        self, epoch, conf, trainloader, optimizer, criterion, scaler, scheduler, metrics
    ):
        """
        Trains the model for one epoch.

        Args:
            epoch (int): Current epoch number.
            conf (dict): Configuration dictionary containing training settings.
            trainloader (DataLoader): DataLoader for the training dataset.
            optimizer (torch.optim.Optimizer): Optimizer used for training.
            criterion (callable): Loss function used for training.
            scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            metrics (callable): Function to compute metrics for evaluation.

        Returns:
            dict: Dictionary containing training metrics and loss for the epoch.
        """
        profile_training = conf["trainer"].get("profile_training", False)

        batches_per_epoch = conf["trainer"]["batches_per_epoch"]
        grad_max_norm = conf["trainer"].get("grad_max_norm", 0.0)
        amp = conf["trainer"]["amp"]
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False
        forecast_length = conf["data"]["forecast_len"]
        ensemble_size = conf["trainer"].get("ensemble_size", 1)
        if ensemble_size > 1:
            logger.info(f"ensemble training with ensemble_size {ensemble_size}")
        distributed_ensemble_mode = conf["trainer"]["type"] == "goes10km-distributed-ensemble"
        if distributed_ensemble_mode:
            logger.info("using distributed ensemble training mode")
        # in distributed ensemble mode, the effective ensemble size computed by the loss is ensemble_size * num_gpus
        
        logger.info(f"Using grad-max-norm value: {grad_max_norm}")

        # number of diagnostic variables
        # varnum_diag = len(conf["data"]["diagnostic_variables"])

        # # number of dynamic forcing + forcing + static
        # static_dim_size = (
        #     len(conf["data"]["dynamic_forcing_variables"])
        #     + len(conf["data"]["forcing_variables"])
        #     + len(conf["data"]["static_variables"])
        # )

        # [Optional] retain graph for multiple backward passes
        retain_graph = conf["data"].get("retain_graph", False)

        # update the learning rate if epoch-by-epoch updates that dont depend on a metric
        if (
            conf["trainer"]["use_scheduler"]
            and conf["trainer"]["scheduler"]["scheduler_type"] == "lambda"
        ):
            scheduler.step()

        # ------------------------------------------------------- #
        # clamp to remove outliers
        if conf["data"]["data_clamp"] is None:
            flag_clamp = False
        else:
            flag_clamp = True
            clamp_min = float(conf["data"]["data_clamp"][0])
            clamp_max = float(conf["data"]["data_clamp"][1])


        # set up a custom tqdm
        if not isinstance(trainloader.dataset, IterableDataset):
            # Check if the dataset has its own batches_per_epoch method
            if hasattr(trainloader.dataset, "batches_per_epoch"):
                dataset_batches_per_epoch = trainloader.dataset.batches_per_epoch()
            elif hasattr(trainloader.sampler, "batches_per_epoch"):
                dataset_batches_per_epoch = trainloader.sampler.batches_per_epoch()
            else:
                dataset_batches_per_epoch = len(trainloader)
                logger.info(f"len trainloader {dataset_batches_per_epoch}")
            # Use the user-given number if not larger than the dataset
            batches_per_epoch = (
                batches_per_epoch
                if 0 < batches_per_epoch < dataset_batches_per_epoch
                else dataset_batches_per_epoch
            )
        logger.debug(f"batches_per_epoch: {batches_per_epoch}")


        self.model.train()

        dl = iter(trainloader)
        results_dict = defaultdict(list)

        batch_group_generator = tqdm.tqdm(
            range(batches_per_epoch), total=batches_per_epoch, leave=True
        )

        for steps in range(batches_per_epoch):
            logs = {}
            loss = 0.
            stop_forecast = False
            y_pred = None  # Place holder that gets updated after first roll-out

            start = time.time()
            batch = next(dl)
            logger.debug(batch["datetime"])
            mode = batch["mode"][0]
            while not stop_forecast:
                logger.debug(f"current mode: {mode}")
                logger.debug(batch.keys())
                
                if "era5" in batch.keys():
                    batch_era5 = batch["era5"]

                if mode == "init":
                    x = batch["x"].to(self.device).float()

                    if ensemble_size > 1:
                        x = torch.repeat_interleave(x, ensemble_size, 0)
                    # --------------------------------------------- #
                    # ensemble x on initialization
                    # copies each sample in the batch ensemble_size number of times.
                    # if samples in the batch are ordered (x,y,z) then the result tensor is (x, x, ..., y, y, ..., z,z ...)
                    # WARNING: needs to be used with a loss that can handle x with b * ensemble_size samples and y with b samples
                
                    if "era5" in batch.keys():
                        era5_static = batch_era5["static"].to(self.device)

                if flag_clamp:
                    x = torch.clamp(x, min=clamp_min, max=clamp_max)
                # load era5 forcing
                # concat order is prognostic, static, forcing
                if "era5" in batch.keys():
                    x_era5 = torch.concat([batch_era5["prognostic"].to(self.device),
                                           era5_static.detach().clone(),
                                           batch_era5["dynamic_forcing"].to(self.device)],
                                           dim=1).float()
                    forcing_t_delta = batch_era5["timedelta_seconds"].to(self.device).float()

                    if ensemble_size > 1:
                        x_era5 = torch.repeat_interleave(x_era5, ensemble_size, 0)
                        forcing_t_delta = torch.repeat_interleave(forcing_t_delta, ensemble_size, 0)
                    if flag_clamp:
                        x_era5 = torch.clamp(x_era5, min=clamp_min, max=clamp_max)

                
                xload_time = time.time()

                with torch.autocast(device_type="cuda", enabled=amp):
                    y_pred = self.model(x, x_era5=x_era5, forcing_t_delta=forcing_t_delta) if "era5" in batch.keys() else self.model(x)

                batch = next(dl)
                mode = batch["mode"][0]
                logger.debug(f"current mode: {mode}")

                # only load y-truth data if we intend to backprop (default is every step gets grads computed
                if mode == "y" or mode == "stop":
                    # calculate rolling loss
                    y = batch["y"].to(self.device)
                    # --------------------------------------------- #
                    # clamp
                    if flag_clamp:     
                        y = torch.clamp(y, min=clamp_min, max=clamp_max)

                    # compute loss
                    with torch.autocast(enabled=amp, device_type="cuda"):
                        y = y.to(y_pred.dtype)
                        if not distributed_ensemble_mode:
                            total_loss = criterion(y, y_pred).mean()
                        else: # distributed ensemble mode
                            lat_size = y.shape[3]
                            batch_size = y.shape[0] // ensemble_size
                            world_size = dist.get_world_size()

                            total_loss = 0
                            total_std = 0
                            for i in range(lat_size):
                                # Slice the tensors
                                y_pred_slice = y_pred[:, :, :, i : i + 1].contiguous()
                                y_slice = y[:, :, :, i : i + 1].contiguous()

                                # Gather the tensor
                                # gather concats on dim=0, so the gathered tensor is
                                # x1, x1, y1, y1, z1, z1, ..., x2, x2, y2, y2, z2, z2
                                y_pred_slice = gather_tensor(y_pred_slice)
                                y_pred_slice = y_pred_slice.view(world_size, batch_size, ensemble_size, *y_pred_slice.shape[1:])
                                y_pred_slice = y_pred_slice.permute(1,2,0, *range(3, y_pred_slice.ndim))
                                y_pred_slice = y_pred_slice.reshape(ensemble_size * batch_size * world_size, *y_pred_slice.shape[3:])
                                # gather ensemble members from across the ranks and reorders them
                                # Compute loss for this slice

                                loss = criterion(y_slice.to(y_pred_slice.dtype), y_pred_slice).mean() / lat_size
                                total_loss += loss

                                # Compute the std over ensemble dim
                                std = (y_pred_slice
                                       .detach()
                                       .view(batch_size, ensemble_size * world_size, *y_pred_slice.shape[1:])
                                       .std(dim=1)
                                       .mean() / lat_size
                                )
                                total_std += std
                            # track std
                            accum_log(logs, {"std": total_std.item()})

                    # track the loss
                    accum_log(logs, {"loss": total_loss.item()})

                    # compute gradients
                    scaler.scale(total_loss).backward(retain_graph=retain_graph)

                if distributed:
                    torch.distributed.barrier()

                # Discard current computational graph, which still 
                # exists (through y_pred reference) if `forecast_step` not in `backprop_on_timestep`
                if not retain_graph:
                    y_pred = y_pred.detach()

                # stop after X steps
                if mode == "stop":
                    stop_forecast = True
                    break
                else:
                    x = y_pred
                

            if distributed:
                torch.distributed.barrier()

            # Grad norm clipping
            scaler.unscale_(optimizer)
            if grad_max_norm == "dynamic":
                # Compute local L2 norm
                local_norm = torch.norm(
                    torch.stack(
                        [
                            p.grad.detach().norm(2)
                            for p in self.model.parameters()
                            if p.grad is not None
                        ]
                    )
                )

                # All-reduce to get global norm across ranks
                if distributed:
                    dist.all_reduce(local_norm, op=dist.ReduceOp.SUM)
                global_norm = local_norm.sqrt()  # Compute total global norm

                # Clip gradients using the global norm
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=global_norm
                )
            elif grad_max_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=grad_max_norm
                )

            # Step optimizer
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if profile_training and self.rank==0:
                end_time = time.time()
                load_time = xload_time - start
                fwd_time = end_time - xload_time
                logger.info(f'''load/fwd time: {load_time:.2f}s/{fwd_time:.2f}s = {load_time/fwd_time:.1}''')
                
                allocated_mem = torch.cuda.memory.max_memory_allocated(self.device) / 1024**3
                reserved_mem = torch.cuda.memory.max_memory_reserved(self.device) / 1024**3
                logger.info(f'''Memory allocated: {allocated_mem} GB''')
                logger.info(f'''Memory reserved: {reserved_mem} GB''')

            # Metrics
            # if distributed_ensemble_mode=False ensemble metrics computed by metrics here
            metrics_dict = metrics(y_pred, y)
            for name, value in metrics_dict.items():
                if name in ["std", "loss"] and distributed_ensemble_mode:
                    pass
                else:
                    value = torch.Tensor([value])
                    value = value.to(self.device, non_blocking=True)
                    if distributed:
                        dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                    results_dict[f"train_{name}"].append(value[0].item())

            batch_loss = torch.Tensor([logs["loss"]])
            batch_loss = batch_loss.to(self.device)
            if distributed:
                dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
            results_dict["train_loss"].append(batch_loss[0].item())
            results_dict["train_forecast_len"].append(forecast_length + 1)

            # TODO: get per channel std and loss for the entire distributed ensemble
            if distributed_ensemble_mode:
                batch_std = torch.Tensor([logs["std"]]).to(self.device)
                dist.all_reduce(batch_std, dist.ReduceOp.AVG, async_op=False)
                results_dict["train_std"].append(batch_std[0].item())

            if not np.isfinite(np.mean(results_dict["train_loss"])):
                print(
                    results_dict["train_loss"],
                    batch["x"].shape,
                    batch["y"].shape,
                    batch["index"],
                )
                try:
                    raise optuna.TrialPruned()
                except Exception as E:
                    raise E

            # agg the results
            to_print = "Epoch: {} train_loss: {:.6f} train_acc: {:.6f} train_mae: {:.6f} forecast_len: {:.6f}".format(
                epoch,
                np.mean(results_dict["train_loss"]),
                np.mean(results_dict["train_acc"]),
                np.mean(results_dict["train_mae"]),
                forecast_length + 1,
            )
            if ensemble_size > 1 or distributed_ensemble_mode:
                to_print += f" std: {np.mean(results_dict['train_std']):.6f}"
            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]["lr"])


            if self.rank == 0: #update tqdm
                batch_group_generator.update(1)
                batch_group_generator.set_description(to_print)

            if (
                conf["trainer"]["use_scheduler"]
                and conf["trainer"]["scheduler"]["scheduler_type"] in update_on_batch
            ):
                scheduler.step()

            logger.debug(f"forward/backward/optimizer time, {time.time() - start}s")


        #  Shutdown the progbar
        batch_group_generator.close()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict

    def validate(self, epoch, conf, valid_loader, criterion, metrics):
        """
        Validates the model on the validation dataset.

        Args:
            epoch (int): Current epoch number.
            conf (dict): Configuration dictionary containing validation settings.
            valid_loader (DataLoader): DataLoader for the validation dataset.
            criterion (callable): Loss function used for validation.
            metrics (callable): Function to compute metrics for evaluation.

        Returns:
            dict: Dictionary containing validation metrics and loss for the epoch.
        """
        
        self.model.eval()
        amp = conf["trainer"]["amp"]

        profile_training = conf["trainer"].get("profile_training", False)

        # number of diagnostic variables
        # varnum_diag = len(conf["data"]["diagnostic_variables"])

        # # number of dynamic forcing + forcing + static
        # static_dim_size = (
        #     len(conf["data"]["dynamic_forcing_variables"])
        #     + len(conf["data"]["forcing_variables"])
        #     + len(conf["data"]["static_variables"])
        # )

        valid_batches_per_epoch = conf["trainer"]["valid_batches_per_epoch"]
        history_len = (
            conf["data"]["valid_history_len"]
            if "valid_history_len" in conf["data"]
            else conf["history_len"]
        )
        forecast_len = (
            conf["data"]["valid_forecast_len"]
            if "valid_forecast_len" in conf["data"]
            else conf["forecast_len"]
        )
        ensemble_size = conf["trainer"].get("ensemble_size", 1)
        distributed_ensemble_mode = conf["trainer"]["type"] == "goes10km-distributed-ensemble"
        if distributed_ensemble_mode:
            logger.info("using distributed ensemble training mode")
        # in distributed ensemble mode, the effective ensemble size computed by the loss is ensemble_size * num_gpus

        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False

        results_dict = defaultdict(list)

        # set up a custom tqdm
        if not isinstance(valid_loader.dataset, IterableDataset):
            # Check if the dataset has its own batches_per_epoch method
            if hasattr(valid_loader.dataset, "batches_per_epoch"):
                dataset_batches_per_epoch = valid_loader.dataset.batches_per_epoch()
            elif hasattr(valid_loader.sampler, "batches_per_epoch"):
                dataset_batches_per_epoch = valid_loader.sampler.batches_per_epoch()
            else:
                dataset_batches_per_epoch = len(valid_loader)
            # Use the user-given number if not larger than the dataset
            valid_batches_per_epoch = (
                valid_batches_per_epoch
                if 0 < valid_batches_per_epoch < dataset_batches_per_epoch
                else dataset_batches_per_epoch
            )

        # ------------------------------------------------------- #
        # clamp to remove outliers
        if conf["data"]["data_clamp"] is None:
            flag_clamp = False
        else:
            flag_clamp = True
            clamp_min = float(conf["data"]["data_clamp"][0])
            clamp_max = float(conf["data"]["data_clamp"][1])

        # ====================================================== #

        batch_group_generator = tqdm.tqdm(
            range(valid_batches_per_epoch), total=valid_batches_per_epoch, leave=True
        )

        stop_forecast = False
        dl = cycle(valid_loader)
        with torch.no_grad():
            for steps in range(valid_batches_per_epoch):
                loss = 0.
                stop_forecast = False
                y_pred = None  # Place holder that gets updated after first roll-out

                batch = next(dl)

                mode = batch["mode"][0]
                while not stop_forecast:
                    logger.debug(f"current mode: {mode}")
                    logger.debug(batch.keys())
                    
                    if "era5" in batch.keys():
                        batch_era5 = batch["era5"]

                    if mode == "init":
                        x = batch["x"].to(self.device).float()

                        if ensemble_size > 1:
                            x = torch.repeat_interleave(x, ensemble_size, 0)
                        # --------------------------------------------- #
                        # ensemble x on initialization
                        # copies each sample in the batch ensemble_size number of times.
                        # if samples in the batch are ordered (x,y,z) then the result tensor is (x, x, ..., y, y, ..., z,z ...)
                        # WARNING: needs to be used with a loss that can handle x with b * ensemble_size samples and y with b samples
                    
                        if "era5" in batch.keys():
                            era5_static = batch_era5["static"].to(self.device)

                        
                    # add era5 forcing to the tensor
                    # concat order is prognostic, static, forcing
                    if "era5" in batch.keys():
                        x_era5 = torch.concat([batch_era5["prognostic"].to(self.device),
                                            era5_static,
                                            batch_era5["dynamic_forcing"].to(self.device)],
                                            dim=1).float()
                        forcing_t_delta = batch_era5["timedelta_seconds"].to(self.device).float()

                        if ensemble_size > 1:
                            x_era5 = torch.repeat_interleave(x_era5, ensemble_size, 0)
                            forcing_t_delta = torch.repeat_interleave(forcing_t_delta, ensemble_size, 0)

                    
                    
                    # --------------------------------------------- #
                    # clamp
                    if flag_clamp:
                        x = torch.clamp(x, min=clamp_min, max=clamp_max)

                    with torch.autocast(device_type="cuda", enabled=amp):
                        y_pred = self.model(x, x_era5=x_era5, forcing_t_delta=forcing_t_delta) if "era5" in batch.keys() else self.model(x)

                    # ================================================================================== #
                    # scope of reaching the final forecast_len

                    batch = next(dl)
                    mode = batch["mode"][0]

                    # only load y-truth data if we intend to backprop (default is every step gets grads computed
                    if mode == "stop":
                        stop_forecast = True
                    # ----------------------------------------------------------------------- #
                    # creating `y` tensor for loss compute
                        y = batch["y"].to(self.device)

                        # --------------------------------------------- #
                        # clamp
                        if flag_clamp:
                            y = torch.clamp(y, min=clamp_min, max=clamp_max)

                        # ----------------------------------------------------------------------- #
                        # calculate rolling loss
                        with torch.autocast(enabled=amp, device_type="cuda"):
                            if not distributed_ensemble_mode:
                                total_loss = criterion(y, y_pred).mean()
                            else: # distributed ensemble mode
                                lat_size = y.shape[3]
                                batch_size = y.shape[0] // ensemble_size
                                world_size = dist.get_world_size()

                                total_loss = 0
                                total_std = 0
                                for i in range(lat_size):
                                    # Slice the tensors
                                    y_pred_slice = y_pred[:, :, :, i : i + 1].contiguous()
                                    y_slice = y[:, :, :, i : i + 1].contiguous()

                                    # Gather the tensor
                                    # gather concats on dim=0, so the gathered tensor is
                                    # x1, x1, y1, y1, z1, z1, ..., x2, x2, y2, y2, z2, z2
                                    y_pred_slice = gather_tensor(y_pred_slice)
                                    y_pred_slice = y_pred_slice.view(world_size, batch_size, ensemble_size, *y_pred_slice.shape[1:])
                                    y_pred_slice = y_pred_slice.permute(1,2,0, *range(3, y_pred_slice.ndim))
                                    y_pred_slice = y_pred_slice.reshape(ensemble_size * batch_size * world_size, *y_pred_slice.shape[3:])
                                    # gather ensemble members from across the ranks and reorders them
                                    # Compute loss for this slice

                                    loss = criterion(y_slice.to(y_pred_slice.dtype), y_pred_slice).mean() / lat_size
                                    total_loss += loss

                                    # Compute the std over ensemble dim
                                    std = (y_pred_slice
                                        .detach()
                                        .view(batch_size, ensemble_size * world_size, *y_pred_slice.shape[1:])
                                        .std(dim=1)
                                        .mean() / lat_size
                                    )
                                    total_std += std

                        # Metrics
                        # metrics_dict = metrics(y_pred, y.float)
                        metrics_dict = metrics(y_pred.float(), y.float())

                        for name, value in metrics_dict.items():
                            value = torch.Tensor([value]).to(
                                self.device, non_blocking=True
                            )

                            if distributed:
                                dist.all_reduce(
                                    value, dist.ReduceOp.AVG, async_op=False
                                )

                            results_dict[f"valid_{name}"].append(value[0].item())

                        assert mode == "stop"
                        break  # stop after X steps
                    else:
                        x = y_pred.detach()
                if profile_training and self.rank==0:
                    
                    allocated_mem = torch.cuda.memory.max_memory_allocated(self.device) / 1024**3
                    reserved_mem = torch.cuda.memory.max_memory_reserved(self.device) / 1024**3
                    logger.info(f'''Memory allocated: {allocated_mem} GB''')
                    logger.info(f'''Memory reserved: {reserved_mem} GB''')
                        

                batch_loss = torch.Tensor([total_loss.item()]).to(self.device)

                if distributed:
                    torch.distributed.barrier()

                results_dict["valid_loss"].append(batch_loss[0].item())
                results_dict["valid_forecast_len"].append(forecast_len + 1)

                stop_forecast = False

                # print to tqdm
                to_print = "Epoch: {} valid_loss: {:.6f} valid_acc: {:.6f} valid_mae: {:.6f}".format(
                    epoch,
                    np.mean(results_dict["valid_loss"]),
                    np.mean(results_dict["valid_acc"]),
                    np.mean(results_dict["valid_mae"]),
                )
                ensemble_size = conf["trainer"].get("ensemble_size", 0)
                if ensemble_size > 1:
                    to_print += f" std: {np.mean(results_dict['valid_std']):.6f}"
                if self.rank == 0:
                    batch_group_generator.update(1)
                    batch_group_generator.set_description(to_print)

        # Shutdown the progbar
        batch_group_generator.close()

        # Wait for rank-0 process to save the checkpoint above
        if distributed:
            torch.distributed.barrier()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict
