import gc
import logging
from collections import defaultdict
from datetime import datetime

import numpy as np
import optuna
import tqdm
import torch
import torch.fft
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch.utils.data import IterableDataset

from credit.output_downscaling import OutputWrangler
from credit.scheduler import update_on_batch
from credit.trainers.utils import cycle, accum_log
from credit.trainers.base_trainer import BaseTrainer
from credit.data import concat_and_reshape, reshape_only
# from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer
from credit.datasets.count_channels import count_channels

logger = logging.getLogger(__name__)


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
                (not yet updated for downscaling)

        """
        super().__init__(model, rank)
        # Add any additional initialization if needed
        logger.info("Loading a downscaling trainer class")

    ###################################

    # this stuff ought to move into init() rather than being rerun each time...
    def setup(self, conf):
        dconf = conf["data"]
        tconf = conf["trainer"]

        self.batches_per_epoch = tconf["batches_per_epoch"]
        self.grad_max_norm = tconf.get("grad_max_norm", 0.0)
        self.amp = tconf["amp"]
        self.distributed = True if tconf["mode"] in ["fsdp", "ddp"] else False

        self.ensemble_size = tconf.get("ensemble_size", 1)
        if self.ensemble_size > 1:
            logger.info(f"ensemble training with ensemble_size {self.ensemble_size}")
        logger.info(f"Using grad-max-norm value: {self.grad_max_norm}")


        self.forecast_length = dconf["forecast_len"]

        self.ccount = count_channels(conf)

        return

    ###################################

    # Training function
    def train_one_epoch(self,
                        epoch,
                        conf,
                        trainloader,
                        optimizer,
                        criterion,
                        scaler,
                        scheduler,
                        metrics,
                        ):

        self.setup(conf)

        # update the learning rate if epoch-by-epoch updates don't depend on a metric
        if(conf["trainer"]["use_scheduler"] and
           conf["trainer"]["scheduler"]["scheduler_type"] == "lambda"):
            scheduler.step()

        # setup custom tqdm progress meter

        if not isinstance(trainloader.dataset, IterableDataset):
            # Check if the dataset has its own batches_per_epoch method
            if hasattr(trainloader.dataset, "batches_per_epoch"):
                dataset_batches_per_epoch = trainloader.dataset.batches_per_epoch()
            elif hasattr(trainloader.sampler, "batches_per_epoch"):
                dataset_batches_per_epoch = trainloader.sampler.batches_per_epoch()
            else:
                dataset_batches_per_epoch = len(trainloader)
            # Use the user-given number if not larger than the dataset
            batches_per_epoch = (
                self.batches_per_epoch
                if 0 < self.batches_per_epoch < dataset_batches_per_epoch
                else dataset_batches_per_epoch
            )

        batch_group_generator = tqdm.tqdm(range(self.batches_per_epoch),
                                          total=self.batches_per_epoch,
                                          leave=True)

        self.model.train()

        dl= cycle(trainloader)
        results_dict = defaultdict(list)
        for steps in range(self.batches_per_epoch):
            logs = {}
            loss = 0

            y_pred = None     # placeholder for predicted values

            batch = next(dl)

            x = batch['x'].to(self.device)

            # if we were doing multistep training, everything from
            # this point to grad norm clipping would go into a loop
            # that iterated over multiple steps.  Check conus404_data
            # verison of this file before 2025-04-24 for details.

            # predict with the model
            with autocast(enabled=self.amp):
                y_pred = self.model(x)
                
            y = batch['y'].to(self.device)

            with autocast(enabled=self.amp):
                loss = criterion(y.to(y_pred.dtype), y_pred).mean()

            # track the loss
            accum_log(logs, {"loss": loss.item()})

            # compute gradients
            scaler.scale(loss).backward()

            if self.distributed:
                torch.distributed.barrier()

            # Grad norm clipping
            scaler.unscale_(optimizer)
            if self.grad_max_norm == "dynamic":
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
                if self.distributed:
                    dist.all_reduce(local_norm, op=dist.ReduceOp.SUM)
                global_norm = local_norm.sqrt()  # Compute total global norm

                # Clip gradients using the global norm
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=global_norm
                )
            elif self.grad_max_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_max_norm
                )

            # Step optimizer
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Metrics
            metrics_dict = metrics(y_pred, y)
            for name, value in metrics_dict.items():
                value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                if self.distributed:
                    dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                results_dict[f"train_{name}"].append(value[0].item())

            batch_loss = torch.Tensor([logs["loss"]]).cuda(self.device)
            if self.distributed:
                dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
            results_dict["train_loss"].append(batch_loss[0].item())
            results_dict["train_forecast_len"].append(self.forecast_length + 1)

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


            # aggregate the results
            to_print = (
                f"Epoch: {epoch}".format(epoch) +
                f" train_loss: {np.mean(results_dict['train_loss']):.6f}" +
                f" train_acc: {np.mean(results_dict['train_acc']):.6f}" +
                f" train_mae: {np.mean(results_dict['train_mae']):.6f}" +
                f" finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                # f" forecast_len: {self.forecast_length+1:.6f}"
            )

            ensemble_size = conf["trainer"].get("ensemble_size", 0)
            if ensemble_size > 1:
                to_print += f" std: {np.mean(results_dict['train_std']):.6f}"
            to_print += f" lr: {optimizer.param_groups[0]['lr']:.12f}"
            if self.rank == 0:
                batch_group_generator.update(1)
                batch_group_generator.set_description(to_print)

            if (
                conf["trainer"]["use_scheduler"]
                and conf["trainer"]["scheduler"]["scheduler_type"] in update_on_batch
            ):
                scheduler.step()

        #  Shutdown the progbar
        batch_group_generator.close()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        # write last training sample & prediction to file every so often
        if conf['trainer']['save_data']:
            saveconf = conf['trainer']['save_data']
            if epoch % saveconf['frequency'] == 0:
                wrangler = OutputWrangler(trainloader.dataset, **saveconf['output'])
                wrangler.process(batch['y'], batch['dates'], prefix=f"ep{epoch}.target")
                wrangler.process(y_pred.cpu().detach(), batch['dates'], prefix=f"ep{epoch}.predicted")

        return results_dict

    ###################################

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

        ## update me VV
        # number of diagnostic variables
        varnum_diag = len(conf["data"]["diagnostic_variables"])

        # number of dynamic forcing + forcing + static
        static_dim_size = (
            len(conf["data"]["dynamic_forcing_variables"])
            + len(conf["data"]["forcing_variables"])
            + len(conf["data"]["static_variables"])
        )
        # ^^

        valid_batches_per_epoch = conf["trainer"]["valid_batches_per_epoch"]
        history_len = conf["history_len"]
        forecast_len = conf["forecast_len"]

        # ensemble_size = conf["trainer"].get("ensemble_size", 1)

        #distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False

        results_dict = defaultdict(list)

        # begin common block 1
        # ------------------------------------------------------- #
        # clamp to remove outliers
        if conf["data"]["data_clamp"] is None:
            flag_clamp = False
        else:
            flag_clamp = True
            clamp_min = float(conf["data"]["data_clamp"][0])
            clamp_max = float(conf["data"]["data_clamp"][1])

        # ====================================================== #
        # postblock opts outside of model
        # skip all this for now

        # post_conf = conf["model"]["post_conf"]
        # flag_mass_conserve = False
        # flag_water_conserve = False
        # flag_energy_conserve = False
        #
        # if post_conf["activate"]:
        #     if post_conf["global_mass_fixer"]["activate"]:
        #         if post_conf["global_mass_fixer"]["activate_outside_model"]:
        #             logger.info("Activate GlobalMassFixer outside of model")
        #             flag_mass_conserve = True
        #             opt_mass = GlobalMassFixer(post_conf)
        #
        #     if post_conf["global_water_fixer"]["activate"]:
        #         if post_conf["global_water_fixer"]["activate_outside_model"]:
        #             logger.info("Activate GlobalWaterFixer outside of model")
        #             flag_water_conserve = True
        #             opt_water = GlobalWaterFixer(post_conf)
        #
        #     if post_conf["global_energy_fixer"]["activate"]:
        #         if post_conf["global_energy_fixer"]["activate_outside_model"]:
        #             logger.info("Activate GlobalEnergyFixer outside of model")
        #             flag_energy_conserve = True
        #             opt_energy = GlobalEnergyFixer(post_conf)
        #
        # ====================================================== #

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

        batch_group_generator = tqdm.tqdm(
            range(valid_batches_per_epoch), total=valid_batches_per_epoch, leave=True
        )
        # end common block 1

        stop_forecast = False
        dl = cycle(valid_loader)
        with torch.no_grad():
            for steps in range(valid_batches_per_epoch):
                loss = 0
                stop_forecast = False
                y_pred = None  # Place holder that gets updated after first roll-out
                while not stop_forecast:
                    batch = next(dl)
                    forecast_step = batch["forecast_step"].item()
                    stop_forecast = batch["stop_forecast"].item()
                    if forecast_step == 1:
                        # Initialize x and x_surf with the first time step
                        if "x_surf" in batch:
                            # combine x and x_surf
                            # input: (batch_num, time, var, level, lat, lon), (batch_num, time, var, lat, lon)
                            # output: (batch_num, var, time, lat, lon), 'x' first and then 'x_surf'
                            x = concat_and_reshape(batch["x"], batch["x_surf"]).to(
                                self.device
                            )  # .float()
                        else:
                            # no x_surf
                            x = reshape_only(batch["x"]).to(self.device)  # .float()
                        # --------------------------------------------- #
                        # ensemble x and x_surf on initialization
                        # copies each sample in the batch ensemble_size number of times.
                        # if samples in the batch are ordered (x,y,z) then the result tensor is (x, x, ..., y, y, ..., z,z ...)
                        # WARNING: needs to be used with a loss that can handle x with b * ensemble_size samples and y with b samples
                        ensemble_size = conf["trainer"].get("ensemble_size", 0)
                        if ensemble_size > 1:
                            x = torch.repeat_interleave(x, ensemble_size, 0)

                    # add forcing and static variables (regardless of fcst hours)
                    if "x_forcing_static" in batch:
                        # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                        x_forcing_batch = (
                            batch["x_forcing_static"]
                            .to(self.device)
                            .permute(0, 2, 1, 3, 4)
                        )  # .float()
                        # ---------------- ensemble ----------------- #
                        # ensemble x_forcing_batch for concat. see above for explanation of code
                        if ensemble_size > 1:
                            x_forcing_batch = torch.repeat_interleave(
                                x_forcing_batch, ensemble_size, 0
                            )
                        # --------------------------------------------- #

                        # concat on var dimension
                        x = torch.cat((x, x_forcing_batch), dim=1)

                    # --------------------------------------------- #
                    # clamp
                    if flag_clamp:
                        x = torch.clamp(x, min=clamp_min, max=clamp_max)

                    y_pred = self.model(x)

                    # ============================================= #
                    # postblock opts outside of model

                    # # backup init state
                    # if flag_mass_conserve:
                    #     if forecast_step == 1:
                    #         x_init = x.clone()
                    #
                    # # mass conserve using initialization as reference
                    # if flag_mass_conserve:
                    #     input_dict = {"y_pred": y_pred, "x": x_init}
                    #     input_dict = opt_mass(input_dict)
                    #     y_pred = input_dict["y_pred"]
                    #
                    # # water conserve use previous step output as reference
                    # if flag_water_conserve:
                    #     input_dict = {"y_pred": y_pred, "x": x}
                    #     input_dict = opt_water(input_dict)
                    #     y_pred = input_dict["y_pred"]
                    #
                    # # energy conserve use previous step output as reference
                    # if flag_energy_conserve:
                    #     input_dict = {"y_pred": y_pred, "x": x}
                    #     input_dict = opt_energy(input_dict)
                    #     y_pred = input_dict["y_pred"]
                    # ============================================= #

                    # ================================================================================== #
                    # scope of reaching the final forecast_len
                    if forecast_step == (forecast_len + 1):
                        # ----------------------------------------------------------------------- #
                        # creating `y` tensor for loss compute
                        if "y_surf" in batch:
                            y = concat_and_reshape(batch["y"], batch["y_surf"]).to(
                                self.device
                            )
                        else:
                            y = reshape_only(batch["y"]).to(self.device)

                        if "y_diag" in batch:
                            # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                            y_diag_batch = (
                                batch["y_diag"].to(self.device).permute(0, 2, 1, 3, 4)
                            )  # .float()

                            # concat on var dimension
                            y = torch.cat((y, y_diag_batch), dim=1)

                        # --------------------------------------------- #
                        # clamp
                        if flag_clamp:
                            y = torch.clamp(y, min=clamp_min, max=clamp_max)

                        # ----------------------------------------------------------------------- #
                        # calculate rolling loss
                        loss = criterion(y.to(y_pred.dtype), y_pred).mean()

                        # Metrics
                        # metrics_dict = metrics(y_pred, y.float)
                        metrics_dict = metrics(y_pred.float(), y.float())

                        for name, value in metrics_dict.items():
                            value = torch.Tensor([value]).cuda(
                                self.device, non_blocking=True
                            )

                            if self.distributed:
                                dist.all_reduce(
                                    value, dist.ReduceOp.AVG, async_op=False
                                )

                            results_dict[f"valid_{name}"].append(value[0].item())

                        assert stop_forecast
                        break  # stop after X steps

                    # ================================================================================== #
                    # scope of keep rolling out

                    # step-in-step-out
                    elif history_len == 1:
                        # cut diagnostic vars from y_pred, they are not inputs
                        if "y_diag" in batch:
                            x = y_pred[:, :-varnum_diag, ...].detach()
                        else:
                            x = y_pred.detach()

                    # multi-step in
                    else:
                        if static_dim_size == 0:
                            x_detach = x[:, :, 1:, ...].detach()
                        else:
                            x_detach = x[:, :-static_dim_size, 1:, ...].detach()

                        # cut diagnostic vars from y_pred, they are not inputs
                        if "y_diag" in batch:
                            x = torch.cat(
                                [x_detach, y_pred[:, :-varnum_diag, ...].detach()],
                                dim=2,
                            )
                        else:
                            x = torch.cat([x_detach, y_pred.detach()], dim=2)

                batch_loss = torch.Tensor([loss.item()]).cuda(self.device)

                if self.distributed:
                    torch.distributed.barrier()

                results_dict["valid_loss"].append(batch_loss[0].item())
                results_dict["valid_forecast_len"].append(forecast_len + 1)

                # stop_forecast = False

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
        if self.distributed:
            torch.distributed.barrier()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict
