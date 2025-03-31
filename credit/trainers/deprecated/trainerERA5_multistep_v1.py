import gc
import logging
from collections import defaultdict

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
import optuna

import os
import pandas as pd
import torch
from credit.models.checkpoint import TorchFSDPCheckpointIO
from credit.scheduler import update_on_epoch
from credit.trainers.utils import cleanup


class Trainer(BaseTrainer):
    def __init__(self, model: torch.nn.Module, rank: int):
        super().__init__(model, rank)
        # Add any additional initialization if needed
        logging.info("Loading a multi-step trainer class")

    # Training function.
    def train_one_epoch(
        self,
        epoch,
        conf,
        trainloader,
        optimizer,
        criterion,
        scaler,
        scheduler,
        metrics,
        forecast_length=0,
    ):
        batches_per_epoch = conf["trainer"]["batches_per_epoch"]
        amp = conf["trainer"]["amp"]
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False

        # update the learning rate if epoch-by-epoch updates that dont depend on a metric
        if (
            conf["trainer"]["use_scheduler"]
            and conf["trainer"]["scheduler"]["scheduler_type"] == "lambda"
        ):
            scheduler.step()

        # set up a custom tqdm
        if isinstance(trainloader.dataset, IterableDataset):
            # we sample forecast termination with probability p during training
            # trainloader.dataset.set_rollout_prob(rollout_p)
            pass
        else:
            batches_per_epoch = (
                batches_per_epoch
                if 0 < batches_per_epoch < len(trainloader)
                else len(trainloader)
            )

        batch_group_generator = tqdm.tqdm(
            range(batches_per_epoch), total=batches_per_epoch, leave=True
        )

        self.model.train()

        dl = cycle(trainloader)

        static = None
        results_dict = defaultdict(list)

        for steps in range(batches_per_epoch):
            logs = {}
            loss = 0
            stop_forecast = False
            y_pred = None  # Place holder that gets updated after first roll-out

            with autocast(enabled=amp):
                while not stop_forecast:
                    batch = next(dl)

                    for i, forecast_hour in enumerate(batch["forecast_hour"]):
                        if (
                            forecast_hour == 0
                        ):  # use true x -- initial condition time-step
                            x_atmo = batch["x"]
                            x_surf = batch["x_surf"]
                            x = self.model.concat_and_reshape(x_atmo, x_surf).to(
                                self.device
                            )
                        else:  # use model's predictions
                            if x.shape[2] > 1:
                                # discard any statics from x here as they will get added below from batch
                                print(i, forecast_hour)
                                x_detach = x[:, :, 1:].detach()
                                atmos_vars = y_pred.shape[1]
                                x = torch.cat(
                                    [x_detach[:, :atmos_vars], y_pred.detach()], dim=2
                                )
                            else:
                                x = y_pred.detach()

                        if "static" in batch:
                            if static is None:
                                static = (
                                    batch["static"]
                                    .to(self.device)
                                    .unsqueeze(2)
                                    .expand(-1, -1, x.shape[2], -1, -1)
                                    .float()
                                )  # [batch, num_stat_vars, hist_len, lat, lon]
                            x = torch.cat((x, static.clone()), dim=1)

                        if "TOA" in batch:
                            toa = batch["TOA"].to(self.device)
                            if x.shape[2] == 1:  # Sequential with time = 1
                                toa = toa.unsqueeze(1).unsqueeze(1)
                            elif x.shape[2] == 2:  # Sequential with time = 2
                                toa = toa.unsqueeze(1)
                            x = torch.cat([x, toa], dim=1)

                        # predict with the model
                        y_pred = self.model(x)

                        # calculate rolling loss
                        y_atmo = batch["y"]
                        y_surf = batch["y_surf"]

                        y = self.model.concat_and_reshape(y_atmo, y_surf).to(
                            self.device
                        )

                        loss = criterion(y.to(y_pred.dtype), y_pred).mean()

                        # compute gradients
                        scaler.scale(loss).backward()

                        if distributed:
                            torch.distributed.barrier()

                        # stop after X steps
                        stop_forecast = batch["stop_forecast"][i]

                    if stop_forecast:
                        break

                # scale, accumulate, backward
                # scaler.scale(loss).backward()
                accum_log(logs, {"loss": loss.item()})

                if distributed:
                    torch.distributed.barrier()

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Metrics
                metrics_dict = metrics(y_pred.float(), y.float())
                for name, value in metrics_dict.items():
                    value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                    if distributed:
                        dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                    results_dict[f"train_{name}"].append(value[0].item())

            batch_loss = torch.Tensor([logs["loss"]]).cuda(self.device)
            if distributed:
                dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
            results_dict["train_loss"].append(batch_loss[0].item())
            results_dict["train_forecast_len"].append(forecast_length + 1)

            if not np.isfinite(np.mean(results_dict["train_loss"])):
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
            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]["lr"])
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

        return results_dict

    def validate(self, epoch, conf, valid_loader, criterion, metrics):
        self.model.eval()

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
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False

        results_dict = defaultdict(list)

        # set up a custom tqdm
        if isinstance(valid_loader.dataset, IterableDataset):
            valid_batches_per_epoch = valid_batches_per_epoch
        else:
            valid_batches_per_epoch = (
                valid_batches_per_epoch
                if 0 < valid_batches_per_epoch < len(valid_loader)
                else len(valid_loader)
            )

        batch_group_generator = tqdm.tqdm(
            range(valid_batches_per_epoch), total=valid_batches_per_epoch, leave=True
        )

        static = None
        stop_forecast = False
        with torch.no_grad():
            for k, batch in enumerate(valid_loader):
                y_pred = None  # Place holder that gets updated after first roll-out
                for _, i in enumerate(batch["forecast_hour"]):
                    if i == 0:  # use true x -- initial condition time-step
                        x_atmo = batch["x"]
                        x_surf = batch["x_surf"]
                        x = self.model.concat_and_reshape(x_atmo, x_surf).to(
                            self.device
                        )
                    else:  # use model's predictions
                        if x.shape[2] > 1:
                            # discard any statics from x here as they will get added below from batch
                            x_detach = x[:, :, 1:].detach()
                            atmos_vars = y_pred.shape[1]
                            x = torch.cat(
                                [x_detach[:, :atmos_vars], y_pred.detach()], dim=2
                            )
                        else:
                            x = y_pred.detach()

                    if "static" in batch:
                        if static is None:
                            static = (
                                batch["static"]
                                .to(self.device)
                                .unsqueeze(2)
                                .expand(-1, -1, x.shape[2], -1, -1)
                                .float()
                            )  # [batch, num_stat_vars, hist_len, lat, lon]
                        x = torch.cat((x, static.clone()), dim=1)

                    if "TOA" in batch:
                        toa = batch["TOA"].to(self.device)
                        if x.shape[2] == 1:  # Sequential with time = 1
                            toa = toa.unsqueeze(1).unsqueeze(1)
                        elif x.shape[2] == 2:  # Sequential with time = 2
                            toa = toa.unsqueeze(1)
                        x = torch.cat([x, toa], dim=1)

                    y_pred = self.model(x)

                    # stop after user-defined number of steps
                    if i == forecast_len:
                        y_atmo = batch["y"]
                        y_surf = batch["y_surf"]
                        y = self.model.concat_and_reshape(y_atmo, y_surf).to(
                            self.device
                        )

                        loss = criterion(y.to(y_pred.dtype), y_pred).mean()

                        # Metrics
                        metrics_dict = metrics(y_pred.float(), y.float())
                        for name, value in metrics_dict.items():
                            value = torch.Tensor([value]).cuda(
                                self.device, non_blocking=True
                            )
                            if distributed:
                                dist.all_reduce(
                                    value, dist.ReduceOp.AVG, async_op=False
                                )
                            results_dict[f"valid_{name}"].append(value[0].item())
                        stop_forecast = True
                        break

                if not stop_forecast:
                    continue

                batch_loss = torch.Tensor([loss.item()]).cuda(self.device)
                if distributed:
                    torch.distributed.barrier()
                results_dict["valid_loss"].append(batch_loss[0].item())
                stop_forecast = False

                # print to tqdm
                to_print = "Epoch: {} valid_loss: {:.6f} valid_acc: {:.6f} valid_mae: {:.6f}".format(
                    epoch,
                    np.mean(results_dict["valid_loss"]),
                    np.mean(results_dict["valid_acc"]),
                    np.mean(results_dict["valid_mae"]),
                )
                if self.rank == 0:
                    batch_group_generator.update(1)
                    batch_group_generator.set_description(to_print)

                if k // history_len >= valid_batches_per_epoch and k > 0:
                    break

        # Shutdown the progbar
        batch_group_generator.close()

        # Wait for rank-0 process to save the checkpoint above
        if distributed:
            torch.distributed.barrier()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict

    def fit(
        self,
        conf,
        train_loader,
        valid_loader,
        optimizer,
        train_criterion,
        valid_criterion,
        scaler,
        scheduler,
        metrics,
        rollout_scheduler=None,
        trial=False,
    ):
        save_loc = conf["save_loc"]
        start_epoch = conf["trainer"]["start_epoch"]
        epochs = conf["trainer"]["epochs"]
        skip_validation = (
            conf["trainer"]["skip_validation"]
            if "skip_validation" in conf["trainer"]
            else False
        )

        # Reload the results saved in the training csv if continuing to train
        if start_epoch == 0:
            results_dict = defaultdict(list)
        else:
            results_dict = defaultdict(list)
            saved_results = pd.read_csv(f"{save_loc}/training_log.csv")
            for key in saved_results.columns:
                if key == "index":
                    continue
                results_dict[key] = list(saved_results[key])

        for epoch in range(start_epoch, epochs):
            logging.info(f"Beginning epoch {epoch}")

            if not isinstance(train_loader.dataset, IterableDataset):
                train_loader.sampler.set_epoch(epoch)
            else:
                train_loader.dataset.set_epoch(epoch)
                # if rollout_scheduler is not None:
                #     conf['trainer']['stop_rollout'] = rollout_scheduler(epoch, epochs)
                #     train_loader.dataset.set_rollout_prob(conf['trainer']['stop_rollout'])

            ############
            #
            # Train
            #
            ############

            train_results = self.train_one_epoch(
                epoch,
                conf,
                train_loader,
                optimizer,
                train_criterion,
                scaler,
                scheduler,
                metrics,
                conf["data"]["forecast_len"],
            )

            ############
            #
            # Validation
            #
            ############

            if skip_validation:
                valid_results = train_results

            else:
                valid_results = self.validate(
                    epoch, conf, valid_loader, valid_criterion, metrics
                )

            #################
            #
            # Save results
            #
            #################

            # update the learning rate if epoch-by-epoch updates

            if (
                conf["trainer"]["use_scheduler"]
                and conf["trainer"]["scheduler"]["scheduler_type"] in update_on_epoch
            ):
                if conf["trainer"]["scheduler"]["scheduler_type"] == "plateau":
                    scheduler.step(results_dict["valid_acc"][-1])
                else:
                    scheduler.step()

            # Put things into a results dictionary -> dataframe

            results_dict["epoch"].append(epoch)
            for name in ["loss", "acc", "mae"]:
                results_dict[f"train_{name}"].append(
                    np.mean(train_results[f"train_{name}"])
                )
                results_dict[f"valid_{name}"].append(
                    np.mean(valid_results[f"valid_{name}"])
                )
            results_dict["train_forecast_len"].append(
                np.mean(train_results["train_forecast_len"])
            )
            results_dict["lr"].append(optimizer.param_groups[0]["lr"])

            df = pd.DataFrame.from_dict(results_dict).reset_index()

            # Save the dataframe to disk

            if trial:
                df.to_csv(
                    os.path.join(
                        f"{save_loc}",
                        "trial_results",
                        f"training_log_{trial.number}.csv",
                    ),
                    index=False,
                )
            else:
                df.to_csv(os.path.join(f"{save_loc}", "training_log.csv"), index=False)

            ############
            #
            # Checkpoint
            #
            ############

            if not trial:
                if conf["trainer"]["mode"] != "fsdp":
                    if self.rank == 0:
                        # Save the current model

                        logging.info(
                            f"Saving model, optimizer, grad scaler, and learning rate scheduler states to {save_loc}"
                        )

                        state_dict = {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict()
                            if conf["trainer"]["use_scheduler"]
                            else None,
                            "scaler_state_dict": scaler.state_dict(),
                        }
                        torch.save(state_dict, f"{save_loc}/checkpoint.pt")

                else:
                    logging.info(
                        f"Saving FSDP model, optimizer, grad scaler, and learning rate scheduler states to {save_loc}"
                    )

                    # Initialize the checkpoint I/O handler

                    checkpoint_io = TorchFSDPCheckpointIO()

                    # Save model and optimizer checkpoints

                    checkpoint_io.save_unsharded_model(
                        self.model,
                        os.path.join(save_loc, "model_checkpoint.pt"),
                        gather_dtensor=True,
                        use_safetensors=False,
                        rank=self.rank,
                    )
                    checkpoint_io.save_unsharded_optimizer(
                        optimizer,
                        os.path.join(save_loc, "optimizer_checkpoint.pt"),
                        gather_dtensor=True,
                        rank=self.rank,
                    )

                    # Still need to save the scheduler and scaler states, just in another file for FSDP

                    state_dict = {
                        "epoch": epoch,
                        "scheduler_state_dict": scheduler.state_dict()
                        if conf["trainer"]["use_scheduler"]
                        else None,
                        "scaler_state_dict": scaler.state_dict(),
                    }

                    torch.save(state_dict, os.path.join(save_loc, "checkpoint.pt"))

                # This needs updated!
                # valid_loss = np.mean(valid_results["valid_loss"])
                # # save if this is the best model seen so far
                # if (self.rank == 0) and (np.mean(valid_loss) == min(results_dict["valid_loss"])):
                #     if conf["trainer"]["mode"] == "ddp":
                #         shutil.copy(f"{save_loc}/checkpoint_{self.device}.pt", f"{save_loc}/best_{self.device}.pt")
                #     elif conf["trainer"]["mode"] == "fsdp":
                #         if os.path.exists(f"{save_loc}/best"):
                #             shutil.rmtree(f"{save_loc}/best")
                #         shutil.copytree(f"{save_loc}/checkpoint", f"{save_loc}/best")
                #     else:
                #         shutil.copy(f"{save_loc}/checkpoint.pt", f"{save_loc}/best.pt")

            # clear the cached memory from the gpu
            torch.cuda.empty_cache()
            gc.collect()

            training_metric = "train_loss" if skip_validation else "valid_loss"

            # Stop training if we have not improved after X epochs (stopping patience)
            best_epoch = [
                i
                for i, j in enumerate(results_dict[training_metric])
                if j == min(results_dict[training_metric])
            ][0]
            offset = epoch - best_epoch
            if offset >= conf["trainer"]["stopping_patience"]:
                logging.info(f"Trial {trial.number} is stopping early")
                break

            # Stop training if we get too close to the wall time
            if "stop_after_epoch" in conf["trainer"]:
                if conf["trainer"]["stop_after_epoch"]:
                    break

        training_metric = "train_loss" if skip_validation else "valid_loss"

        best_epoch = [
            i
            for i, j in enumerate(results_dict[training_metric])
            if j == min(results_dict[training_metric])
        ][0]

        result = {k: v[best_epoch] for k, v in results_dict.items()}

        if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
            cleanup()

        return result
