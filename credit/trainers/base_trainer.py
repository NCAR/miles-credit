"""
base_trainer.py
-------------------------------------------------------
Content:
    - Trainer
        - train_one_epoch
        - validate
        - fit
"""

import os
import gc
import shutil
import logging
from collections import defaultdict

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from collections import OrderedDict

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LRScheduler

from credit.models.checkpoint import TorchFSDPCheckpointIO, copy_checkpoint
from credit.scheduler import update_on_epoch
from credit.trainers.utils import cleanup

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    def __init__(self, model: torch.nn.Module, rank: int):
        """
        Abstract base class for training and validating machine learning models.

        This class provides a framework for training, validating, and saving model checkpoints.
        It supports both single-GPU and distributed training. Subclasses must implement
        the `train_one_epoch` and `validate` methods.

        Attributes:
            model (torch.nn.Module): The model to be trained.
            rank (int): The rank of the process in distributed training.
            device (torch.device): The device on which to train the model (CPU or GPU).
        """
        super(BaseTrainer, self).__init__()
        self.model = model
        self.rank = rank
        self.device = (
            torch.device(f"cuda:{rank % torch.cuda.device_count()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    @abstractmethod
    def train_one_epoch(
        self,
        epoch: int,
        conf: Dict[str, Any],
        trainloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        scaler: torch.cuda.amp.GradScaler,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        metrics: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Train the model for one epoch.

        Args:
            epoch (int): The current epoch number.
            conf (Dict[str, Any]): The configuration dictionary.
            trainloader (torch.utils.data.DataLoader): The training data loader.
            optimizer (torch.optim.Optimizer): The optimizer.
            criterion (torch.nn.Module): The loss function.
            scaler (torch.cuda.amp.GradScaler): The gradient scaler for mixed precision training.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler.
            metrics (Dict[str, Any]): The metrics to track during training.

        Returns:
            Dict[str, float]: A dictionary containing the training results.
        """
        raise NotImplementedError

    @abstractmethod
    def validate(
        self,
        epoch: int,
        conf: Dict[str, Any],
        valid_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        metrics: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Validate the model on the validation set.

        Args:
            epoch (int): The current epoch number.
            conf (Dict[str, Any]): The configuration dictionary.
            valid_loader (torch.utils.data.DataLoader): The validation data loader.
            criterion (torch.nn.Module): The loss function.
            metrics (Dict[str, Any]): The metrics to track during validation.

        Returns:
            Dict[str, float]: A dictionary containing the validation results.
        """
        raise NotImplementedError

    def save_checkpoint(self, save_loc: str, state_dict: Dict[str, Any]) -> None:
        """
        Save a checkpoint of the model.

        Args:
            save_loc (str): The location to save the checkpoint.
            state_dict (Dict[str, Any]): The state dictionary to save.
        """
        torch.save(state_dict, f"{save_loc}/checkpoint.pt")
        logger.info(f"Saved checkpoint to {save_loc}/checkpoint.pt")

    def save_fsdp_checkpoint(self, save_loc: str, state_dict: Dict[str, Any]) -> None:
        """
        Save a checkpoint for FSDP training.

        Args:
            save_loc (str): The location to save the checkpoint.
            state_dict (Dict[str, Any]): The state dictionary to save.
        """
        from credit.models.checkpoint import TorchFSDPCheckpointIO

        checkpoint_io = TorchFSDPCheckpointIO()

        checkpoint_io.save_unsharded_model(
            self.model,
            os.path.join(save_loc, "model_checkpoint.pt"),
            gather_dtensor=True,
            use_safetensors=False,
            rank=self.rank,
        )
        logger.info(f"Saved FSDP model checkpoint to {save_loc}/model_checkpoint.pt")

        torch.save(state_dict, os.path.join(save_loc, "checkpoint.pt"))
        logger.info(
            f"Saved FSDP scheduler and scaler states to {save_loc}/checkpoint.pt"
        )

    def fit(
        self,
        conf: Dict[str, Any],
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer: Optimizer,
        train_criterion: torch.nn.Module,
        valid_criterion: torch.nn.Module,
        scaler: GradScaler,
        scheduler: LRScheduler,
        metrics: Dict[str, Any],
        rollout_scheduler: Optional[callable] = None,
        trial: bool = False,
    ) -> Dict[str, Any]:
        """
        Fit the model to the data.

        Args:
            conf (Dict[str, Any]): Configuration dictionary.
            train_loader (DataLoader): DataLoader for training data.
            valid_loader (DataLoader): DataLoader for validation data.
            optimizer (Optimizer): The optimizer to use for training.
            train_criterion (torch.nn.Module): Loss function for training.
            valid_criterion (torch.nn.Module): Loss function for validation.
            scaler (GradScaler): Gradient scaler for mixed precision training.
            scheduler (_LRScheduler): Learning rate scheduler.
            metrics (Dict[str, Any]): Dictionary of metrics to track during training.
            rollout_scheduler (Optional[callable]): Function to schedule rollout probability, if applicable.
            trial (bool): Whether this is a trial run (e.g., for hyperparameter tuning).

        Returns:
            Dict[str, Any]: Dictionary containing the best results from training.
        """

        # convert $USER to the actual user name
        conf["save_loc"] = save_loc = os.path.expandvars(conf["save_loc"])

        # training hyperparameters
        start_epoch = conf["trainer"]["start_epoch"]
        epochs = conf["trainer"]["epochs"]
        skip_validation = (
            conf["trainer"]["skip_validation"]
            if "skip_validation" in conf["trainer"]
            else False
        )
        flag_load_weights = conf["trainer"]["load_weights"]

        # Check if 'training_metric' and 'training_metric_direction' exist in the config
        training_metric = conf["trainer"].get(
            "training_metric", "train_loss" if skip_validation else "valid_loss"
        )
        direction = conf["trainer"].get("training_metric_direction", "min")
        logger.info(
            f"The training metric being used is {training_metric} which has direction {direction}"
        )
        direction = min if direction == "min" else max

        # Check if we are saving user-defined variable metrics
        save_metric_vars = conf["trainer"].get("save_metric_vars", [])

        # =========================================== #
        # user can specify to run a fixed number of epochs
        if "num_epoch" in conf["trainer"]:
            logger.info(
                "The current job will run {} epochs max".format(
                    conf["trainer"]["num_epoch"]
                )
            )
        else:
            conf["trainer"]["num_epoch"] = 1e8
        # =========================================== #

        # Reload the results saved in the training csv if continuing to train
        if (start_epoch == 0) or (flag_load_weights is False):
            results_dict = defaultdict(list)
            # Set start_epoch to the length of the training log and train for one epoch
            # This is a manual override, you must use train_one_epoch = True
            if (
                "train_one_epoch" in conf["trainer"]
                and conf["trainer"]["train_one_epoch"]
            ):
                epochs = 1
        else:
            results_dict = defaultdict(list)
            saved_results = pd.read_csv(os.path.join(save_loc, "training_log.csv"))

            # Set start_epoch to the length of the training log and train for one epoch
            # This is a manual override, you must use train_one_epoch = True
            if (
                "train_one_epoch" in conf["trainer"]
                and conf["trainer"]["train_one_epoch"]
            ):
                start_epoch = len(saved_results)
                epochs = start_epoch + 1

            for key in saved_results.columns:
                if key == "index":
                    continue
                results_dict[key] = list(saved_results[key])

        count = 0
        for epoch in range(start_epoch, epochs):
            if count >= conf["trainer"]["num_epoch"]:
                logger.info(
                    "Completed {} epochs, exiting".format(conf["trainer"]["num_epoch"])
                )
                break

            # ========================= #
            # backup the previous epoch
            # ========================= #
            if count > 0 and conf["trainer"]["save_backup_weights"]:
                if self.rank == 0:
                    # checkpoint.pt
                    shutil.copyfile(
                        os.path.join(save_loc, "checkpoint.pt"),
                        os.path.join(save_loc, "backup_checkpoint.pt"),
                    )

                    # model_checkpoint.pt and optimizer_checkpoint.pt
                    if conf["trainer"]["mode"] == "fsdp":
                        shutil.copyfile(
                            os.path.join(save_loc, "model_checkpoint.pt"),
                            os.path.join(save_loc, "backup_model_checkpoint.pt"),
                        )

                        shutil.copyfile(
                            os.path.join(save_loc, "optimizer_checkpoint.pt"),
                            os.path.join(save_loc, "backup_optimizer_checkpoint.pt"),
                        )

            logger.info(f"Beginning epoch {epoch}")

            # set the epoch in the dataset and sampler to ensure distribured randomness is handled correctly
            if hasattr(train_loader, "sampler") and hasattr(
                train_loader.sampler, "set_epoch"
            ):
                train_loader.sampler.set_epoch(epoch)  # Start a new forecast

            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(
                    epoch
                )  # Ensure we don't start in the middle of a forecast epoch-over-epoch

            # Valid shouldnt depend on epoch, but we need to call set_epoch for consistency
            if not conf["trainer"]["skip_validation"]:
                with torch.no_grad():
                    # set the epoch in the dataset and sampler to ensure distributed randomness is handled correctly
                    if hasattr(valid_loader, "sampler") and hasattr(
                        valid_loader.sampler, "set_epoch"
                    ):
                        valid_loader.sampler.set_epoch(epoch)

                    if hasattr(valid_loader.dataset, "set_epoch"):
                        valid_loader.dataset.set_epoch(epoch)

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

            results_dict["epoch"].append(epoch)

            # Save metrics for select variables
            required_metrics = [
                "loss",
                "acc",
                "mae",
                "forecast_len",
            ]  # Base required metrics
            if isinstance(save_metric_vars, list) and len(save_metric_vars) > 0:
                names = [
                    key.replace("train_", "")
                    for key in train_results.keys()
                    if any(var in key for var in save_metric_vars)
                ]
            elif isinstance(save_metric_vars, bool) and save_metric_vars:
                names = [key.replace("train_", "") for key in train_results.keys()]
            else:
                names = []
            names = list(set(names + required_metrics))

            for name in names:
                results_dict[f"train_{name}"].append(
                    np.mean(train_results[f"train_{name}"])
                )
                if skip_validation:
                    continue
                results_dict[f"valid_{name}"].append(
                    np.mean(valid_results[f"valid_{name}"])
                )
            results_dict["lr"].append(optimizer.param_groups[0]["lr"])

            # update the learning rate if epoch-by-epoch updates

            if (
                conf["trainer"]["use_scheduler"]
                and conf["trainer"]["scheduler"]["scheduler_type"] in update_on_epoch
            ):
                if conf["trainer"]["scheduler"]["scheduler_type"] == "plateau":
                    scheduler.step(results_dict[training_metric][-1])
                else:
                    scheduler.step()

            # Create pandas df

            # Find the maximum length among all lists
            max_len = max(len(lst) for lst in results_dict.values())

            # Prepend NaNs to lists that are shorter than max_len
            padded_dict = OrderedDict()
            for key, lst in results_dict.items():
                if len(lst) < max_len:
                    padded_dict[key] = [np.nan] * (max_len - len(lst)) + lst
                else:
                    padded_dict[key] = lst

            df = pd.DataFrame.from_dict(padded_dict).reset_index()

            # Save the dataframe to disk

            if trial:  # If using ECHO-opt, save to the trial_results directory
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

                        logger.info(
                            f"Saving model, optimizer, grad scaler, and learning rate scheduler states to {save_loc}"
                        )
                        if conf["trainer"]["mode"] == "ddp":
                            model_state_dict = self.model.module.state_dict()
                        else:
                            model_state_dict = self.model.state_dict()
                        state_dict = {
                            "epoch": epoch,
                            "model_state_dict": model_state_dict,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict()
                            if conf["trainer"]["use_scheduler"]
                            else None,
                            "scaler_state_dict": scaler.state_dict(),
                        }
                        torch.save(state_dict, f"{save_loc}/checkpoint.pt")

                        if conf.get("trainer", {}).get("save_every_epoch", False):
                            copy_checkpoint(os.path.join(save_loc, "checkpoint.pt"), epoch)

                else:
                    logger.info(
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

                    if conf.get("trainer", {}).get("save_every_epoch", False):
                            copy_checkpoint(os.path.join(save_loc, "model_checkpoint.pt"), epoch)

            # clear the cached memory from the gpu
            torch.cuda.empty_cache()
            gc.collect()
            count += 1

            if skip_validation:
                pass
            else:
                # Stop training if we have not improved after X epochs (stopping patience)
                best_epoch = [
                    i
                    for i, j in enumerate(results_dict[training_metric])
                    if j == direction(results_dict[training_metric])
                ][0]
                offset = epoch - best_epoch

                # ==================== #
                # backup the best epoch
                # ==================== #
                if offset == 0 and conf["trainer"]["save_best_weights"]:
                    if self.rank == 0:
                        # checkpoint.pt
                        shutil.copyfile(
                            os.path.join(save_loc, "checkpoint.pt"),
                            os.path.join(save_loc, "best_checkpoint.pt"),
                        )

                        # model_checkpoint.pt and optimizer_checkpoint.pt
                        if conf["trainer"]["mode"] == "fsdp":
                            shutil.copyfile(
                                os.path.join(save_loc, "model_checkpoint.pt"),
                                os.path.join(save_loc, "best_model_checkpoint.pt"),
                            )

                            shutil.copyfile(
                                os.path.join(save_loc, "optimizer_checkpoint.pt"),
                                os.path.join(save_loc, "best_optimizer_checkpoint.pt"),
                            )

                # ==================== #
                # early stopping block
                # ==================== #
                if offset >= conf["trainer"]["stopping_patience"]:
                    logger.info(
                        "Best {} were in epoch {}; current epoch is {}; early stopping.".format(
                            training_metric, best_epoch, epoch
                        )
                    )
                    break

            # ==================== #
            # stop after one epoch
            # ==================== #
            if "stop_after_epoch" in conf["trainer"]:
                if conf["trainer"]["stop_after_epoch"]:
                    break

        best_epoch = [
            i
            for i, j in enumerate(results_dict[training_metric])
            if j == direction(results_dict[training_metric])
        ][0]

        result = {k: v[best_epoch] for k, v in results_dict.items()}

        if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
            cleanup()

        return result
