"""
train.py
-------------------------------------------------------
"""

import os
import sys
import yaml
import wandb
import optuna
import shutil
import logging
import warnings

from pathlib import Path
from argparse import ArgumentParser
from echo.src.base_objective import BaseObjective

import torch
from torch.cuda.amp import GradScaler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from credit.distributed import distributed_model_wrapper, setup, get_rank_info

from credit.seed import seed_everything
from credit.loss import VariableTotalLoss2D

from credit.scheduler import load_scheduler
from credit.trainers import load_trainer
from credit.parser import credit_main_parser, training_data_check
from credit.datasets.load_dataset_and_dataloader import load_dataset, load_dataloader

from credit.metrics import LatWeightedMetrics
from credit.pbs import launch_script, launch_script_mpi
from credit.models import load_model
from credit.models.checkpoint import FSDPOptimizerWrapper, TorchFSDPCheckpointIO


warnings.filterwarnings("ignore")


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# https://stackoverflow.com/questions/59129812/how-to-avoid-cuda-out-of-memory-in-pytorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_model_states_and_optimizer(conf, model, device):
    """
    Load the model states, optimizer, scheduler, and gradient scaler.

    Args:
        conf (dict): Configuration dictionary containing training parameters.
        model (torch.nn.Module): The model to be trained.
        device (torch.device): The device (CPU or GPU) where the model is located.

    Returns:
        tuple: A tuple containing the updated configuration, model, optimizer, scheduler, and scaler.
    """

    # convert $USER to the actual user name
    conf["save_loc"] = save_loc = os.path.expandvars(conf["save_loc"])

    # training hyperparameters
    learning_rate = float(conf["trainer"]["learning_rate"])
    weight_decay = float(conf["trainer"]["weight_decay"])
    amp = conf["trainer"]["amp"]

    # load weights / states flags
    load_weights = (
        False
        if "load_weights" not in conf["trainer"]
        else conf["trainer"]["load_weights"]
    )
    load_optimizer_conf = (
        False
        if "load_optimizer" not in conf["trainer"]
        else conf["trainer"]["load_optimizer"]
    )
    load_scaler_conf = (
        False
        if "load_scaler" not in conf["trainer"]
        else conf["trainer"]["load_scaler"]
    )
    load_scheduler_conf = (
        False
        if "load_scheduler" not in conf["trainer"]
        else conf["trainer"]["load_scheduler"]
    )

    #  Load an optimizer, gradient scaler, and learning rate scheduler, the optimizer must come after wrapping model using FSDP
    if not load_weights:  # Loaded after loading model weights when reloading
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        if conf["trainer"]["mode"] == "fsdp":
            optimizer = FSDPOptimizerWrapper(optimizer, model)
        scheduler = load_scheduler(optimizer, conf)
        scaler = (
            ShardedGradScaler(enabled=amp)
            if conf["trainer"]["mode"] == "fsdp"
            else GradScaler(enabled=amp)
        )

    # Multi-step training case -- when starting, only load the model weights (then after load all states)
    elif load_weights and not (
        load_optimizer_conf or load_scaler_conf or load_scheduler_conf
    ):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        # FSDP checkpoint settings
        if conf["trainer"]["mode"] == "fsdp":
            logging.info(
                f"Loading FSDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}"
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
            )
            optimizer = FSDPOptimizerWrapper(optimizer, model)
            checkpoint_io = TorchFSDPCheckpointIO()
            checkpoint_io.load_unsharded_model(
                model, os.path.join(save_loc, "model_checkpoint.pt")
            )
        else:
            # DDP settings
            ckpt = os.path.join(save_loc, "checkpoint.pt")
            checkpoint = torch.load(ckpt, map_location=device)
            if conf["trainer"]["mode"] == "ddp":
                logging.info(
                    f"Loading DDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}"
                )
                model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                logging.info(
                    f"Loading model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}"
                )
                model.load_state_dict(checkpoint["model_state_dict"])
        # Load the learning rate scheduler and mixed precision grad scaler
        scheduler = load_scheduler(optimizer, conf)
        scaler = (
            ShardedGradScaler(enabled=amp)
            if conf["trainer"]["mode"] == "fsdp"
            else GradScaler(enabled=amp)
        )

    # load optimizer and grad scaler states
    else:
        ckpt = os.path.join(save_loc, "checkpoint.pt")
        checkpoint = torch.load(ckpt, map_location=device)

        # FSDP checkpoint settings
        if conf["trainer"]["mode"] == "fsdp":
            logging.info(
                f"Loading FSDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}"
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
            )
            optimizer = FSDPOptimizerWrapper(optimizer, model)
            checkpoint_io = TorchFSDPCheckpointIO()
            checkpoint_io.load_unsharded_model(
                model, os.path.join(save_loc, "model_checkpoint.pt")
            )
            if (
                "load_optimizer" in conf["trainer"]
                and conf["trainer"]["load_optimizer"]
            ):
                checkpoint_io.load_unsharded_optimizer(
                    optimizer, os.path.join(save_loc, "optimizer_checkpoint.pt")
                )

        else:
            # DDP settings
            if conf["trainer"]["mode"] == "ddp":
                logging.info(
                    f"Loading DDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}"
                )
                model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                logging.info(
                    f"Loading model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}"
                )
                model.load_state_dict(checkpoint["model_state_dict"])
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
            )
            if (
                "load_optimizer" in conf["trainer"]
                and conf["trainer"]["load_optimizer"]
            ):
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler = load_scheduler(optimizer, conf)
        scaler = (
            ShardedGradScaler(enabled=amp)
            if conf["trainer"]["mode"] == "fsdp"
            else GradScaler(enabled=amp)
        )

        # Update the config file to the current epoch
        if "reload_epoch" in conf["trainer"] and conf["trainer"]["reload_epoch"]:
            conf["trainer"]["start_epoch"] = checkpoint["epoch"] + 1

        if conf["trainer"]["start_epoch"] > 0:
            # Only reload the scheduler state if not starting over from epoch 0
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            # reload the AMP gradient scaler
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Enable updating the lr if not using a policy
    if (
        conf["trainer"]["update_learning_rate"]
        if "update_learning_rate" in conf["trainer"]
        else False
    ):
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

    return conf, model, optimizer, scheduler, scaler


def main(rank, world_size, conf, backend, trial=False):
    """
    Main function to set up training and validation processes.

    Args:
        rank (int): Rank of the current process.
        world_size (int): Number of processes participating in the job.
        conf (dict): Configuration dictionary containing model, data, and training parameters.
        backend (str): Backend to be used for distributed training.
        trial (bool, optional): Flag for whether this is an Optuna trial. Defaults to False.

    Returns:
        Any: The result of the training process.
    """

    # convert $USER to the actual user name
    conf["save_loc"] = os.path.expandvars(conf["save_loc"])

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["trainer"]["mode"], backend)

    # infer device id from rank
    device = (
        torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Config settings
    seed = conf["seed"]
    seed_everything(seed)

    # Load the dataset using the provided dataset_type
    train_dataset = load_dataset(conf, rank=rank, world_size=world_size, is_train=True)
    valid_dataset = load_dataset(conf, rank=rank, world_size=world_size, is_train=False)

    # Load the dataloader
    train_loader = load_dataloader(
        conf, train_dataset, rank=rank, world_size=world_size, is_train=True
    )
    valid_loader = load_dataloader(
        conf, valid_dataset, rank=rank, world_size=world_size, is_train=False
    )

    # model
    m = load_model(conf)

    # have to send the module to the correct device first
    m.to(device)

    # move out of eager-mode
    if conf["trainer"].get("compile", False):
        m = torch.compile(m)

    # Wrap in DDP or FSDP module, or none
    model = distributed_model_wrapper(conf, m, device)

    # Load model weights (if any), an optimizer, scheduler, and gradient scaler
    conf, model, optimizer, scheduler, scaler = load_model_states_and_optimizer(
        conf, model, device
    )

    # Train and validation losses
    train_criterion = VariableTotalLoss2D(conf)
    valid_criterion = VariableTotalLoss2D(conf, validation=True)

    # Set up some metrics
    metrics = LatWeightedMetrics(conf)

    # Initialize a trainer object
    trainer_cls = load_trainer(conf)
    trainer = trainer_cls(model, rank)

    # Fit the model
    result = trainer.fit(
        conf,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        train_criterion=train_criterion,
        valid_criterion=valid_criterion,
        scaler=scaler,
        scheduler=scheduler,
        metrics=metrics,
        trial=trial,  # Optional
    )

    return result


class Objective(BaseObjective):
    """
    Optuna objective class for hyperparameter optimization.

    Attributes:
        config (dict): Configuration dictionary containing training parameters.
        metric (str): Metric to optimize, defaults to "val_loss".
        device (str): Device for training, defaults to "cpu".
    """

    def __init__(self, config, metric="val_loss", device="cpu"):
        """
        Initialize the Objective class.

        Args:
            config (dict): Configuration dictionary containing training parameters.
            metric (str, optional): Metric to optimize. Defaults to "val_loss".
            device (str, optional): Device for training. Defaults to "cpu".
        """

        # Initialize the base class
        BaseObjective.__init__(self, config, metric, device)

    def train(self, trial, conf):
        """
        Train the model using the given trial configuration.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.
            conf (dict): Configuration dictionary for the current trial.

        Returns:
            Any: The result of the training process.
        """

        conf["model"]["dim_head"] = conf["model"]["dim"]
        conf["model"]["vq_codebook_dim"] = conf["model"]["dim"]

        try:
            return main(0, 1, conf, trial=trial)

        except Exception as E:
            if "CUDA" in str(E) or "non-singleton" in str(E):
                logging.warning(
                    f"Pruning trial {trial.number} due to CUDA memory overflow: {str(E)}."
                )
                raise optuna.TrialPruned()
            elif "non-singleton" in str(E):
                logging.warning(
                    f"Pruning trial {trial.number} due to shape mismatch: {str(E)}."
                )
                raise optuna.TrialPruned()
            else:
                logging.warning(f"Trial {trial.number} failed due to error: {str(E)}.")
                raise E


if __name__ == "__main__":
    description = "Train a segmengation model on a hologram data set"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        "--config",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )
    parser.add_argument(
        "-l",
        dest="launch",
        type=int,
        default=0,
        help="Submit workers to PBS.",
    )
    parser.add_argument(
        "-w",
        "--wandb",
        dest="wandb",
        type=int,
        default=0,
        help="Use wandb. Default = False",
    )
    parser.add_argument(
        "--backend",
        type=str,
        help="Backend for distribted training.",
        default="nccl",
        choices=["nccl", "gloo", "mpi"],
    )
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    launch = int(args_dict.pop("launch"))
    backend = args_dict.pop("backend")
    use_wandb = int(args_dict.pop("wandb"))

    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Load the configuration and get the relevant variables
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    # ======================================================== #
    # handling config args
    conf = credit_main_parser(
        conf, parse_training=True, parse_predict=False, print_summary=False
    )
    training_data_check(conf, print_summary=False)
    # ======================================================== #

    # Create directories if they do not exist and copy yml file
    save_loc = os.path.expandvars(conf["save_loc"])
    os.makedirs(save_loc, exist_ok=True)

    if not os.path.exists(os.path.join(save_loc, "model.yml")):
        shutil.copy(config, os.path.join(save_loc, "model.yml"))

    # Launch PBS jobs
    if launch:
        # Where does this script live?
        script_path = Path(__file__).absolute()
        if conf["pbs"]["queue"] == "casper":
            logging.info("Launching to PBS on Casper")
            launch_script(config, script_path)
        else:
            logging.info("Launching to PBS on Derecho")
            launch_script_mpi(config, script_path)
        sys.exit()

    if use_wandb:  # this needs updated
        wandb.init(
            # set the wandb project where this run will be logged
            project="Derecho parallelism",
            name=f"Worker {os.environ['RANK']} {os.environ['WORLD_SIZE']}",
            # track hyperparameters and run metadata
            config=conf,
        )

    seed = conf["seed"]
    seed_everything(seed)

    local_rank, world_rank, world_size = get_rank_info(conf["trainer"]["mode"])
    main(world_rank, world_size, conf, backend)
