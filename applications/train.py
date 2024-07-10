'''
train.py 
-------------------------------------------------------
'''
import os
import sys
import glob
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
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from credit.distributed import distributed_model_wrapper

from credit.seed import seed_everything
from credit.loss import VariableTotalLoss2D
from credit.data import ERA5Dataset, ERA5_and_Forcing_Dataset, Dataset_BridgeScaler
from credit.transforms import load_transforms
from credit.scheduler import load_scheduler, annealed_probability

from credit.trainer import Trainer
# <-------------- the new pipeline
from credit.trainer_new import Trainer as Trainer_New

from credit.metrics import LatWeightedMetrics
from credit.pbs import launch_script, launch_script_mpi
from credit.models import load_model
from credit.models.checkpoint import (
    FSDPOptimizerWrapper,
    TorchFSDPCheckpointIO
)


warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

os.environ['NCCL_SHM_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'


# https://stackoverflow.com/questions/59129812/how-to-avoid-cuda-out-of-memory-in-pytorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def setup(rank, world_size, mode):
    logging.info(f"Running {mode.upper()} on rank {rank} with world_size {world_size}.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def load_dataset_and_sampler(conf, files, world_size, rank, is_train, seed=42):

    # convert $USER to the actual user name
    conf['save_loc'] = os.path.expandvars(conf['save_loc'])

    # number of previous lead time inputs
    history_len = conf["data"]["history_len"]
    valid_history_len = conf["data"]["valid_history_len"]
    history_len = history_len if is_train else valid_history_len

    # number of lead times to forecast
    forecast_len = conf["data"]["forecast_len"]
    valid_forecast_len = conf["data"]["valid_forecast_len"]
    forecast_len = forecast_len if is_train else valid_forecast_len

    # optional setting: max_forecast_len
    max_forecast_len = None if "max_forecast_len" not in conf["data"] else conf["data"]["max_forecast_len"]

    # optional setting: skip_periods
    skip_periods = None if "skip_periods" not in conf["data"] else conf["data"]["skip_periods"]

    # optional setting: one_shot
    one_shot = None if "one_shot" not in conf["data"] else conf["data"]["one_shot"]

    # shufle dataloader if training
    shuffle = is_train
    name = "Train" if is_train else "Valid"

    # data preprocessing utils
    transforms = load_transforms(conf)

    # quantile transform using BridgeScaler
    if conf["data"]["scaler_type"] == "quantile-cached":
        dataset = Dataset_BridgeScaler(
            conf,
            conf_dataset='bs_years_train' if is_train else 'bs_years_val',
            transform=transforms
        )

    else:
        # Z-score
        dataset = ERA5Dataset(
            filenames=files,
            history_len=history_len,
            forecast_len=forecast_len,
            skip_periods=skip_periods,
            one_shot=one_shot,
            max_forecast_len=max_forecast_len,
            transform=transforms
        )

    # Pytorch sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        shuffle=shuffle,
        drop_last=True
    )
    logging.info(f" Loaded a {name} ERA dataset, and a distributed sampler (forecast length = {forecast_len + 1})")

    return dataset, sampler

def load_dataset_and_sampler_zscore_only(conf, all_ERA_files, surface_files, diagnostic_files, world_size, rank, is_train, seed=42):

    # convert $USER to the actual user name
    conf['save_loc'] = os.path.expandvars(conf['save_loc'])

    # ======================================================== #
    # parse intputs
    
    # file names
    varname_upper_air = conf['data']['variables']
    
    if ('forcing_variables' in conf['data']) and (len(conf['data']['forcing_variables']) > 0):
        forcing_files = conf['data']['save_loc_forcing']
        varname_forcing = conf['data']['forcing_variables']
    else:
        forcing_files = None
        varname_forcing = None
    
    if ('static_variables' in conf['data']) and (len(conf['data']['static_variables']) > 0):
        static_files = conf['data']['save_loc_static']
        varname_static = conf['data']['static_variables']
    else:
        static_files = None
        varname_static = None
    
    if surface_files is not None:
        varname_surface = conf['data']['surface_variables']
    else:
        varname_surface = None
        
    if diagnostic_files is not None:
        varname_diagnostic = conf['data']['diagnostic_variables']
    else:
        varname_diagnostic = None
        
    # number of previous lead time inputs
    history_len = conf["data"]["history_len"]
    valid_history_len = conf["data"]["valid_history_len"]

    # number of lead times to forecast
    forecast_len = conf["data"]["forecast_len"]
    valid_forecast_len = conf["data"]["valid_forecast_len"]
    
    if is_train:
        history_len = history_len
        forecast_len = forecast_len
        # print out training / validation
        name = "training"
    else:
        history_len = valid_history_len
        forecast_len = valid_forecast_len
        name = 'validation'
        
    # max_forecast_len
    if "max_forecast_len" not in conf["data"]:
        max_forecast_len = None
    else:
        max_forecast_len = conf["data"]["max_forecast_len"]

    # skip_periods
    if "skip_periods" not in conf["data"]:
        skip_periods = None
    else:
        skip_periods = conf["data"]["skip_periods"]
        
    # one_shot
    if "one_shot" not in conf["data"]:
        one_shot = None
    else:
        one_shot = conf["data"]["one_shot"]

    # shufle
    shuffle = is_train
    
    # data preprocessing utils
    transforms = load_transforms(conf)

    # Z-score
    dataset = ERA5_and_Forcing_Dataset(
        varname_upper_air=varname_upper_air,
        varname_surface=varname_surface,
        varname_forcing=varname_forcing,
        varname_static=varname_static,
        varname_diagnostic=varname_diagnostic,
        filenames=all_ERA_files,
        filename_surface=surface_files,
        filename_forcing=forcing_files,
        filename_static=static_files,
        filename_diagnostic=diagnostic_files,
        history_len=history_len,
        forecast_len=forecast_len,
        skip_periods=skip_periods,
        one_shot=one_shot,
        max_forecast_len=max_forecast_len,
        transform=transforms
    )
    
    # Pytorch sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        shuffle=shuffle,
        drop_last=True
    )
    
    logging.info(f" Loaded a {name} ERA dataset, and a distributed sampler (forecast length = {forecast_len + 1})")

    return dataset, sampler


def load_model_states_and_optimizer(conf, model, device):

    # convert $USER to the actual user name
    conf['save_loc'] = save_loc = os.path.expandvars(conf['save_loc'])

    # training hyperparameters
    start_epoch = conf['trainer']['start_epoch']
    learning_rate = float(conf['trainer']['learning_rate'])
    weight_decay = float(conf['trainer']['weight_decay'])
    amp = conf['trainer']['amp']

    # load weights falg
    load_weights = False if 'load_weights' not in conf['trainer'] else conf['trainer']['load_weights']

    #  Load an optimizer, gradient scaler, and learning rate scheduler, the optimizer must come after wrapping model using FSDP
    #if start_epoch == 0 and not load_weights: 
    if not load_weights:  # Loaded after loading model weights when reloading
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
        if conf["trainer"]["mode"] == "fsdp":
            optimizer = FSDPOptimizerWrapper(optimizer, model)
        scheduler = load_scheduler(optimizer, conf)
        scaler = ShardedGradScaler(enabled=amp) if conf["trainer"]["mode"] == "fsdp" else GradScaler(enabled=amp)

    # load optimizer and grad scaler states
    else:
        ckpt = os.path.join(save_loc, "checkpoint.pt")
        checkpoint = torch.load(ckpt, map_location=device)

        # FSDP checkpoint settings
        if conf["trainer"]["mode"] == "fsdp":
            logging.info(f"Loading FSDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
            optimizer = FSDPOptimizerWrapper(optimizer, model)
            checkpoint_io = TorchFSDPCheckpointIO()
            checkpoint_io.load_unsharded_model(model, os.path.join(save_loc, "model_checkpoint.pt"))
            if 'load_optimizer' in conf['trainer'] and conf['trainer']['load_optimizer']:
                checkpoint_io.load_unsharded_optimizer(optimizer, os.path.join(save_loc, "optimizer_checkpoint.pt"))

        else:
            # DDP settings
            if conf["trainer"]["mode"] == "ddp":
                logging.info(f"Loading DDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")
                model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                logging.info(f"Loading model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")
                model.load_state_dict(checkpoint["model_state_dict"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
            if 'load_optimizer' in conf['trainer'] and conf['trainer']['load_optimizer']:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler = load_scheduler(optimizer, conf)
        scaler = ShardedGradScaler(enabled=amp) if conf["trainer"]["mode"] == "fsdp" else GradScaler(enabled=amp)

        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # Enable updating the lr if not using a policy
    if (conf["trainer"]["update_learning_rate"] if "update_learning_rate" in conf["trainer"] else False):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    return model, optimizer, scheduler, scaler


def main(rank, world_size, conf, trial=False):

    # convert $USER to the actual user name
    conf['save_loc'] = os.path.expandvars(conf['save_loc'])
    
    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["trainer"]["mode"])
    
    # infer device id from rank
    
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(rank % torch.cuda.device_count())
    
    # Config settings
    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)
    
    train_batch_size = conf['trainer']['train_batch_size']
    valid_batch_size = conf['trainer']['valid_batch_size']
    thread_workers = conf['trainer']['thread_workers']
    valid_thread_workers = conf['trainer']['valid_thread_workers'] if 'valid_thread_workers' in conf['trainer'] else thread_workers
    
    # get file names
    all_ERA_files = sorted(glob.glob(conf["data"]["save_loc"]))
    
    # <------------------------------------------ std_new
    if conf['data']['scaler_type'] == 'std_new':
    
        if "save_loc_surface" in conf["data"]:
            surface_files = sorted(glob.glob(conf["data"]["save_loc_surface"]))
        else:
            surface_files = None
    
        if "save_loc_diagnostic" in conf["data"]:
            diagnostic_files = sorted(glob.glob(conf["data"]["save_loc_diagnostic"]))
        else:
            diagnostic_files = None
    
    
    # -------------------------------------------------- #
    # import training / validation years from conf
    
    if 'train_years' in conf['data']:
        train_years_range = conf['data']['train_years']
    else:
        train_years_range = [1979, 2014]
    
    if 'valid_years' in conf['data']:
        valid_years_range = conf['data']['valid_years']
    else:
        valid_years_range = [2014, 2018]
    
    # convert year info to str for file name search
    train_years = [str(year) for year in range(train_years_range[0], train_years_range[1])]
    valid_years = [str(year) for year in range(valid_years_range[0], valid_years_range[1])]
    
    # Filter the files for training / validation
    train_files = [file for file in all_ERA_files if any(year in file for year in train_years)]
    valid_files = [file for file in all_ERA_files if any(year in file for year in valid_years)]
    
    # <----------------------------------- std_new
    if conf['data']['scaler_type'] == 'std_new':
        train_surface_files = [file for file in surface_files if any(year in file for year in train_years)]
        valid_surface_files = [file for file in surface_files if any(year in file for year in valid_years)]
        
        train_diagnostic_files = [file for file in diagnostic_files if any(year in file for year in train_years)]
        valid_diagnostic_files = [file for file in diagnostic_files if any(year in file for year in valid_years)]
    
    # load dataset and sampler
    # <----------------------------------- std_new
    if conf['data']['scaler_type'] == 'std_new':
        # training set and sampler
        train_dataset, train_sampler = load_dataset_and_sampler_zscore_only(conf, 
                                                                            train_files, 
                                                                            train_surface_files, 
                                                                            train_diagnostic_files, 
                                                                            world_size, rank, is_train=True)
        # validation set and sampler
        valid_dataset, valid_sampler = load_dataset_and_sampler_zscore_only(conf, 
                                                                            valid_files, 
                                                                            valid_surface_files, 
                                                                            valid_diagnostic_files,
                                                                            world_size, rank, is_train=False)
    else:
        train_dataset, train_sampler = load_dataset_and_sampler(conf, train_files, world_size, rank, is_train=True)
        valid_dataset, valid_sampler = load_dataset_and_sampler(conf, valid_files, world_size, rank, is_train=False)
    
    # setup the dataloder for this process
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True if thread_workers > 0 else False,
        num_workers=thread_workers,
        drop_last=True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        sampler=valid_sampler,
        pin_memory=False,
        num_workers=valid_thread_workers,
        drop_last=True
    )

    # model

    m = load_model(conf)

    # have to send the module to the correct device first

    m.to(device)
    # m = torch.compile(m)

    # Wrap in DDP or FSDP module, or none

    model = distributed_model_wrapper(conf, m, device)

    # Load model weights (if any), an optimizer, scheduler, and gradient scaler

    model, optimizer, scheduler, scaler = load_model_states_and_optimizer(conf, model, device)

    # Train and validation losses

    train_criterion = VariableTotalLoss2D(conf)
    valid_criterion = VariableTotalLoss2D(conf, validation=True)

    # Optional load stopping probability annealer

    # Set up some metrics

    metrics = LatWeightedMetrics(conf)

    # Initialize a trainer object
    # <----------------------------------- replace
    if conf['data']['scaler_type'] == 'std_new':
        trainer = Trainer_New(model, rank, module=(conf["trainer"]["mode"] == "ddp"))
    else:
        trainer = Trainer(model, rank, module=(conf["trainer"]["mode"] == "ddp"))

    # Fit the model

    result = trainer.fit(
        conf,
        train_loader,
        valid_loader,
        optimizer,
        train_criterion,
        valid_criterion,
        scaler,
        scheduler,
        metrics,
        rollout_scheduler=annealed_probability,
        trial=trial
    )

    return result


class Objective(BaseObjective):
    def __init__(self, config, metric="val_loss", device="cpu"):

        # Initialize the base class
        BaseObjective.__init__(self, config, metric, device)

    def train(self, trial, conf):

        conf['model']['dim_head'] = conf['model']['dim']
        conf['model']['vq_codebook_dim'] = conf['model']['dim']

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
        help="Use wandb. Default = False"
    )
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    launch = int(args_dict.pop("launch"))
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

    # Create directories if they do not exist and copy yml file
    save_loc = os.path.expandvars(conf["save_loc"])
    os.makedirs(save_loc, exist_ok=True)

    if not os.path.exists(os.path.join(save_loc, "model.yml")):
        shutil.copy(config, os.path.join(save_loc, "model.yml"))

    # Launch PBS jobs
    if launch:
        # Where does this script live?
        script_path = Path(__file__).absolute()
        if conf['pbs']['queue'] == 'casper':
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
            config=conf
        )

    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        main(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), conf)
    else:
        main(0, 1, conf)
