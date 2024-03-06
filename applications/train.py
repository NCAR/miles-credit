import warnings
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.distributed as dist
import functools
from echo.src.base_objective import BaseObjective

from argparse import ArgumentParser
from pathlib import Path

import torch.fft
import logging
import shutil

import wandb
import glob
import os
import sys
import yaml
import optuna

from torch.cuda.amp import GradScaler
from torch.distributed.fsdp import StateDictType
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed.checkpoint as DCP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
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

from torchvision import transforms

from credit.models import load_model
from credit.loss import VariableTotalLoss2D
from credit.data import ERA5Dataset
from credit.transforms import ToTensor, NormalizeState
from credit.scheduler import load_scheduler, annealed_probability
from credit.trainer import Trainer
from credit.metrics import LatWeightedMetrics
from credit.pbs import launch_script, launch_script_mpi
from credit.seed import seed_everything


warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def setup(rank, world_size, mode):
    logging.info(f"Running {mode.upper()} on rank {rank} with world_size {world_size}.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def load_dataset_and_sampler(conf, files, world_size, rank, is_train, seed=42):
    history_len = conf["data"]["history_len"]
    forecast_len = conf["data"]["forecast_len"]
    valid_history_len = conf["data"]["valid_history_len"]
    valid_forecast_len = conf["data"]["valid_forecast_len"]
    time_step = None if "time_step" not in conf["data"] else conf["data"]["time_step"]
    one_shot = None if "one_shot" not in conf["data"] else conf["data"]["one_shot"]

    history_len = history_len if is_train else valid_history_len
    forecast_len = forecast_len if is_train else valid_forecast_len
    shuffle = is_train
    name = "Train" if is_train else "Valid"

    dataset = ERA5Dataset(
        filenames=files,
        history_len=history_len,
        forecast_len=forecast_len,
        skip_periods=time_step,
        one_shot=one_shot,
        transform=transforms.Compose([
            NormalizeState(conf["data"]["mean_path"], conf["data"]["std_path"]),
            ToTensor(history_len=history_len, forecast_len=forecast_len),
        ]),
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        shuffle=shuffle,
        drop_last=True
    )
    logging.info(f"{name} (forecast length = {forecast_len}): Loaded an ERA dataset, and a distributed sampler")

    return dataset, sampler


def distributed_model_wrapper(conf, vae, device):

    if conf["trainer"]["mode"] == "fsdp":

        if "use_rotary" in conf["model"]:
            from credit.rvt import Attention, FeedForward
        else:
            from credit.vit2d import Attention, FeedForward

        # Define two policies
        auto_wrap_policy1 = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Attention, FeedForward}
        )

        auto_wrap_policy2 = functools.partial(
            size_based_auto_wrap_policy, min_num_params=1_000
        )

        def combined_auto_wrap_policy(module, recurse, nonwrapped_numel):
            # Define a new policy that combines the two
            return auto_wrap_policy1(module, recurse, nonwrapped_numel) or auto_wrap_policy2(module, recurse, nonwrapped_numel)

        model = FSDP(
            vae,
            use_orig_params=True,  # needed if using torch.compile
            auto_wrap_policy=combined_auto_wrap_policy,
            # mixed_precision=torch.distributed.fsdp.MixedPrecision(
            #     param_dtype=torch.float16,
            #     reduce_dtype=torch.float16,
            #     buffer_dtype=torch.float16,
            #     #cast_forward_inputs=True
            # ),
            # sharding_strategy=ShardingStrategy.FULL_SHARD  # Zero3. Zero2 = ShardingStrategy.SHARD_GRAD_OP
            cpu_offload=CPUOffload(offload_params=True)
        )

        # activation checkpointing on the transformer blocks
        # https://pytorch.org/blog/efficient-large-scale-training-with-pytorch/

#         non_reentrant_wrapper = functools.partial(
#             checkpoint_wrapper,
#             checkpoint_impl=CheckpointImpl.NO_REENTRANT,
#         )

#         check_fn = lambda submodule: (isinstance(submodule, Attention) or isinstance(submodule, FeedForward))

#         apply_activation_checkpointing(
#             model,
#             checkpoint_wrapper_fn=non_reentrant_wrapper,
#             check_fn=check_fn
#         )

    elif conf["trainer"]["mode"] == "ddp":
        model = DDP(vae, device_ids=[device])
    else:
        model = vae

    return model


def load_model_and_optimizer(conf, model, device):

    start_epoch = conf['trainer']['start_epoch']
    save_loc = os.path.expandvars(conf['save_loc'])
    learning_rate = float(conf['trainer']['learning_rate'])
    weight_decay = float(conf['trainer']['weight_decay'])
    amp = conf['trainer']['amp']
    load_weights = False if 'load_weights' not in conf['trainer'] else conf['trainer']['load_weights']

    #  Load an optimizer, gradient scaler, and learning rate scheduler, the optimizer must come after wrapping model using FSDP
    if start_epoch == 0 and not load_weights:  # Loaded after loading model weights when reloading
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
        scheduler = load_scheduler(optimizer, conf)
        scaler = ShardedGradScaler(enabled=amp) if conf["trainer"]["mode"] == "fsdp" else GradScaler(enabled=amp)

    # load optimizer and grad scaler states
    else:
        ckpt = os.path.join(save_loc, "checkpoint.pt")
        checkpoint = torch.load(ckpt, map_location=device)

        if conf["trainer"]["mode"] == "fsdp":
            logging.info(f"Loading FSDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")

            # wait for all workers to get the model loaded
            torch.distributed.barrier()

            # tell torch how we are loading the data in (e.g. sharded states)
            FSDP.set_state_dict_type(
                model,
                StateDictType.SHARDED_STATE_DICT,
            )
            # different from ``torch.load()``, DCP requires model state_dict prior to loading to get
            # the allocated storage and sharding information.
            state_dict = {
                "model_state_dict": model.state_dict(),
            }
            DCP.load_state_dict(
                state_dict=state_dict,
                storage_reader=DCP.FileSystemReader(os.path.join(save_loc, "checkpoint")),
            )
            model.load_state_dict(state_dict["model_state_dict"])

            # Load the optimizer here on all ranks
            # https://github.com/facebookresearch/fairscale/issues/1083
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
            curr_opt_state_dict = checkpoint["optimizer_state_dict"]
            optim_shard_dict = model.get_shard_from_optim_state_dict(curr_opt_state_dict)
            # https://www.facebook.com/pytorch/videos/part-5-loading-and-saving-models-with-fsdp-full-state-dictionary/421104503278422/
            # says to use scatter_full_optim_state_dict
            optimizer.load_state_dict(optim_shard_dict)

        elif conf["trainer"]["mode"] == "ddp":
            logging.info(f"Loading DDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")
            model.module.load_state_dict(checkpoint["model_state_dict"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
            if 'load_optimizer' in conf['trainer'] and conf['trainer']['load_optimizer']:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

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

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = learning_rate

    return model, optimizer, scheduler, scaler


def main(rank, world_size, conf, trial=False):

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

    # datasets (zarr reader)

    all_ERA_files = sorted(glob.glob(conf["data"]["save_loc"]))
    #filenames = list(map(os.path.basename, all_ERA_files))
    #all_years = sorted([re.findall(r'(?:_)(\d{4})', fn)[0] for fn in filenames])

    # Specify the years for each set
    #if conf["data"][train_test_split]:
    #    normalized_split = conf["data"][train_test_split] / sum(conf["data"][train_test_split])
    #    n_years = len(all_years)
    #    train_years, sklearn.model_selection.train_test_split¶

    train_years = [str(year) for year in range(1979, 2014)]
    valid_years = [str(year) for year in range(2014, 2018)]  # can make CV splits if we want to later on
    test_years = [str(year) for year in range(2018, 2022)]  # same as graphcast -- always hold out

    # train_years = [str(year) for year in range(1995, 2013) if year != 2007]
    # valid_years = [str(year) for year in range(2014, 2015)]  # can make CV splits if we want to later on
    # test_years = [str(year) for year in range(2015, 2016)]  # same as graphcast -- always hold out

    # Filter the files for each set

    train_files = [file for file in all_ERA_files if any(year in file for year in train_years)]
    valid_files = [file for file in all_ERA_files if any(year in file for year in valid_years)]
    test_files = [file for file in all_ERA_files if any(year in file for year in test_years)]

    # load dataset and sampler

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

    vae = load_model(conf)

    num_params = sum(p.numel() for p in vae.parameters())
    if rank == 0:
        logging.info(f"Number of parameters in the model: {num_params}")
    #summary(vae, input_size=(channels, height, width))

    # have to send the module to the correct device first

    vae.to(device)
    #vae = torch.compile(vae)

    # Wrap in DDP or FSDP module, or none

    model = distributed_model_wrapper(conf, vae, device)

    # Load an optimizer, scheduler, and gradient scaler from disk if epoch > 0

    model, optimizer, scheduler, scaler = load_model_and_optimizer(conf, model, device)

    # Train and validation losses

    train_criterion = VariableTotalLoss2D(conf)
    valid_criterion = VariableTotalLoss2D(conf, validation=True)

    # Optional load stopping probability annealer

    # Set up some metrics

    metrics = LatWeightedMetrics(conf)

    # Initialize a trainer object

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
        "--world-size",
        type=int,
        default=4,
        help="Number of processes (world size) for multiprocessing"
    )
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    launch = int(args_dict.pop("launch"))

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

#     wandb.init(
#         # set the wandb project where this run will be logged
#         project="Derecho parallelism",
#         name=f"Worker {os.environ["RANK"]} {os.environ["WORLD_SIZE"]}"
#         # track hyperparameters and run metadata
#         config=conf
#     )

    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        main(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), conf)
    else:
        main(0, 1, conf)
