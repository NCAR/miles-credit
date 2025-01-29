import warnings
import os
import sys
import yaml
import shutil
import logging

from pathlib import Path
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from torchsummary import summary

from credit.models import load_model
from credit.pbs import launch_script, launch_script_mpi
from credit.seed import seed_everything
from credit.distributed import distributed_model_wrapper


warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# https://stackoverflow.com/questions/59129812/how-to-avoid-cuda-out-of-memory-in-pytorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def setup(rank, world_size, mode):
    logging.info(f"Running {mode.upper()} on rank {rank} with world_size {world_size}.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(rank, world_size, conf, frames=1, height=640, width=1280):
    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["trainer"]["mode"])

    # Config settings
    single_level_vars = [
        "surface_variables",
        "static_variables",
        "diagnostic_variables",
        "dynamic_forcing_variables",
    ]
    channels = conf["model"]["levels"] * len(conf["data"]["variables"])
    channels += sum(len(conf["data"].get(key, [])) for key in single_level_vars)

    # Set device

    device = (
        torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Load model

    m = load_model(conf)

    # send the module to the correct device first

    m.to(device)

    # Wrap if using DDP or FSDP

    model = distributed_model_wrapper(conf, m, device)

    try:
        summary(model, input_size=(channels, frames, height, width))
    except RuntimeError as e:
        if "CUDA" in str(e):
            logging.warning(f"CUDA out of memory error occurred: {e}.")
        else:
            logging.warning(f"An error occurred: {e}")


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
        "-t",
        "--num_timesteps",
        dest="t",
        type=int,
        default=1,
        help="The number of time steps the model reqiures on input",
    )
    parser.add_argument(
        "-lat",
        "--latitide",
        dest="lat",
        type=int,
        default=640,
        help="The number of pixels along latitude (default: 640)",
    )
    parser.add_argument(
        "-lon",
        "--longitude",
        dest="lon",
        type=int,
        default=1280,
        help="The number of pixels along longitude (default: 1240)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default=0,
        help="Update the config to use none, DDP, or FSDP",
    )

    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    launch = int(args_dict.pop("launch"))
    num_timesteps = int(args_dict.pop("t"))
    image_height = int(args_dict.pop("lat"))
    image_width = int(args_dict.pop("lon"))
    mode = str(args_dict.pop("mode"))

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

    # Update config using override options
    if mode in ["none", "ddp", "fsdp"]:
        logging.info(f"Setting the running mode to {mode}")
        conf["trainer"]["mode"] = mode

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

    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        world_size = 1

    main(
        rank=rank,
        world_size=world_size,
        conf=conf,
        frames=num_timesteps,
        height=image_height,
        width=image_width,
    )
