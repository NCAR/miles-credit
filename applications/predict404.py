# ---------- #
# System
import os
import sys
import logging
import warnings
from pathlib import Path
from argparse import ArgumentParser
import yaml

# ---------- #
# Numerics
import xarray as xr

# ---------- #
# AI libs
import torch
from torchvision import transforms

# ---------- #
# credit
from credit.data404 import CONUS404Dataset
from credit.models import load_model
from credit.transforms404 import ToTensor, NormalizeState
from credit.seed import seed_everything
from credit.pbs import launch_script, launch_script_mpi

# ---------- #

logging.basicConfig(
    format="%(asctime)s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def predict(rank, world_size, conf, dataset):
    """Loads model defined in configuration object and applies it to
    data in dataset (using test split defined in conf).  Returns a
    list of xarray objects, one for each predicted output.

    """
    autoregressive = conf["predict"]["autoregressive"]

    # infer device id from rank
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        device = torch.device("cpu")

    # load model
    logging.info("Loading model")
    model = load_model(conf, load_weights=True).to(device)

    # switch to evaluation mode
    model.eval()

    # storage for outputs from model
    xarraylist = []

    # do predictions
    with torch.no_grad():
        outdims = ["t", "vars", "z", "y", "x"]  # todo: squeeze bottom_top

        # model inference loop
        logging.info("Beginning inference loop")

        for index in range(len(dataset)):  # noqa C0200
            if autoregressive and index > 0:
                xin = torch.cat((xin[:, :, 1:3, :, :], yout[:, :, 0:1, :, :]), dim=2)  # noqa F821
            else:
                xin = dataset[index]["x"].unsqueeze(0).to(device)

            yout = model(xin)
            y = state_transformer.inverse_transform(yout.cpu())
            xarr = xr.DataArray(y, dims=outdims)
            xarraylist.append(xarr)

            # # Update the input
            # # setup for next iteration, transform to z-space and send to device
            # y_pred = state_transformer.transform_array(y_pred).to(device)
            #
            # if history_len == 1:
            #     x = y_pred.detach()
            # else:
            #     # use multiple past forecast steps as inputs
            #     static_dim_size = abs(x.shape[1] - y_pred.shape[1])
            #         # static channels will get updated on next pass
            #     x_detach = x[:, :-static_dim_size, 1:].detach()
            #     x = torch.cat([x_detach, y_pred.detach()], dim=2)
            #
            # # Explicitly release GPU memory
            # torch.cuda.empty_cache()
            # gc.collect()

    # if distributed:
    #     torch.distributed.barrier()

    return xarraylist


if __name__ == "__main__":
    description = "Rollout AI-NWP forecasts"
    parser = ArgumentParser(description=description)
    # -------------------- #
    # parser args: -c, -l, -w
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
        help="Number of processes (world size) for multiprocessing",
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default=0,
        help="Update the config to use none, DDP, or FSDP",
    )

    parser.add_argument(
        "-nd",
        "--no-data",
        type=str,
        default=0,
        help="If set to True, only pandas CSV files will we saved for each forecast",
    )

    # parse
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    launch = int(args_dict.pop("launch"))
    mode = str(args_dict.pop("mode"))
    no_data = 0 if "no-data" not in args_dict else int(args_dict.pop("no-data"))

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

    # Update config using override options
    if mode in ["none", "ddp", "fsdp"]:
        logger.info(f"Setting the running mode to {mode}")
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

    # main branch if not launching starts here

    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    logging.info("Loading C404 dataset")

    # Preprocessing transformations
    if conf["data"]["scaler_type"] == "std":
        state_transformer = NormalizeState(conf)
    else:
        state_transformer = NormalizeState_Quantile(conf)  # noqa F821
    transform = transforms.Compose(
        [
            state_transformer,
            ToTensor(conf),
        ]
    )

    ds = CONUS404Dataset(
        varnames=conf["data"]["variables"],
        history_len=conf["data"]["history_len"],
        forecast_len=conf["data"]["forecast_len"],
        transform=transform,
        start=conf["predict"]["start"],
        finish=conf["predict"]["finish"],
    )

    # if mode in ["fsdp", "ddp"]:
    #     xarraylist = predict(
    #         rank = int(os.environ["RANK"]),
    #         world_size = int(os.environ["WORLD_SIZE"]),
    #         conf = conf,
    #         dataset = ds
    #     )
    # else:
    logging.info("Starting prediction")
    xarraylist = predict(rank=0, world_size=1, conf=conf, dataset=ds)
    logging.info("Prediction finished")

    # reconstruct xarray dataset

    xcat = xr.concat(xarraylist, dim="t")
    xcat = xcat.rename({"t": ds.tdimname})
    xcat = xcat.assign_coords({"vars": ds.varnames})

    ds_out = xcat.to_dataset(dim="vars")

    sep = "."
    filename = sep.join(
        [
            os.path.basename(conf["save_loc"]),
            "C404",
            conf["predict"]["start"],
            conf["predict"]["finish"],
            "nc",
        ]
    )
    save_path = os.path.join(conf["save_loc"], filename)

    logging.info("Writing results to file")
    ds_out.to_netcdf(
        path=save_path,
        format="NETCDF4",
        engine="netcdf4",
        encoding={
            v: {"zlib": True, "complevel": 1, "dtype": "float"}
            for v in conf["data"]["variables"]
        },
        unlimited_dims="Time",
        compute=True,
    )

    logging.info("Done!")
