from argparse import ArgumentParser
import logging
import os
from os.path import join
import shutil

import yaml
import torch

def model_checkpoint_to_checkpoint(save_loc):
    checkpoint_to_convert = join(save_loc, "convert_checkpoint.pt")

    if not os.path.isfile(checkpoint_to_convert):
        logging.info("copying model_checkpoint.pt to stage for conversion\n")
        ckpt = join(save_loc, "model_checkpoint.pt")
        shutil.copy(ckpt, checkpoint_to_convert)

    state_dict = torch.load(checkpoint_to_convert)
    dst_checkpoint = join(save_loc, "checkpoint.pt")
    torch.save({"model_state_dict": state_dict},
               dst_checkpoint)
    logging.info(f"converted model_checkpoint to {dst_checkpoint}")

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

    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")

    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    # Create directories if they do not exist and copy yml file
    save_loc = os.path.expandvars(conf["save_loc"])
    
    model_checkpoint_to_checkpoint(save_loc)
    
