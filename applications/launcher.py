from argparse import ArgumentParser
import os
import yaml


if __name__ == "__main__":
    description = "Generic job launcher"
    parser = ArgumentParser(description=description)
    # -------------------- #
    # parser args: -c, -l,
    parser.add_argument(
        "script",
        dest="script_path",
        type=str,
        default=False,
        help="Path to the script to run",
    )

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

    # parse
    args = parser.parse_args()
    args_dict = vars(args)
    script_path = args_dict.pop("script_path")
    config = args_dict.pop("model_config")
    launch = int(args_dict.pop("launch"))

    # get config file
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    save_loc = conf["save_loc"]
    target_loc = conf.get("target_loc", "")

    # scenarios 
    # note: ALWAYS launch in the target loc
    # - save_loc to target_loc
    # - save_loc only/ no target loc
    # - save_loc == target_loc



    if target_loc == save_loc:
        # do
        pass

    