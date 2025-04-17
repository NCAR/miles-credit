import yaml
import os

from credit.parser import credit_main_parser

TEST_FILE_DIR = "/".join(os.path.abspath(__file__).split("/")[:-1])
CONFIG_FILE_DIR = os.path.join("/".join(os.path.abspath(__file__).split("/")[:-2]),
                      "config")

def test_main_parser():
    config = os.path.join(CONFIG_FILE_DIR, "example-v2025.2.0.yml")
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = credit_main_parser(conf, print_summary=True) # parser will copy model configs to post_conf


if __name__ == "__main__":
    pass