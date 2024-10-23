import pytest
import yaml
import os

from credit.parser import CREDIT_main_parser, training_data_check

TEST_FILE_DIR = "/".join(os.path.abspath(__file__).split("/")[:-1])
CONFIG_FILE_DIR = os.path.join("/".join(os.path.abspath(__file__).split("/")[:-2]),
                      "config")

def test_main_parser():
    config = os.path.join(CONFIG_FILE_DIR, "example_skebs.yml")
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = CREDIT_main_parser(conf, print_summary=True) # parser will copy model configs to post_conf
    return conf

@pytest.mark.skip(reason="need to be on glade filesystem")
def test_training_data_check():
    conf = test_main_parser()
    conf = training_data_check(conf, print_summary=True)

    assert conf["data"]["level_list"]

if __name__ == "__main__":
    test_training_data_check()