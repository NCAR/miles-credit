from os.path import expandvars, dirname
from importlib.resources import files


def get_meta_file_path(meta_file: str):
    """
    Handles relative path and environment variable expansion for metadata files.
    If your path includes environment variables, they are filled in.
    If no directories are provided, this script assumes that the file is in
    `credit.metadata` and will find the installed credit.metadata path for you.

    Args:
        meta_file: Path to the metadata file.

    Returns:

    """
    meta_file_exp = expandvars(meta_file)
    if not dirname(meta_file_exp):
        meta_file_exp = str(files("credit.metadata").joinpath(meta_file_exp))
    return meta_file_exp
