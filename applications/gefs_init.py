import argparse
from multiprocessing import Pool
from credit.gefs import download_gefs_run, process_member, download_single_gefs_run
from functools import partial
from os.path import join, exists
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser(
        description="Initialize CREDIT models with GEFS data"
    )
    parser.add_argument(
        "-d",
        "--date",
        required=True,
        help="Initialization date in YYYY-MM-DD HHMM format.",
    )
    parser.add_argument(
        "-p",
        "--path",
        required=True,
        help="Path to where raw GEFS data will be downloaded.",
    )
    parser.add_argument(
        "-o",
        "--out",
        required=True,
        help="Path to where processed GEFS data will be saved.",
    )
    parser.add_argument(
        "-w",
        "--weights",
        required=True,
        help="Path to ESMF_RegridWeightGen regrid weight file.",
    )
    parser.add_argument(
        "-m",
        "--members",
        type=int,
        default=30,
        help="Number of GEFS perturbation members to download.",
    )
    parser.add_argument(
        "-n", "--nprocs", type=int, default=1, help="Number of processes to use."
    )
    parser.add_argument(
        "-v",
        "--variables",
        type=str,
        default="ps,t,sphum,liq_wat,ice_wat,rainwat,snowwat,graupel,u_s,v_w,slmsk,tsea,fice,t2m,q2m,zh",
        help="Variables to use separated by commas.",
    )
    parser.add_argument(
        "-r",
        "--rename_dict_file",
        type=str,
        default="",
        help="YAML file containing mappings between GEFS and destination model variable names.",
    )
    parser.add_argument(
        "-t",
        "--meta_file",
        type=str,
        default="",
        help="YAML file containing metadata for regridded variables.",
    )
    parser.add_argument(
        "-u",
        "--vertical",
        type=str,
        default="",
        help="netCDF file containing vertical.",
    )

    args = parser.parse_args()
    init_date_str = args.date
    init_date = pd.Timestamp(init_date_str)
    download_path = args.path
    out_path = args.out
    weight_file = args.weights
    n_pert_members = args.members
    rename_dict_file = args.rename_dict_file
    vertical_file = args.vertical
    meta_file = args.meta_file
    variables = args.variables.split(",")
    # download_gefs_run(init_date_str, download_path, n_pert_members)
    member_names = ["c00"] + [f"p{m:02d}" for m in range(1, n_pert_members + 1)]
    init_date_path = init_date.strftime("gefs.%Y%m%d/%H")
    full_out_path = join(out_path, init_date_path)
    # if not exists(full_out_path):
    #     os.makedirs(full_out_path)
    with Pool(args.nprocs) as pool:
    # for init_date_str in pd.date_range("2022-05-20", "2022-05-31", freq="12h").strftime("%Y-%m-%dT%H00"):
    #     print(f"Running for {init_date_str}")
    #     init_date = pd.Timestamp(init_date_str)
        init_date_path = init_date.strftime("gefs.%Y%m%d/%H")
        full_out_path = join(out_path, init_date_path)
        if not exists(full_out_path):
            os.makedirs(full_out_path)
            pool.map(
                partial(download_single_gefs_run,
                        init_date_str,
                        download_path,
                ),
                member_names,
            )
            pool.map(
                partial(
                    process_member,
                    member_path=download_path,
                    out_path=full_out_path,
                    init_date_str=init_date_str,
                    variables=variables,
                    weight_file=weight_file,
                    rename_dict_file=rename_dict_file,
                    meta_file=meta_file,
                    vertical_level_file=vertical_file,
                ),
                member_names,
            )
    return


if __name__ == "__main__":
    main()