# Forecast Verification Workflow

This subfolder contains scripts to facilitate the verification of newly produced forecasts. This is an evolving section of the repository, and contributions are welcome to enhance the pipeline and expand verification capabilities.

Forecast verification is compute-intensive and often requires parallel processing. Our strategy leverages many small queued jobs to handle the workload efficiently. This document outlines the steps required to verify a forecast against the ERA5 dataset and compare it to your forecast system.

The process is driven by a **YAML configuration file** located in **./verification/** and named **verif\_config.yml**. This file must be customized extensively before initiating any verification steps.

---

## Step 00 – Adjust the YAML Configuration

Most fields in the `ERA5` and `IFS` sections of the YAML file can remain as they are, but the following areas require your attention and adjustment:

1. **qsub Section**

   - **`qsub_loc`** – Path to the directory for qsub scripts (typically `./verification/qsub/`).
   - **`scripts_loc`** – Path to the directory containing verification scripts.
   - **`project_code`** – Your project code (required for submitting jobs to the cluster).
   - **`conda_env`** – Name of the conda environment used for running the scripts.

2. **forecastmodel Section**

   - **`save_loc_rollout`** – Path to the directory where your generated forecasts are saved.
   - **`verif_variables`** – List of variables you wish to verify (ensure these match your forecast output).

---

## Step 01 – Generate and Run QSUB Scripts

Navigate to the **`./verification/verification/`** directory, where you will find four Jupyter notebooks named **`qsub_STEP00_*.ipynb`**. These notebooks generate the qsub scripts found in the **`./verification/qsub/`** directory.

### Key Scripts to Run:

- **STEP\_00** – Gathers forecast data (required before proceeding).
- **STEP\_02** – Generates RMSE and ACC metrics.

These scripts must be executed sequentially.

### Running QSUB Scripts:

1. After generating the qsub scripts via the notebooks, navigate to the **`./verification/qsub/`** directory.
2. Execute the following scripts via bash:
   ```bash
   bash step00_gather_ForecastModel_all.sh
   bash step02_RMSE_MF_all.sh
   bash step02_ACC_MF_all.sh
   ```
3. **`step00_gather_ForecastModel_all.sh`** must complete before running the other scripts.

---

## Expected Results

Upon completion of each stage:

1. **After Forecast Gathering:**
   - Forecasts will be gathered into individual NetCDF (`*.nc`) files in the location specified in the `qsub` section of the **YAML file**.

2. **After RMSE and ACC Computation:**
   - RMSE and ACC NetCDF files will be saved in the directory defined by the **`save_loc_verif`** field under the `forecastmodel` section of the YAML file.

## Troubleshooting

Forecast verification can take several days, especially for multi-year data. If errors occur, consider the following:

1. **Directory Permissions & Existence**\
   Ensure all directories specified in the YAML file exist and have appropriate write permissions. Create them manually if necessary.

   ```python
   import os
   os.makedirs(path_verif, exist_ok=True)
   ```

2. **Post-Gather Checks**

   - After running the gather script, verify that all forecast files have been created and contain the correct data and size.
   - If you encounter files with abnormally small sizes, delete them and rerun the gather script. Files that already exist will **not** be overwritten.
   - This will be much faster than the first run, as the files that already exist will be skipped. 

3. **Monitoring Job Progress**

   - Use cluster job monitoring tools to track progress and troubleshoot errors.
   - For failed jobs, inspect the `.err` files in the qsub directory for detailed logs.

---

## Additional Notes

- This process heavily relies on **parallel computing environments** like NCAR's Casper/Derecho clusters. Ensure you are familiar with the cluster's queuing and submission systems (PBS/SLURM).
- The workflow is designed to be flexible. Users are encouraged to adapt scripts to suit their specific verification needs.

If additional clarification or sections are needed (e.g., explanation of the verification metrics or variable definitions), feel free to reach out or contribute directly to this repository.

