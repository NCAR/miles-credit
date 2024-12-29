# Location of qsub folder 

## Step 01 – Generate and Run QSUB Scripts

Navigate to the **`./verification/verification/`** directory, where you will find four Jupyter notebooks named **`qsub_STEP00_*.ipynb`**. These notebooks generate the qsub scripts found in the **`./verification/qsub/`** directory.

### Key Scripts to Run:

- **STEP\_00** – Gathers forecast data (required before proceeding).
- **STEP\_02** – Generates RMSE and ACC metrics.

These scripts must be executed sequentially.

### Running QSUB Scripts:

1. After generating the qsub scripts via the notebooks, navigate to the **`./verification/qsub/`**  (you are here now) directory.
2. Execute the following scripts via bash:
   ```bash
   bash step00_gather_ForecastModel_all.sh
   bash step02_RMSE_MF_all.sh
   bash step02_ACC_MF_all.sh
   ```
3. **`step00_gather_ForecastModel_all.sh`** must complete before running the other scripts.

---