## Hello!

Below we outline the notebooks used to gen the QSUB scripts, in **`./scripts`** are the main files which drive the calculations. Adjustments can be made there.

---
## QSUB Jupyter Notebooks – Generating Job Scripts for Forecast Verification

This directory contains Jupyter notebooks designed to generate and submit qsub scripts for various stages of the forecast verification process. These notebooks facilitate job scheduling and resource allocation on HPC systems, streamlining the process of gathering, processing, and verifying forecast data.

---
### Notebooks Overview

The primary function of the notebooks in this folder is to automate the creation of bash scripts (`.sh`) that submit jobs to the PBS queueing system. This approach allows for efficient parallelization, ensuring multiple forecasts are processed concurrently.

**Notebook Naming Convention:**
- **`qsub_STEP00_*.ipynb`** – Responsible for gathering forecast model data.
- **`qsub_STEP02_*.ipynb`** – Generates RMSE and ACC qsub scripts for model verification.
---

### How to Use These Notebooks

1. **Setup & Prerequisites**  
   Ensure the following prerequisites are met before running the notebooks:  
   - **Configured YAML file** (`verif_config.yml`) with correct paths, project codes, and environment settings.  
   - **Conda environment** activated (defined in the YAML under `conda_env`).  
   - Appropriate access to the cluster and necessary permissions for submitting jobs.

   **Example Activation:**
   ```bash
   conda activate credit
   ```

2. **Navigating the Workflow**  
   - Start by opening the `qsub_STEP00_jobs.ipynb` notebook to generate scripts for gathering forecast data.  
   - Follow by executing the `qsub_STEP02_*` notebooks for computing RMSE and ACC after the gather phase completes.

3. **Running the Notebooks**  
   - Execute cells sequentially within the notebook.  
   - Each notebook will output `.sh` scripts into the `./verification/qsub/` directory.

4. **Submitting QSUB Jobs**  
   Once scripts are generated, submit them to the cluster queue:  
   ```bash
   bash step00_gather_ForecastModel_all.sh
   bash step02_RMSE_MF_all.sh
   bash step02_ACC_MF_all.sh
   ```

---

### Notebook Breakdown – `qsub_STEP00_jobs.ipynb`

**Purpose:**  
Generates qsub scripts to gather forecast data from various sources and formats it for further verification.  

**Key Sections:**  
- **Config Loading:** Loads the YAML configuration to set paths, environment, and project-specific parameters.  
- **Script Generation Loop:** Iterates over forecast indices (`INDs`) to create individual qsub scripts for each chunk of data.  
- **Output:**  
   - Scripts are saved in the `qsub_loc` directory specified in the YAML.  
   - Example script: `verif_ZES_WX_001.sh`

**Critical Code Example:**
```python
for i, ind_start in enumerate(INDs[:-1]):
    
    ind_end = INDs[i+1]
    
    f = open('{}gather_ForecastModel_{:03d}.sh'.format(conf['qsub']['qsub_loc'], i), 'w') 
    
    heads = '''#!/bin/bash -l

#PBS -N gather_ForecastModel
#PBS -A {project_code}
#PBS -l walltime=23:59:59
#PBS -l select=1:ncpus=4:mem=32GB
#PBS -q casper
#PBS -o gather_ForecastModel.log
#PBS -e gather_ForecastModel.err

module load conda
conda activate {conda_env}
cd {scripts_loc}
python STEP00_gather_ForecastModel.py {ind_start} {ind_end}
'''.format(project_code=conf['qsub']['project_code'],
           conda_env=conf['qsub']['conda_env'],
           scripts_loc=conf['qsub']['scripts_loc'], 
           ind_start=ind_start, 
           ind_end=ind_end)
    
    print(heads, file=f)    
    f.close()

f = open('{}step00_gather_ForecastModel_all.sh'.format(conf['qsub']['qsub_loc']), 'w')

for i, ind_start in enumerate(INDs[:-1]):
    print('qsub gather_ForecastModel_{:03d}.sh'.format(i), file=f)
    
f.close()
```

---

### Keys to Running These Notebooks

- **Ensure Sequential Execution:**  
   - `STEP00` scripts **must** be submitted and completed **before** proceeding to `STEP02` scripts.  
   - Failure to adhere to this order will result in missing forecast data during RMSE/ACC calculations.  
- **Directory Existence:**  
   - The directories where NetCDF files are saved (`save_loc_verif`) must exist.  
   - Use `os.makedirs(path, exist_ok=True)` to create directories if needed.  
- **Cluster Specifics:**  
   - These scripts are optimized for NCAR’s Cheyenne/Derecho clusters. Adjust for other HPC environments if necessary.

---

### Troubleshooting

- **Job Failures:**  
   - Review `.err` files in the qsub directory. These contain logs of job failures and error messages.  
- **Missing Files:**  
   - If forecast files appear incomplete, re-run the gather phase (`STEP00`) without fear of overwriting existing valid files.  
- **Memory/CPU Issues:**  
   - Adjust resource allocation by modifying `ncpus` and `mem` in the qsub script templates within the notebooks.

---

### Final Notes

This workflow provides a scalable and efficient method for verifying forecast data on HPC clusters. While designed for internal projects, contributions are encouraged to improve performance, add metrics, or adapt for other clusters.

If you find gaps or areas that require clarification, feel free to submit issues or pull requests to enhance the repository.