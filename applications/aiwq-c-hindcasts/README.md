# AIWQ-C Hindcasts

## Step I: Run `pregenerate_data_parallel.sh`

Pre-generates hindcast initial conditions and ensemble files (2000-01-01 to 2020-12-31) using a PBS job array. Each array index processes 10 consecutive dates.

```bash
mkdir -p logs
# change the project key if needed...
qsub -J 0-766 pregenerate_data_parallel.sh
```
Also check the instructions in pregenerate_data_parallel.sh

For testing, use a smaller job array range (e.g., 2 jobs = 20 dates):

```bash
qsub -J 0-1 pregenerate_data_parallel.sh
```

## Step II: Run `run_forecasts.sh`

Runs 11 ensemble member hindcast forecasts across 4 GPUs in parallel using a PBS job array. Each array index processes 150 consecutive dates (~10.5 hrs per job).

```bash
mkdir -p logs
qsub -J 0-51 run_forecasts.sh        # all dates (2000-01-01 to 2020-12-31)
qsub -J 0-1  run_forecasts.sh        # first 300 dates only (for testing)
```

Key files:
- `run_forecasts.sh` — PBS job array script; loops over dates, launches members in batches of 4 (one per GPU)
- `hindcast_wrapper.py` — Python wrapper; configures and runs a single member via `rollout_realtime.predict`
- `S2Shindcast.yml` — Model/data config (variables, architecture, paths, output location)

Configuration (in `run_forecasts.sh`):
- `DATES_PER_JOB` — dates per array index (default 150)
- `START_DATE` / `END_DATE` — date range
- `N_MEMBERS` — ensemble size (default 11)
- `N_GPUS` — GPUs per node (default 4, must match `#PBS -l ngpus`)
- Conda env path in the `conda activate` line

Logs:
- PBS stdout/stderr: `logs/forecast_{array_index}.log`
- Per-member logs: `logs/forecast_{DATE}_m{MEMBER}.log`

Output goes to the `save_forecast` path in `S2Shindcast.yml` (default: `/glade/derecho/scratch/negins/CAMulator_Hindcast/`), structured as:
```
{save_forecast}/{DATE}T00Z/pred_{MEMBER}_{DATE}T00Z_{HOUR}.nc
```

## How it works

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STEP I: pregenerate_data_parallel.sh                              │
 │                                                                    │
 │  PBS job array (0-766), 10 dates per job                           │
 │  Creates initial conditions + forcing for each date:               │
 │    ICs.{001..011}.{DATE}-00000.nc                                  │
 │    CESM_dynamicforcing_{DATE}.nc                                   │
 └────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STEP II: run_forecasts.sh                        [~5.5 min/date]  │
 │                                                                    │
 │  PBS job array (0-51), 150 dates per job, ~10.5 hrs per job        │
 │  For each date, runs hindcast_wrapper.py per member:               │
 │                                                                    │
 │  PARALLEL (4 GPUs):                                                │
 │   GPU0: Mbr 001 → Mbr 005 → Mbr 009                               │
 │   GPU1: Mbr 002 → Mbr 006 → Mbr 010                               │
 │   GPU2: Mbr 003 → Mbr 007 → Mbr 011                               │
 │   GPU3: Mbr 004 → Mbr 008                                         │
 │                                                                    │
 │   ┌────────┐  ┌────────┐       ┌────────┐                         │
 │   │ Batch 1│─▶│ Batch 2│─▶ ... │ Batch 3│                         │
 │   │ ~71s   │  │ ~71s   │       │ ~71s   │                         │
 │   └────────┘  └────────┘       └────────┘                         │
 │                                                                    │
 │   Each: 46-day rollout @ 6hr steps = 184 timesteps                 │
 └────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Output: CAMulator_Hindcast/{DATE}T00Z/                            │
 │                                                                    │
 │  pred_{MEMBER}_{DATE}T00Z_{HOUR}.nc                                │
 │  184 files per member x 11 members = 2,024 files per date          │
 └─────────────────────────────────────────────────────────────────────┘
```

## Dependencies

The workflow requires access to:
- `/glade/campaign/cesm/development/cross-wg/S2S/sglanvil/forKirsten/subCESMulator/final/` (source CESM data)
- `/glade/derecho/scratch/kjmayer/CREDIT_runs/S2Socnlndatm_MLdata/` (CESM dynamical forcing)
- `/glade/derecho/scratch/kjmayer/DATA/CESMS2S_inits/` (dynamic forcing init files)
- `/glade/campaign/cesm/development/cross-wg/S2S/CESM2/CAMI/RP/` (perturbation files)
- `/glade/work/cbecker/AIWQ_models/checkpoint.pt` (model weights)
