# CAMulator ↔ POP Coupling via CESM2/CDEPS

This document covers the coupling of CAMulator (the CREDIT AI atmosphere) to POP2 (ocean) + CICE5 (sea ice) inside CESM 2.1.5 on Derecho/Casper.

---

## Quickstart

### Step 1 — Get CESM

```bash
git clone -b release-cesm2.1.5-camulator \
    https://github.com/WillyChap/CESM.git my_camulator_cesm
cd my_camulator_cesm
./manage_externals/checkout_externals
```

This pulls a CESM 2.1.5 tree with the two custom components pre-wired:
- **CIME** (`Cambridge-ICCS/cime_je:coupled_camulator`) — DATM CAMULATOR datamode + SST=0 fix + FTorch build hooks
- **CAM** (`WillyChap/CAM:coupled_camulator`) — FTorch-enabled CAM physics

### Step 2 — Create and build the case

Edit the top of `climate/setup_CAMULATOR_GIAF_case.sh` to set your paths:

```bash
CESM_ROOT=/path/to/my_camulator_cesm
CASE_DIR=/glade/work/$USER/cesm/CREDIT/g.e21.CAMULATOR_GIAF_v01
PROJECT=<YOUR_PROJECT>
```

Then run:

```bash
bash setup_CAMULATOR_GIAF_case.sh
```

This runs `create_newcase`, applies all required `xmlchange` commands (CAMULATOR datamode,
PE layout, MPI GPU env vars, CICE ndtd=2), and builds the model (~20-40 min).

### Step 3 — Configure `camulator_config.yml`

```yaml
predict:
  save_forecast: '/glade/derecho/scratch/<USER>/CREDIT/climate_output/'
  init_cond_fast_climate: '/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/init_times/init_condition_tensor_2000-01-01T00Z.pth'
  start_datetime: '2000-01-01 00:00:00'
  timesteps_fast_climate: 1460   # 6-hr steps = 1 year
```

### Step 4 — Launch the CAMulator server (GPU node, before CESM)

```bash
# On a Casper A100 node:
# qsub -I -A <PROJECT> -l select=1:ncpus=32:ngpus=1:mem=250GB \
#      -l walltime=12:00:00 -q casper -l gpu_type=a100_80gb

conda activate credit-coupling
python camulator_server.py \
    --config ./camulator_config.yml \
    --model_name checkpoint.pt00091.pt \
    --rundir /glade/derecho/scratch/$USER/g.e21.CAMULATOR_GIAF_v01/run/ \
    --save_atm_nc camulator_out \
    --daily_mean
```

Or submit as a PBS job: `qsub Start_Cam_Serve.sh`

### Step 5 — Submit CESM

```bash
cd /glade/work/$USER/cesm/CREDIT/g.e21.CAMULATOR_GIAF_v01
./case.submit
```

The server must be running and waiting **before** CESM reaches its first coupling step.

---

## 1000-Foot View

### What is this?

This project implements the world's first coupling of a **machine-learning atmosphere** to a **dynamic ocean + sea ice** system. CAMulator — a neural network trained to emulate CAM6 (NCAR's Community Atmosphere Model) — replaces the prescribed atmospheric forcing in a standard ocean-only CESM2 configuration. The ocean and sea ice respond to CAMulator's evolving atmospheric state, and the resulting sea-surface temperatures feed back into CAMulator at every 6-hour step.

### Component glossary

| Acronym | Full name | Role in this project |
|---|---|---|
| **CESM2** | Community Earth System Model v2 | The framework that orchestrates all components, handles I/O, and manages the time loop |
| **CPL7** | CESM coupler version 7 (MCT-based) | Fortran coupler: remaps fields between component grids, applies bulk formulae, mediates all data exchange |
| **MCT** | Model Coupling Toolkit | Low-level MPI library under CPL7 that handles parallel field interpolation between components |
| **CDEPS / DATM** | Community Data Model for Earth Prediction System / Data ATMosphere | Normally reads prescribed atmospheric forcing from files (CORE2 IAF). **Here replaced by `CAMULATOR` mode** — a custom datamode that calls our Python server instead of reading static files |
| **CAMulator** | CAM6 emulator (Chapman et al. 2025) | Autoregressive 6-hr ML atmosphere. Takes 3D wind/T/Q + surface state → predicts next 6-hr state + surface fluxes |
| **CREDIT** | Community Research Earth Digital Intelligence Twin | The broader project; CAMulator is its flagship atmosphere model |
| **POP2** | Parallel Ocean Program v2 | 3D ocean circulation model on the gx1v7 displaced-pole grid (~1°). Evolves SST, salinity, currents |
| **CICE** | Los Alamos Sea Ice model v5 | Thermodynamic + dynamic sea ice on the same gx1v7 grid as POP2. Exchanges heat, momentum, freshwater with both POP and the atmosphere |
| **GIAF** | G-compset IAF (Interannual Forcing) | The baseline CESM2 compset: active POP2 + CICE, prescribed DATM + DROF. This project swaps DATM for CAMULATOR mode |
| **IAF** | Interannual Atmospheric Forcing | The CORE2 dataset (NCEP winds, GISS radiation, GCGCS precip, 1948–2009) that DATM normally reads. CAMulator replaces this |
| **T62** | Gaussian grid 94×192 | The DATM grid (lat×lon). All DATM→CPL fields live on this grid. POP/CICE are on gx1v7; CPL7 bilinearly remaps between them |
| **gx1v7** | POP/CICE displaced-pole grid | ~1° nominal resolution ocean/ice grid, ~320×384 |

### Data flow (one 6-hour coupling step)

```
POP2/CICE                CPL7 (Fortran/MPI)            DATM (CAMULATOR mode)          camulator_server.py (Python/GPU)
─────────                ──────────────────            ─────────────────────          ────────────────────────────────
advance ocean/ice   ───► remap SST+ifrac              write camulator_sst_in.nc  ───► read sst_in.nc
(gx1v7 grid)             gx1v7 → T62                  write go.flag                   inject SST into CAMulator state
                                                                                       run one 6-hr CAMulator step
                   ◄─── remap a2x fluxes              read  done.flag            ◄─── write camulator_cam_out.nc
receive forcing          T62 → gx1v7                  read  cam_out.nc                write done.flag
apply to ocean/ice                                     unpack fields → CPL av-ects
```

### Why is this scientifically novel?

Standard ocean-only runs (G-compset) use **prescribed** CORE2/JRA55 climatological forcing — the atmosphere never responds to what the ocean does. With CAMulator coupled:
- The atmosphere sees **actual POP2 SSTs** at every 6-hour step
- Air-sea feedbacks are **interactive**: warm SST anomalies can generate more evaporation, change winds, and feed back on ocean heat flux
- CAMulator evolves its own weather — storms, jet streams, MJO-like variability — all responding to the ocean state

This is qualitatively different from AMIP (atmosphere forced by observed SST) or OMIP (ocean forced by prescribed winds): here **both** components are prognostic.

### Current status

Fully working coupled system as of 2026-03. Performance: ~45 SYPD on Derecho (256 CPU ranks) + 1× Casper A100.

- **CESM fork:** `https://github.com/WillyChap/CESM` branch `release-cesm2.1.5-camulator`
- **Case setup script:** `climate/setup_CAMULATOR_GIAF_case.sh` (in this repo)
- **Server script:** `climate/camulator_server.py` (in this repo)
- **Checkpoint:** `/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/CAMulator_models/checkpoint.pt00091.pt`

---

## Goal

Replace the DATM (Data Atmosphere) component in a CESM2 GIAF case with CAMulator — an autoregressive 6-hour CAM6 emulator (Chapman et al. 2025) — so that POP2 and CICE receive realistic, evolving air-sea fluxes from an AI atmosphere rather than prescribed climatological forcing.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  CESM2 CPL7 coupled run  (Fortran/MPI, CPU ranks)               │
│                                                                 │
│  POP2 ──SST──► CPL ──o2x_SST──► DATM (camulator mode)          │
│                                      │                          │
│                                write sst_in.nc                  │
│                                write go.flag                    │
│                                      │  (poll every ~1s)        │
│                                read  done.flag                  │
│                                read  cam_out.nc                 │
│                                      │                          │
│  POP2 ◄──a2x fluxes──── CPL ◄────────┘                         │
└─────────────────────────────────────────────────────────────────┘
                   ▲ done.flag / cam_out.nc
                   │
┌──────────────────┴──────────────────────────────────────────────┐
│  camulator_server.py  (Python, GPU, pre-launched on same node)  │
│                                                                 │
│  loop:                                                          │
│    watch go.flag                                                │
│    read  sst_in.nc  → inject SST into CAMulator state           │
│    run   one 6-hr CAMulator step  (Model_State / CAMulatorStepper)│
│    write cam_out.nc  (taux, tauy, Qnet, P-E, SW, LW, ...)      │
│    write done.flag                                              │
└─────────────────────────────────────────────────────────────────┘
```

### Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| Inference bridge | File-based NetCDF + flag files | Simplest, fully debuggable; 6-hr interval makes I/O overhead negligible |
| Coupling interval | 6 hours (ATM_NCPL=4) | Matches CAMulator's native timestep |
| DATM mode | New `CAMULATOR` case in `datm_comp_mod.F90` | Reuses all existing CPL7 mapping/merging/diagnostics |
| GPU usage | Python server pre-launched on same node | Avoids Fortran-Python interop complexity; CESM uses node CPUs, CAMulator uses GPU |
| Starting compset | `GIAF` (POP2 + CICE + DATM IAF) | Has DATM slot ready; standard OMIP configuration |
| Machine | `--mach derecho` (CESM) + separate GPU node (server) | CESM MPI ranks run on CPU nodes; CAMulator inference runs in a separate Python process on any GPU node sharing GLADE |

---

## Coupling Field Contract

### OCN → ATM (every 6h, via CPL `x2a` / `o2x` av-ect)

| CPL field name | Description | Units | Notes |
|---|---|---|---|
| `x2a_So_t` | Bulk SST from POP surface layer | K | bilinear remap OCN→ATM grid |
| `x2a_Si_ifrac` | Sea-ice fraction from CICE | 0–1 | optional; used to taper fluxes in polar regions |

### ATM → CPL → OCN (every 6h, 6-hour interval means)

| `a2x` field key | Description | Units | Sign convention |
|---|---|---|---|
| `a2x_Sa_taux` | Zonal wind stress | N m⁻² | positive = westward stress on ocean surface |
| `a2x_Sa_tauy` | Meridional wind stress | N m⁻² | positive = northward stress on ocean surface |
| `a2x_Faxa_swnet` | Net downward SW into ocean | W m⁻² | positive = into ocean (downward) |
| `a2x_Faxa_lwdn` | Downward LW at surface | W m⁻² | positive = into ocean |
| `a2x_Faxa_sen` | Sensible heat flux | W m⁻² | positive = into ocean (downward) |
| `a2x_Faxa_lat` | Latent heat flux | W m⁻² | positive = into ocean (downward) |
| `a2x_Faxa_rainl` | Liquid precipitation | kg m⁻² s⁻¹ | P−E handled by CPL |
| `a2x_Faxa_snowl` | Frozen precipitation | kg m⁻² s⁻¹ | partitioned by T < 273.15K |

**Sign note:** CAMulator outputs use atmosphere-upward convention for fluxes (SHFLX, LHFLX positive = upward from surface). These must be **negated** before populating `a2x` fields (ocean downward-positive).

---

## File & Path Reference

| Item | Path |
|---|---|
| CESM2.1.5 checkout | `/glade/work/wchapman/JE_help_cnn/cesm_FTORCH_FORPY_v8/` |
| DATM source | `.../cime/src/components/data_comps/datm/` |
| Case directory (v01, retired) | `/glade/work/wchapman/cesm/CREDIT/g.e21.CAMULATOR_GIAF_v02/` — used `--mach derecho-gpu`, crashed with GTL MPI error |
| Case directory (v02, active) | `/glade/work/wchapman/cesm/CREDIT/g.e21.CAMULATOR_GIAF_v02/` |
| Run directory | `/glade/derecho/scratch/wchapman/g.e21.CAMULATOR_GIAF_v02/run/` |
| Build directory | `/glade/derecho/scratch/wchapman/g.e21.CAMULATOR_GIAF_v02/bld/` |
| CAMulator config | `./camulator_config.yml` (this directory) |
| CAMulator checkpoint | `/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/CAMulator_models/checkpoint.pt00091.pt` |
| Conda env (CESM build) | `/glade/work/wchapman/miniconda3.2/envs/cesmML3.10gpuPD` |
| Conda env (CREDIT/CAMulator) | `/glade/work/wchapman/conda-envs/credit-coupling` |
| Flag file directory | `/glade/derecho/scratch/wchapman/g.e21.CAMULATOR_GIAF_v02/run/` (same as run dir) |
| New DATM mode file | `.../datm/datm_datamode_camulator.F90` ✅ created |
| Modified DATM driver | `.../datm/datm_comp_mod.F90` ✅ use + case added |
| Python server | `./camulator_server.py` ✅ (this directory) |
| Server smoke test | `./test_camulator_server.py` ✅ (one-step test, no CESM needed) |
| Setup script | `./setup_CAMULATOR_GIAF_case.sh` (this directory) |

---

## Step-by-Step Log

### Step 1 — CESM2 case creation ✅

**Date:** 2026-02-22

**Command used:**
```bash
/glade/work/wchapman/JE_help_cnn/cesm_FTORCH_FORPY_v8/cime/scripts/create_newcase \
  --case /glade/work/wchapman/cesm/CREDIT/g.e21.CAMULATOR_GIAF_v01 \
  --mach derecho-gpu \
  --compiler intel \
  --compset GIAF \
  --res T62_g17 \
  --project P03010039 \
  --run-unsupported
```

**Why `--run-unsupported`:** The `derecho-gpu` + `GIAF` + `T62_g17` combination is not in CESM's tested matrix. The warning is harmless.

**Why `--mach derecho-gpu`:** GPU nodes are needed for CAMulator inference. CESM MPI tasks use the node CPUs; the Python server uses the GPU. Both share the GLADE filesystem.

**Compset breakdown:** `2000_DATM%IAF_SLND_CICE_POP2_DROF%IAF_SGLC_WW3`
- `DATM%IAF` — Data Atmosphere, interannually varying (will be replaced by CAMULATOR mode)
- `SLND` — Stub land (not needed for ocean coupling)
- `CICE` — Sea ice model
- `POP2` — Parallel Ocean Program
- `DROF%IAF` — Data runoff
- `SGLC` — Stub glacier
- `WW3` — Wave Watch III

**Resolution:** `T62_g17` = T62 atmosphere grid (~1.9°), gx1v7 tripole ocean grid (~1°)

---

### Step 2 — xmlchanges and case.setup ✅

**Date:** 2026-02-22

```bash
cd /glade/work/wchapman/cesm/CREDIT/g.e21.CAMULATOR_GIAF_v02

./xmlchange NCPL_BASE_PERIOD=day   # 6-hour coupling: 4 exchanges per day
./xmlchange ATM_NCPL=4
./xmlchange OCN_NCPL=4

./xmlchange STOP_OPTION=ndays,STOP_N=2,RESUBMIT=0   # 2-day smoke test
./xmlchange JOB_QUEUE=develop
./xmlchange JOB_WALLCLOCK_TIME=01:00:00
./xmlchange DOUT_S=FALSE           # no archiving during development

./case.setup
```

---

### Step 3 — case.build ✅

**Date:** 2026-02-22

**Required before building:** activate the CESM ML conda env (the modified `buildexe` requires `CONDA_PREFIX`):

```bash
module load conda
conda activate /glade/work/wchapman/miniconda3.2/envs/cesmML3.10gpuPD
cd /glade/work/wchapman/cesm/CREDIT/g.e21.CAMULATOR_GIAF_v02
./case.build
```

**Result:** `MODEL BUILD HAS FINISHED SUCCESSFULLY`
- All components compiled: datm, pop, cice, drof, ww3, slnd, sglc, sesp
- Link step (cesm exe) succeeded with 27 warnings (all harmless)
- Build log: `/glade/derecho/scratch/wchapman/g.e21.CAMULATOR_GIAF_v02/bld/cesm.bldlog.*`

---

### Step 4 — Add CAMULATOR mode to DATM ✅

**Date:** 2026-02-22

#### 4a. Created `datm_datamode_camulator.F90` ✅

**File:** `/glade/work/wchapman/JE_help_cnn/cesm_FTORCH_FORPY_v8/cime/src/components/data_comps/datm/datm_datamode_camulator.F90`

New Fortran module `datm_datamode_camulator_mod` with one public subroutine:

```fortran
subroutine datm_datamode_camulator_run(x2a, a2x, ggrid, gsmap, &
     mpicom, my_task, master_task, logunit, currentYMD, currentTOD, nxg, nyg)
```

**What it does each coupling step:**
1. On first call: caches `So_t` / `Sf_ifrac` field indices in `x2a`; caches all `a2x` output field indices; allocates global arrays; builds MPI gather/scatter structures
2. `MPI_Gatherv` — collects SST and ice fraction from all MPI ranks onto master task
3. Master writes `camulator_sst_in.nc` (vars: `sst [K]`, `ifrac [0-1]`, `ymd`, `tod`)
4. Master writes `camulator_go.flag` (empty sentinel file)
5. Master polls for `camulator_done.flag` every 1 second; aborts after 3600 iterations (1 hr)
6. Master reads `camulator_cam_out.nc` (vars: `u10`, `v10`, `tbot`, `zbot`, `tref`, `qbot`, `pbot`, `fsds`, `flnsd`, `prect`)
7. Safety clamps: wind ≤ 80 m/s, q ≥ 1e-9 kg/kg, fluxes ≥ 0
8. `shr_mpi_bcast` — broadcasts all output arrays to all ranks
9. `MPI_Scatterv` — scatters local portions back to each rank
10. Populates `a2x` fields:
    - `Sa_z = 10.0` (reference height)
    - `Sa_u`, `Sa_v` (10m winds → CPL7 bulk formula computes stress)
    - `Sa_tbot`, `Sa_ptem` (near-surface temperature)
    - `Sa_shum` (specific humidity)
    - `Sa_dens` (from ideal gas law: `p / (287.04 × Tv)`)
    - `Sa_pbot`, `Sa_pslv` (surface pressure)
    - `Faxa_lwdn` (downward LW)
    - `Faxa_swvdr/swndr/swvdf/swndf/swnet` (downwelling SW `fsds` split 28/31/24/17% fractions; coupler applies ocean/ice albedo internally)
    - `Faxa_rainl/rainc/snowl/snowc` (PRECT × 1000; rain/snow by T threshold)

**Build system:** `buildlib` uses `build_cime_component_lib` which auto-discovers all `.F90` files in the datm directory. No `Filepath` or `CMakeLists.txt` changes needed — the new file is picked up automatically.

**Key constants and conventions:**
- `GO_FLAG = 'camulator_go.flag'`, `DONE_FLAG = 'camulator_done.flag'`
- `SST_FILE = 'camulator_sst_in.nc'`, `CAM_FILE = 'camulator_cam_out.nc'`
- SW fractions (from CORE2 convention): VDR=0.28, NDR=0.31, VDF=0.24, NDF=0.17
- Phase 1: provides `Sa_u/Sa_v` winds; CPL7 bulk formula computes wind stress
- Phase 2 (future): bypass bulk formula by directly populating `Faxx_taux/tauy`

#### 4b. Wired into `datm_comp_mod.F90` ✅

**File:** `.../datm/datm_comp_mod.F90`

Two changes made:

**1. Added `use` statement** (after the existing `datm_shr_mod` uses, line ~27):
```fortran
use datm_datamode_camulator_mod, only: datm_datamode_camulator_run
```

**2. Added `case('CAMULATOR')` branch** in the `select case (trim(datamode))` block (before `end select`, after the `CLMNCEP` case):
```fortran
case('CAMULATOR')
   call datm_datamode_camulator_run( &
        x2a        = x2a,          &
        a2x        = a2x,          &
        ggrid      = ggrid,        &
        gsmap      = gsmap,        &
        mpicom     = mpicom,       &
        my_task    = my_task,      &
        master_task= master_task,  &
        logunit    = logunit,      &
        currentYMD = currentYMD,   &
        currentTOD = currentTOD,   &
        nxg        = SDATM%nxg,    &
        nyg        = SDATM%nyg)
```

**Key discovery during implementation:** `So_t` (SST) IS present in the `x2a` attribute vector for DATM — confirmed by reading `seq_flds_mod.F90` line 1071. No coupler modifications are needed to receive SST in DATM.

#### 4c. IAF baseline smoke test ✅

**Date:** 2026-02-22 — **`case.run success`** (job 5174316.desched1)

With `datamode = IAF` still set (CAMULATOR code compiled in but not activated), a 2-day GIAF run completed successfully in ~65 seconds wall time on 4 GPU nodes (`main` queue). This confirms:
- `datm_datamode_camulator.F90` compiles cleanly (1 harmless warning)
- The `use` statement and `case('CAMULATOR')` block in `datm_comp_mod.F90` do not break any existing modes
- The linked cesm exe runs a full IAF GIAF case end-to-end without error

**Queue note:** The `develop` queue only routes to CPU nodes; use `--force JOB_QUEUE=main` for GPU jobs.

---

### Step 5 — Write camulator_server.py ✅

**Date:** 2026-02-22

**Goal:** Python process that runs alongside the CESM job, serving CAMulator inference over the file-based interface.

**Files created:**
- `climate/camulator_server.py` — production server (daemon, runs indefinitely)
- `climate/test_camulator_server.py` — standalone smoke test (one step, no CESM required)

#### Design compared to Quick_Climate.py

`camulator_server.py` follows exactly the same step-loop structure as `Quick_Climate.py`:
- Same `initialize_camulator()` one-time setup
- Same first-step special case (`timestep == 0` → use `initial_state` directly)
- Same `torch.jit.trace` optimization (trace once at startup, reuse frozen graph)
- Same `build_input_with_forcing` + `shift_state_forward` pattern
- Same `stepper.model(model_input.float())` + `_apply_postprocessing` + `inverse_transform` sequence

Key additions vs Quick_Climate.py (because we're driving from real ocean SST):
1. T62 ↔ CAMulator grid remapping (`remap_field` with `scipy.RegularGridInterpolator` + lon wrap-around)
2. Gaussian lat computation (`scipy.special.roots_legendre(94)`)
3. SST normalization and injection via `accessor_input.set_state_var`
4. Flag-file polling loop (`camulator_go.flag` / `camulator_done.flag`)

**Note on `torch.jit.trace`:** The server traces with `state.float()` (real IC tensor, not zeros). Using `torch.zeros_like` triggers a divide-by-zero in spectral norm layers on some PyTorch builds. PyTorch 2.10+ (e.g. `credit-feb2026`) crashes entirely; PyTorch 2.4.1 (`credit-coupling`) traces cleanly.

#### Key facts about CAMulator internals (discovered 2026-02-22)

**Grid dimensions:**
- T62 DATM grid (what CESM sees): 94 lat × 192 lon = **18,048 points** (flattened 1D in NetCDF)
- CAMulator internal grid: **192 lat × 288 lon** = 55,296 points (1° resolution)
- The server must remap T62 → 192×288 for SST injection, then remap 192×288 → T62 for output

**SST is a `dynamic_forcing_variable`** (confirmed in `camulator_config.yml`):
```yaml
dynamic_forcing_variables: ['SOLIN','SST','ICEFRAC','co2vmr_3d']
```
This means SST lives in the normalized forcing tensor at each timestep, **not** in the prognostic state tensor. To inject ocean SST from POP, the server must:
1. Read physical SST [K] from `camulator_sst_in.nc`
2. Remap to 192×288 (bilinear)
3. **Normalize** using the SST mean/std from the model's scaler
4. Call `accessor_input.set_state_var(model_input, 'SST', normalized_sst)` after `build_input_with_forcing`
5. Run the model on the modified input

**Output variables extracted from `prediction` (after `inverse_transform` to physical units):**

| CAMulator var | Units | Conversion for cam_out.nc | DATM `a2x` target |
|---|---|---|---|
| `U` (lowest model lev) | m/s | pick lowest level | `Sa_u` → `u10` |
| `V` (lowest model lev) | m/s | pick lowest level | `Sa_v` → `v10` |
| `T` (lowest model lev) | K | pick lowest level | `Sa_tbot` → `tbot` |
| `T` (lowest model lev) | K | `z_bot = 0.2187 × T_bot` | `Sa_z` → `zbot` |
| `TREFHT` | K | direct | `tref` in nc (diagnostic only, not wired to a2x) |
| `Qtot` (lowest level) | kg/kg | pick lowest level | `Sa_shum` → `qbot` |
| `PS` | Pa | direct | `Sa_pbot` → `pbot` |
| `FSNS` | model units | `/ 21600` → W m⁻²; then `→ FSDS` via albedo inversion | `Faxa_sw*` → `fsds` |
| `FLNS` | model units | `/ -21600` → W m⁻² downward | `Faxa_lwdn` → `flnsd` |
| `PRECT` | m/s liq-eq | direct | `Faxa_rain/snow` → `prect` |

**Unit conversion note (from `Get_Coupling_Vars.py`):**
- `FSNS /= 21600` → converts J/m² per 6h to W/m² (divide by 6hr in seconds)
- `FLNS /= -21600` → FLNS is net upward LW (positive = energy leaving surface), so negating and dividing gives downward LW [W/m²]

**FSNS → FSDS conversion (albedo inversion, Section 11):**
- CAMulator outputs **FSNS** (net SW = downwelling − upwelling, after surface albedo applied by CAM6).
- CPL7 `seq_flux_mct.F90` treats `Faxa_sw*` as **downwelling** SW and applies ocean/ice albedo to compute reflected SW.
- Passing FSNS directly would double-count the surface albedo: SW_absorbed = FSNS × (1 − α_ocean) ≈ FSNS × 0.94.
- Fix: reconstruct FSDS = FSNS / (1 − α_sfc) where α_sfc = (1 − ifrac) × 0.06 + ifrac × 0.60.
- Ice fraction `ifrac` comes from CICE via the same coupling step's `sst_in.nc`, so it is timestep-consistent.
- Floor `1 − α_sfc` at 0.10 and cap FSDS at 1500 W/m² to prevent outliers at high ice fractions.

**Model_State.py API used by the server:**
```python
from Model_State import initialize_camulator, StateVariableAccessor

ctx = initialize_camulator(config_path, model_name='checkpoint.pt00091.pt')
stepper = ctx['stepper']
state = ctx['initial_state']
state_transformer = ctx['state_transformer']
forcing_ds_norm = ctx['forcing_dataset']
static_forcing = ctx['static_forcing']

accessor_input  = StateVariableAccessor(conf, tensor_type='input')
accessor_output = StateVariableAccessor(conf, tensor_type='output')

# Each step:
model_input = stepper.state_manager.build_input_with_forcing(state, dynamic_forcing_t, static_forcing)
accessor_input.set_state_var(model_input, 'SST', normalized_sst_tensor)  # inject ocean SST
with torch.no_grad():
    prediction = stepper.model(model_input.float())
prediction = stepper._apply_postprocessing(prediction, model_input)
prediction_out = state_transformer.inverse_transform(prediction)
# extract vars...
state = stepper.state_manager.shift_state_forward(state, prediction)
```

**First-step special case:** On timestep 0, the initial state already contains forcing — do not call `build_input_with_forcing`, use `state` directly as `model_input`. (Same pattern as `Get_Coupling_Vars.py` line 230–235.)

**Logic:**
```
initialize_camulator(config, checkpoint)
timestep_counter = 0
loop:
    wait for go.flag  (poll with os.path.exists, sleep 0.5s)
    remove go.flag
    read sst_in.nc  → remap T62→192×288 → normalize → inject into model_input
    if timestep_counter == 0:
        model_input = initial_state  (forcing already embedded)
    else:
        model_input = build_input_with_forcing(state, dynamic_forcing_t, static_forcing)
        inject normalized SST into model_input
    run prediction = stepper.model(model_input) + _apply_postprocessing
    inverse_transform prediction
    extract U,V,TS,Qtot,PS,FSNS,FLNS,PRECT from lowest model level
    remap 192×288 → T62 (94×192) → flatten to 18,048 pts
    write cam_out.nc
    write done.flag
    state = shift_state_forward(state, prediction)
    timestep_counter += 1
```

**Test standalone (run before CESM coupling to confirm model loads and inference works):**
```bash
conda activate /glade/work/wchapman/conda-envs/credit-coupling
cd /glade/work/wchapman/Roman_Coupling/credit_feb182026/climate
python test_camulator_server.py \
  --config ./camulator_config.yml \
  --model_name checkpoint.pt00091.pt
# Expect: "SMOKE TEST PASSED" at end; prints field stats for all 8 output variables
```

**Launch (must be running before CESM job starts):**
```bash
conda activate /glade/work/wchapman/conda-envs/credit-coupling
cd /glade/work/wchapman/Roman_Coupling/credit_feb182026/climate
python camulator_server.py \
  --config ./camulator_config.yml \
  --model_name checkpoint.pt00091.pt \
  --rundir /glade/derecho/scratch/wchapman/g.e21.CAMULATOR_GIAF_v02/run/ \
  --init_cond /path/to/init_tensor.pth
```

---

### Step 6 — Switch case to CAMULATOR mode and run 🔲 TODO

This step converts the existing IAF-validated case into a live CAMulator-coupled run.
Do this only after `camulator_server.py` is written and locally tested (Step 5).

#### 6a. Prerequisites checklist

- [x] `camulator_server.py` exists in `climate/` ✅
- [x] Standalone smoke test passed (`credit-coupling` env, Casper A100, 2026-02-23) ✅
- [ ] Initial condition `.pth` file exists for the desired start date:
  ```
  /glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/init_times/
  init_camulator_condition_tensor_YYYY-MM-DDTHhz.pth
  ```
  If not, generate one first:
  ```bash
  conda activate /glade/work/wchapman/conda-envs/credit-coupling
  cd /glade/work/wchapman/Roman_Coupling/credit_feb182026/climate
  python Make_Climate_Initial_Conditions.py -c ./camulator_config.yml \
    --model_name checkpoint.pt00091.pt
  ```

#### 6b. Register CAMULATOR in the DATM namelist definition ✅

**File:** `/glade/work/wchapman/JE_help_cnn/cesm_FTORCH_FORPY_v8/cime/src/components/data_comps/datm/cime_config/namelist_definition_datm.xml`

`buildnml` validates `datamode` against a whitelist in this XML file. `CAMULATOR` must be added in two places:

**1. `valid_values` attribute (line ~2767):**
```xml
<valid_values>CLMNCEP,COPYALL,CORE2_NYF,CORE2_IAF,CORE_IAF_JRA,CORE_RYF_JRA,NULL,CAMULATOR</valid_values>
```

**2. `<values>` block (after the CPLHIST entry, line ~2820):**
```xml
<value datm_mode="CAMULATOR">CAMULATOR</value>
```

Without this, `./case.build` aborts with:
```
Invalid values ['CAMULATOR']
ERROR: Variable 'datamode' from file 'user_nl_datm' has invalid value ["'CAMULATOR'"].
```

This is a one-time change to the CESM source tree — it persists across rebuilds.

#### 6c. Register CAMULATOR in the Fortran runtime whitelist ✅

**File:** `/glade/work/wchapman/JE_help_cnn/cesm_FTORCH_FORPY_v8/cime/src/components/data_comps/datm/datm_shr_mod.F90`

There is a **second** datamode validation in `datm_shr_read_namelists` (subroutine in `datm_shr_mod.F90`) that is separate from the XML buildnml check. This one fires at runtime (during `datm_comp_init`) and will abort with:

```
(datm_comp_init)  ERROR illegal datm datamode = CAMULATOR
```

Fix: add `CAMULATOR` to the `if` block at line ~160:

```fortran
    if (trim(datamode) == 'NULL'      .or. &
         trim(datamode) == 'CORE2_NYF' .or. &
         trim(datamode) == 'CORE2_IAF' .or. &
         trim(datamode) == 'CORE_IAF_JRA' .or. &
         trim(datamode) == 'CORE_RYF_JRA' .or. &
         trim(datamode) == 'CLMNCEP'   .or. &
         trim(datamode) == 'COPYALL'   .or. &
         trim(datamode) == 'CAMULATOR' ) then   ! ← added
```

**Two-layer validation summary:**

| Layer | File | When it fires | Error message |
|---|---|---|---|
| 1 (build) | `namelist_definition_datm.xml` | `./case.build` / `buildnml` | `Invalid values ['CAMULATOR']` |
| 2 (runtime) | `datm_shr_mod.F90` | CESM startup, `datm_comp_init` | `ERROR illegal datm datamode = CAMULATOR` |

Both must be patched. After editing `datm_shr_mod.F90`, a full rebuild is required so the new Fortran gets compiled into the `cesm.exe`.

#### 6d. Set `datamode = 'CAMULATOR'` in the CESM namelist

```bash
# Append to user_nl_datm (idempotent — check file first to avoid duplicates)
echo "datamode = 'CAMULATOR'" >> \
  /glade/work/wchapman/cesm/CREDIT/g.e21.CAMULATOR_GIAF_v02/user_nl_datm
```

Verify:
```bash
cat /glade/work/wchapman/cesm/CREDIT/g.e21.CAMULATOR_GIAF_v02/user_nl_datm
```

The file should contain `datamode = 'CAMULATOR'` and nothing else active (no conflicting `datamode` lines from the IAF placeholder comments).

#### 6e. Rebuild

The Fortran is already compiled; this just regenerates namelists with the new datamode:

```bash
module load conda && conda activate /glade/work/wchapman/miniconda3.2/envs/cesmML3.10gpuPD
cd /glade/work/wchapman/cesm/CREDIT/g.e21.CAMULATOR_GIAF_v02
./case.build
```

#### 6f. Launch the Python server on a GPU node

The server must be running **before** `./case.submit`. It watches the CESM run directory for `camulator_go.flag`.

Option A — run interactively in a separate terminal on the same GPU node:
```bash
# In a separate terminal / screen session on a derecho-gpu node
conda activate /glade/work/wchapman/conda-envs/credit-coupling
cd /glade/work/wchapman/Roman_Coupling/credit_feb182026/climate
python camulator_server.py \
  --config ./camulator_config.yml \
  --model_name checkpoint.pt00091.pt \
  --rundir /glade/derecho/scratch/wchapman/g.e21.CAMULATOR_GIAF_v02/run/ \
  --init_cond /glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/init_times/init_camulator_condition_tensor_0001-01-01T00z.pth
```

Option B — wrap in a PBS job that runs alongside the CESM job (for longer runs):
```bash
# Submit server job first, note the node it lands on, then submit CESM job
# (both jobs must share the same GLADE filesystem — they do on Derecho)
```

#### 6g. Submit the CESM job

```bash
cd /glade/work/wchapman/cesm/CREDIT/g.e21.CAMULATOR_GIAF_v02
./case.submit
```

The DATM will immediately begin cycling go.flag → done.flag with the Python server.

Watch the coupler log for CAMULATOR messages:
```bash
tail -f /glade/derecho/scratch/wchapman/g.e21.CAMULATOR_GIAF_v02/run/cpl.log.*
```

Watch the Python server terminal for inference timing and any errors.

#### 6h. Verification checklist (2-day = 8 coupling steps)

- [ ] 8 `camulator_go.flag` / `camulator_done.flag` cycles complete without timeout (1hr limit)
- [ ] `camulator_sst_in.nc` has plausible SST values (270–305 K range)
- [ ] `camulator_cam_out.nc` has plausible winds (~2–15 m/s), FSNS (~0–400 W m⁻²), FLNSD (~200–450 W m⁻²)
- [ ] No NaNs in `a2x` fields — check coupler log:
  ```bash
  grep -i "nan\|inf\|abort" /glade/derecho/scratch/wchapman/g.e21.CAMULATOR_GIAF_v02/run/cpl.log.*
  ```
- [ ] `CaseStatus` ends with `case.run success`
- [ ] SST tendency in POP has correct sign relative to net heat flux
- [ ] Global mean Qnet plausible (~−10 to +10 W m⁻² acceptable for smoke test)

---

## Rebuild Cheatsheet

After any source modification:

```bash
# Always activate conda first
module load conda && conda activate /glade/work/wchapman/miniconda3.2/envs/cesmML3.10gpuPD

cd /glade/work/wchapman/cesm/CREDIT/g.e21.CAMULATOR_GIAF_v02

# Clean rebuild (required for CPL7 driver changes: cime_comp_mod.F90, component_mod.F90, etc.)
./case.build --clean && ./case.build

# Fast relink only (if only datm_comp_mod.F90 / datm_datamode_camulator.F90 changed)
cd /glade/derecho/scratch/wchapman/g.e21.CAMULATOR_GIAF_v02/bld
gmake -j 8 cesm
```

**Which files require a clean rebuild vs fast relink:**

| Changed file | Rebuild needed |
|---|---|
| `datm_datamode_camulator.F90` | fast relink (`gmake -j8 cesm`) |
| `datm_comp_mod.F90` | fast relink |
| `datm_shr_mod.F90` | fast relink |
| `cime_comp_mod.F90` | **clean rebuild** (`--clean`) |
| `component_mod.F90` | **clean rebuild** |
| `prep_atm_mod.F90` | **clean rebuild** |

---

## Troubleshooting

### Build fails with `KeyError: 'CONDA_PREFIX'`
Forgot to activate conda before building. Run:
```bash
module load conda && conda activate /glade/work/wchapman/miniconda3.2/envs/cesmML3.10gpuPD
```

### `create_newcase` stops with "untested compset/grid"
Add `--run-unsupported` to the command.

### `WARNING: User-selected machine 'derecho-gpu' does not match probed machine 'derecho'`
Harmless — occurs on login nodes. The case is correctly configured for derecho-gpu.

### `MPIDI_CRAY_init: GPU_SUPPORT_ENABLED is requested, but GTL library is not linked`

Cray MPICH (`8.1.27`, as shipped in `ncarenv/23.09`) enables GPU-aware transport on all Derecho nodes by default. `cesm.exe` is not linked with the GTL library, so it aborts at MPI init.

**Required fix — three env vars in `env_mach_specific.xml`** (no rebuild needed, applies to any case):

```xml
<environment_variables>
  ...
  <env name="MPICH_GPU_SUPPORT_ENABLED">0</env>
  <env name="FI_CXI_DISABLE_HOST_REGISTER">1</env>
  <env name="MPICH_SMP_SINGLE_COPY_MODE">NONE</env>
</environment_variables>
```

All three are necessary:
- `MPICH_GPU_SUPPORT_ENABLED=0` — disables cray-mpich GPU transport requirement
- `FI_CXI_DISABLE_HOST_REGISTER=1` — prevents libfabric CXI from trying to register GPU memory (without this, the above causes a SIGSEGV in `fi_getinfo` → `ofi_hmem_init`)
- `MPICH_SMP_SINGLE_COPY_MODE=NONE` — disables single-copy mode which also has CUDA dependencies

**Setting only `MPICH_GPU_SUPPORT_ENABLED=0` causes a SIGSEGV** in libfabric (crash in `fi_getinfo`/`ofi_hmem_init`) because libfabric still tries to initialize CUDA HMEM. All three variables must be set together.

After editing, just resubmit (`./case.submit`) — no rebuild required. Confirmed working on v03 (stock IAF, 2026-02-23) and applied to v02 (CAMULATOR mode).

**Note on v01 vs v02:** v01 used `--mach derecho-gpu` and also hit this error. Switching to `--mach derecho` (v02) landed on a CPU node but still crashed for the same reason. The fix is these three env vars, not the machine type. Use `--mach derecho` for all future CESM GIAF cases (CPU nodes are sufficient — CESM never needs the GPU).

### Build fails with `Invalid values ['CAMULATOR']`
The XML namelist definition hasn't been patched. See Step 6b — add `CAMULATOR` to `valid_values` and `<values>` block in `namelist_definition_datm.xml`.

### Runtime abort: `(datm_comp_init) ERROR illegal datm datamode = CAMULATOR`
The Fortran runtime whitelist in `datm_shr_mod.F90` hasn't been patched. See Step 6c — add `.or. trim(datamode) == 'CAMULATOR'` to the `if` block in `datm_shr_read_namelists`, then rebuild.

### DATM aborts with "unknown datamode" (other)
`datamode = 'CAMULATOR'` set in `user_nl_datm` but the new mode hasn't been compiled in yet. Rebuild after adding the new Fortran code.

### `case.submit` fails with "Job violates queue resource limits"
The `develop` queue only accepts CPU jobs. GPU nodes require the `main` routing queue:
```bash
cd /glade/work/wchapman/cesm/CREDIT/g.e21.CAMULATOR_GIAF_v02
./xmlchange --force JOB_QUEUE=main
./case.submit
```

### Python server misses `go.flag`
Check that `--rundir` in `camulator_server.py` matches `RUNDIR` in the CESM case:
```bash
cd /glade/work/wchapman/cesm/CREDIT/g.e21.CAMULATOR_GIAF_v02
./xmlquery RUNDIR
```

---

## Step 7 — First live coupled run and SST=0 fix 🔲 STILL DEBUGGING

### 7a. First coupled run ✅ (2026-02-23)

Launched `camulator_server.py` on a GPU node and submitted v02. The go/done flag handshake worked perfectly — **8 coupling cycles completed without timeout.** Inference per step: ~0.19s (after JIT trace warmup of ~23s at step 0).

However, all steps showed `SST min=0.0 K max=0.0 K mean=0.0 K`. The atmospheric outputs diverged unphysically (FSNS growing from 239 → 444 W/m², |U10| from 7 → 18 m/s over 8 steps) because CAMulator was being driven with zero SST instead of realistic ocean temperatures.

### 7b. Root cause: CPL7 does not exchange x2a with DATA atmosphere ❌

**The architectural problem:** In the CPL7/MCT driver (`cime_comp_mod.F90`), the coupler→atmosphere data exchange is guarded by `atm_prognostic`:

```fortran
! cime_comp_mod.F90  line ~3436 (ORIGINAL)
if (iamin_CPLALLATMID .and. atm_prognostic) then
   call component_exch(atm, flow='x2c', ...)   ! sends merged x2a to ATM
endif
```

For a **DATA atmosphere (DATM)**, `atm_prognostic = .false.`. This means:
1. `prep_atm_mrg` IS called and correctly fills the coupler's `x2c_cx` buffer with POP's WOA13 SST mapped to the T62 grid
2. But `component_exch(atm, flow='x2c')` is **never called** for DATM
3. DATM's internal `x2c_cc` buffer stays at zero throughout the run
4. `datm_datamode_camulator.F90` reads `x2a_So_t = 0` on every step

This is a CPL7 architectural gap: DATA atmosphere components are designed to SEND forcing to the ocean and never need to RECEIVE ocean state. No standard DATM mode (IAF, CORE2, JRA) uses `x2a` fields, so this gap was never hit before CAMULATOR.

**Confirmed by tracing the full data path:**
- POP initializes from WOA13 (`ts_WOA13v2_jan_ic_gx1v7_20170706.ieeer8`) — SST is non-zero ✅
- `prep_atm_calc_o2x_ax` remaps `o2x_ox%So_t` (POP grid) → `o2x_ax%So_t` (T62 grid) ✅
- `prep_atm_mrg` merges `o2x_ax%So_t * ofrac` → `atm%x2c_cx%So_t` ✅
- `component_exch(atm, flow='x2c')` **SKIPPED** for DATA ATM ❌ → `x2c_cc%So_t = 0` forever

### 7c. Full root cause: TWO guards block x2a delivery to DATA ATM

There are **two separate `atm_prognostic` guards** in `cime_comp_mod.F90` that both need to be changed. The ATM SETUP-SEND section (shared by ALL coupling options including RASM_OPTION1) looks like this:

```fortran
! --- ATM prep-merge (ORIGINAL, lines ~3394) ---
if (iamin_CPLID .and. atm_prognostic) then           ! GUARD #1
    ...
    call prep_atm_calc_o2x_ax(fractions_ox, ...)     ! maps POP SST to T62 grid
    ...
    call prep_atm_mrg(infodata, fractions_ax, ...)   ! merges into x2c_cx
    ...
endif

! --- CPL → ATM exchange (ORIGINAL, lines ~3436) ---
if (iamin_CPLALLATMID .and. atm_prognostic) then     ! GUARD #2
    call component_exch(atm, flow='x2c', ...)        ! maps x2c_cx → x2c_cc
endif
```

For DATA ATM (`atm_prognostic = .false.`, `atm_present = .true.`):
- **Guard #1 blocks prep-merge** → `x2c_cx` is never populated → stays zero
- **Guard #2 blocks exchange** → `x2c_cc` is never updated even if cx had data

After the first clean rebuild (Guard #2 only fixed), `x2c_cx` was still zeros because Guard #1 was still blocking `prep_atm_mrg`. The exchange ran but mapped zeros.

**Data path with both guards blocking:**
```
POP SST (WOA13) → prep_atm_calc_o2x_ax → o2x_ax ─BLOCKED─► x2c_cx  ─BLOCKED─► x2c_cc = 0
```

**Data path with both guards fixed:**
```
POP SST (WOA13) → prep_atm_calc_o2x_ax → o2x_ax → prep_atm_mrg → x2c_cx → component_exch → x2c_cc → DATM
```

### 7d. Two-part fix applied to cime_comp_mod.F90 🔧 (rebuild in progress)

**File:** `/glade/work/wchapman/JE_help_cnn/cesm_FTORCH_FORPY_v8/cime/src/drivers/mct/main/cime_comp_mod.F90`

**Fix #1 — ATM prep-merge guard (line ~3394):**
```fortran
! ORIGINAL:
if (iamin_CPLID .and. atm_prognostic) then

! MODIFIED:
if (iamin_CPLID .and. (atm_prognostic .or. atm_present)) then
   ! NOTE: atm_present so DATA atmosphere (DATM CAMULATOR) also runs the
   ! full prep-merge block: maps POP SST to T62 grid and merges into x2c_cx.
   ! Standard DATA modes (IAF, CORE2) have no consequence — they ignore x2a.
```

**Fix #2 — CPL→ATM exchange guard (line ~3440):**
```fortran
! ORIGINAL:
if (iamin_CPLALLATMID .and. atm_prognostic) then

! MODIFIED:
if (iamin_CPLALLATMID .and. (atm_prognostic .or. atm_present)) then
   ! NOTE: so that x2c_cx (now populated by fix #1) is mapped to x2c_cc
   ! where DATM's datm_datamode_camulator_run reads So_t.
```

Both changes are in the "ATM SETUP-SEND" block (lines 3385–3451), which has **no** `cpl_seq_option` conditional — it is shared by CESM1_ORIG, CESM1_MOD, and RASM_OPTION1 alike. The fix is correct for all coupling modes.

**Clean rebuild triggered** (`./case.build --clean && ./case.build`) on 2026-02-23. After rebuild, SST **still 0.0 K** — both `atm_prognostic` guards are fixed but something else upstream is blocking the ocean→ATM field path.

### 7e. `ocn_c2_atm` hypothesis — DISPROVED ✅ (2026-02-24)

The hypothesis was that `ocn_c2_atm = .false.` for DATA ATM. **This is wrong.**

In `cime_comp_mod.F90` at line ~1469:
```fortran
if (ocn_present) then
   if (atm_prognostic) ocn_c2_atm = .true.
   if (atm_present   ) ocn_c2_atm = .true.  ! set for DATA ATM too!
endif
```

So `ocn_c2_atm = .true.` for DATA ATM, and `prep_atm_calc_o2x_ax` IS called. The `o2x_ax` array IS populated with POP SST mapped to the T62 grid.

**Build confirmed:** Source modified 13:36, object compiled 13:45, exe linked 13:45 on 2026-02-23. Both guards (Fix #1 and Fix #2) were in the binary used for run 5183301.

### 7f. True root cause: `xao_ax` null pointer — `prep_atm_mrg` never called ❌ (2026-02-24)

With both guards fixed, the ATM SETUP-SEND block now executes for DATA ATM. But inside the block:

```fortran
! cime_comp_mod.F90 line ~3425
if (associated(xao_ax)) then          ! <── THIS IS .FALSE. FOR DATA ATM
   call prep_atm_mrg(infodata, fractions_ax, xao_ax=xao_ax, ...)
endif
```

**Why is `xao_ax` null?**

`xao_ax` is a module-level pointer in `cime_comp_mod.F90` (line 195): `type(mct_aVect), pointer :: xao_ax(:) => null()`.

It is only assigned to `prep_aoflux_get_xao_ax()` in the INIT phase (line ~1974):
```fortran
! cime_comp_mod.F90 lines ~1943-1985
if (atm_prognostic) then          ! <── entire INIT block is ATM_PROGNOSTIC guarded
   if (iamin_CPLID) then
      ...
      if (lnd_present .or. ocn_present) then
         xao_ax => prep_aoflux_get_xao_ax()   ! sets pointer — skipped for DATA ATM
         ...
      endif
   endif
endif  ! atm_prognostic
```

For DATA ATM (`atm_prognostic = .false.`), this INIT block is skipped entirely. `xao_ax` stays `null()`. Later in the time loop, `if (associated(xao_ax))` = `.false.` → `prep_atm_mrg` is never called → `x2c_cx%So_t` stays zero → SST = 0.

**Important:** `prep_aoflux_init` (which allocates the underlying `xao_ax` array in `prep_aoflux_mod`) IS called unconditionally (line ~1882, guarded only by `iamin_CPLID`). So `prep_aoflux_get_xao_ax()` always returns an associated pointer. The `cime_comp_mod.F90` local pointer just needs to be assigned.

**Data path with Fix #3 needed:**
```
o2x_ax (populated) → if (associated(xao_ax)) → NULL! → prep_atm_mrg SKIPPED → x2c_cx = 0
```

### 7g. Fix #3 — assign `xao_ax` before `prep_atm_mrg` call (2026-02-24) 🔧

**File:** `/glade/work/wchapman/JE_help_cnn/cesm_FTORCH_FORPY_v8/cime/src/drivers/mct/main/cime_comp_mod.F90`

**Change (inside our new `atm_present` prep-merge block, before line ~3431):**

```fortran
! ADDED (before the if (associated(xao_ax)) check):
! NOTE: For DATA ATM (atm_present but not atm_prognostic), xao_ax is null
! because the INIT phase assignment (line ~1974) is guarded by atm_prognostic.
! prep_aoflux_init always allocates xao_ax unconditionally, so we can always
! retrieve the pointer here before the associated() check.
xao_ax => prep_aoflux_get_xao_ax()

if (associated(xao_ax)) then
   call prep_atm_mrg(infodata, fractions_ax, xao_ax=xao_ax, timer_mrg='CPL:atmprep_mrgx2a')
endif
```

**Expected after rebuild:** On first call, `prep_atm_merge` prints its Summary to `cpl.log` → diagnostic confirmation. SST in DATM should become non-zero.

**Three-part fix summary (all three required):**

| Fix | Location (cime_comp_mod.F90) | What it does |
|---|---|---|
| Fix #1 | Line ~3394: `atm_prognostic` → `(atm_prognostic .or. atm_present)` | Allows prep-merge block to run for DATA ATM |
| Fix #2 | Line ~3446: `atm_prognostic` → `(atm_prognostic .or. atm_present)` | Allows `component_exch(x2c)` to run for DATA ATM |
| Fix #3 | Line ~3429: add `xao_ax => prep_aoflux_get_xao_ax()` | Ensures `xao_ax` pointer is set so `prep_atm_mrg` is actually called |

**Status as of 2026-02-24: Fix #3 applied to source. Needs clean rebuild and rerun.**

**Status as of 2026-02-25: THREE-PART FIX CONFIRMED WORKING.** 10-day run (job 5197394) completed at 45 SYPD. `SUCCESSFUL TERMINATION` written to `fort.99` (not `cesm.log` — run script false-alarms but science is correct).

---

## Section 8. CICE CFL crash fix — `ndtd = 2` (2026-02-25)

### 8a. Symptom

After ~12 days of coupled integration (job 5198188), CICE aborted with:
```
ERROR: remap transport: bad departure points
  Global i and j:  54  379
  dpx, dpy =  15946.4   -1142.4
  HTN(i,j), HTN(i+1,j) = 15848.7   15704.2
```

The ice departure point (`dpx = 15946 m`) exceeded the grid-cell width (`HTN = 15848 m`), i.e., Courant number > 1 in the remap transport scheme. Location is in the Arctic (high-latitude region of the gx1v7 grid).

### 8b. Root cause

With `ndtd = 1` (default), CICE dynamics and advection run **once per 6-hr coupling interval** (dt = 21600 s). The CFL limit at the crash location is:

```
U_max = HTN / dt = 15848 / 21600 ≈ 0.73 m/s
```

Ice entered the run at max speed ~0.63 m/s (from restart) and was gradually accelerated by CAMulator wind stress over 12 days until it exceeded 0.73 m/s. `camulator_cam_out.nc` at crash time showed max wind speed **27.6 m/s** with 7 points > 25 m/s.

### 8c. Fix — `ndtd = 2` in `user_nl_cice`

**File:** `/glade/work/wchapman/cesm/CREDIT/g.e21.CAMULATOR_GIAF_v02/user_nl_cice`

```fortran
ndtd = 2
```

This is a **runtime namelist parameter** — no rebuild required. With `ndtd = 2`, CICE subcycles dynamics and advection **twice** per coupling interval (effective dt = 10800 s), doubling the CFL limit to ~1.47 m/s. The `ndte = 120` EVP subcycles scale with `ndtd` automatically.

The change lives in `user_nl_cice` (the case directory) so it persists across all future submissions and is not overwritten when namelists are regenerated. After editing `user_nl_cice`, regenerate namelists before the next submit:

```bash
cd /glade/work/wchapman/cesm/CREDIT/g.e21.CAMULATOR_GIAF_v02
./preview_namelists
```

This writes `ndtd = 2` into the run directory's `ice_in` automatically.

---

## Section 9. Reference height fix — `Sa_z = 10m → 50m` (2026-02-25, superseded by Section 11)

### Problem

`datm_datamode_camulator.F90` was hardcoding `Sa_z = 10.0 m` (the CORE2 IAF convention, where forcing fields really are at 10m). CAMulator outputs U, V, T, Q from its **bottom model level**, not from 10m. With `Sa_z = 10m`, CPL7's bulk formula (`shr_flux_atmOcn` in `shr_flux_mod.F90`) computes `alz = log(10/10) = 0` — **no Monin-Obukhov height correction at all**.

### Initial fix (intermediate step)

Changed hardcoded value from 10 → 50 m:
```fortran
a2x%rAttr(kz, n) = 50.0_R8   ! was 10.0_R8
```

This was later found to be wrong in magnitude and to use TREFHT (2 m diagnostic) instead of the true bottom level T. See **Section 11** for the correct dynamic implementation.

---

## Section 10. Dynamic z_bot and T_bottom_level fix (2026-02-25)

### What we found

Checking the actual CAM6 L32 hybrid coordinate coefficients from the statics file:
```
hybi[-2] = 0.98511219   (top interface of the bottom model layer)
hybi[-1] = 1.00000000   (surface)
```

**Bottom model level midpoint** (hypsometric formula, PS-independent):
```
p_mid_frac = 0.5 * (hybi[-2] + 1.0) = 0.99255610
z_bot = (Rd/g) * (-ln(p_mid_frac)) * T_bot
      ≈ 0.2187 * T_bot   [m, with T_bot in K]
```

| Condition | T_bot (K) | z_bot (m) | alz = ln(z/10) |
|---|---|---|---|
| Polar winter | 230 | 50.3 | 1.62 |
| Mid-latitude | 280 | 61.2 | 1.81 |
| Tropics      | 305 | 66.7 | 1.90 |
| **Old fixed value** | — | **50** | **1.61** |

The fixed 50 m was accidentally correct for polar regions but **12–18% too low for mid-latitudes and tropics**, causing undercorrection of wind stress and heat fluxes in warmer regions.

**Second issue found simultaneously:** `tbot` (used as `Sa_tbot` → `thbot` in bulk formula) was set to **TREFHT** (the 2 m diagnostic from CAM's own stability parameterization), but the bulk formula expects **temperature at the bottom model level**. Using TREFHT with z_bot ≈ 60 m is physically inconsistent — it mixes a 2 m diagnostic with a 60 m reference height.

### Fix (2026-02-25)

**Two changes together:**

**1. Python — `camulator_server.py`:**
- Load `hybi[-2]` from statics file at startup; precompute `Z_BOT_SCALE = (Rd/g)*(-ln(0.5*(hybi[-2]+1.0)))` ≈ 0.2187 m/K
- Extract `T_bot_cam = T[0, -1, 0]` (bottom model level T, not TREFHT)
- Compute `z_bot_cam = Z_BOT_SCALE * T_bot_cam` (spatially varying 2D field)
- Write `tbot` = T_bot to `cam_out.nc` (replaces TREFHT)
- Write `zbot` = z_bot to `cam_out.nc` (new field)
- Also write `tref` = TREFHT to `cam_out.nc` for diagnostics (not wired to any a2x field yet)

**2. Fortran — `datm_datamode_camulator.F90`:**
- Add `g_zbot` global array (default 61.0 m) and `local_zbot` local array
- Read `zbot` from `cam_out.nc` in `read_cam_nc`
- Bcast + scatter `g_zbot` to all MPI ranks
- Replace hardcoded `a2x%rAttr(kz, n) = 50.0_R8` with `local_zbot(n)`
- Density uses `local_tbot(n)` (now bottom level T rather than TREFHT — more consistent with u/v/q)

**Requires rebuild:** `gmake -j8 cesm` from the bld directory (DATM source change only).

### Key formula

```python
# In camulator_server.py startup:
hybi_bot    = float(hybi[-2])               # 0.98511219 for CAM6 L32
p_mid_frac  = 0.5 * (hybi_bot + 1.0)       # 0.99255610
Z_BOT_SCALE = (287.058 / 9.80616) * (-np.log(p_mid_frac))   # 0.21872 m/K

# Each coupling step:
T_bot_cam = accessor_output.get_state_var(prediction_out, "T")[0, -1, 0].cpu().numpy()
z_bot_cam = Z_BOT_SCALE * T_bot_cam   # (192, 288) field, values ~50–67 m
```

---

## Section 11. FSNS → FSDS albedo inversion for shortwave (2026-02-25)

### Problem

CAMulator outputs **FSNS** (net SW at surface = downwelling − reflected, after CAM6's own surface albedo).
CPL7's `seq_flux_mct.F90` (lines 852–859) treats the `Faxa_sw*` fields as **downwelling** SW and applies the ocean/ice albedo from POP/CICE to compute reflected SW:

```fortran
swupc = a2x_o%rAttr(index_a2x_Faxa_swndr,n)*(-anidr) + ...   ! reflected SW
swdnc = a2x_o%rAttr(index_a2x_Faxa_swndr,n) + ...            ! total downwelling SW
```

If we pass FSNS as if it were downwelling SW, the ocean absorbs:
```
SW_absorbed_wrong = FSNS × (1 − α_ocean) ≈ FSNS × 0.94   (open ocean α ≈ 0.06)
```
instead of the correct `FSNS`. This is a **~6% underestimate** for open ocean and larger errors over sea ice.

### Fix — albedo inversion in Python server (2026-02-25)

We have CICE ice fraction (`ifrac`) from the **same coupling step** (already in `sst_in.nc`). Use it to estimate the effective surface albedo and invert:

```python
_alpha_ocean = 0.06     # open-water SW albedo (consistent with CICE5 default)
_alpha_ice   = 0.60     # effective sea-ice SW albedo (includes melt ponds)
_alpha_sfc   = (1.0 - ifrac_flat) * _alpha_ocean + ifrac_flat * _alpha_ice
_one_minus_a = np.maximum(1.0 - _alpha_sfc, 0.10)   # floor prevents explosion
fsds = np.where(fsns > 0.0, fsns / _one_minus_a, 0.0)   # zero at night
fsds = np.minimum(fsds, 1500.0)                           # physical cap
```

The field written to `cam_out.nc` is renamed **`fsds`** (downwelling SW); Fortran reads it and splits it into the four `Faxa_sw*` bands using CORE2 fractions (VDR=0.28, NDR=0.31, VDF=0.24, NDF=0.17). The coupler then correctly applies ocean/ice albedo to derive absorbed SW.

### Accuracy notes

| Surface type | α_sfc | FSDS/FSNS ratio | Error before fix |
|---|---|---|---|
| Open ocean | 0.06 | 1.064 | −6% |
| 50% ice | 0.33 | 1.493 | −33% |
| Full ice cover | 0.60 | 2.500 | −60% |

The `_alpha_ice = 0.60` is intentionally lower than bare ice (0.65–0.80) to account for melt ponds; tunable once we have reference data. The floor `0.10` on `(1 − α)` limits FSDS/FSNS ≤ 10× even for dense ice.

### Files changed

- `camulator_server.py` — step (k2): reconstruct `fsds` from `fsns + ifrac_flat`; `write_cam_nc` parameter `fsns → fsds`
- `datm_datamode_camulator.F90` — `g_fsns → g_fsds`; `nc_get` reads `'fsds'`; `swnet → swdn` in a2x loop

**Requires rebuild:** DATM source change only (`gmake -j8 cesm` from bld/).

---

## Section 12. Verifying Ocean ↔ Atmosphere Communication

The coupled system is only useful if the two components are genuinely influencing each other. The checks below go from quick sanity tests to deeper physical verification.

### 9a. Quick checks (every run)

**1. Server log `ocn_mean` changes over time**
The server prints per-step diagnostics to stdout:
```
SST  min=271.x K  max=303.x K  ocn_mean=286.x K  (land pts: 5xxx)  date=19xx-xx-xx
FSNS  mean= xxx.x W/m²  FSDS  mean= xxx.x W/m²  FLNSD mean= xxx.x W/m²  |U_bot| mean= x.xx m/s  zbot mean= xx.x m
```
- `ocn_mean` should NOT be constant across steps — it should drift slowly as POP evolves SST
- `FSDS` should be ≥ `FSNS` (always, since FSDS = FSNS / (1−α)); ratio ≈ 1.06 over open ocean
- `|U_bot|` and `FSNS`/`FSDS` should vary day-to-day reflecting CAMulator's weather
- `land pts` should stay ~5000–6000 (the fixed continental mask); if it jumps to 18048 → SST channel broken

**2. `camulator_sst_in.nc` — SST reaching CAMulator**
```python
import xarray as xr
ds = xr.open_dataset('.../run/camulator_sst_in.nc')
ds['sst'].plot()   # should show spatially varying ocean SST, land=0
ds['ifrac'].plot() # should show sea ice fraction in polar regions
```
- Ocean points should be 271–305 K
- Land points should be 0 (masked out in server before normalization)
- If all values are 0 → CPL7 SST mapping is broken (three-part cime_comp_mod fix not applied)
- If all values are 283 K → land fill is leaking into ocean points (OCEAN_MIN_K threshold wrong)

**3. `camulator_cam_out.nc` — CAMulator forcing reaching CESM**
```python
ds = xr.open_dataset('.../run/camulator_cam_out.nc')
# Check all 10 fields are non-trivial:
# u10, v10, tbot, zbot, tref, qbot, pbot, fsds, flnsd, prect
```
- `u10`/`v10`: ±5–25 m/s, spatially structured (jets, storm tracks)
- `tbot`: ~240–310 K (bottom model level T); `zbot`: ~50–67 m (dynamic); `tref`: ~250–305 K (TREFHT, diagnostic)
- `fsds`: 0–1500 W/m² downwelling SW (higher than FSNS by ~1/0.94 over ocean); should be 0 in polar night
- `flnsd`: 200–450 W/m², smoothly varying (should now use Stefan-Boltzmann formula)
- `prect`: 0–5×10⁻⁵ m/s, precipitation bands visible

### 9b. Intermediate checks (after multi-week runs)

**4. POP SST evolves from initial conditions**
Compare `pop.h.nday1` SST on day 1 vs day 30:
```bash
# Days since start should show SST anomalies developing
ncdump -v TEMP /glade/derecho/scratch/.../run/g.e21.CAMULATOR_GIAF_v02.pop.h.nday1.0001-01-*.nc | head
```
- SST should **not** be identical to the initial condition after 30 days
- Anomaly patterns should be spatially correlated with `camulator_cam_out.nc` FSDS/FLNSD

**5. Compare CAMulator output against CORE2 IAF climatology**
The CORE2 forcing files live at `/glade/campaign/cesm/cesmdata/inputdata/ocn/iaf/`. Load a NCEP wind file and compare against `camulator_cam_out.nc` for the same calendar date:
```python
import xarray as xr
core2 = xr.open_dataset('.../ncep.u_10.T62.1948.nc')  # CORE2 u-wind
cam   = xr.open_dataset('.../camulator_cam_out.nc')    # CAMulator u-wind
# Spatial patterns should be broadly similar; amplitudes within factor ~2
```

**6. CICE ice extent responds to forcing**
```python
ds_ice = xr.open_dataset('.../g.e21.CAMULATOR_GIAF_v02.cice.h.*.nc')
ds_ice['aice'].sum(['ni','nj']).plot()   # total ice area over time
```
- Should show seasonal cycle if run is long enough
- If ice area is static → CICE not receiving valid forcing

### 9c. Definitive coupling test (2-member experiment)

Run two identical simulations starting from the same restart, but with:
- **Control:** CAMulator uses POP2 SST (this run)
- **Decoupled:** CAMulator uses fixed SST (set `OCEAN_MIN_K` very high so persistent IC SST is always used)

If `ocn_mean` diverges between members over weeks → the feedback loop is real. This is the gold-standard confirmation that the system is fully coupled rather than one-way forced.

---

## Next Steps (as of 2026-02-25)

| # | Task | Status |
|---|---|---|
| 1 | Run standalone server smoke test on GPU node | ✅ DONE (2026-02-23) |
| 2 | Verify initial condition `.pth` exists for coupling start date | ✅ DONE |
| 3 | Set `datamode = 'CAMULATOR'` in `user_nl_datm` | ✅ DONE |
| 4 | Patch XML whitelist (`namelist_definition_datm.xml`) — Step 6b | ✅ DONE |
| 5 | Patch Fortran runtime whitelist (`datm_shr_mod.F90`) — Step 6c | ✅ DONE (2026-02-23) |
| 6 | Recreate case as `--mach derecho` (v02) + fix MPI GTL issue in `env_mach_specific.xml` (3 vars) | ✅ DONE (2026-02-23) |
| 6a | Confirm MPI fix on clean v03 IAF baseline — `case.run success` (job 5182424, 2026-02-23) | ✅ CONFIRMED |
| 7 | First live coupled run — go/done handshake confirmed; SST=0 bug identified | ✅ DONE |
| 7a | Two-part CPL7 fix: Guards #1 and #2 (`atm_prognostic` → `atm_present`) applied | ✅ DONE |
| 7b | `ocn_c2_atm` hypothesis disproved — flag IS `.true.` for DATA ATM | ✅ DONE |
| 7c | Fix #3: `xao_ax => prep_aoflux_get_xao_ax()` added before `prep_atm_mrg` call | ✅ DONE — rebuilt & confirmed |
| 8 | **10-day coupled run confirmed working** — 45 SYPD, SST non-zero, restarts clean | ✅ DONE (job 5197394, 2026-02-25) |
| 9 | FLNSD fix in `camulator_server.py`: `ε σ T⁴ + FLNS/DT_SEC` (Stefan-Boltzmann) | ✅ DONE (2026-02-25) |
| 10 | CICE CFL crash at day 12 — fix `ndtd = 1 → 2` in `user_nl_cice` (no rebuild) | ✅ DONE (2026-02-25) |
| 11 | Fix `Sa_z` reference height: `10m → 50m` in `datm_datamode_camulator.F90` (intermediate) | ✅ DONE (2026-02-25) |
| 12 | Dynamic z_bot + T_bottom_level fix: `z_bot = 0.2187*T_bot`, write zbot+tbot+tref to nc, read in Fortran | ✅ DONE (2026-02-25) |
| 13 | FSNS → FSDS albedo inversion (Section 11): reconstruct downwelling SW from net SW + CICE ifrac | ✅ DONE (2026-02-25) |
| 14 | Rebuild DATM after Fortran changes: `gmake -j8 cesm` from bld dir | 🔲 TODO |
| 15 | Run longer test (30+ days) to confirm stability with `ndtd=2` + dynamic `zbot` + `T_bot` + FSDS | 🔲 TODO |
| 16 | Compare CAMulator output fields vs CORE2 IAF reference data | 🔲 TODO |
