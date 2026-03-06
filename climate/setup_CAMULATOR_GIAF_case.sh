#!/bin/bash
# =============================================================================
# setup_CAMULATOR_GIAF_case.sh
#
# Creates and configures a GIAF (POP2 + CICE + DATM) CESM2.1.5 case on
# derecho as the starting point for CAMulator-ATM coupling.
#
# Usage:
#   bash setup_CAMULATOR_GIAF_case.sh          # create + setup + build
#   bash setup_CAMULATOR_GIAF_case.sh nocreate # skip create_newcase (case exists)
#   bash setup_CAMULATOR_GIAF_case.sh nobuild  # skip case.build (setup only)
#
# Prerequisites — modifications outside this script:
#   1. cime_comp_mod.F90 three-part SST=0 fix (see MEMORY.md / coupling docs)
#      File: $CESM_ROOT/cime/src/drivers/mct/main/cime_comp_mod.F90
#   2. datm_datamode_camulator.F90 wired into DATM (see coupling docs)
#   3. camulator_server.py placed in $RUNDIR and launched on a GPU node
#      before ./case.submit
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION — update these as needed
# =============================================================================
CESM_ROOT=/glade/work/wchapman/JE_help_cnn/camulator_sandbox
CASE_DIR=/glade/work/wchapman/cesm/CREDIT/g.e21.CAMULATOR_SBV01
PROJECT=P03010039
MACH=derecho
COMPILER=intel
COMPSET=GIAF          # 2000_DATM%IAF_SLND_CICE_POP2_DROF%IAF_SGLC_WW3
RES=T62_g17           # T62 ATM grid, gx1v7 tripole ocean (1-deg POP)

# =============================================================================
# STEP 1 — create_newcase
# =============================================================================
if [[ "$1" != "nocreate" && "$1" != "nobuild" ]]; then
    echo "==> Creating case: $CASE_DIR"
    ${CESM_ROOT}/cime/scripts/create_newcase \
        --case     ${CASE_DIR} \
        --mach     ${MACH} \
        --compiler ${COMPILER} \
        --compset  ${COMPSET} \
        --res      ${RES} \
        --project  ${PROJECT} \
        --run-unsupported
else
    echo "==> Skipping create_newcase (case assumed to exist at $CASE_DIR)"
fi

cd ${CASE_DIR}

# =============================================================================
# STEP 2 — xmlchanges (all must be before case.setup for PE layout)
# =============================================================================
echo "==> Applying xmlchanges..."

# --- Coupling interval: 6-hour (4 couplings per day) ---
./xmlchange NCPL_BASE_PERIOD=day
./xmlchange ATM_NCPL=4
./xmlchange OCN_NCPL=4

# --- Short smoke test: 2 days = 8 coupling steps ---
./xmlchange STOP_OPTION=nyears
./xmlchange STOP_N=1
./xmlchange RESUBMIT=35

# --- Queue and walltime (use 'main' with longer walltime for production) ---
./xmlchange JOB_QUEUE=main
./xmlchange JOB_WALLCLOCK_TIME=00:50:00

# --- Turn off archiving for development ---
./xmlchange DOUT_S=FALSE

# --- PE layout: ATM/CPL/ICE/WAV/GLC/ROF/LND on PEs 0-127,
#                OCN on PEs 128-255 (confirmed working at 45 SYPD on Derecho) ---
./xmlchange NTASKS_ATM=128,NTASKS_CPL=128,NTASKS_ICE=128
./xmlchange NTASKS_OCN=128,NTASKS_WAV=128,NTASKS_GLC=128
./xmlchange NTASKS_ROF=128,NTASKS_LND=128
./xmlchange ROOTPE_OCN=128

# =============================================================================
# STEP 3 — case.setup
# =============================================================================
echo "==> Running case.setup..."
./case.setup

# =============================================================================
# STEP 4 — patch env_mach_specific.xml with MPI GPU-compatibility fixes
#
# These three env vars are required for MPI to work correctly on Derecho
# when a GPU node (camulator_server.py) participates in the job via flags:
#   MPICH_GPU_SUPPORT_ENABLED=0  — disables GPU-aware MPI (not needed, avoids hangs)
#   FI_CXI_DISABLE_HOST_REGISTER=1 — avoids CXI host-register conflicts
#   MPICH_SMP_SINGLE_COPY_MODE=NONE — prevents SMP shared-memory copy issues
# =============================================================================
echo "==> Patching env_mach_specific.xml with MPI fixes..."

for nameval in \
    "MPICH_GPU_SUPPORT_ENABLED:0" \
    "FI_CXI_DISABLE_HOST_REGISTER:1" \
    "MPICH_SMP_SINGLE_COPY_MODE:NONE"
do
    varname="${nameval%%:*}"
    varval="${nameval##*:}"
    if ! grep -q "name=\"${varname}\"" env_mach_specific.xml; then
        sed -i "s|</environment_variables>|    <env name=\"${varname}\">${varval}</env>\n  </environment_variables>|" \
            env_mach_specific.xml
        echo "    Added ${varname}=${varval}"
    else
        echo "    Already present: ${varname}"
    fi
done

# =============================================================================
# STEP 5 — user namelists
# =============================================================================
echo "==> Writing user namelists..."

# --- user_nl_datm: activate CAMULATOR data mode ---
cat >> user_nl_datm << 'EOF'
datamode = 'CAMULATOR'
EOF

# --- user_nl_cice: subcycle CICE dynamics twice per 6-hr coupling interval.
#     dt_dyn = 10800 s (half the 6-hr step).  This doubles the remap-transport
#     CFL limit from ~0.73 m/s to ~1.47 m/s, preventing "bad departure points"
#     crashes driven by CAMulator wind stress in the Arctic. ---
cat >> user_nl_cice << 'EOF'
ndtd = 2
EOF

# user_nl_pop:  no changes required
# user_nl_cpl:  no changes required

# =============================================================================
# STEP 6 — case.build (optional)
# =============================================================================
if [[ "$1" != "nobuild" ]]; then
    echo "==> Running case.build (this takes ~20-40 minutes)..."
    ./case.build
    echo "==> Build complete."
    echo ""
    echo "    Run directory: $(./xmlquery RUNDIR --value)"
    echo ""
    echo "    Before submitting:"
    echo "      1. Copy camulator_server.py to \$(./xmlquery RUNDIR --value)"
    echo "      2. Launch camulator_server.py on a GPU node (Casper A100)"
    echo "      3. Then: cd ${CASE_DIR} && ./case.submit"
else
    echo "==> Skipping case.build (run './case.build' manually from $CASE_DIR)"
fi

echo ""
echo "==> Done. Case is at: ${CASE_DIR}"
