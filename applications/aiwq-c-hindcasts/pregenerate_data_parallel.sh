#!/bin/bash -l
#PBS -N pregen_hindcast
#PBS -l select=1:mem=64GB:ncpus=4
#PBS -l walltime=00:25:00
#PBS -A SCSG0001
#PBS -q casper@casper-pbs
#PBS -o logs/pregen_^array_index^.log
#PBS -j oe

# ============================================================================
# Notes from Negin (2026-02-06)
# PARALLEL Pre-generation using PBS Job Array
#
# To submit all dates (2000-01-01 to 2020-12-31):
#   mkdir -p logs
#   qsub -J 0-766 pregenerate_data_parallel.sh
#
# Monitor: qstat -u $USER
# Check failures: cat logs/dates.txt
# Safe to resubmit — already-completed dates are skipped.
#
# Splits ~7,670 dates across 767 array jobs (10 dates per job)
# Each job processes 10 consecutive dates
#
# Array indices: 0-766 (767 jobs total)
# Dates per job: 10
# Total dates: ~7,670 (2000-01-01 to 2020-12-31)
#
# Estimated time: ~10-15 minutes per job (10 dates × ~1 min each)
# ============================================================================

module load conda
conda activate /glade/derecho/scratch/negins/conda-envs/miles-credit-for-kirsten

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
DATES_PER_JOB=10
START_DATE="2000-01-01"
LOG_DIR="${SCRIPT_DIR}/logs"

FAILED_DATES="$LOG_DIR/dates.txt"

mkdir -p "$LOG_DIR"

# Calculate date range for this array job
start_sec=$(date -d "$START_DATE" +%s)
job_start_sec=$((start_sec + PBS_ARRAY_INDEX * DATES_PER_JOB * 86400))
job_end_sec=$((job_start_sec + (DATES_PER_JOB - 1) * 86400))

# Don't go past 2020-12-31
max_end_sec=$(date -d "2020-12-31" +%s)
if [ "$job_end_sec" -gt "$max_end_sec" ]; then
    job_end_sec=$max_end_sec
fi

echo "============================================"
echo "Job Array Index: $PBS_ARRAY_INDEX"
echo "Processing dates: $(date -d @$job_start_sec +%Y-%m-%d) to $(date -d @$job_end_sec +%Y-%m-%d)"
echo "Started at: $(date)"
echo "============================================"

current_sec=$job_start_sec
count=0

while [ "$current_sec" -le "$job_end_sec" ]; do
    DATE_FLAG=$(date -d "@$current_sec" +%Y-%m-%dT0000)
    DATE_SHORT=$(date -d "@$current_sec" +%Y-%m-%d)

    count=$((count + 1))
    echo ""
    echo "[$count/$DATES_PER_JOB] Processing: $DATE_SHORT ($(date +%H:%M:%S))"

    # Output paths
    DATA_DIR="/glade/derecho/scratch/${USER}/DATA/CESMS2S_inits/${DATE_SHORT}"
    STEP1_INIT="${DATA_DIR}/CESM_${DATE_SHORT}.nc"
    STEP1_FORCING="${DATA_DIR}/CESM_dynamicforcing_${DATE_SHORT}.nc"
    ENSEMBLE_LAST="${DATA_DIR}/ICs.011.${DATE_SHORT}-00000.nc"
    CESM_FILE="/glade/campaign/cesm/development/cross-wg/S2S/sglanvil/forKirsten/subCESMulator/final/CESM_${DATE_SHORT}.nc"
    TIMEOUT=120

    # Skip if CESM source missing
    if [ ! -f "$CESM_FILE" ]; then
        echo "  ${DATE_SHORT}: CESM file not found, skipping..."
        current_sec=$((current_sec + 86400))
        continue
    fi

    # --- Step 1: prepare_hindcastinit.py ---
    if [ -f "$STEP1_INIT" ] && [ -f "$STEP1_FORCING" ]; then
        echo $STEP1_FORCING
        echo "  ${DATE_SHORT}: Step 1 outputs already exist, skipping..."
    else
        echo "  ${DATE_SHORT}: Step 1 prepare_hindcastinit.py..."
        step1_start=$(date +%s)
        timeout $TIMEOUT python "${SCRIPT_DIR}/prepare_hindcastinit.py" -d "$DATE_FLAG"
        step1_exit=$?
        step1_elapsed=$(( $(date +%s) - step1_start ))
        if [ $step1_exit -eq 124 ]; then
            echo "  ${DATE_SHORT}: Step 1 KILLED after ${step1_elapsed}s (timeout ${TIMEOUT}s)"
        else
            echo "  ${DATE_SHORT}: Step 1 finished in ${step1_elapsed}s (exit code: $step1_exit)"
        fi

        # Verify Step 1 produced both output files
        if [ ! -f "$STEP1_INIT" ] || [ ! -f "$STEP1_FORCING" ]; then
            echo "  ${DATE_SHORT}: FAILED (Step 1 missing outputs)"
            echo "$DATE_SHORT" >> "$FAILED_DATES"
            current_sec=$((current_sec + 86400))
            continue
        fi
    fi

    # --- Step 2: make_ensemble_hindcast.py ---
    echo "  ${DATE_SHORT}: Step 2 make_ensemble_hindcast.py..."
    step2_start=$(date +%s)
    timeout $TIMEOUT python "${SCRIPT_DIR}/make_ensemble_hindcast.py" -d "$DATE_FLAG"
    step2_exit=$?
    step2_elapsed=$(( $(date +%s) - step2_start ))
    if [ $step2_exit -eq 124 ]; then
        echo "  ${DATE_SHORT}: Step 2 KILLED after ${step2_elapsed}s (timeout ${TIMEOUT}s)"
    else
        echo "  ${DATE_SHORT}: Step 2 finished in ${step2_elapsed}s (exit code: $step2_exit)"
    fi

    # Verify final output
    if [ -f "$ENSEMBLE_LAST" ]; then
        echo "  ${DATE_SHORT}: SUCCESS"
    else
        echo "  ${DATE_SHORT}: FAILED (ensemble files not produced)"
        echo "$DATE_SHORT" >> "$FAILED_DATES"
    fi

    current_sec=$((current_sec + 86400))
done

echo ""
echo "============================================"
echo "Completed at: $(date)"
echo "============================================"
