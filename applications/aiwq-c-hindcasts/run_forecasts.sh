#!/bin/bash -l
#PBS -N run_hindcast_forecasts
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -l walltime=12:00:00
#PBS -A "SCSG0001"
#PBS -q main@desched1
#PBS -o logs/forecast_^array_index^.log
#PBS -j oe

# ============================================================================
# Run hindcast forecasts — Job Array version
#
# Each array index processes DATES_PER_JOB consecutive dates.
# Runs 11 ensemble members per date, 4 at a time (one per GPU).
#
# Usage:
#   mkdir -p logs
#   qsub -J 0-51 run_forecasts.sh        # all dates (2000-01-01 to 2020-12-31)
#   qsub -J 0-0   run_forecasts.sh       # first 150 dates only (for testing)
#
# Timing (from job 5015053):
#   ~1.3-1.6 min per ensemble member, ~4.2 min per date
#   DATES_PER_JOB=150 → ~10.5 hrs per job → 52 array tasks
#
# ============================================================================

module load conda
conda activate /glade/derecho/scratch/negins/conda-envs/miles-credit-for-kirsten

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="${SCRIPT_DIR}/hindcast_wrapper.py"

# Configuration
DATES_PER_JOB=150
START_DATE="2000-01-01"
END_DATE="2020-12-31"
CONFIG="${SCRIPT_DIR}/S2Shindcast.yml"
LOG_DIR="${SCRIPT_DIR}/logs"
N_GPUS=4
N_MEMBERS=11

mkdir -p "$LOG_DIR"

# Calculate date range for this array job
start_sec=$(date -d "$START_DATE" +%s)
max_end_sec=$(date -d "$END_DATE" +%s)
job_start_sec=$((start_sec + PBS_ARRAY_INDEX * DATES_PER_JOB * 86400))
job_end_sec=$((job_start_sec + (DATES_PER_JOB - 1) * 86400))

# Don't go past END_DATE
if [ "$job_end_sec" -gt "$max_end_sec" ]; then
    job_end_sec=$max_end_sec
fi

# Skip if this array index is entirely past the end date
if [ "$job_start_sec" -gt "$max_end_sec" ]; then
    echo "Array index $PBS_ARRAY_INDEX is past $END_DATE, nothing to do."
    exit 0
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
    echo "[$count/$DATES_PER_JOB] Processing forecast for: $DATE_SHORT ($(date +%H:%M:%S))"

    # Check if ensemble files exist (pre-generated)
    ENSEMBLE_CHECK="/glade/derecho/scratch/negins/DATA/CESMS2S_inits/${DATE_SHORT}/ICs.011.${DATE_SHORT}-00000.nc"
    FORCING_CHECK="/glade/derecho/scratch/negins/DATA/CESMS2S_inits/${DATE_SHORT}/CESM_dynamicforcing_${DATE_SHORT}.nc"

    if [ -f "$ENSEMBLE_CHECK" ] && [ -f "$FORCING_CHECK" ]; then

        # Launch members in batches of N_GPUS
        for member in $(seq 1 $N_MEMBERS); do
            member_str=$(printf "%03d" $member)
            gpu_id=$(( (member - 1) % N_GPUS ))

            echo "  Launching member $member_str on GPU $gpu_id"

            CUDA_VISIBLE_DEVICES=$gpu_id \
            torchrun --master-port=$((RANDOM % 10000 + 20000)) "$WRAPPER" \
                -d "$DATE_FLAG" \
                -c "$CONFIG" \
                --member "$member_str" \
                >> "$LOG_DIR/forecast_${DATE_SHORT}_m${member_str}.log" 2>&1 &

            # When N_GPUS processes are running, wait for the batch to finish
            if [ $((member % N_GPUS)) -eq 0 ]; then
                echo "  Waiting for batch (members $((member - N_GPUS + 1))-${member}) ..."
                wait
            fi
        done

        # Wait for any remaining members in the last partial batch
        wait
        echo "  Completed all members for $DATE_SHORT"

    else
        echo "  WARNING: Pre-generated data not found for $DATE_SHORT, skipping..."
        echo "    Missing ensemble: $ENSEMBLE_CHECK"
        echo "    Missing forcing: $FORCING_CHECK"
    fi

    # Increment by one day
    current_sec=$((current_sec + 86400))
done

echo ""
echo "============================================"
echo "Forecast runs complete!"
echo "Finished at: $(date)"
echo "Processed $count dates"
echo "============================================"
