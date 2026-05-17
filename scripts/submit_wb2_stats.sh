#!/usr/bin/env bash
# submit_wb2_stats.sh
# ---------------------------------------------------------------------------
# Three-phase WB2 ERA5 stats pipeline:
#
# Phase 1 — One PBS job per year (2000–2018), each saves a partial Welford pkl.
# Phase 2 — Merge job combines all partials into mean.nc / std.nc.
# Phase 3 — Tendency job computes xi and tend_std.nc (depends on Phase 2).
#
# Usage:
#   bash scripts/submit_wb2_stats.sh
#
# Final outputs:
#   /glade/campaign/cisl/aiml/credit/static_scalers/
#     wb2_era5_1440x721_2000_2018_v2_mean.nc
#     wb2_era5_1440x721_2000_2018_v2_std.nc
#     wb2_era5_1440x721_2000_2018_v2_xi.nc
#     wb2_era5_1440x721_2000_2018_v2_tend_std.nc
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT="NAML0001"
QUEUE="casper"
NCPUS=8
N_WORKERS=4
PYTHON="/glade/u/home/schreck/.conda/envs/credit-casper/bin/python"

SCRIPT="$(realpath "$(dirname "$0")/compute_wb2_stats.py")"
OUT_DIR="/glade/campaign/cisl/aiml/credit/static_scalers/wb2_stats_partial"
FINAL_DIR="/glade/campaign/cisl/aiml/credit/static_scalers"
LOG_DIR="/glade/derecho/scratch/${USER}/wb2_stats_logs"

mkdir -p "$OUT_DIR" "$LOG_DIR"

YEARS=(2000 2001 2002 2003 2004 2005 2006 2007 2008 2009
       2010 2011 2012 2013 2014 2015 2016 2017 2018)

job_ids=()

# ---------------------------------------------------------------------------
# Phase 1 — one job per year
# ---------------------------------------------------------------------------
for YEAR in "${YEARS[@]}"; do
    START="${YEAR}-01-01"
    END="${YEAR}-12-31"
    TAG="${YEAR}_v2"

    JOB_ID=$(qsub <<EOF
#PBS -N wb2stats_${YEAR}
#PBS -A ${PROJECT}
#PBS -q ${QUEUE}
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=${NCPUS}:mem=256GB
#PBS -o ${LOG_DIR}/wb2stats_${YEAR}.o
#PBS -e ${LOG_DIR}/wb2stats_${YEAR}.e
#PBS -j oe

${PYTHON} ${SCRIPT} compute \
    --start ${START} \
    --end   ${END} \
    --n_workers ${N_WORKERS} \
    --chunk_size 20 \
    --out_dir ${OUT_DIR} \
    --tag ${TAG} \
    --partial_only
EOF
    )
    echo "Submitted ${YEAR}: ${JOB_ID}"
    job_ids+=("${JOB_ID%%.*}")  # strip queue suffix
done

# Build afterok dependency string
DEP=$(IFS=:; echo "${job_ids[*]}")

# ---------------------------------------------------------------------------
# Phase 2 — merge job (afterok: all Phase 1 jobs)
# ---------------------------------------------------------------------------
MERGE_JID=$(qsub -W depend=afterok:"${DEP}" <<EOF
#PBS -N wb2stats_merge
#PBS -A ${PROJECT}
#PBS -q ${QUEUE}
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=4:mem=16GB
#PBS -o ${LOG_DIR}/wb2stats_merge.o
#PBS -e ${LOG_DIR}/wb2stats_merge.e
#PBS -j oe

${PYTHON} ${SCRIPT} merge \
    --partials ${OUT_DIR}/partial_*_v2.pkl \
    --out_dir  ${FINAL_DIR} \
    --tag      2000_2018_v2
EOF
)
echo ""
echo "Phase 2 (merge): ${MERGE_JID}"

# ---------------------------------------------------------------------------
# Phase 3 — tendency/xi job (afterok: Phase 2 merge)
# ---------------------------------------------------------------------------
TEND_JID=$(qsub -W depend=afterok:"${MERGE_JID%%.*}" <<EOF
#PBS -N wb2stats_xi
#PBS -A ${PROJECT}
#PBS -q ${QUEUE}
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=${NCPUS}:mem=128GB
#PBS -o ${LOG_DIR}/wb2stats_xi.o
#PBS -e ${LOG_DIR}/wb2stats_xi.e
#PBS -j oe

${PYTHON} ${SCRIPT} tendency \
    --std_path ${FINAL_DIR}/wb2_era5_1440x721_2000_2018_v2_std.nc \
    --start    2000-01-01 \
    --end      2018-12-31 \
    --chunk_size 20 \
    --n_workers  ${N_WORKERS} \
    --out_dir  ${FINAL_DIR} \
    --tag      2000_2018_v2
EOF
)
echo "Phase 3 (xi):   ${TEND_JID}"
echo ""
echo "Final outputs → ${FINAL_DIR}/wb2_era5_1440x721_2000_2018_v2_{mean,std,xi,tend_std}.nc"
