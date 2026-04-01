#!/bin/bash
# launch_coupled_run.sh
# ---------------------
# Waits for the CAMulator server to signal readiness, then submits the CESM case.
#
# The server writes camulator_server_ready.flag to rundir once model loading,
# JIT tracing, and all initialisation are complete.  This script polls for
# that flag and calls ./case.submit as soon as it appears.
#
# Run this on a Derecho login node (or as a small PBS job) after starting the
# server on Casper.  No cross-cluster scheduler queries needed.
#
# Usage:
#   ./launch_coupled_run.sh \
#       --rundir  /glade/derecho/scratch/wchapman/g.e21.CAMULATOR_GIAF_v02/run \
#       --casedir /glade/work/wchapman/cesm/CREDIT/g.e21.CAMULATOR_GIAF_v02   \
#       [--timeout 3600]   [--poll 15]

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
RUNDIR=""
CASEDIR="$(pwd)"
TIMEOUT=3600    # seconds to wait before giving up (default 1 h)
POLL_SLEEP=15   # seconds between checks
READY_FLAG="camulator_server_ready.flag"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 --rundir <cesm_rundir> [--casedir <cesm_casedir>] [--timeout <secs>] [--poll <secs>]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --rundir)   RUNDIR="$2";      shift 2 ;;
        --casedir)  CASEDIR="$2";     shift 2 ;;
        --timeout)  TIMEOUT="$2";     shift 2 ;;
        --poll)     POLL_SLEEP="$2";  shift 2 ;;
        -h|--help)  usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "$RUNDIR" ]] && { echo "ERROR: --rundir is required."; usage; }

READY_PATH="$RUNDIR/$READY_FLAG"

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  CAMulator launch script"
echo "  rundir    : $RUNDIR"
echo "  casedir   : $CASEDIR"
echo "  ready flag: $READY_PATH"
echo "  timeout   : ${TIMEOUT}s  |  poll interval: ${POLL_SLEEP}s"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Poll loop
# ---------------------------------------------------------------------------
elapsed=0
while [[ $elapsed -lt $TIMEOUT ]]; do

    if [[ -f "$READY_PATH" ]]; then
        echo "$(date '+%H:%M:%S')  Server ready flag detected!"
        echo "$(date '+%H:%M:%S')  Submitting CESM case from: $CASEDIR"
        echo ""
        cd "$CASEDIR"
        ./case.submit
        echo ""
        echo "$(date '+%H:%M:%S')  case.submit completed successfully."
        exit 0
    fi

    remaining=$((TIMEOUT - elapsed))
    echo "$(date '+%H:%M:%S')  Waiting for server... (${elapsed}s elapsed, ${remaining}s remaining)"
    sleep "$POLL_SLEEP"
    elapsed=$((elapsed + POLL_SLEEP))

done

echo ""
echo "$(date '+%H:%M:%S')  ERROR: timed out after ${TIMEOUT}s — server ready flag never appeared."
echo "  Check that the CAMulator server is running on Casper and writing to:"
echo "  $READY_PATH"
exit 1
