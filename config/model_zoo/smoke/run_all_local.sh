#!/bin/bash
# Run all model-zoo smoke tests locally (one at a time, logs to per-model files).
# Intended for use on casper29 when the PBS queue is backed up.
# Usage: bash run_all_local.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "[dry-run] Would run the following configs:"
fi

CONFIGS=(
    smoke_stormer_casper.yml
    smoke_arches_casper.yml
    smoke_sfno_casper.yml
    smoke_climax_casper.yml
    smoke_fourcastnet_casper.yml
    smoke_fourcastnet3_casper.yml
    smoke_fengwu_casper.yml
    smoke_swinrnn_casper.yml
    smoke_itransformer_casper.yml
    smoke_mambavision_casper.yml
    smoke_graphcast_casper.yml
    smoke_healpix_casper.yml
    smoke_corrdiff_casper.yml
    smoke_fuxi_ens_casper.yml
    smoke_aurora_casper.yml
    smoke_pangu_casper.yml
    smoke_aifs_casper.yml
    smoke_nextgen_wxformer_casper.yml
)

PASS=()
FAIL=()

for cfg in "${CONFIGS[@]}"; do
    CFG_PATH="${SCRIPT_DIR}/${cfg}"
    MODEL="${cfg%.yml}"

    SAVE_LOC="$(python -c "import yaml, os; c=yaml.safe_load(open('${CFG_PATH}')); print(os.path.expandvars(c['save_loc']))")"
    mkdir -p "${SAVE_LOC}"
    cp "${CFG_PATH}" "${SAVE_LOC}/model.yml"
    LOG="${SAVE_LOC}/smoke.log"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  ${cfg}  →  ${LOG}"
        continue
    fi

    echo -n "Running ${MODEL} ... "
    if credit_train -c "${SAVE_LOC}/model.yml" > "${LOG}" 2>&1; then
        echo "PASS  (log: ${LOG})"
        PASS+=("${MODEL}")
    else
        echo "FAIL  (log: ${LOG})"
        FAIL+=("${MODEL}")
    fi
done

if [[ $DRY_RUN -eq 1 ]]; then
    exit 0
fi

echo ""
echo "=============================="
echo "Results: ${#PASS[@]} passed, ${#FAIL[@]} failed"
if [[ ${#PASS[@]} -gt 0 ]]; then
    echo "PASS: ${PASS[*]}"
fi
if [[ ${#FAIL[@]} -gt 0 ]]; then
    echo "FAIL: ${FAIL[*]}"
    exit 1
fi
