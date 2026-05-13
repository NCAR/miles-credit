#!/bin/bash
# Submit all model-zoo smoke tests to Casper PBS queue.
# Usage: bash submit_all_casper.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "[dry-run] Would submit the following jobs:"
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

# Resolve the credit_train script path
TRAIN_SCRIPT="$(python -c "import credit.applications.train as m; import inspect; print(inspect.getfile(m))")"

for cfg in "${CONFIGS[@]}"; do
    CFG_PATH="${SCRIPT_DIR}/${cfg}"
    JOB_NAME="${cfg%.yml}"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  ${cfg}"
        continue
    fi

    # Build per-job PBS script and submit
    SAVE_LOC="$(python -c "import yaml; c=yaml.safe_load(open('${CFG_PATH}')); print(c['save_loc'].replace('\$USER', '${USER}'))" 2>/dev/null || \
               python -c "import yaml; import os; c=yaml.safe_load(open('${CFG_PATH}')); print(os.path.expandvars(c['save_loc']))")"

    PBS_CFG="$(python -c "import yaml; c=yaml.safe_load(open('${CFG_PATH}')); p=c['pbs']; print('\n'.join([f\"{k}: {v}\" for k,v in p.items()]))")"
    CONDA="$(python   -c "import yaml; c=yaml.safe_load(open('${CFG_PATH}')); print(c['pbs']['conda'])")"
    NCPUS="$(python   -c "import yaml; c=yaml.safe_load(open('${CFG_PATH}')); print(c['pbs']['ncpus'])")"
    NGPUS="$(python   -c "import yaml; c=yaml.safe_load(open('${CFG_PATH}')); print(c['pbs']['ngpus'])")"
    MEM="$(python     -c "import yaml; c=yaml.safe_load(open('${CFG_PATH}')); print(c['pbs']['mem'])")"
    GPU_TYPE="$(python -c "import yaml; c=yaml.safe_load(open('${CFG_PATH}')); print(c['pbs']['gpu_type'])")"
    WALLTIME="$(python -c "import yaml; c=yaml.safe_load(open('${CFG_PATH}')); print(c['pbs']['walltime'])")"
    QUEUE="$(python   -c "import yaml; c=yaml.safe_load(open('${CFG_PATH}')); print(c['pbs']['queue'])")"
    PROJECT="$(python -c "import yaml; c=yaml.safe_load(open('${CFG_PATH}')); print(c['pbs']['project'])")"

    mkdir -p "${SAVE_LOC}"
    cp "${CFG_PATH}" "${SAVE_LOC}/model.yml"

    PBS_SCRIPT=$(cat <<EOF
#!/bin/bash -l
#PBS -N ${JOB_NAME}
#PBS -l select=1:ncpus=${NCPUS}:ngpus=${NGPUS}:mem=${MEM}:gpu_type=${GPU_TYPE}
#PBS -l walltime=${WALLTIME}
#PBS -A ${PROJECT}
#PBS -q ${QUEUE}
#PBS -j oe
#PBS -k eod
#PBS -o ${SAVE_LOC}/${JOB_NAME}.out

source ~/.bashrc
conda activate ${CONDA}

credit_train -c ${SAVE_LOC}/model.yml
EOF
)

    LAUNCH_SH="${SAVE_LOC}/launch.sh"
    echo "${PBS_SCRIPT}" > "${LAUNCH_SH}"
    JOBID=$(qsub "${LAUNCH_SH}")
    echo "Submitted ${cfg}  →  ${JOBID}"
done
