#!/bin/bash
# Submit pretrained model smoke tests to Casper PBS queue.
# Usage: bash submit_pretrained_casper.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "[dry-run] Would submit the following jobs:"
fi

CONFIGS=(
    smoke_stormer_pretrained_casper.yml
    smoke_climax_pretrained_casper.yml
    smoke_fourcastnet_pretrained_casper.yml
    smoke_aurora_pretrained_casper.yml
    smoke_fuxi_pretrained_casper.yml
)

for cfg in "${CONFIGS[@]}"; do
    CFG_PATH="${SCRIPT_DIR}/${cfg}"
    JOB_NAME="${cfg%.yml}"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  ${cfg}"
        continue
    fi

    SAVE_LOC="$(python -c "import yaml; c=yaml.safe_load(open('${CFG_PATH}')); print(c['save_loc'].replace('\$USER', '${USER}'))" 2>/dev/null || \
               python -c "import yaml; import os; c=yaml.safe_load(open('${CFG_PATH}')); print(os.path.expandvars(c['save_loc']))")"

    CONDA="$(python    -c "import yaml; c=yaml.safe_load(open('${CFG_PATH}')); print(c['pbs']['conda'])")"
    NCPUS="$(python    -c "import yaml; c=yaml.safe_load(open('${CFG_PATH}')); print(c['pbs']['ncpus'])")"
    NGPUS="$(python    -c "import yaml; c=yaml.safe_load(open('${CFG_PATH}')); print(c['pbs']['ngpus'])")"
    MEM="$(python      -c "import yaml; c=yaml.safe_load(open('${CFG_PATH}')); print(c['pbs']['mem'])")"
    GPU_TYPE="$(python -c "import yaml; c=yaml.safe_load(open('${CFG_PATH}')); print(c['pbs']['gpu_type'])")"
    WALLTIME="$(python -c "import yaml; c=yaml.safe_load(open('${CFG_PATH}')); print(c['pbs']['walltime'])")"
    QUEUE="$(python    -c "import yaml; c=yaml.safe_load(open('${CFG_PATH}')); print(c['pbs']['queue'])")"
    PROJECT="$(python  -c "import yaml; c=yaml.safe_load(open('${CFG_PATH}')); print(c['pbs']['project'])")"

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

python -m credit.applications.train_gen2 -c ${SAVE_LOC}/model.yml
EOF
)

    LAUNCH_SH="${SAVE_LOC}/launch.sh"
    echo "${PBS_SCRIPT}" > "${LAUNCH_SH}"
    JOBID=$(qsub "${LAUNCH_SH}")
    echo "Submitted ${cfg}  →  ${JOBID}"
done
