#!/bin/bash -l
#PBS -N credit-env-casper
#PBS -l select=1:ncpus=8:mem=32GB
#PBS -l walltime=02:00:00
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe
#PBS -k eod

# ---------------------------------------------------------------------------
# Build a miles-credit conda env on Casper.
#
# Usage:
#   qsub scripts/build_env_casper.sh
#   # or with custom source/target:
#   SOURCE_ENV=/path/to/env TARGET_ENV=/path/to/new/env qsub scripts/build_env_casper.sh
#
# SOURCE_ENV: base env to clone (must have PyTorch + CUDA stack)
# TARGET_ENV: where the new env is created (defaults to $WORK/conda-envs/credit-casper)
# REPO:       miles-credit repo root (defaults to directory containing this script's parent)
# ---------------------------------------------------------------------------

set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
SOURCE_ENV=${SOURCE_ENV:-/glade/u/home/schreck/.conda/envs/credit-casper}
TARGET_ENV=${TARGET_ENV:-/glade/work/$USER/conda-envs/credit-casper}

echo "============================================================"
echo "Building credit-casper conda env"
echo "  source : $SOURCE_ENV"
echo "  target : $TARGET_ENV"
echo "  repo   : $REPO"
echo "============================================================"

module load conda/latest

if [ -d "$TARGET_ENV" ]; then
    echo "WARNING: $TARGET_ENV already exists — removing and rebuilding."
    conda env remove -p "$TARGET_ENV" -y
fi

echo "[1/3] Cloning base env …"
conda create --clone "$SOURCE_ENV" -p "$TARGET_ENV" -y
echo "Clone done."

echo "[2/3] Installing miles-credit in editable mode …"
source activate "$TARGET_ENV"

pip install --upgrade pip setuptools wheel --quiet
pip uninstall -y miles-credit 2>/dev/null || true
pip install -e "${REPO}[ask,serve,develop]" --no-build-isolation

echo "[3/3] Verifying install …"
python -c "import credit; print('credit version:', credit.__version__)"
python -c "from credit.trainers.preflight import check_dataloader_startup; print('preflight OK')"

echo "============================================================"
echo "SUCCESS: env ready at $TARGET_ENV"
echo ""
echo "To activate:"
echo "  module load conda/latest"
echo "  conda activate $TARGET_ENV"
echo "============================================================"
