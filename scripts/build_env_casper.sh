#!/bin/bash -l
#PBS -N credit-main-env-casper
#PBS -l select=1:ncpus=8:mem=32GB
#PBS -l walltime=02:00:00
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe
#PBS -k eod

# ---------------------------------------------------------------------------
# Build a dedicated conda env for miles-credit-main on Casper.
#
# Source:  /glade/u/home/schreck/.conda/envs/credit-casper
# Target:  /glade/work/schreck/conda-envs/credit-main-casper
#
# The env is a full clone so the PyTorch / CUDA stack is preserved exactly.
# miles-credit is then installed in *editable* mode from the local repo so
# that git pulls are picked up without a reinstall.
# ---------------------------------------------------------------------------

set -euo pipefail

REPO=/glade/work/schreck/repos/miles-credit-main
SOURCE_ENV=/glade/u/home/schreck/.conda/envs/credit-casper
TARGET_ENV=/glade/work/schreck/conda-envs/credit-main-casper

echo "============================================================"
echo "Building credit-main-casper"
echo "  source : $SOURCE_ENV"
echo "  target : $TARGET_ENV"
echo "  repo   : $REPO"
echo "============================================================"

module load conda/latest

# ---- Clone -----------------------------------------------------------------
if [ -d "$TARGET_ENV" ]; then
    echo "WARNING: $TARGET_ENV already exists — removing and rebuilding."
    conda env remove -p "$TARGET_ENV" -y
fi

echo "[1/3] Cloning base env …"
conda create --clone "$SOURCE_ENV" -p "$TARGET_ENV" -y
echo "Clone done."

# ---- Activate & update miles-credit ----------------------------------------
echo "[2/3] Installing miles-credit in editable mode …"
source activate "$TARGET_ENV"

# Upgrade pip/setuptools first — older cloned envs may have versions
# that cannot parse pyproject.toml license = {text = "..."} correctly
pip install --upgrade pip setuptools wheel --quiet

# Remove any previously installed (non-editable) miles-credit
pip uninstall -y miles-credit 2>/dev/null || true

# Install from repo with all optional extras
pip install -e "${REPO}[ask,serve,develop]" --no-build-isolation

echo "[3/3] Verifying install …"
python -c "import credit; print('credit version:', credit.__version__)"
python -c "from credit.trainers.preflight import check_dataloader_startup; print('preflight OK')"
python -c "import fastapi; print('fastapi OK')"
python -c "import anthropic; print('anthropic OK')"

echo "============================================================"
echo "SUCCESS: credit-main-casper is ready at $TARGET_ENV"
echo ""
echo "To activate:"
echo "  conda activate $TARGET_ENV"
echo "  # or add this alias to ~/.bashrc:"
echo "  alias credit-main='conda activate $TARGET_ENV'"
echo "============================================================"
