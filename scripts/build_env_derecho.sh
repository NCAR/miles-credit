#!/bin/bash -l
#PBS -N credit-main-env-derecho
#PBS -l select=1:ncpus=8:mem=64GB
#PBS -l walltime=02:00:00
#PBS -A NAML0001
#PBS -q main
#PBS -j oe
#PBS -k eod

# ---------------------------------------------------------------------------
# Build a dedicated conda env for miles-credit-main on Derecho.
#
# Source:  /glade/work/benkirk/conda-envs/credit-derecho-torch28-nccl221
# Target:  /glade/work/schreck/conda-envs/credit-main-derecho
#
# The env is a full clone so the custom Cray-MPICH PyTorch 2.8 build
# (from /glade/work/benkirk/…) is preserved exactly.
# miles-credit is then installed in *editable* mode from the local repo so
# that git pulls are picked up without a reinstall.
# ---------------------------------------------------------------------------

set -euo pipefail

REPO=/glade/work/schreck/repos/miles-credit-main
SOURCE_ENV=/glade/work/benkirk/conda-envs/credit-derecho-torch28-nccl221
TARGET_ENV=/glade/work/schreck/conda-envs/credit-main-derecho

echo "============================================================"
echo "Building credit-main-derecho"
echo "  source : $SOURCE_ENV"
echo "  target : $TARGET_ENV"
echo "  repo   : $REPO"
echo "============================================================"

module load ncarenv/24.12 gcc/12.4.0 ncarcompilers craype cray-mpich/8.1.29 \
            cuda/12.3.2 conda/latest cudnn/9.2.0.82-12 mkl/2025.0.1

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

# Remove any previously installed (non-editable) miles-credit
pip uninstall -y miles-credit 2>/dev/null || true

# Install from repo with all optional extras
pip install -e "${REPO}[ask,serve,develop]" --no-build-isolation

echo "[3/3] Verifying install …"
python -c "import credit; print('credit version:', credit.__version__)"
python -c "from credit.trainers.preflight import check_dataloader_startup; print('preflight OK')"
python -c "import fastapi; print('fastapi OK')"
python -c "import anthropic; print('anthropic OK')"

# Verify the Derecho-specific MPI torch still works
python -c "import torch; import torch.distributed; print('torch:', torch.__version__, '| distributed OK')"

echo "============================================================"
echo "SUCCESS: credit-main-derecho is ready at $TARGET_ENV"
echo ""
echo "To activate:"
echo "  module load ncarenv/24.12 gcc/12.4.0 ncarcompilers craype cray-mpich/8.1.29 cuda/12.3.2 conda/latest"
echo "  conda activate $TARGET_ENV"
echo "============================================================"
