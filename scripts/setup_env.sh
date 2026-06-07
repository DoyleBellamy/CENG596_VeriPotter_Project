#!/bin/bash
# Creates the 'longeval' conda environment with all dependencies.
# Usage: bash scripts/setup_env.sh

set -e

CONDA=$(conda info --base)/bin/conda
PROJECT=$(cd "$(dirname "$0")/.." && pwd)

echo "=== Creating conda environment 'longeval' ==="
$CONDA create -y -n longeval python=3.10

echo "=== Activating environment ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate longeval

echo "=== Installing Java (required by PyTerrier) ==="
$CONDA install -y -c conda-forge openjdk=21

echo "=== Installing base packages ==="
pip install \
    python-terrier==0.13.0 \
    click \
    "tira>=0.0.160" \
    "tirex-tracker>=0.2.14" \
    openai \
    numpy \
    pandas \
    flask

echo "=== Installing pyterrier-dr ==="
pip install pyterrier-dr

echo "=== Installing ir-datasets-longeval from local clone ==="
pip install -e "$PROJECT/ir-datasets-longeval"

echo "=== Done. Activate with: conda activate longeval ==="
