#!/bin/bash
# Train the transformer in WSL (Ubuntu) on the GPU. Run from a WSL terminal:
#
#   bash /mnt/d/GITHUB_SOFTWARE/SPTnano-core/wsl/train_transformer.sh --smoke_test   # quick check, a few minutes
#   bash /mnt/d/GITHUB_SOFTWARE/SPTnano-core/wsl/train_transformer.sh                # full run, about a day
#
# Any extra options are passed to scripts/train.py (e.g. --out_dir /mnt/d/TRANSFORMER_DEVELOPMENT/saved_data/my_model).
# First time only:  conda env create -f environment_wsl.yml
set -e

CONDA_DIR="$HOME/miniconda3"     # change if conda is installed somewhere else
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate nanoSPTlinux

cd "$(dirname "$0")/.."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "No GPU found - training will be very slow"
python scripts/train.py "$@"
