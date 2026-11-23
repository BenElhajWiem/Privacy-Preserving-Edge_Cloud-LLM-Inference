#!/bin/bash
#SBATCH --job-name=gapdeleg-components
#SBATCH --gpus=0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/%x-%j.out

set -eo pipefail

source ${SCRATCH:-$HOME/scratch}/envs/gapdeleg/bin/activate

python scripts/pupa_components.py --out runs/pupa-components "$@"
echo "finished $(date)"
