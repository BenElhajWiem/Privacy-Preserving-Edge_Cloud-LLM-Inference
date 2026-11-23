#!/bin/bash
#SBATCH --job-name=ollama-pull
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/%x-%j.out

set -eo pipefail


. cluster/ollama_env.sh
for m in "$@"; do ollama pull $m 2>&1 | tail -1; done
ollama list
