#!/bin/bash
#SBATCH --job-name=ollama-create
#SBATCH --gpus=0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-00:15:00
#SBATCH --output=logs/%x-%j.out
set -eo pipefail

NAME=${1:?usage: sbatch cluster/create_model.sh <name> <modelfile>}
FILE=${2:?usage: sbatch cluster/create_model.sh <name> <modelfile>}
. cluster/ollama_env.sh
ollama create "$NAME" -f "$FILE" 2>&1 | tail -2
echo "== template lines that mention a date:"
ollama show "$NAME" --template | grep -n -i "date" || true
echo "== model layer (must equal the base model's):"
ollama show "$NAME" --modelfile | grep -m1 "^FROM"
ollama list | grep -F "${NAME%%:*}"
