#!/bin/bash
#SBATCH --job-name=gapdeleg-sweep
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -eo pipefail


TASKS=${1:?usage: sbatch --array=<ids> cluster/sweep.sh <task file>}
line=$(grep -v -E '^[[:space:]]*(#|$)' "$TASKS" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
[ -n "$line" ] || { echo "no task $SLURM_ARRAY_TASK_ID in $TASKS"; exit 1; }

SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/cache/huggingface XDG_CACHE_HOME=$SCRATCH/cache PIP_CACHE_DIR=$SCRATCH/cache/pip
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

. cluster/ollama_env.sh
if [[ $line =~ --edge[=\ ]([^ ]+) ]]; then
  m=${BASH_REMATCH[1]}
  ollama pull $m 2>&1 | tail -1 || { ollama show $m >/dev/null || exit 1; echo "pull failed, using local copy: $(ollama list | grep -F "$m ")"; }
fi

source $SCRATCH/envs/gapdeleg/bin/activate
echo "task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID on $(hostname -s), started $(date): python $line"
python $line
echo "finished $(date)"
