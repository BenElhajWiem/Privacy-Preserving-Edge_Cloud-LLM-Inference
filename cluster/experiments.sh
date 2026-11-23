#!/bin/bash
#SBATCH --job-name=gapdeleg
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x-%j.out

set -eo pipefail


MODELS=${MODELS:-qwen3:1.7b}
SCRATCH=${SCRATCH:-$HOME/scratch}
export HF_HOME=$SCRATCH/cache/huggingface XDG_CACHE_HOME=$SCRATCH/cache PIP_CACHE_DIR=$SCRATCH/cache/pip
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

. cluster/ollama_env.sh
for m in $MODELS; do
  ollama pull $m 2>&1 | tail -1 || { ollama show $m >/dev/null || exit 1; echo "pull failed, using local copy: $(ollama list | grep -F "$m ")"; }
done

source $SCRATCH/envs/gapdeleg/bin/activate
echo "job $SLURM_JOB_ID on $(hostname -s), models: $MODELS, args: $*, started $(date)"
python scripts/run_experiment.py "$@"
echo "finished $(date)"
