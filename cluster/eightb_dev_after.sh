#!/bin/bash
#SBATCH --job-name=gapdeleg-8b-dev
#SBATCH --gpus=0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/%x-%j.out
set -eo pipefail
source ${SCRATCH:-$HOME/scratch}/envs/gapdeleg/bin/activate
D=runs/eightb-dev
MODELS="llama3.1-8b qwen3-8b ministral3-8b"

echo "== 1. method without the gap loop"
python scripts/gap_loop_ablation.py $(for m in $MODELS; do echo $D/method/open-$m $D/method/closed-$m; done) --out $D/method/no-gap-loop

echo "== 2. judge (one batch)"
python scripts/judge_runs.py $(for m in $MODELS; do echo $D/edge_only-$m $D/redact_send-$m $D/method/no-gap-loop/open-$m $D/method/no-gap-loop/closed-$m; done) --batch

echo "== 3. cascade and AutoMix thresholds"
for m in $MODELS; do
  python scripts/threshold_curves.py --scores runs/scores-dev/$m --edge-only $D/edge_only-$m --redact $D/redact_send-$m \
    --out runs/scores-dev/$m --register configs/baseline_thresholds.json
done

echo "== 4. the method's thresholds"
for m in $MODELS; do
  python scripts/method_curves.py --open $D/method/no-gap-loop/open-$m --closed $D/method/no-gap-loop/closed-$m \
    --out $D/method-curves/$m
done
echo "finished $(date)"
