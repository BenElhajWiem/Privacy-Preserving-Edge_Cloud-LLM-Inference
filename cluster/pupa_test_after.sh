#!/bin/bash
#SBATCH --job-name=gapdeleg-pupatest
#SBATCH --gpus=0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/%x-%j.out

set -eo pipefail

source ${SCRATCH:-$HOME/scratch}/envs/gapdeleg/bin/activate

T=runs/pupa-test
MODELS="qwen3-1.7b qwen3-4b-q4_K_M qwen3-4b-instruct-2507 llama3.2-3b gemma3-4b"
BASELINES=$(ls -d $T/*/ | grep -vE "/(method|ci)/$" | sed 's|/$||')

echo "== 1. method without the gap loop"
python scripts/gap_loop_ablation.py $(for m in $MODELS; do echo $T/method/open-$m $T/method/closed-$m; done) --out $T/method/no-gap-loop

echo "== 2. judge (one batch)"
python scripts/judge_runs.py $BASELINES $(for m in $MODELS; do echo $T/method/no-gap-loop/open-$m $T/method/no-gap-loop/closed-$m; done) --batch

echo "== 3. method at its dev thresholds"
for m in $MODELS; do
  python scripts/method_curves.py --open $T/method/no-gap-loop/open-$m --closed $T/method/no-gap-loop/closed-$m \
    --fixed runs/pupa-quality-dev/curves/$m/selected.json --out $T/method/curves/$m
done

echo "== 4. baseline table and intervals"
python scripts/run_baselines.py --collect $BASELINES --out $T
python scripts/bootstrap_ci.py $T/table.json --against redact_send \
  --extra $(for m in $MODELS; do for r in best_quality zero_violation; do echo "method_$r=$T/method/curves/$m/fixed/$r.json"; done; done) \
  --out $T/ci
echo "finished $(date)"
