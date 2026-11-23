#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
ROOT=${1:?checkout with runs/ and .env}
PY=$ROOT/.venv/bin/python
R=$ROOT/runs/famb-test
LOG=$R/attacker-logs
mkdir -p "$LOG"
MODELS="qwen3-1.7b qwen3-4b-q4_K_M qwen3-4b-instruct-2507-q4_K_M llama3.2-3b gemma3-4b-it-q4_K_M"

attack() {
  local name=$1 budget=$2
  shift 2
  "$PY" scripts/attack_stored.py "$@" --budget "$budget" --root "$ROOT" > "$LOG/$name.log" 2>&1
}

wait_all() {
  local fail=0
  for p in "$@"; do wait "$p" || fail=1; done
  return $fail
}

echo "== stage 1, $(date)"
attack privfin-prior 2.5 "$R/cloud-only/privfin" $(for m in $MODELS; do echo "$R/edge-only/privfin-$m"; done) &
p1=$!
attack privmed-prior 3 "$R/cloud-only/privmed" \
  $(for m in $MODELS; do echo "$R/edge-only/privmed-$m" "$R/method/rows/privmed_test__$m.json"; done) &
p2=$!
wait_all $p1 $p2 || { echo "stage 1 failed: see $LOG"; exit 1; }

echo "== stage 2, $(date)"
pids=()
for m in $MODELS; do
  attack "$m" 3.9 "$R/redact-send/privfin-$m" "$R/method/rows/privfin_test__$m.json" "$R/redact-send/privmed-$m" &
  pids+=($!)
done
wait_all "${pids[@]}" || { echo "stage 2 failed: see $LOG"; exit 1; }

echo "== fresh attacker calls per process"
grep -H "fresh attacker calls" "$LOG"/*.log
echo "finished $(date)"
