#!/bin/bash
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
WAIT_MIN=${WAIT_MIN:-240}
unset OLLAMA_HOST_URL

ready() { pmset -g batt | grep -q "AC Power" && [ "$(pmset -g | awk '/lowpowermode/ {print $2}')" = "0" ]; }
for ((i = 0; i < WAIT_MIN; i++)); do ready && break; [ $i -eq 0 ] && echo "$(date) waiting for mains power and Low Power Mode off"; sleep 60; done
ready || { echo "$(date) not on mains power with Low Power Mode off after $WAIT_MIN min: not measuring"; exit 1; }
echo "$(date) conditions met: $(pmset -g batt | head -1)"

COMMON="--ids-file configs/mac_cost_ids.txt --no-judge --no-attacker --measure-memory"
run() {
    local out=$1; shift
    if [ -f "$out/table.json" ] && grep -q '"run"' "$out/table.json"; then echo "skip $out (done)"; return; fi
    ready || { echo "$(date) power conditions lost before $out: stopping"; exit 1; }
    echo "$(date) start $out"
    $PY scripts/run_baselines.py "$@" $COMMON --out "$out" > "$out.log" 2>&1 || echo "$(date) FAILED $out (see $out.log)"
}
mkdir -p runs/mac-cost
for m in qwen3:1.7b qwen3:4b-q4_K_M; do
    t=${m//:/-}
    for s in edge_only redact_send cascade automix papillon; do
        run runs/mac-cost/$s-$t --systems $s --edge $m
    done
    run runs/mac-cost/prism-$t --systems prism --prism-seed 0 --edge $m
done
run runs/mac-cost/papillon-llama3.2-1b --systems papillon --edge llama3.2:1b --papillon-prompts llama_32_1b
run runs/mac-cost/papillon-llama3.2-3b --systems papillon --edge llama3.2:3b --papillon-prompts llama_32_3b

run_method() {
    local out=$1
    if [ -f "$out/table.json" ] && grep -q '"run"' "$out/table.json"; then echo "skip $out (done)"; return; fi
    ready || { echo "$(date) power conditions lost before $out: stopping"; exit 1; }
    echo "$(date) start $out"
    mkdir -p "$out"
    $PY scripts/mac_cost_method.py --set "$2" --edge "$3" --out "$out" > "$out.log" 2>&1 || echo "$(date) FAILED $out (see $out.log)"
}
for m in qwen3:1.7b qwen3:4b-q4_K_M; do
    for set in pupa privfin privmed; do
        run_method runs/mac-cost/method_$set-${m//:/-} $set $m
    done
done
echo "$(date) all on-device cost runs done"
