# Privacy-Preserving Edge_Cloud LLM Inference

Code for *Gap-Aware Delegation: Privacy-Preserving Edge–Cloud Language Model Inference with an On-Device Egress
Guard and Value-Free Method Requests*.

A small model on the device answers first and asks a cloud model only for what it cannot do well. An on-device
labeler marks each private span as public, abstractable or local-only, and a fail-closed egress guard checks every
message before it leaves. When the private data is the hard part, the cloud returns a value-free method, such as a
small program, and the device applies it to the private values.

## Install

Python 3.12 and [Ollama](https://ollama.com).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
```

For the PAPILLON and PRISM baselines, also install their dependencies and fetch their authors' code:

```bash
pip install -e ".[baselines]"
bash env/get_external.sh
```

## Try it

**1. Unit tests** (no model, no key):

```bash
pytest
```

**2. Offline demo** (no model, no key). On two made-up requests it shows what may leave the device, the egress
guard's decision, the leakage check, a cloud reply restored on the device, and a cloud-written program run on private
numbers in the on-device sandbox:

```bash
python scripts/demo.py
```

**3. The full method with local models only** (no key). A larger local model plays the cloud:

```bash
ollama pull qwen3:1.7b && ollama pull qwen3:4b-q4_K_M

# Family A (PUPA): the method, and two baselines on the same requests
python scripts/run_method.py --edge qwen3:1.7b --cloud-backend ollama --cloud qwen3:4b-q4_K_M \
    --gate open --sample 5 --no-judge --out runs/try-method
python scripts/run_baselines.py --systems edge_only,redact_send --edge qwen3:1.7b \
    --cloud-backend ollama --cloud qwen3:4b-q4_K_M --sample 5 --no-judge --no-attacker --out runs/try-baselines

# Family B (PrivFin): the cloud returns a program, the device runs it on the private numbers
python scripts/famb_run.py --idea formula --variant try --system gapdeleg.formula.formula:make \
    --system-args prompt=py_ex,mask=clean --data privfin_dev --edge qwen3:1.7b \
    --local --local-cloud-model qwen3:4b-q4_K_M --limit 5
```

Each run writes `runs/<id>/`: `config.yaml`, `metadata.json` (exact models and their digests), `outgoing.jsonl`
(everything that left the device), `guard.jsonl` (every guard decision), `results.jsonl` and `summary.json`. The
leakage metric reads only `outgoing.jsonl`.

## Results

`results/` holds the summary results behind the paper's tables and figures: for each system and edge model, the
scores with their 95% bootstrap intervals, and the paired tests against redact-and-send. To print the main tables:

```bash
python scripts/show_results.py
```

| Folder | What it holds |
|---|---|
| `family_a_test/` | every system on the family A test set (`ci.json`), the method's thresholds (`method_curves/`), and the gate, guard and labeler analyses (`components.json`) |
| `family_a_pilot/` | every system on the pilot, and the method with and without the gap loop |
| `family_a_dev/` | the method with and without the gap loop on the development split |
| `baselines_pilot/` | the baselines on the pilot, and the cloud-sampling variance check |
| `baseline_thresholds/` | the cascade and AutoMix thresholds chosen on the development split |
| `family_b_test/` | the family B test sets: exact match, hard violations, the decision rule, and the attacker |
| `eightb_dev/` | the three 8B edge models on the development split |
| `on_device_cost/` | on-device time and memory on the laptop |

## Full runs

These use Claude Sonnet 5.5 as the cloud model and the attacker, and GPT-5.5 as the quality judge. Copy `.env.example`
to `.env` and fill in your own keys:

- `GAPDELEG_ANTHROPIC_API_KEY`: the cloud model and the attacker;
- `GAPDELEG_OPENAI_API_KEY`: the quality judge (not needed with `--no-judge`);
- `HF_TOKEN`: only for gated edge models.

Other models can be used with `--cloud <model>`, or with `GAPDELEG_CLOUD_MODEL` and `GAPDELEG_JUDGE_MODEL` in `.env`.
Edge models are served by Ollama at `http://localhost:11434` unless `OLLAMA_HOST_URL` says otherwise.
`run_method.py` stops a run once its fresh cloud calls cost more than `--max-cloud-usd` (default 5), and the family B
runners have fixed caps. Check the setup with `python env/check_env.py`.

```bash
# Family A: the method, gate open and closed; thresholds are chosen on the development split
python scripts/run_method.py --edge qwen3:1.7b --data pupa_new --split dev --sample 0 --gate open --out runs/method/open
python scripts/run_method.py --edge qwen3:1.7b --data pupa_new --split dev --sample 0 --gate closed --out runs/method/closed
python scripts/gap_loop_ablation.py runs/method/open runs/method/closed --out runs/method/no-gap-loop
python scripts/method_curves.py --open runs/method/no-gap-loop/open --closed runs/method/no-gap-loop/closed \
    --out runs/method/curves

# Family A: the baselines (edge-only, cloud-only, redact-and-send, cascade, AutoMix, PAPILLON, PRISM)
python scripts/run_baselines.py --edge qwen3:1.7b --ids-file configs/pupa_test_ids.txt --sample 0 --out runs/baselines
python scripts/bootstrap_ci.py runs/baselines/table.json --against redact_send --out runs/baselines

# Family B: the method and the baselines on the test sets, then the comparison
python scripts/famb_method.py --data privfin_test --edge qwen3:1.7b
python scripts/famb_baselines.py --data privfin_test --sample 0 --systems redact_send --edge qwen3:1.7b \
    --no-attacker --no-judge
python scripts/famb_compare.py
```

The cascade and AutoMix thresholds come from `scripts/score_edge.py` and `scripts/threshold_curves.py` on the
development split; the registered ones are in `configs/baseline_thresholds.json`. Cloud responses are cached under
`runs/cache/`, so a rerun with unchanged messages costs nothing. `cluster/sweep.sh` runs one command per line of a
task file as a Slurm array job (set `SCRATCH`, and pass your account and partition to `sbatch`).

## Layout

```
gapdeleg/
  privacy/      policy.yaml, labeling_policy.txt, detectors, labeler, egress guard
  pipeline/     the method (run.py), value-free method requests (method_request.py),
                egress.py (the only path to the cloud: guard, then cloud, then log)
  formula/      the PrivFin method: a cloud-written program run in an on-device sandbox
  confidence/   the gate's confidence signal
  split/        typed placeholders and their restoration
  baselines/    edge-only, cloud-only, redact-and-send, cascade, AutoMix, PAPILLON, PRISM
  backends/     Ollama, MLX, llama.cpp, Anthropic, OpenAI; cloud backends refuse to run without a guard ticket
  metrics/      leakage, quality, attacker, bootstrap intervals, cost
  data/, build/ data schema, loaders and dataset construction
scripts/        experiments, analyses and the offline demo
cluster/        Slurm jobs
configs/        request ids and registered thresholds
data/           processed datasets and frozen splits
inspection/     the two human-check sheets
results/        the summary results behind the paper's tables
tests/          unit tests
```

## Data

Every request the experiments use is in the repository, already labeled, so no download is needed to run them:

- `data/processed/`: PUPA (family A, re-labeled public / abstractable / local-only, with character offsets),
  PrivFin and PrivMed (family B, with synthetic personas), and the frozen splits in `splits.json`. The loaders read
  them with `gapdeleg.data.load("processed", name=..., split=...)`.
- `configs/pupa_test_ids.txt`: the family A test requests.
- `data/review/`: the annotation review (second annotator, adjudication). `data/judge_validation/`: the quality
  judge's validation set.
- `inspection/`: the two human-check sheets, scored by `scripts/judge_agreement.py` and `scripts/label_agreement.py`.

`bash env/get_external.sh` fetches, at the versions used, the PAPILLON and PRISM code the two baselines need and the
sources the datasets were built from (PUPA, GSM8K, gretelai/symptom_to_diagnosis). Rebuilding the datasets
(`python -m gapdeleg.build.pupa_labels`, `privfin`, `privmed`) calls the annotator models and needs both API keys.

## Licenses

The code and our labels, personas and splits are under the MIT license ([LICENSE](LICENSE)). The request texts keep
their sources' licenses:

| Files | Source | License |
|---|---|---|
| `pupa_*.jsonl`, `data/review/`, `data/judge_validation/` | PUPA (Li et al., 2025), built from WildChat (Zhao et al., 2024) | PUPA: MIT. WildChat: ODC-BY 1.0; cite WildChat along with PUPA |
| `privfin_*.jsonl` | GSM8K (Cobbe et al., 2021) | MIT |
| `privmed_test.jsonl`, `privmed_dev.jsonl` | gretelai/symptom_to_diagnosis | Apache-2.0 |
| `privmed_prism.jsonl` | PRISM's medical prompts, partly adapted from PrivacyRestore | PRISM: MIT; PrivacyRestore states no license |
