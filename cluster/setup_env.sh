#!/bin/bash
set -eo pipefail


SCRATCH=${SCRATCH:-$HOME/scratch}
ENV=$SCRATCH/envs/gapdeleg
mkdir -p $SCRATCH/envs $SCRATCH/tmp $SCRATCH/cache/pip
export TMPDIR=$SCRATCH/tmp TMP=$SCRATCH/tmp PIP_CACHE_DIR=$SCRATCH/cache/pip

[ -d $ENV ] || python -m venv $ENV
source $ENV/bin/activate
pip install -q --upgrade pip
grep -v '^llama-cpp-python' env/requirements-cloud.txt > $TMPDIR/gapdeleg-req.txt
pip install -q -r $TMPDIR/gapdeleg-req.txt
CMAKE_BUILD_PARALLEL_LEVEL=${SLURM_CPUS_PER_TASK:-4} pip install -q 'llama-cpp-python>=0.3' \
  || echo "WARNING: llama-cpp-python did not build; only the llama_cpp backend needs it"
pip install -q "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
for m in en_core_web_lg en_core_web_sm; do
  python -c "import $m" 2>/dev/null \
    || pip install -q "https://github.com/explosion/spacy-models/releases/download/${m}-3.8.0/${m}-3.8.0-py3-none-any.whl"
done

python -m pytest -q -m "not integration"
echo "env ready on $(hostname -s): $ENV"
