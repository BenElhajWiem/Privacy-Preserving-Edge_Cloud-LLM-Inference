#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
VENV=${GAPDELEG_VENV:-$HOME/venv}

if [ ! -x "$VENV/bin/python" ]; then
  uv venv -q --python 3.12 "$VENV"
fi
source "$VENV/bin/activate"
uv pip install -q -r "$REPO/env/requirements-cloud.txt"
for m in en_core_web_lg en_core_web_sm; do
  python -c "import $m" 2>/dev/null || uv pip install -q "https://github.com/explosion/spacy-models/releases/download/${m}-3.8.0/${m}-3.8.0-py3-none-any.whl"
done

bash "$REPO/env/get_external.sh"
echo "setup done; run: $VENV/bin/python $REPO/env/check_env.py"
