#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
EXTERNAL=${GAPDELEG_EXTERNAL:-$REPO/external}
mkdir -p "$EXTERNAL"

fetch() {
  local dir="$EXTERNAL/$(basename "$1")"
  [ -d "$dir/.git" ] || git clone -q "https://github.com/$1.git" "$dir"
  git -C "$dir" checkout -q "$2"
}

fetch Columbia-NLP-Lab/PAPILLON 594e7ebe5ab87090916a53425a2c2ac0f4ec6239
fetch Junfei-Z/PRISM 49df4f9ad641e66bb362e3286034ed951455136b
fetch openai/grade-school-math 3101c7d5072418e28b9008a6636bde82a006892c
python -c "from huggingface_hub import snapshot_download; snapshot_download('gretelai/symptom_to_diagnosis', repo_type='dataset', local_dir='$EXTERNAL/symptom_to_diagnosis', allow_patterns=['*.jsonl'])"
echo "external sources are in $EXTERNAL"
