import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env(path=ENV_FILE):
    if not path.exists():
        return []
    loaded = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key == "ANTHROPIC_API_KEY":
            raise ValueError("Use GAPDELEG_ANTHROPIC_API_KEY in .env, not ANTHROPIC_API_KEY")
        if value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
