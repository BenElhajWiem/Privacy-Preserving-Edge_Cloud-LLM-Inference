from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import yaml

from gapdeleg import REPO_ROOT


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def power_state() -> dict:
    if platform.system() != "Darwin":
        return {}
    try:
        batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
        conf = subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=5).stdout
        low = next((ln.split()[-1] for ln in conf.splitlines() if "lowpowermode" in ln), None)
        return {"on_battery": "Battery Power" in batt, "low_power_mode": low == "1"}
    except Exception:
        return {}


class RunDir:
    def __init__(self, root: str | Path = REPO_ROOT / "runs", run_id: str | None = None, name: str = "run"):
        self.id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
        self.path = Path(root) / self.id
        self.path.mkdir(parents=True, exist_ok=True)

    @property
    def outgoing(self) -> Path:
        return self.path / "outgoing.jsonl"

    @property
    def guard_log(self) -> Path:
        return self.path / "guard.jsonl"

    @property
    def results(self) -> Path:
        return self.path / "results.jsonl"

    def write_config(self, cfg: dict) -> None:
        (self.path / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    def write_metadata(self, models: dict) -> None:
        import psutil
        meta = {"run_id": self.id, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
                "platform": platform.platform(), "machine": platform.machine(),
                "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
                "power": power_state(), "models": models}
        (self.path / "metadata.json").write_text(json.dumps(meta, indent=1))

    def append_result(self, rec: dict) -> None:
        with open(self.results, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def write_json(self, name: str, obj) -> Path:
        p = self.path / name
        p.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
        return p
