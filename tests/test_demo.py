import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_demo_runs_offline():
    out = subprocess.run([sys.executable, str(ROOT / "scripts/demo.py")], capture_output=True, text=True,
                         check=True).stdout
    assert "full request    -> blocked" in out
    assert "masked request  -> allowed" in out
    assert "masked request  -> hard violation: False" in out
    assert "Answer: 5250" in out
