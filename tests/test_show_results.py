import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_show_results_prints_every_section():
    out = subprocess.run([sys.executable, str(ROOT / "scripts/show_results.py")], capture_output=True, text=True,
                         check=True).stdout
    for title in ("Family A test set", "Family A pilot", "Family B test sets", "Family B attacker"):
        assert f"=== {title}" in out
