"""P2-2: verify requirements.txt matches pyproject.toml dashboard extras."""
import subprocess
import sys
from pathlib import Path


def test_requirements_in_sync():
    """requirements.txt must match pyproject.toml dashboard optional-deps."""
    result = subprocess.run(
        [sys.executable, "scripts/sync_requirements.py", "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"requirements.txt is out of sync with pyproject.toml.\n"
        f"Run: python scripts/sync_requirements.py\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
