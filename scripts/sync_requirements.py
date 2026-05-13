"""Regenerate requirements.txt from pyproject.toml dashboard optional-deps.

Uses tomllib (stdlib, Python 3.11+) to parse pyproject.toml safely,
including PEP 508 environment markers and quoted commas.

Usage:
  python scripts/sync_requirements.py          # write requirements.txt in place
  python scripts/sync_requirements.py --check  # dry-run; exit 1 if file would change
"""
import argparse
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"

HEADER = (
    "# Source of truth: pyproject.toml [project.optional-dependencies] dashboard\n"
    "# Keep in sync via scripts/sync_requirements.py + tests/test_requirements_sync.py\n"
)

# These base deps are excluded from requirements.txt (dev-only / not needed at runtime).
_EXCLUDED_BASE_DEPS = {"matplotlib", "seaborn", "jupyter"}


def _dep_name(dep: str) -> str:
    """Extract bare package name from a PEP 508 dep specifier."""
    import re
    return re.split(r"[><=!;@\s]", dep)[0].strip().lower()


def generate() -> str:
    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)

    base: list[str] = data.get("project", {}).get("dependencies", [])
    dash: list[str] = (
        data.get("project", {})
        .get("optional-dependencies", {})
        .get("dashboard", [])
    )

    if not base:
        sys.exit("ERROR: [project].dependencies is empty or missing in pyproject.toml")
    if not dash:
        sys.exit("ERROR: [project.optional-dependencies].dashboard is empty or missing in pyproject.toml")

    filtered_base = [d for d in base if _dep_name(d) not in _EXCLUDED_BASE_DEPS]
    all_deps = filtered_base + dash
    return HEADER + "\n".join(all_deps) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="dry-run; exit 1 if out of sync")
    args = parser.parse_args()

    generated = generate()

    if args.check:
        current = REQUIREMENTS.read_text() if REQUIREMENTS.exists() else ""
        if current != generated:
            print("requirements.txt is out of sync with pyproject.toml dashboard deps.")
            print("Run: python scripts/sync_requirements.py")
            sys.exit(1)
        print("requirements.txt is in sync.")
    else:
        REQUIREMENTS.write_text(generated)
        print(f"Written: {REQUIREMENTS}")


if __name__ == "__main__":
    main()
