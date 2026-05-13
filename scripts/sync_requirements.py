"""Regenerate requirements.txt from pyproject.toml dashboard optional-deps.

Usage:
  python scripts/sync_requirements.py          # write requirements.txt in place
  python scripts/sync_requirements.py --check  # dry-run; exit 1 if file would change
"""
import argparse
import re
import sys
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


def _extract_deps(toml_text: str, section_pattern: str) -> list[str]:
    m = re.search(section_pattern, toml_text, re.DOTALL)
    if not m:
        return []
    raw = m.group(1)
    deps = [d.strip().strip('"').strip("'") for d in raw.split(",") if d.strip().strip('"').strip("'")]
    return [d for d in deps if d]


def _dep_name(dep: str) -> str:
    return re.split(r"[><=!;]", dep)[0].strip().lower()


def generate() -> str:
    toml = PYPROJECT.read_text()
    base = _extract_deps(toml, r'dependencies\s*=\s*\[([^\]]+)\]')
    dash = _extract_deps(toml, r'\[project\.optional-dependencies\].*?dashboard\s*=\s*\[([^\]]+)\]')
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
