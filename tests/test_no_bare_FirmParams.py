"""T-15 meta-test: Verify that non-test firm_ai_abm/*.py files do not have
unpinned FirmParams() calls (without tasks_per_worker= explicitly set).

Any bare FirmParams() call in validate.py, viz.py, production.py, or other
firm_ai_abm/*.py modules would silently use the new defaults (tpw=5, p=0.22),
breaking the legacy Stage 1-5 validation numerics.
"""
import re
import pathlib

import pytest


_FIRM_AI_ABM = pathlib.Path(__file__).parent.parent / "firm_ai_abm"


def _collect_module_files():
    """Return all non-test .py files under firm_ai_abm/."""
    return [
        p for p in _FIRM_AI_ABM.glob("*.py")
        if not p.name.startswith("test_")
    ]


def _is_in_docstring(lines: list[str], target_idx: int) -> bool:
    """Return True if line at target_idx is inside a multi-line docstring."""
    in_docstring = False
    triple = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i == target_idx:
            return in_docstring
        if not in_docstring:
            # Check if this line opens a docstring
            for q in ('"""', "'''"):
                count = stripped.count(q)
                if count >= 1:
                    # Odd count → opens (or closes) a docstring
                    if count % 2 == 1:
                        in_docstring = True
                        triple = q
                        break
        else:
            # We're in a docstring — look for the closing triple
            if triple in stripped:
                in_docstring = False
                triple = None
    return in_docstring


def test_validate_uses_pinned_defaults():
    """Every FirmParams( call in executable code in firm_ai_abm/*.py must pin tasks_per_worker=.

    Calls in docstrings and comments are excluded.
    """
    failing_sites = []

    for py_file in _collect_module_files():
        source = py_file.read_text(encoding="utf-8")
        lines = source.splitlines()

        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Skip comment lines
            if stripped.startswith("#"):
                continue

            # Skip lines inside docstrings
            if _is_in_docstring(lines, lineno - 1):
                continue

            # Skip docstring delimiter lines themselves
            if stripped.startswith('"""') or stripped.startswith("'''"):
                continue

            # Find FirmParams( occurrences in this line
            if "FirmParams(" not in line:
                continue

            # Allow: function signature parameters (type annotation, not a call)
            if re.search(r"def\s+\w+.*FirmParams", line):
                continue

            # Check if this line or adjacent context pins tasks_per_worker
            call_start = lineno - 1  # 0-indexed
            context_lines = lines[call_start : call_start + 15]
            context = "\n".join(context_lines)

            if "tasks_per_worker=" not in context:
                failing_sites.append(f"{py_file.name}:{lineno}: {stripped[:80]}")

    assert not failing_sites, (
        "Unpinned FirmParams() calls found in firm_ai_abm/ (no tasks_per_worker= set).\n"
        "These will silently use new defaults (tpw=5, p=0.22) and break Stage 1-5 numerics.\n"
        "Add tasks_per_worker=10 (and p=1.0) to each call site:\n"
        + "\n".join(f"  {s}" for s in failing_sites)
    )


def test_validate_uses_pinned_N():
    """Every FirmParams( call in executable code in firm_ai_abm/*.py must pin N=.

    After the N=500 default change (D-01), any unguarded FirmParams() in validate.py
    or other modules would run Phase-1 checks at N=500, breaking parity fixtures
    captured at N=100. This meta-test guards against future regressions.
    Calls in docstrings and comments are excluded. Tests/ files are exempt.
    """
    failing_sites = []

    for py_file in _collect_module_files():
        source = py_file.read_text(encoding="utf-8")
        lines = source.splitlines()

        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()

            if stripped.startswith("#"):
                continue

            if _is_in_docstring(lines, lineno - 1):
                continue

            if stripped.startswith('"""') or stripped.startswith("'''"):
                continue

            if "FirmParams(" not in line:
                continue

            if re.search(r"def\s+\w+.*FirmParams", line):
                continue

            call_start = lineno - 1
            context_lines = lines[call_start : call_start + 15]
            context = "\n".join(context_lines)

            if "N=" not in context:
                failing_sites.append(f"{py_file.name}:{lineno}: {stripped[:80]}")

    assert not failing_sites, (
        "FirmParams() calls without explicit N= found in firm_ai_abm/.\n"
        "These will silently use the new default (N=500) and may break Phase-1 parity.\n"
        "Add N=100 (or appropriate value) to each call site:\n"
        + "\n".join(f"  {s}" for s in failing_sites)
    )
