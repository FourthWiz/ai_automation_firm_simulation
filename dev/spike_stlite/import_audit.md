# T-03: Pyodide Import Compatibility Audit

**Date:** 2026-05-17  
**Kernel version audited:** HEAD at task T-03 implementation  
**Files audited:** 15 kept `.py` files (17 total − 2 stripped = 15)  
**Strip list:** `viz.py` (matplotlib), `validate.py` (subprocess) — both confirmed module-scope incompatible

## Verdict

**Zero BLOCK verdicts.** All 15 kept files are Pyodide-compatible.

## Per-File Results

| File | External deps (module-scope) | Verdict |
|---|---|---|
| `__init__.py` | — (uses lazy imports for simulate/workers/review/dp/margin) | compatible |
| `adjustment.py` | `math`, `numpy` | compatible |
| `config.py` | `math`, `dataclasses` | compatible |
| `dashboard.py` | `math`, `numpy`, `pandas`, `plotly.graph_objects`, `plotly.subplots` | compatible |
| `dp_optimizer.py` | `math`, `numpy` | compatible |
| `firm.py` | `dataclasses`, `numpy` | compatible |
| `forward_sim.py` | `copy`, `math`, `numpy` | compatible |
| `margin_optimizer.py` | `copy`, `math`, `numpy` | compatible |
| `production.py` | `math`, `enum.IntEnum`, `numpy` | compatible |
| `review.py` | `math`, `warnings`, `numpy` | compatible |
| `simulate.py` | `math`, `numpy`, `pandas` | compatible |
| `strategy.py` | `numpy` | compatible |
| `tasks.py` | `numpy` | compatible |
| `theme.py` | — (no external imports) | compatible |
| `workers.py` | `dataclasses`, `numpy` | compatible |

## Stripped Files (not bundled)

| File | Reason stripped |
|---|---|
| `viz.py` | `import matplotlib` at module scope — incompatible with Pyodide |
| `validate.py` | `import subprocess` at module scope — incompatible with Pyodide |

## Danger-class scan results

Checked all 15 kept files for: `multiprocessing`, `threading`, `concurrent.futures`, `subprocess`, `requests`, `urllib3`

**Result:** Zero matches in any kept file. Only `validate.py` (stripped) has subprocess.

## Pyodide wheel availability

- `numpy` — included in Pyodide core distribution ✓
- `pandas` — included in Pyodide core distribution ✓
- `plotly` — pre-bundled in `@stlite/browser@1.7.3` build ✓ (do NOT add to `requirements=` in mount config)
- No `pyarrow` dependency anywhere in kernel (grep verified; R-12 closed)

## New imports check (since 2026-05-17)

All imports verified at HEAD. No new incompatible imports found. Re-run this audit if any `firm_ai_abm/*.py` file is modified before T-08 (bundle step).
