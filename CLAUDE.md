# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Agent-based simulation of tech firms responding to AI. Firms choose between human, AI-augmented, and automated production modes across a portfolio of tasks. A labor market layer adds worker mobility and wage spillovers. Goal: discover which strategies win under which conditions, and how that answer differs for shareholders vs. workers.

**Current state:** Design phase. All five `.md` files are design documents; no source code exists yet. Start with `00_overview.md`, then the relevant phase doc.

## Phase roadmap

| Phase | Focus | Doc |
|---|---|---|
| 1 | Single firm, task-based, three modes — validate core production function | `01_phase1_single_firm.md` |
| 2 | Two competing firms, endogenous labor market | `02_phase2_two_firms.md` |
| 3 | N=20 firms, continuous strategy dials, skill heterogeneity | `03_phase3_market.md` |
| 4 | Welfare metrics, sensitivity sweeps, paper writeup | `04_phase4_welfare_writeup.md` |

## Tech stack

- Python 3.11+, numpy, pandas (numerical core)
- matplotlib/seaborn (paper figures), plotly (interactive/GIF)
- Jupyter for development, plain scripts for parameter sweeps
- pytest (Phase 4)
- **Not using:** mesa, agentpy, PyTorch, sklearn — plain numpy is fast enough and avoids abstraction tax

## Planned repo layout (build incrementally, don't scaffold all at once)

```
firm-ai-abm/
  firm_ai_abm/
    config.py       # Default parameters (FirmParams dataclass)
    tasks.py        # Task generation (alpha, beta arrays)
    firm.py         # Firm class, production function (vectorized numpy)
    strategy.py     # Strategy dials, mode allocation rules
    worker.py       # Worker class, reservation wages (Phase 2+)
    market.py       # Hiring, firing, matching (Phase 2+)
    wage.py         # Wage-setting heuristics (Phase 2+)
    simulate.py     # Time loop
    sweep.py        # Sensitivity sweeps (Phase 3+)
    welfare.py      # Welfare metrics (Phase 4)
    viz.py          # Plotting
  notebooks/
  results/sweeps/   # CSVs from sensitivity runs
  results/figures/  # PNG/SVG outputs
  paper/
  pyproject.toml
```

## Core model mechanics

**Task-based production (Acemoglu & Restrepo 2018):** Each firm has N tasks, heterogeneous in automatability `α_i` and augmentability `β_i`. Per task, firm picks mode `m_i ∈ {H, A, T}`:

| Mode | Productivity | Cost/task/period |
|---|---|---|
| H (human) | `q_h` | `w / N` |
| A (augmented) | `q_h · (1 + g · β_i)` | `w/N + c_aug` |
| T (automated) | `q_a · α_i` | `c_auto` |

Output is additive across tasks: `Y = Σ productivity(m_i)`. Profit `π = p·Y − C`.

**Core vectorization pattern** (use this from day one, Phase 3 runs thousands of simulations):
```python
def productivity_vec(modes, alpha, beta, params):
    p = np.where(modes == 0, params.q_h, 0)
    p = np.where(modes == 1, params.q_h * (1 + params.g * beta), p)
    p = np.where(modes == 2, params.q_a * alpha, p)
    return p
```

## Phase 1 validation checks (must pass before Phase 2)

These are the ground truth for correctness. If any fail, fix before proceeding:

1. **Constant baseline:** All-H with `q_a=0, g=0, c_aug=0` must produce constant per-period profit = `p·q_h·N − w·K − F`.
2. **Greedy dominance:** With all switching costs = 0, Greedy-profit must ≥ every other strategy at every period.
3. **Monotonicity in `q_a`:** Higher `q_a` monotonically increases Automate-all profit; leaves Augment-all unchanged.
4. **Monotonicity in `w`:** Higher `w` monotonically decreases All-H profit; leaves Automate-all unchanged.
5. **Numeraire invariance:** Multiplying all monetary params (w, c_aug, c_auto, c_fire, c_hire, c_train, F, p) by 2 must scale all profits by exactly 2.
6. **Adjustment cost integration:** With `c_train=100`, Greedy must refuse to switch modes.

## Pre-committed findings to test

The model exists to test these three (falsifying is also a result):

1. **Coordination failure:** Individually rational firing can produce collective race-to-bottom worse than augmentation.
2. **Augment-vs-automate tipping:** A productivity threshold exists above which augment dominates; skill heterogeneity shifts it.
3. **Welfare wedge:** Profit-maximizing strategy diverges from median-worker-welfare strategy as automation costs fall.

Every design decision should connect to testing one of these. If a feature doesn't, defer it.

## Running the code (once it exists)

```bash
# Development
jupyter notebook notebooks/01_phase1_single_firm.ipynb

# Run a specific simulation script
.venv/bin/python -m firm_ai_abm.simulate

# Sensitivity sweeps (Phase 3+)
.venv/bin/python -m firm_ai_abm.sweep

# Tests (Phase 4)
.venv/bin/pytest
```

## Scope discipline

Each phase doc lists explicit "do NOT build" items. Respect them — the weekend pace depends on it. If scope creep is tempting, add a note to the phase doc under "Extensions" and move on.
