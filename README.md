# FirmBehavior

Agent-based simulation of tech firms responding to AI. Firms choose between human, AI-augmented, and automated production modes across a portfolio of tasks. A labor market layer adds worker mobility and wage spillovers.

## Install

Kernel only (simulation + analysis):

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

Dashboard (adds Streamlit):

```bash
.venv/bin/pip install -e .[dashboard]
```

## Run the dashboard

```bash
.venv/bin/streamlit run app.py
```

Opens at `http://localhost:8501`. The sidebar exposes all 22 simulation parameters grouped into 7 expanders. Click **Run** to execute a simulation; plots update in a 4×2 grid.

## Validate

```bash
.venv/bin/python validate_tier_a.py
```

Runs the Phase 1 validation checks (constant baseline, greedy dominance, monotonicity, numeraire invariance, adjustment cost integration).

## Test

```bash
.venv/bin/pytest
```

Runs the full test suite (Phase 1 kernel tests + Phase 1.5 stage tests + dashboard tests).

## Design docs

- [`00_overview.md`](00_overview.md) — project overview and research questions
- [`01_phase1_single_firm.md`](01_phase1_single_firm.md) — Phase 1 single-firm model
- [`02_phase2_two_firms.md`](02_phase2_two_firms.md) — Phase 2 two-firm competition
- [`03_phase3_market.md`](03_phase3_market.md) — Phase 3 N-firm market
- [`04_phase4_welfare_writeup.md`](04_phase4_welfare_writeup.md) — Phase 4 welfare metrics
- [`CLAUDE.md`](CLAUDE.md) — codebase conventions and tech stack
