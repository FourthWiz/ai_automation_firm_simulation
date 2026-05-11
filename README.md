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

## Margin calibration recipe

Default params produce very high margins (~80–90%) because the task/worker leverage is large (100 tasks, 10 workers). Use this recipe to explore near-realistic margins:

1. Set `w` to `8.0` (raise wage pressure).
2. Set `c_auto` to `0.6` (default 0.4 leaves margins too high at scale).
3. Set `F` to `5` (moderate fixed overhead).
4. Keep default `q_h=1.0`, `q_a=1.2`, `g=0.5`, `p=1.0`.
5. Strategy: `greedy_profit`. Seed: `0`.

Worked example: at N=100, K=10, `seed=0`, the above parameters yield margin ≈ 16% (verified: `pi.mean() / Y.mean() ≈ 0.162`). The firm converges to ~75% A-mode + ~17% T-mode.

**Firing-threshold guidance:**

At default sigmas (`sigma_theta=0.2`, `sigma_w=0.05`), `firing_threshold=0.0` produces 1–2 firings then reaches steady state. `firing_threshold > 0.5` (with default `w=1.0`) may cause monotonic K decay. If K drops near 0, reduce `firing_threshold` or increase `T_review`.

**Multi-batch wage drift note:**

In high-turnover scenarios (many fire+replace cycles), `wage_bill` may drift slightly from the `w*K` baseline due to sampling variance. Reduce `sigma_w` or `sigma_theta` to minimize drift.

## Design docs

- [`00_overview.md`](00_overview.md) — project overview and research questions
- [`01_phase1_single_firm.md`](01_phase1_single_firm.md) — Phase 1 single-firm model
- [`02_phase2_two_firms.md`](02_phase2_two_firms.md) — Phase 2 two-firm competition
- [`03_phase3_market.md`](03_phase3_market.md) — Phase 3 N-firm market
- [`04_phase4_welfare_writeup.md`](04_phase4_welfare_writeup.md) — Phase 4 welfare metrics
- [`CLAUDE.md`](CLAUDE.md) — codebase conventions and tech stack
