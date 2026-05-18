---
title: Firm Behavior Under AI Transformation
emoji: 🤖
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# FirmBehavior

Agent-based simulation of tech firms responding to AI. Firms choose between human, AI-augmented, and automated production modes across a portfolio of tasks. A labor market layer adds worker mobility and wage spillovers.

## Public simulator

An interactive browser-based version of this simulation is available. The dashboard lets you explore how a single firm navigates the human-vs-AI production decision in real time.


**What it shows:**

- **Outcomes tab** — per-period and cumulative profit over 60 simulated periods
- **Workforce tab** — how the firm's headcount evolves, including hiring and firing events
- **Tasks & modes tab** — the mix of human (H), AI-augmented (A), and fully automated (T) tasks over time
- **Wages tab** — individual wage distributions and mean accumulated earnings
- **Worker heterogeneity tab** — the distribution of worker skill (theta) and how it shifts

**Key dials to try:**

- **AI productivity ceiling (q_a)** — raise this above 1.0 to make automation more attractive; at q_a ≈ 1.5 most strategies shift heavily into T-mode
- **Augmentation gain (g)** — how much a human worker benefits from AI tools; higher g favors A-mode (augmented) over pure automation
- **Automation cost per task (c_auto)** — the per-task fee for running an AI agent; lowering this past a threshold tips the firm from augmentation into automation
- **Wage rate (w)** — raising wages makes human labor more expensive relative to automation

Hit **▶ Run simulation** after adjusting any dial. The KPI strip at the top updates instantly with cumulative profit, final workforce size, and the dominant production mode.

> **Note on first load:** if the app is hosted on the free tier it may take ~30 seconds to wake up after a period of inactivity. Give it a moment.

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

Opens at `http://localhost:8501`. Primary controls (strategy, firm size, AI productivity, wages, costs) are in the main panel. An **Advanced parameters** expander below holds 24 additional controls. Click **▶ Run simulation** to run; 13 charts update across 5 tabs.

## Test

```bash
.venv/bin/pytest
```

Runs the full test suite (Phase 1 kernel tests + Phase 1.5 stage tests + dashboard tests).

## Default recalibration (Stage 6)

Defaults recalibrated: `tasks_per_worker=5`, `p=0.22` (F unchanged at 5.0). All-H baseline is intentionally loss-making at these params (≈ −3 per period); strategies that augment or automate tasks improve profit materially. Set `tasks_per_worker=10, p=1.0` in tests that depend on Stage 1–5 numerical fixtures.

## Margin calibration recipe

At `tasks_per_worker=10, p=1.0` (legacy numerics), margins are very high (~80–90%) due to task/worker leverage. Use this recipe to explore near-realistic margins:

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
