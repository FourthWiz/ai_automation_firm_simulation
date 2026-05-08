# Tech Firms Under AI — Agent-Based Model: Project Overview

This is the master design document. Phase docs (`01_phase1_*.md` ... `04_phase4_*.md`) build on the notation, theoretical anchor, and tech stack defined here. If you're starting a new conversation with Claude on this project, paste this file first.

---

## 1. Project goal

Build an agent-based simulation of tech firms responding to AI by reallocating production between human, AI-augmented, and fully automated modes. Layer a labor market on top so worker mobility, layoffs, and wage spillovers emerge endogenously. Use the model to study which firm strategies win under which conditions — and how the answer differs for shareholders vs. workers.

**Dual deliverable:**
1. Shareable visualizations and animations as the model develops (a thread of posts, one per phase).
2. A paper-shaped writeup with lit review, formal model, sensitivity analysis, and one or two crisp findings.

**Pace:** weekend tinkering, ~2 months end-to-end at relaxed cadence.

---

## 2. Theoretical anchor

The model uses a **task-based production function** following Acemoglu & Restrepo (2018, "The Race Between Man and Machine"). Core idea: a firm produces output by completing a continuum of tasks. For each task, the firm chooses among production modes (human, augmented human, AI/agency). Strategy differences become differences in **mode allocation** across tasks — making it natural to compare augment-heavy, fire-heavy, and automate-heavy firms inside the same formalism.

This matters because it lets you smoothly interpolate between strategies rather than hardcoding firm "types." You can find emergent strategy regions instead of imposing them.

**Other lit to skim (don't read in full):**
- Acemoglu & Restrepo (2020) on automation and employment.
- Korinek & Stiglitz on AI and inequality.
- Aghion, Jones & Jones on AI and economic growth.
- Mortensen-Pissarides matching theory for the labor market layer.
- ABM in economics: Tesfatsion (handbook), Dawid & Gemkow on labor-market ABM.

---

## 3. Phase roadmap

| Phase | Weekends | Focus | Deliverable | Doc |
|---|---|---|---|---|
| 1 | 1–2 | Single firm, task-based, three modes | Profit trajectory under each strategy | `01_phase1_single_firm.md` |
| 2 | 3–4 | Two firms competing for workers | Animated worker-flow diagram | `02_phase2_two_firms.md` |
| 3 | 5–7 | N-firm market with continuous strategy dials | Heatmap of "who wins" over a 2D parameter slice | `03_phase3_market.md` |
| 4 | 8–9 | Welfare metrics, sensitivity sweeps, writeup | Paper draft + post thread | `04_phase4_welfare_writeup.md` |

Each phase ships something visual and postable. The paper draws from the accumulated simulations.

---

## 4. Notation (used throughout)

Variables that appear across multiple phase docs:

| Symbol | Meaning | First introduced |
|---|---|---|
| `N` | number of tasks per firm | Phase 1 |
| `T` | simulation horizon (periods) | Phase 1 |
| `i` | task index, `i = 1..N` | Phase 1 |
| `α_i` | task automatability ∈ [0,1] | Phase 1 |
| `β_i` | task augmentability ∈ [0,1] | Phase 1 |
| `m_i` | mode for task i ∈ {H, A, T} (Human, Augmented, auTomated) | Phase 1 |
| `q_h` | human productivity per task (numeraire) | Phase 1 |
| `q_a` | AI productivity ceiling | Phase 1 |
| `g` | augmentation gain factor | Phase 1 |
| `c_aug` | augmentation cost per task per period | Phase 1 |
| `c_auto` | automation cost per task per period | Phase 1 |
| `c_fire`, `c_hire`, `c_train` | adjustment costs | Phase 1 |
| `w` | wage (exogenous in P1, endogenous from P2) | Phase 1 |
| `p` | output price | Phase 1 |
| `Y_f`, `R_f`, `C_f`, `π_f` | firm output, revenue, cost, profit | Phase 1 |
| `K_total` | total worker pool | Phase 2 |
| `s` | worker switching cost (fraction of wage) | Phase 2 |
| `w_res` | worker reservation wage | Phase 2 |
| `σ_w` | worker skill ∈ [0,1] | Phase 3 |
| `θ_f` | firm strategy dials, vector ∈ [0,1]^4 | Phase 3 |
| `F` | number of firms | Phase 3 |

**Numeraire choice:** set `q_h = 1` and `p = 1`. Everything else scales relative.

---

## 5. Tech stack

- **Language:** Python 3.11+
- **Numerical:** numpy, pandas
- **Static viz:** matplotlib, seaborn (paper-ready)
- **Interactive viz:** plotly (post-ready, GIF-able via kaleido)
- **Animation:** matplotlib `FuncAnimation` to GIF, or plotly frames
- **Notebook:** Jupyter for development, scripts for sweeps
- **Optional:** Streamlit only at phase 4 if you want a public dashboard

**Not using:** `mesa`, `agentpy`, or other ABM frameworks. At your scales (≤1000 workers, ≤50 firms, ≤500 periods), plain numpy runs a full simulation in seconds. Frameworks add abstraction tax and don't pay back. Keep classes small and explicit.

**Not using:** pytorch, jax, sklearn. No learning involved at first. Heuristic agents are realistic and tractable.

**Repo layout (target by end of phase 3):**
```
firm-ai-abm/
  firm_ai_abm/
    __init__.py
    tasks.py        # Task generation
    firm.py         # Firm class, production function
    worker.py       # Worker class, reservation wages
    strategy.py     # Strategy dials, mode allocation
    market.py       # Hiring, firing, matching
    wage.py         # Wage-setting heuristics
    welfare.py      # Welfare metrics (added P4)
    simulate.py     # Time loop
    sweep.py        # Sensitivity sweeps (added P3)
    viz.py          # Plotting
    config.py       # Default parameters
  notebooks/
    01_phase1_single_firm.ipynb
    02_phase2_two_firms.ipynb
    03_phase3_market.ipynb
    04_phase4_writeup.ipynb
  results/
    sweeps/         # CSVs from sensitivity runs
    figures/        # PNG/SVG outputs
  paper/
    main.md         # Draft paper
    refs.bib
  README.md
  pyproject.toml
```

Don't build all of this on day one. Add modules as phases need them.

---

## 6. Pre-committed candidate findings

Pick these now, before coding, so the model has a spine. The model doesn't have to confirm them — finding them false is also a finding worth publishing.

**Finding 1 — Coordination failure in firing.** In markets with high worker mobility, individually rational firing decisions can produce a collective outcome where total surplus is lower than under augmentation. Each firm fires expecting to rehire cheap, but firing depresses peer firms' wages too, creating a race to the bottom.

**Finding 2 — Augment-vs-automate tipping.** There's a productivity threshold for AI augmentation above which augment-heavy strategies dominate, and below which automate-heavy strategies dominate. Worker skill heterogeneity shifts the threshold: more heterogeneous skill pushes the line toward augmentation.

**Finding 3 — Welfare wedge.** The profit-maximizing firm strategy diverges from the median-worker-welfare-maximizing strategy as automation costs fall. The size of the gap is quantifiable and is a function of switching frictions.

These three are what the model is built to test. Every design decision should be traceable back to "does this affect our ability to test these?" If a feature doesn't connect to a finding, defer it to extensions.

---

## 7. What this model is NOT for

- **Forecasting** actual labor market outcomes — calibration is impossible at pet-project scale. Use it as a thinking tool.
- **Macroeconomics** beyond the modeled firms (no monetary policy, no aggregate demand).
- **Capital structure** or financial markets.
- **Political economy** — regulation, unionization, taxes — possible extensions but not core.
- **Replacing existing empirical work** — complement, don't compete.

State this honestly in the paper. It strengthens the contribution rather than weakens it.

---

## 8. How to work this with Claude

When picking up a phase in a future Claude conversation:

1. Open the relevant phase doc (`01_phase1_single_firm.md`, etc.).
2. Paste it into Claude along with this overview.
3. Ask Claude to either: (a) scaffold the code, (b) review your draft code against the design, or (c) help debug a specific equation.

Each phase doc contains:
- The conceptual model.
- The math (in markdown-friendly notation).
- An algorithm in pseudocode.
- Default parameter values.
- Validation checks (run these before believing any output).
- The visualization deliverable.
- Things to deliberately NOT build at that phase (scope discipline).
- Open design questions to resolve before coding.

If the design changes mid-phase, edit the phase doc directly so future-you has the actual model documented, not the original plan.
