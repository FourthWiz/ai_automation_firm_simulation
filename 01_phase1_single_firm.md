# Phase 1 — Single Firm, Task-Based Production

**Weekends:** 1–2.
**Deliverable:** profit-trajectory chart comparing strategies under one firm.
**Status of model after phase 1:** core production function works; can compare pure strategies; no labor market yet.

> Read `00_overview.md` first for project context, theoretical anchor, and notation.

---

## 1. Phase goal

Build a one-firm simulator where the firm chooses, for each task, among (human, augment, automate) production modes. Run the simulation under each pure strategy and compare profit trajectories. Visualize.

**This is the riskiest design phase.** If the production function doesn't meaningfully distinguish strategies, nothing downstream will work. Spend most of weekend 1 on the production function. Don't move to phase 2 until validation checks pass.

---

## 2. Conceptual model

A firm produces by completing `N` tasks (e.g., `N = 100`). Tasks are heterogeneous in two dimensions:

- **Automatability** `α_i ∈ [0, 1]`: how amenable to automation. `α = 0` means hard for AI alone, `α = 1` means trivially automatable.
- **Augmentability** `β_i ∈ [0, 1]`: how much an AI tool helps a human do it. `β = 0` means no benefit, `β = 1` means large benefit.

`α_i` and `β_i` are drawn once at firm creation and held fixed. Suggested distributions:
- `α_i ~ Uniform(0, 1)`
- `β_i ~ Uniform(0, 1)`
- Optionally with negative correlation `ρ ≈ -0.3` (tasks easy to automate are also less amplified by augmentation, and vice versa). Add this in a sensitivity check.

For each task `i` per period, the firm picks `m_i ∈ {H, A, T}`:

| Mode | Productivity per task | Cost per task per period |
|---|---|---|
| H (human) | `q_h` | `0` (wages are per worker, not per task — see §4) |
| A (augmented human) | `q_h · (1 + g · β_i)` | `c_aug` |
| T (auTomated) | `q_a · α_i` | `c_auto` |

**Note on wages.** Wages are charged per worker, not per task. Worker count `K` is computed from §4's integerization rule (`K = ceil(#H_or_A / tasks_per_worker)`). The simulate loop adds `w · K` to total cost per period, separately from the per-task cost column above. This makes the §10 baseline formula (`p · q_h · N − w · K − F`) follow directly from this table plus the per-period `w · K` charge plus the fixed cost `F`.

**Output:**
```
Y = Σ_i productivity(m_i)
```

**Revenue:** `R = p · Y`.

**Cost:**
```
C = Σ_i cost(m_i) + adjustment_costs(t) + F
```
where `F` is fixed cost (rent, overhead).

**Profit per period:** `π = R - C`.
**Cumulative profit:** `Π_T = Σ_t π_t`.

---

## 3. Adjustment costs

Switching the mode of a task incurs a cost paid in the period of the change. Phase 1 uses two distinct accounting rules — per-task for training, per-worker (lumpy) for hiring and firing:

- `H → A`: training cost `c_train` per task converted (per-task — training is task-specific).
- `H → T`, `A → T`, `T → H`, `T → A`: NO per-task cost. Hiring and firing are paid once per period at the worker level, after the new workforce `K_new` is computed:
  - firing cost: `c_fire · max(0, K_prev − K_new)`
  - hiring cost: `c_hire · max(0, K_new − K_prev)`

Per-task `c_fire/N` and `c_hire/N` charges are **not** used. Workers are integer (§4); a few task switches that don't cross a worker boundary cost zero in hire/fire — this is intentional and shows up in greedy behavior.

These matter because pure strategies will look much better than they should without them. Without firing costs, switching to "automate-all" is free; with them, the timing of the switch matters.

---

## 4. Workers (minimal at phase 1)

Phase 1 simplification: a firm has `K` identical workers. `K` is determined by the number of tasks in mode H or A. Wages `w` are exogenous and fixed. No worker heterogeneity, no labor market — just a count that drives hiring/firing costs.

Concretely:
- Each worker covers `N/K` tasks worth of capacity, where `K = ceil(#H_or_A_tasks / tasks_per_worker)`.
- Default `tasks_per_worker = 10` so `K = #H_or_A / 10` rounded up.
- A new worker costs `c_hire`; firing a worker costs `c_fire`.

This integerization matters: you can't fire half a worker. Switching a few tasks may not change `K` at all.

---

## 5. Strategies

A strategy is a rule for choosing `m_i`. Phase 1 studies five:

1. **All-Human (baseline):** `m_i = H` for all `i`.
2. **Augment-all:** `m_i = A` for all `i`.
3. **Automate-all:** `m_i = T` for all `i`.
4. **Greedy-profit:** for each task, pick the mode that maximizes per-task profit at current prices. Recomputed each period.
5. **Greedy with switching cost:** as 4, but only switches mode if the gain exceeds the adjustment cost amortized over `n_amortize = 6` periods.

Strategies 4–5 are the interesting ones — they show what an "optimal" myopic firm does.

---

## 6. Time evolution

Per period (period = month, conceptually):

1. Firm observes external state: `w`, `c_aug`, `c_auto`, `p` (constants in phase 1, will vary later).
2. Firm decides `{m_i}` for this period using its strategy.
3. Adjustment costs paid for any mode changes since last period.
4. Workforce `K_t` recomputed from new `{m_i}`. Hiring/firing costs paid.
5. Production: compute `Y`, `R`, `C`, `π`.
6. Cumulative profit updated.
7. Next period.

Run for `T = 60` periods (5 years).

---

## 7. Default parameters

| Symbol | Meaning | Start value | Sensitivity range |
|---|---|---|---|
| `N` | number of tasks | 100 | fixed |
| `T` | periods | 60 | fixed |
| `q_h` | human productivity per task | 1.0 | numeraire |
| `q_a` | AI productivity ceiling | 1.2 | 0.5–2.0 (key dial) |
| `g` | augmentation gain | 0.5 | 0–1.5 (key dial) |
| `w` | wage | 1.0 | 0.5–2.0 |
| `c_aug` | per-task augmentation cost | 0.05 | 0–0.3 |
| `c_auto` | per-task automation cost | 0.4 | 0.1–1.5 (key dial) |
| `c_fire` | firing cost per worker | 2.0 | 0.5–5.0 |
| `c_hire` | hiring cost per worker | 0.5 | 0–2.0 |
| `c_train` | training cost per task | 0.1 | 0–0.5 |
| `F` | fixed cost per period | 5.0 | fixed |
| `p` | output price | 1.0 | numeraire |
| `tasks_per_worker` | task capacity | 10 | fixed |

Two numeraires: `w = 1` (price of labor) or `p = 1` (price of output). Pick one; here we set both because output and labor are both 1.0 — that's fine, scaling all monetary parameters by a constant should leave profit unchanged. Use this as a validation check.

---

## 8. Algorithm (pseudocode)

```python
firm = Firm(N=100, alpha=sample_alpha(), beta=sample_beta(), params=defaults)

for strategy in [all_H, all_A, all_T, greedy, greedy_with_switching]:
    firm.reset()
    history = []
    for t in range(T):
        new_modes = strategy.decide(firm, t)
        adj_cost = compute_adjustment_cost(firm.modes, new_modes, params)
        firm.modes = new_modes
        firm.K = compute_workforce(new_modes, params)
        Y = sum(productivity(firm.modes[i], firm.alpha[i], firm.beta[i], params) for i in range(N))
        C = sum(cost(firm.modes[i], params) for i in range(N)) + adj_cost + F
        pi = p * Y - C
        history.append({"t": t, "Y": Y, "C": C, "pi": pi, "K": firm.K, "modes": new_modes.copy()})
    save(strategy.name, history)

plot_profit_trajectories(all histories)
```

---

## 9. Code structure

Phase 1 is small. One notebook with classes inline is fine. As a starter:

```python
# firm_ai_abm/firm.py
from dataclasses import dataclass, field
import numpy as np

@dataclass
class FirmParams:
    N: int = 100
    q_h: float = 1.0
    q_a: float = 1.2
    g: float = 0.5
    w: float = 1.0
    c_aug: float = 0.05
    c_auto: float = 0.4
    c_fire: float = 2.0
    c_hire: float = 0.5
    c_train: float = 0.1
    F: float = 5.0
    p: float = 1.0
    tasks_per_worker: int = 10

@dataclass
class Firm:
    params: FirmParams
    alpha: np.ndarray  # shape (N,)
    beta: np.ndarray   # shape (N,)
    modes: np.ndarray = None  # shape (N,), values in {0,1,2} for H,A,T
    K: int = 0
    history: list = field(default_factory=list)

    def reset(self):
        self.modes = np.zeros(self.params.N, dtype=int)  # all H
        self.K = self.params.N // self.params.tasks_per_worker
        self.history = []
```

Productivity and cost as vectorized numpy:

```python
def productivity_vec(modes, alpha, beta, params):
    p = np.where(modes == 0, params.q_h, 0)              # H
    p = np.where(modes == 1, params.q_h * (1 + params.g * beta), p)  # A
    p = np.where(modes == 2, params.q_a * alpha, p)      # T
    return p

def cost_vec(modes, params):
    c = np.where(modes == 0, params.w / params.N, 0)
    c = np.where(modes == 1, params.w / params.N + params.c_aug, c)
    c = np.where(modes == 2, params.c_auto, c)
    return c
```

Vectorize so the per-period cost is microseconds. Premature optimization is fine here because phase 3 needs many runs.

---

## 10. Validation checks

Before believing any output, run these:

1. **Constant profit baseline.** All-H with `q_a = 0`, `g = 0`, `c_aug = 0` should give a perfectly constant per-period profit equal to `p · q_h · N − w · K − F`. Verify analytically and numerically.
2. **Greedy dominance.** With switching cost `c_train = c_fire = 0`, Greedy-profit should dominate or equal every other strategy at every period. If not, your greedy logic is wrong.
3. **Monotonicity in `q_a`.** Increasing `q_a` should monotonically increase Automate-all profit and weakly affect Augment-all (it doesn't appear in A's productivity). If Automate-all profit goes down with higher `q_a`, the cost function is wrong.
4. **Monotonicity in `w`.** Increasing `w` should monotonically decrease All-H profit and not affect Automate-all.
5. **Numeraire invariance.** Multiply `w`, `c_aug`, `c_auto`, `c_fire`, `c_hire`, `c_train`, `F`, and `p` all by 2.0. Profit trajectories should also scale by 2.0 exactly. If they don't, there's a bug.
6. **Adjustment cost shows up.** Run Greedy with `c_train = 100`. The strategy should refuse to switch modes. If it still switches, the adjustment-cost integration is wrong.

If any check fails, fix before phase 2. Phase 2 inherits all of phase 1's machinery, so bugs compound.

---

## 11. Output: postable visualization

**Primary figure:** line chart with five lines (one per strategy), x-axis = time, y-axis = cumulative profit. Different colors and markers per strategy. Annotated with key parameter values in a small text box.

**Secondary figure:** small-multiples (3×1) of the same chart for three values of one key dial — recommended `q_a ∈ {0.8, 1.2, 1.6}`. This shows how strategy ranking depends on AI productivity. **This is the more interesting post**, because it makes the "it depends" point visually.

**Tertiary figure (optional):** stacked bar showing mode mix over time for the Greedy strategy. Demonstrates how the firm shifts toward automation as it explores.

Style notes:
- Matplotlib with a clean theme. Avoid gridlines clutter.
- Strategy names in legend, not labels on lines (lines too crowded).
- Save as PNG (web post) and SVG (paper).
- Resolution at least 300 dpi.

---

## 12. Things to deliberately NOT build at phase 1

Resist the urge:

- ❌ Multiple firms (phase 2)
- ❌ Heterogeneous workers (phase 3)
- ❌ Endogenous wages (phase 2)
- ❌ Reservation wages, labor-market matching (phase 2/3)
- ❌ Welfare metrics (phase 4)
- ❌ Streamlit / dashboards (phase 4)
- ❌ Fancy ML strategies — heuristic strategies are the point
- ❌ Bankruptcy / firm exit (phase 3)
- ❌ Changing parameters over time within a run (phase 4)

Keeping phase 1 minimal is what makes the weekend pace work. The discipline pays off when phase 2 builds on a solid foundation.

---

## 13. Open design questions to resolve before coding

1. **Output aggregation:** additive across tasks (as written) or CES-like with substitution? Additive is simpler and what's specified above; CES is more realistic but harder to interpret. **Decision: start additive. Revisit if results are uninteresting.**
2. **Worker count `K`:** should it equal #(H or A tasks)/tasks_per_worker, or be decoupled (workers do many tasks)? **Decision: `K = ceil(#H_or_A / tasks_per_worker)`. Decouple later if needed.**
3. **Adjustment cost:** paid once or amortized? **Decision: paid once in the period of the change.** Greedy strategies that look ahead can amortize internally.
4. **Bankruptcy:** firm with persistent negative profit? **Decision: no bankruptcy at phase 1.** Cumulative profit can go arbitrarily negative.
5. **Stochasticity:** noise in productivity per period? **Decision: no, deterministic at phase 1.** Add noise at phase 3 if it adds something.

If you change any of these decisions during implementation, edit this doc.

---

## 14. Definition of "phase 1 done"

Move to phase 2 when:

- All six validation checks pass.
- The primary line chart is saved and looks presentable.
- The small-multiples chart by `q_a` is saved.
- The code is in a state you'd be willing to share publicly (rough is fine, broken is not).
- You've made one observation about strategy ranking that surprised you. (Write it down — it might be a paper finding.)

If you're at the end of weekend 2 and aren't done: cut scope (skip Greedy-with-switching, skip the secondary chart) but make sure validation passes.
