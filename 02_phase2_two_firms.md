# Phase 2 — Two Firms Competing for Workers

**Weekends:** 3–4.
**Deliverable:** animated worker-flow diagram + dual-firm profit comparison.
**Status of model after phase 2:** workers move endogenously, wages adjust, layoff cascades possible.

> Read `00_overview.md` and `01_phase1_single_firm.md` first.

---

## 1. Phase goal

Add a second firm and a shared worker pool. Workers can switch between firms in response to wage and stability differences. Watch worker flows as the two firms execute different strategies. The animated worker-flow diagram is the deliverable — it's the most visually striking output of the project.

---

## 2. What's new vs phase 1

- **Two firms** `F1`, `F2`. Each has its own (potentially different) strategy.
- **A worker pool** of size `K_total`. Workers still homogeneous in skill (heterogeneity is phase 3) but they have firm-specific tenure and a reservation wage.
- **Endogenous wages.** Each firm sets a wage to balance attraction/retention with cost.
- **Active labor market.** Workers can be hired, fired, quit, or remain unemployed.
- **Layoff cascades** become observable (a layoff at F1 lowers wages for everyone, since unemployed workers will take less).

Phase 1's production function is unchanged. The new mechanics happen between firms.

---

## 3. Worker model

A worker has state:

| Field | Type | Description |
|---|---|---|
| `id` | int | unique identifier |
| `employer` | enum | F1, F2, or `unemployed` |
| `tenure` | int | periods at current employer (0 if unemployed) |
| `w_res` | float | reservation wage |
| `last_wage` | float | last wage actually earned (0 if never employed) |

**Reservation wage rule (phase 2 simple version):**

- **Employed worker:** `w_res = current_wage · (1 - quit_buffer)`. They quit immediately if poached at `(1 + ε) · current_wage` or higher.
- **Unemployed worker:** reservation wage decays each period of unemployment toward a floor `w_floor`:
  ```
  w_res ← w_res · (1 - decay_rate) + w_floor · decay_rate
  ```
  with `decay_rate = 0.1`, `w_floor = 0.3`.

**Switching cost:** when a worker changes firm (poach), they lose fraction `s` of their first period's wage as a search cost. This dampens unproductive churn.

---

## 4. Firm wage setting

Each firm sets a single wage `w_F` per period. No within-firm wage dispersion at phase 2.

**Wage rule (heuristic, not optimal):**

```python
def wage_rule(firm, prev_wage, recent_turnover, target_turnover, Δ_w, w_min, w_max):
    if recent_turnover > target_turnover:
        return min(prev_wage * (1 + Δ_w), w_max)
    elif recent_turnover < target_turnover * 0.5 and firm.profit_pressure():
        return max(prev_wage * (1 - Δ_w), w_min)
    else:
        return prev_wage
```

`recent_turnover` = (#workers who left in last 3 periods) / (mean workforce in last 3 periods).

`profit_pressure()` returns True if cumulative profit over last 6 periods is below the firm's threshold. Threshold is a parameter; default = 0.

This is intentionally heuristic. Optimal wage-setting under the full dynamic problem is intractable, and heuristic firms are also closer to how real firms behave.

---

## 5. Hiring, firing, quitting

**Hiring.** Each period, firm computes desired workforce `K_F* = ceil(#H_or_A_tasks_F / tasks_per_worker)`. If `K_F* > K_F`, the firm posts vacancies for the difference. It hires from unemployed pool first; if pool insufficient, it can poach by raising wage next period (current period vacancies stay unfilled).

**Firing.** If `K_F* < K_F`, the firm fires `K_F − K_F*` workers. Pays `c_fire` per fired worker. Default firing rule: lowest tenure first (last in, first out).

**Quitting.** A worker quits if any *other* firm offers `≥ (1 + ε) · w_F_current`. They switch immediately, paying the switching cost `s · w_new`. This is checked each period after wage-setting.

**Order within a period:**
1. Firms decide modes (`m_i`).
2. Firms decide wages.
3. Firings (workforce shrinks).
4. Quits (workers leave for higher-wage firm).
5. Hires (firms fill vacancies from unemployed pool).
6. Production.
7. Tenure & reservation wage updates.

This ordering means a worker fired in step 3 can be hired by a competitor in step 5 — that's intentional, and you'll see it in the visualization.

---

## 6. Mass conservation invariant

At all times: `K_total = #(employed at F1) + #(employed at F2) + #unemployed`.

**Verify this every period in code (assert).** It's the easiest bug to introduce.

---

## 7. New parameters

| Symbol | Meaning | Start value | Sensitivity range |
|---|---|---|---|
| `K_total` | total workers | 200 | 100–500 |
| `s` | switching cost (fraction of wage) | 0.1 | 0–0.5 (key dial) |
| `ε` | poach threshold | 0.1 | 0.02–0.3 |
| `quit_buffer` | wage drop tolerated before quitting | 0.1 | 0.05–0.2 |
| `target_turnover` | firm wage-setting target | 0.05 | 0–0.15 |
| `Δ_w` | wage adjustment step | 0.05 | 0.01–0.2 |
| `w_min` | wage floor | 0.3 | fixed |
| `w_max` | wage ceiling | 3.0 | fixed |
| `decay_rate` | reservation wage decay when unemployed | 0.1 | fixed |
| `w_floor` | reservation wage floor | 0.3 | fixed |

---

## 8. Algorithm (pseudocode)

```python
state = init_state(firms=[F1, F2], K_total=200)
for t in range(T):
    # 1. Modes
    for f in state.firms:
        f.modes = f.strategy.decide(f, t)
    # 2. Wages
    for f in state.firms:
        f.wage = wage_rule(f, prev_wage=f.wage, recent_turnover=f.turnover_3p(), ...)
    # 3. Firings
    for f in state.firms:
        f.K_target = compute_workforce_target(f.modes, params)
        if f.K_target < len(f.workers):
            fired = fire_lowest_tenure(f, len(f.workers) - f.K_target)
            move_to_unemployed(fired)
            f.cost += c_fire * len(fired)
    # 4. Quits (cross-firm poaching)
    poaches = []
    for w in employed_workers(state):
        offers = [(g.wage, g) for g in state.firms if g != w.employer and g.has_vacancy()]
        if offers and max(offers)[0] >= (1 + ε) * w.last_wage:
            poaches.append((w, max(offers, key=lambda x: x[0])[1]))
    for (w, target_firm) in poaches:
        move_worker(w, source=w.employer, target=target_firm)
        w.last_wage *= (1 - s)  # switching cost
    # 5. Hires from unemployed
    for f in state.firms:
        vacancies = max(0, f.K_target - len(f.workers))
        candidates = [w for w in state.unemployed if w.w_res <= f.wage]
        hired = pick(candidates, vacancies)  # FCFS or random
        for w in hired:
            move_worker(w, source=None, target=f)
            f.cost += c_hire
    # 6. Production
    for f in state.firms:
        compute_production(f, params)  # phase 1 logic
    # 7. Tenure + reservation wage updates
    for w in all_workers(state):
        if w.employed:
            w.tenure += 1
            w.last_wage = w.employer.wage
            w.w_res = w.last_wage * (1 - quit_buffer)
        else:
            w.tenure = 0
            w.w_res = w.w_res * (1 - decay_rate) + w_floor * decay_rate
    # 8. Record
    state.record(t)
    assert mass_conservation(state)

animate(state.history)
```

---

## 9. Code structure

Add to module:

```
firm_ai_abm/
  ...phase 1...
  worker.py        # Worker dataclass, reservation wage logic
  market.py        # Hiring, firing, quitting, mass conservation check
  wage.py          # Firm wage rule
  state.py         # State container, history recording
notebooks/
  02_phase2_two_firms.ipynb
```

Keep firms and workers as dataclasses, not full classes. Logic lives in module-level functions that take dataclasses and parameters. This pattern stays simple and ports cleanly to phase 3 (adding a third firm changes nothing structurally).

---

## 10. Validation checks

Run before believing any output:

1. **Mass conservation every period** (already an assert).
2. **Identical firms, no flows.** If both firms run the same strategy with the same wage and `quit_buffer = 0`, no worker should move between them. Verify.
3. **Wage attracts.** If F1 wage = 1.0 and F2 wage = 1.5 (held fixed), workers should flow F1 → F2 over time and stop when F1 is empty or F2 is at capacity.
4. **Firing cascade.** If F1 lays off 50% of workers, F2 should be able to hire from the resulting unemployment pool — but at a wage potentially lower than before, because reservation wages decay.
5. **Switching cost dampens churn.** Increasing `s` from 0.1 to 0.5 should reduce total worker movements per period. Quantify.
6. **Reduces to phase 1.** Setting `K_total = ∞` and disabling poaching should make F1 and F2 behave exactly like two independent phase-1 simulations. Verify on profits.

---

## 11. Output: postable visualization

**Primary deliverable: animated worker-flow diagram.**

Layout:
- Left circle: F1 (sized proportional to current workforce). Shows current wage, current strategy.
- Right circle: F2 (same).
- Bottom area: unemployed pool (sized proportional to count).
- Workers as small dots flowing between regions.
- Each frame = 1 period.
- Color-code workers by tenure (new = bright, long-tenured = darker), or by previous firm.

Below the animation: dual-line chart showing F1 (blue) vs F2 (orange) cumulative profit over the same time axis.

Aim: 60 frames, 30-second loop, exported as GIF. Tools:
- **matplotlib FuncAnimation → GIF** via Pillow writer. Slow to render, simple to code.
- **plotly with frames** → cleaner output, harder to fine-tune.
- **manim** for community.manim. Beautiful, time-consuming. Skip unless you already know it.

**Recommendation:** matplotlib + Pillow first. Polish if time allows.

**Secondary figure (still):** "wage divergence" plot. X = time, Y = wage at each firm. Lines diverge then reconverge as poaching equilibrates. Often shows damped oscillation, which is interesting.

**Tertiary (post-friendly):** flow Sankey diagram aggregating worker movements over the full simulation. F1 → F2, F1 → unemployed, etc. Use plotly's Sankey trace.

---

## 12. Suggested experiments to run at phase 2

Set up these contrasts and post a thread about each:

1. **Augment vs. fire-and-rehire.** F1 = Augment-all, F2 = "fire then automate" (e.g., Greedy with low `c_fire` perception). Compare profits and worker outcomes.
2. **Same strategy, different wages.** Both F1 and F2 = Augment-all. F1 starts at low wage, F2 at high. Watch the wage convergence.
3. **Asymmetric layoff shock.** Both run Greedy. Halfway through, exogenously jump `c_auto` down by 50% (simulating a sudden AI improvement). Watch which firm reacts faster and what happens to workers.

Each is a good post and a paragraph in the eventual paper.

---

## 13. Things to deliberately NOT build yet

- ❌ More than 2 firms (phase 3)
- ❌ Skill heterogeneity (phase 3)
- ❌ Continuous strategy dials (phase 3)
- ❌ Reservation wages updating from peers, not just own employer (phase 3)
- ❌ Bankruptcy / firm exit (phase 3)
- ❌ Endogenous output price (phase 3)
- ❌ Welfare metrics (phase 4)

---

## 14. Open design questions to resolve before coding

1. **Tie-breaking when poaching.** Multiple firms might offer above reservation. **Decision: pick the highest wage; if tied, pick at random.**
2. **Vacancy delay.** Should there be a lag between posting a vacancy and filling it? **Decision: no delay at phase 2.** Add at phase 3 if matching frictions are interesting.
3. **Order of firings vs. quits within a period.** **Decision: firings first, then quits, then hires.** Means a fired worker becomes part of the unemployed pool that competing firms can hire from in the same period — which is the dynamic we want.
4. **Worker preference for stability.** Should workers prefer their current firm even at slightly lower wages (loyalty / risk aversion)? **Decision: not at phase 2 — it's captured implicitly by `quit_buffer`.** Could be richer at phase 3.
5. **Bankrupt firms.** **Decision: no bankruptcy at phase 2** — assume infinite credit.

---

## 15. Definition of "phase 2 done"

Move to phase 3 when:

- All validation checks pass.
- Animation renders correctly and looks like something you'd post.
- At least one of the suggested experiments produces an interesting result (worker-flow oscillation, wage convergence, layoff cascade).
- You've spotted at least one second-order effect you didn't expect — write it down for the paper.

If at end of weekend 4 and not done: skip experiment 3 (asymmetric shock) and the Sankey diagram. The animation + dual profit chart is enough to post.
