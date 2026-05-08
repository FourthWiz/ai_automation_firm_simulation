# Phase 3 — N-Firm Market with Continuous Strategy Dials

**Weekends:** 5–7.
**Deliverable:** "who wins" heatmap over a 2D parameter slice — the central paper figure.
**Status of model after phase 3:** the full ABM is operational; sweeps generate publishable results.

> Read `00_overview.md`, `01_phase1_single_firm.md`, and `02_phase2_two_firms.md` first.

---

## 1. Phase goal

Scale to `F = 20` firms. Replace pure strategies with continuous strategy dials. Add worker skill heterogeneity. Add reservation-wage updating that responds to peers, not just own employer. Build sensitivity-sweep infrastructure (run hundreds of simulations under different parameter vectors). Generate the central paper figure: a heatmap showing which strategy region wins under different parameter regimes.

This is the most ambitious phase. Three weekends, not two. Don't compress it — phase 4 depends on having clean sweep output.

---

## 2. What's new vs phase 2

- **`F = 20` firms** instead of 2. Strategy varies across firms.
- **Continuous strategy dials** `θ_f ∈ [0,1]^4` per firm — replacing discrete strategies.
- **Worker skill heterogeneity** `σ_w ∈ [0,1]`.
- **Reservation wages update from peer signals,** not just own employer history.
- **Endogenous output price** with downward-sloping demand (so automation can't trivially win on volume).
- **Bankruptcy and firm entry** to keep `F` bounded.
- **Sensitivity-sweep infrastructure** — running 1000+ simulations and aggregating results.

---

## 3. Strategy dials

Replace discrete strategies with a vector `θ_f ∈ [0,1]^4`:

| Index | Dial | Meaning |
|---|---|---|
| `θ_f[0]` | automation propensity | bias toward putting tasks in mode T |
| `θ_f[1]` | augmentation propensity | bias toward A |
| `θ_f[2]` | firing aggressiveness | willingness to fire when modes shift; also firing-first vs attrition |
| `θ_f[3]` | wage premium | markup over local market wage to attract talent |

**Mode allocation rule:** for each task `i`, compute three scores and pick the highest.

```
score_i(H) = q_h(σ̄_F) − w_F / N
score_i(A) = q_h(σ̄_F)·(1 + g·β_i) − w_F / N − c_aug + θ_f[1]·b
score_i(T) = q_a·α_i − c_auto + θ_f[0]·b
```

`σ̄_F` = mean skill of workers at firm F. `b` is a "bias bonus" parameter, e.g., `b = 0.3`, that lets dials meaningfully tilt the choice without overpowering economics. Tune so that with `θ_f = (0.5, 0.5, ·, ·)`, scores roughly balance; with `θ_f = (1, 0, ·, ·)`, automation strongly preferred.

**Wage rule with `θ_f[3]`:**
```
w_F = market_wage · (1 + θ_f[3]·premium_scale) + adjustment_from_phase2_rule
```

`market_wage` is a rolling average of wages at active firms over last 3 periods. `premium_scale = 0.3` means a firm with `θ_f[3] = 1` pays 30% above market.

**Firing aggressiveness `θ_f[2]`:**
- High `θ_f[2]`: firm fires whenever workforce target drops, fires `K_F − K_F*` immediately.
- Low `θ_f[2]`: firm uses attrition — doesn't fire actively; lets natural quits shrink workforce. Pays no firing cost but may run with excess workers (paying them anyway).

Intermediate values are a mix.

---

## 4. Mapping archetypes to θ regions

The four phase-1 archetypes correspond to:

| Archetype | θ ≈ |
|---|---|
| Low-cost retain + augment | `(0.1, 0.8, 0.2, 0.0)` |
| High-cost fire + cheap rehire + augment | `(0.4, 0.4, 0.9, -0.2)` (use clamped values, or shift dial range to [-0.3, 1]) |
| High-cost retain + augment | `(0.1, 0.8, 0.1, 0.3)` |
| Full automation | `(0.95, 0.0, 0.7, -0.5)` |

These are *centroids*. Sample firms from a Dirichlet around them, or uniformly from `[0,1]^4` and let the heatmap show which region wins.

**Decision:** for sweep runs, sample firms uniformly from `[0,1]^4` (with `θ_f[3] ∈ [-0.5, 0.5]`) so the heatmap is unbiased.

---

## 5. Worker skill heterogeneity

Each worker has skill `σ_w ∈ [0,1]`. Skill modifies productivity:

- In mode H: `q_h(σ_w) = q_h_base · (0.5 + σ_w)`, so skill range is 0.5–1.5× base.
- In mode A: `q_h(σ_w)·(1 + g·β_i)` — augmentation amplifies skill linearly.
- In mode T: irrelevant (no human worker on that task).

**Distribution:** `σ_w ~ Beta(2, 2)` initially. This is unimodal at 0.5 with mass tapering to 0 and 1 — a reasonable proxy for a labor pool.

**Hiring with skill matching:** when a firm posts a vacancy, it prefers higher-skill workers (sort candidates by skill, take top `vacancies`). Workers prefer firms paying higher wage (sort offers by wage). At each round, do a top-down matching.

---

## 6. Reservation wage with peer signals

Each worker has a peer network of `d = 5` other workers (random graph, fixed across simulation). Each period, worker `w` observes:

- `peer_median_wage` = median of last-period wages of their `d` peers (zero if peer unemployed).
- `peer_layoff_rate` = fraction of peers fired in last 3 periods.

**New reservation wage rule:**
```
w_res ← (1 - λ_peer) · w_res_personal + λ_peer · peer_median_wage · (1 - 0.5·peer_layoff_rate)
```

`w_res_personal` is the phase-2 rule. `λ_peer = 0.3` weighting on peer signals. `peer_layoff_rate` dampens reservation wage when peers are getting fired (signaling caution).

**This is the mechanism that produces cascades and spillovers.** Without it, layoffs at one firm don't directly affect wages at others — they only affect through the unemployed pool. With it, worker expectations propagate through the network.

---

## 7. Endogenous output price

Without endogenous price, automation always wins on volume — high `Y_F` directly increases revenue. Real markets have downward-sloping demand. Introduce:

```
p_t = p_0 · (Y_t_total / Y_0_baseline)^(-1/η)
```

`Y_t_total = Σ_F Y_F`. `Y_0_baseline` is total output at t = 0 (or a fixed reference). `η = 2` (demand elasticity).

Effect: when many firms automate and total output rises, price falls. Revenue per unit drops. This bounds the value of automation.

---

## 8. Bankruptcy and firm entry

A firm exits if its cumulative profit over last `k_bankrupt = 6` periods is below `Π_bankrupt = -50`. Its workers all become unemployed.

To maintain `F = 20`, new firms enter at rate `λ_entry = 1` firm per period when below target. New firms are initialized with:
- Fresh `θ_f` sampled uniformly (or from successful firms — see open question).
- Initial wage = current market median wage.
- Initial workforce = 0; they hire from unemployed pool over subsequent periods.

---

## 9. New parameters

| Symbol | Meaning | Start value |
|---|---|---|
| `F` | number of firms | 20 |
| `K_total` | workers | 1000 |
| `θ_f` | strategy dials | sampled per firm |
| `σ_w` | worker skill | Beta(2,2) |
| `b` | dial bias bonus in mode scoring | 0.3 |
| `premium_scale` | wage premium scaling | 0.3 |
| `Π_bankrupt` | bankruptcy threshold | -50 |
| `k_bankrupt` | consecutive bad periods | 6 |
| `λ_entry` | firm entry rate | 1/period when below `F` |
| `d` | peer network degree | 5 |
| `λ_peer` | peer signal weight | 0.3 |
| `η` | demand elasticity | 2 |
| `p_0`, `Y_0_baseline` | demand calibration | as needed |

---

## 10. Algorithm (pseudocode)

```python
state = init_state(F=20, K_total=1000, theta_distribution="uniform")
build_peer_network(state.workers, degree=5)

for t in range(T):
    # 1. Modes
    for f in state.firms:
        f.modes = mode_allocation(f.theta, f.workers, params)
    # 2. Wages (with theta dial)
    market_wage = median_recent_wages(state, lookback=3)
    for f in state.firms:
        f.wage = compute_wage(f.theta, market_wage, f.recent_turnover, params)
    # 3. Firings
    for f in state.firms:
        target = workforce_target(f.modes, params)
        if target < len(f.workers):
            n_fire = (len(f.workers) - target) * f.theta[2]  # aggressiveness scales firing
            fire_lowest_tenure(f, ceil(n_fire))
    # 4. Quits / poaching
    process_poaches(state.firms, state.workers, params)
    # 5. Hires (skill-sorted)
    for f in state.firms:
        vacancies = max(0, target - len(f.workers))
        candidates = sorted(
            [w for w in state.unemployed if w.w_res <= f.wage],
            key=lambda w: -w.skill
        )
        for w in candidates[:vacancies]:
            move_worker(w, target=f)
    # 6. Production with skill
    Y_total = 0
    for f in state.firms:
        f.Y = sum(productivity_with_skill(f.modes[i], f.alpha[i], f.beta[i], f.workers, params) for i in range(N))
        Y_total += f.Y
    # 7. Endogenous price
    p_t = p_0 * (Y_total / Y_0_baseline) ** (-1/eta)
    # 8. Profit
    for f in state.firms:
        f.profit = p_t * f.Y - f.cost
    # 9. Bankruptcy + entry
    for f in state.firms[:]:
        if f.cum_profit_recent(k_bankrupt) < Pi_bankrupt:
            close_firm(f, state)
    while len(state.firms) < F:
        spawn_new_firm(state, params)
    # 10. Update peer signals + reservation wages
    update_peer_signals(state.workers)
    update_reservation_wages(state.workers, params)
    # 11. Record
    state.record(t)
    assert mass_conservation(state)
```

---

## 11. Sensitivity-sweep infrastructure

This is essential for phase 3 and phase 4. Build it as a separate module.

```python
# firm_ai_abm/sweep.py
def run_sweep(grid, n_replicates=20, T=60, base_params=defaults):
    """grid: dict of param_name -> list of values."""
    results = []
    for combo in itertools.product(*grid.values()):
        params = base_params.copy()
        for name, val in zip(grid.keys(), combo):
            setattr(params, name, val)
        for r in range(n_replicates):
            seed = hash((tuple(combo), r))
            state = run_simulation(params, T=T, seed=seed)
            summary = summarize(state)
            summary.update(dict(zip(grid.keys(), combo)))
            summary["replicate"] = r
            results.append(summary)
    return pd.DataFrame(results)

def summarize(state):
    # Aggregate over second half of run
    half = state.T // 2
    return {
        "median_profit_per_firm": ...,
        "median_wage": ...,
        "gini_wage": ...,
        "employment_rate": ...,
        "best_theta_region": classify_winner(state, half),
        "total_surplus": ...,
        "n_bankruptcies": ...,
    }
```

**Default sweep for the central figure:**
- `c_auto ∈ {0.1, 0.2, ..., 1.0}` — 10 values
- `s ∈ {0.0, 0.05, ..., 0.45}` — 10 values
- 20 replicates each
- Total: 100 cells × 20 reps = 2000 runs
- Each run takes ~0.5–2 seconds for `F = 20`, `K_total = 1000`, `T = 60`. Total: ~30–120 minutes. Feasible on a laptop overnight.

Save results as a parquet/CSV. Each row = one run.

---

## 12. Validation checks

1. **Reduces to phase 2.** Setting `F = 2`, no peer network (`λ_peer = 0`), no bankruptcy, identical skill (`σ_w = 0.5` for all) should reproduce phase-2 results.
2. **Mass conservation** every period across all firms + unemployed.
3. **All-automate market.** A market where every firm has `θ_f[0] = 1` should drive total wages toward zero (or very low).
4. **Theta affects allocation.** Increasing `θ_f[0]` of one firm while holding all else fixed should shift its mode mix toward T.
5. **Demand response.** Doubling `Y_0_baseline` should change `p` but leave qualitative ranking of strategies unchanged.
6. **Determinism with seed.** Same params + same seed → identical results bit-for-bit.

---

## 13. Output: the central paper figure

**The heatmap.**

X-axis: `c_auto` (automation cost).
Y-axis: `s` (worker switching friction).
Color: which strategy region wins (categorical, 4 colors for the four archetypes plus a "no clear winner" category).

How to compute "winning strategy" for each cell:
1. For each run, classify each firm's `θ_f` into one of 4 archetypes by nearest centroid.
2. For each archetype, compute median cumulative profit across firms in that archetype, across all replicates in the cell.
3. The winning archetype = max median profit. If second-place is within 10%, label "no clear winner."

Alternative (and arguably more interesting): plot `profit_augment_heavy − profit_automate_heavy` as a continuous color. Diverging colormap, centered at zero. Shows the *intensity* of who wins, not just the identity.

**Recommended:** produce both. Categorical heatmap for paper, diverging continuous for posts.

**Subplots:** the same heatmap stratified by:
- Skill heterogeneity (low Beta(5,5) vs high Beta(0.5,0.5) vs default Beta(2,2)).
- Peer network strength (`λ_peer ∈ {0, 0.3, 0.6}`).

This shows how the boundary moves as those parameters change, which is exactly the structural insight the paper claims.

---

## 14. Things to deliberately NOT build at phase 3

- ❌ Welfare metrics (phase 4)
- ❌ Multiple welfare definitions side by side (phase 4)
- ❌ Streamlit / public dashboard (phase 4)
- ❌ Strategy *learning* (firms updating their `θ` over time). Could be an extension but adds complexity that obscures the basic findings. Stick to fixed `θ` per firm at phase 3.
- ❌ Multi-good markets / firm differentiation. One good, one demand curve.
- ❌ Skill investment / training by workers. Skill is fixed at birth.

---

## 15. Open design questions

1. **Entry of new firms.** Sampled `θ` uniformly or biased toward currently-successful firms (mimicry)? **Decision: uniform at phase 3.** Mimicry is a worthwhile extension that can be added without breaking anything else.
2. **Peer network — static or dynamic?** **Decision: static random graph at phase 3.** Workers don't change peer groups when they switch firms. Dynamic version is an extension.
3. **Output price — common across firms or firm-specific?** **Decision: single common price `p_t`.** Firm differentiation is a paper extension.
4. **Demand elasticity `η`.** Default 2 is a guess. Run sensitivity in the sweep.
5. **What if all firms go bankrupt?** **Decision: the simulation continues with new entrants at the spawn rate.** Track time spent below `F` as a diagnostic.
6. **Worker skill investment.** **Decision: skip at phase 3.** Skill is determined at worker birth and fixed.
7. **Stochasticity.** Does production have noise? **Decision: deterministic given seed.** Stochastic productivity adds nothing fundamental and complicates analysis.

---

## 16. Definition of "phase 3 done"

Move to phase 4 when:

- All validation checks pass.
- The central heatmap renders and shows a non-trivial structure (regions, transitions). If it's flat, something is wrong — either the production function doesn't distinguish strategies enough, or the parameter ranges are wrong.
- Sweep infrastructure runs end-to-end and produces a CSV/parquet that downstream code can analyze.
- Skill-stratified subplots show meaningful variation.
- You can articulate one sentence about what the heatmap says — that sentence is the paper's headline finding.

If at end of weekend 7 and not done: cut the skill-stratified subplots first, then the dynamic peer network if you somehow added it, then halve replicates from 20 to 10.
