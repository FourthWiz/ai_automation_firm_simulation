"""Time-loop simulation for Phase 1 / Phase 1.5.

History captured as list-of-dicts -> DataFrame at end-of-run. Phase 1 = ~1800 rows total,
trivially in-memory. Profile in Phase 3; if DataFrame conversion >5% of run time, switch
to pre-allocated numpy columns.

Phase 1.5 Stage 1 additions:
  - theta_per_task threaded through productivity_vec (D-02 multiplicative)
  - wage_bill from ASSIGNED workers only (D-03: deviation from architecture §5 for all_T parity)
  - firm.K is now a read-only property; step 5 uses local K_modes variable
  - History gains 3 new columns: wage_bill, mean_theta, mean_wage

Phase 1.5 Stage 2 additions:
  - adj_cost now receives firm.workforce (per-worker training memory gate, D-02)
  - adj_cost mutates workforce.a_trained as a side effect (D-02; documented in adjustment.py)
  - History gains n_a_trained column (count of workers with a_trained==True, added T-10)
"""
from typing import Callable

import numpy as np
import pandas as pd

from firm_ai_abm.adjustment import adj_cost as compute_adj_cost
from firm_ai_abm.firm import Firm
from firm_ai_abm.production import compute_K, productivity_vec, cost_vec
from firm_ai_abm.workers import task_to_worker_map


def run_simulation(firm: Firm, strategy: Callable, T: int | None = None) -> pd.DataFrame:
    """Run a single strategy for T periods and return the history as a DataFrame.

    Calls `firm.reset()` at entry so that strategies share the same firm instance
    without inheriting stale state from a prior run. Callers do NOT need to reset
    manually before calling this function.

    Per-period sequence (eleven-step loop; stage 4 R-03 timing rule):
      1. new_modes = strategy(firm, t)
      2. prev_modes = firm.modes.copy()   (capture AFTER strategy returns, BEFORE install)
      2b. assert prev_modes == firm.modes (eager contract: strategy must not mutate firm.modes)
      3. period_adj = compute_adj_cost(prev_modes, new_modes, params)
      4. firm.modes = new_modes           (install the proposed modes)
      5. K_modes = compute_K(...)         (mode-derived headcount as LOCAL variable; not stored
                                           on firm — firm.K is now a property of workforce.K)
      6. build theta_per_task from workforce; Y = productivity_vec(..., theta_per_task).sum()
      7. task_costs = cost_vec(...).sum()
      8. wage_bill = sum of wages for ASSIGNED workers only (D-03)
      9. C = task_costs + wage_bill + F + period_adj
      10. pi = p * Y - C
      11. append row to firm.history

    Returns:
        pd.DataFrame with columns {t, Y, C, pi, K, modes, adj_cost, wage_bill, mean_theta,
        mean_wage, n_a_trained}, one row per period. K records the mode-derived headcount
        (compute_K), NOT workforce.K. New columns (wage_bill, mean_theta, mean_wage,
        n_a_trained) are additive; existing downstream code that reads pi and K is unaffected.
        n_a_trained: int count of workers with a_trained==True at end of the period (after
        adj_cost's in-place mutation, D-02).

    Risk citations:
        R-03: prev_modes captured AFTER strategy(firm, t) returns and BEFORE
              firm.modes = new_modes — the load-bearing timing rule for adj_cost.
        R-04: firm.modes.copy() prevents aliasing if a future strategy mutates firm.modes.
        R-09: eager assert (prev_modes == firm.modes).all() pins the contract.
        D-03: wage_bill from ASSIGNED workers only — all_T → no active workers → wage_bill=0,
              preserving byte-parity with Phase 1's w * K = 0 for all_T.
    """
    if T is None:
        T = firm.params.T

    firm.reset()

    for t in range(T):
        # Step 1: strategy proposes new modes (reads firm.modes as prev-period install)
        new_modes = strategy(firm, t)

        # Step 2: capture prev_modes AFTER strategy returns AND BEFORE install (R-03)
        prev_modes = firm.modes.copy()  # defensive copy (R-04)

        # Step 2b: eager contract assertion — strategy must not have mutated firm.modes (R-09)
        assert (prev_modes == firm.modes).all(), (
            f"firm.modes was mutated by strategy between strategy(firm, t) return "
            f"and prev_modes capture at t={t}. This violates the R-03 timing contract."
        )

        # Step 3: compute adjustment cost from prev to new
        # Stage 2: passes firm.workforce so adj_cost uses per-worker training memory
        # (D-02 side effect: workforce.a_trained mutated in-place inside adj_cost)
        period_adj = compute_adj_cost(prev_modes, new_modes, firm.params, firm.workforce)

        # Step 4: install new modes
        firm.modes = new_modes

        # Step 5: mode-derived headcount as local variable (firm.K is now a read-only property)
        K_modes = compute_K(firm.modes, firm.params)

        # Step 6: build theta_per_task from workforce and compute output
        t2w = task_to_worker_map(firm.modes, firm.workforce.K, firm.params.tasks_per_worker)
        # T-slots map to -1; their theta is unused (T-branch in productivity_vec ignores theta)
        # np.where(t2w >= 0, t2w, 0) avoids negative indexing; value at T-slots is multiplied
        # by 0.0 (via the Mode.H / Mode.A conditions) so the spurious index[0] doesn't matter
        theta_per_task = np.where(
            t2w >= 0,
            firm.workforce.theta[np.where(t2w >= 0, t2w, 0)],
            1.0,
        )
        Y = float(productivity_vec(
            firm.modes, firm.alpha, firm.beta, firm.params, theta_per_task=theta_per_task
        ).sum())

        # Step 7: per-task variable costs (c_aug / c_auto; wages NOT included here)
        task_costs = float(cost_vec(firm.modes, firm.params).sum())

        # Step 8: wage bill from ASSIGNED workers only (D-03)
        # all_T → t2w is all-(-1) → active is empty → wage_bill = 0 (Phase 1 parity)
        active = np.unique(t2w[t2w >= 0])
        wage_bill = float(firm.workforce.wage[active].sum()) if active.size > 0 else 0.0

        # Step 9-10: total cost and profit
        C = task_costs + wage_bill + firm.params.F + period_adj
        pi = firm.params.p * Y - C

        # Step 11: append audit row
        firm.history.append({
            "t": t,
            "Y": Y,
            "C": C,
            "pi": pi,
            "K": int(K_modes),           # mode-derived headcount (NOT workforce.K)
            "modes": firm.modes.copy(),
            "adj_cost": period_adj,
            "wage_bill": wage_bill,       # new: sum of assigned-worker wages
            "mean_theta": float(firm.workforce.theta.mean()),  # new: workforce mean theta
            "mean_wage": float(firm.workforce.wage.mean()),    # new: workforce mean wage
            "n_a_trained": int(firm.workforce.a_trained.sum()),  # Stage 2: trained worker count
        })

    df = pd.DataFrame(firm.history)
    return df
