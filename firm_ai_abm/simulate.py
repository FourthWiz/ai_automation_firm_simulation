"""Time-loop simulation for Phase 1.

History captured as list-of-dicts -> DataFrame at end-of-run. Phase 1 = ~1800 rows total,
trivially in-memory. Profile in Phase 3; if DataFrame conversion >5% of run time, switch
to pre-allocated numpy columns.
"""
from typing import Callable

import numpy as np
import pandas as pd

from firm_ai_abm.adjustment import adj_cost as compute_adj_cost
from firm_ai_abm.firm import Firm
from firm_ai_abm.production import compute_K, productivity_vec, cost_vec


def run_simulation(firm: Firm, strategy: Callable, T: int | None = None) -> pd.DataFrame:
    """Run a single strategy for T periods and return the history as a DataFrame.

    Calls `firm.reset()` at entry so that strategies share the same firm instance
    without inheriting stale state from a prior run (architecture D-02). Callers
    do NOT need to reset manually before calling this function.

    Per-period sequence (seven-step loop; stage 4 R-03 timing rule):
      1. new_modes = strategy(firm, t)    (strategy proposes a mode vector; reads firm.modes
                                           as previous-period install — design doc §6)
      2. prev_modes = firm.modes.copy()   (capture AFTER strategy returns, BEFORE install;
                                           defensive copy per R-04 / D-03)
      2b. assert prev_modes == firm.modes (eager contract: nothing mutated firm.modes between
                                           strategy return and this capture; O(N), cheap)
      3. period_adj = compute_adj_cost(prev_modes, new_modes, params)  (single float)
      4. firm.modes = new_modes           (install the proposed modes)
      5. firm.K = compute_K(...)          (recompute worker headcount)
      6. Y = sum(productivity_vec(...))   (gross output)
      7. task_costs = sum(cost_vec(...))  (per-task c_aug / c_auto costs)
      8. wage_cost = w * K               (D-01: wages charged per worker here, NOT in cost_vec)
      9. C = task_costs + wage_cost + F + period_adj
      10. pi = p * Y - C
      11. append row to firm.history

    Returns:
        pd.DataFrame with columns {t, Y, C, pi, K, modes, adj_cost}, one row per period.
        The adj_cost column records the per-period adjustment cost for audit/debugging.
        Adding this column is non-breaking: downstream Tier A code reads pi and K only.
        firm.history is NOT cleared after the run — the caller may inspect it directly.

    Risk citations:
        R-03: prev_modes captured AFTER strategy(firm, t) returns and BEFORE
              firm.modes = new_modes — the load-bearing timing rule for adj_cost.
        R-04: firm.modes.copy() prevents aliasing if a future strategy mutates
              firm.modes in place.
        R-09: eager assert (prev_modes == firm.modes).all() pins the contract that
              nothing mutates firm.modes between strategy return and capture.
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
        period_adj = compute_adj_cost(prev_modes, new_modes, firm.params)

        # Step 4: install new modes
        firm.modes = new_modes

        # Step 5: recompute workforce
        firm.K = compute_K(firm.modes, firm.params)

        # Steps 6-10: production accounting
        Y = float(productivity_vec(firm.modes, firm.alpha, firm.beta, firm.params).sum())
        task_costs = float(cost_vec(firm.modes, firm.params).sum())
        wage_cost = firm.params.w * firm.K
        C = task_costs + wage_cost + firm.params.F + period_adj
        pi = firm.params.p * Y - C

        # Step 11: append audit row (adj_cost column added in stage 4)
        firm.history.append({
            "t": t,
            "Y": Y,
            "C": C,
            "pi": pi,
            "K": int(firm.K),
            "modes": firm.modes.copy(),
            "adj_cost": period_adj,
        })

    df = pd.DataFrame(firm.history)
    return df
