"""Time-loop simulation for Phase 1.

History captured as list-of-dicts -> DataFrame at end-of-run. Phase 1 = ~1800 rows total,
trivially in-memory. Profile in Phase 3; if DataFrame conversion >5% of run time, switch
to pre-allocated numpy columns.
"""
from typing import Callable

import numpy as np
import pandas as pd

from firm_ai_abm.firm import Firm
from firm_ai_abm.production import compute_K, productivity_vec, cost_vec


def run_simulation(firm: Firm, strategy: Callable, T: int | None = None) -> pd.DataFrame:
    """Run a single strategy for T periods and return the history as a DataFrame.

    Calls `firm.reset()` at entry so that strategies share the same firm instance
    without inheriting stale state from a prior run (architecture D-02). Callers
    do NOT need to reset manually before calling this function.

    Per-period sequence:
      1. strategy(firm, t) -> new_modes   (strategy proposes a mode vector)
      2. adj_cost = 0.0                   (Stage 4 wires real adjustment cost — S-4 seam)
      3. firm.modes = new_modes           (install the proposed modes)
      4. firm.K = compute_K(...)          (recompute worker headcount)
      5. Y = sum(productivity_vec(...))   (gross output)
      6. task_costs = sum(cost_vec(...))  (per-task c_aug / c_auto costs)
      7. wage_cost = w * K               (D-01: wages charged per worker here, NOT in cost_vec)
      8. C = task_costs + wage_cost + F + adj_cost
      9. pi = p * Y - C
      10. append row to firm.history

    Returns:
        pd.DataFrame with columns {t, Y, C, pi, K, modes}, one row per period.
        firm.history is NOT cleared after the run — the caller may inspect it directly.
    """
    if T is None:
        T = firm.params.T

    firm.reset()

    for t in range(T):
        new_modes = strategy(firm, t)

        adj_cost = 0.0  # Stage 4 wires real adjustment cost (architecture S-4)

        firm.modes = new_modes
        firm.K = compute_K(firm.modes, firm.params)

        Y = float(productivity_vec(firm.modes, firm.alpha, firm.beta, firm.params).sum())
        task_costs = float(cost_vec(firm.modes, firm.params).sum())
        wage_cost = firm.params.w * firm.K
        C = task_costs + wage_cost + firm.params.F + adj_cost
        pi = firm.params.p * Y - C

        firm.history.append({
            "t": t,
            "Y": Y,
            "C": C,
            "pi": pi,
            "K": int(firm.K),
            "modes": firm.modes.copy(),
        })

    df = pd.DataFrame(firm.history)
    return df
