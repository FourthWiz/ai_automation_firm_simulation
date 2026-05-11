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

Phase 1.5 Stage 3 additions:
  - output_per_worker matrix: shape (T, K_max), float64, NaN-filled for inactive slots
    and pre-hire periods. K_max = N // tasks_per_worker. Accumulated at step 6 each period.
  - Periodic firing review (D-08 timing: START-OF-PERIOD, before strategy call):
    At periods t > 0 where t % T_review == 0 (and T_review is finite):
      1. firing_review returns (fire_indices, c_train_lost_metric)
      2. apply_firings removes fired workers; K MAY SHRINK (Stage 5 D-03)
      3. period_review_fire_cost = c_fire * len(fire_indices) is added to C (charged in pi)
      4. c_train_lost is recorded as a history diagnostic (NOT charged — D-08, D-11)
  - History gains 2 new columns: n_review_fired, c_train_lost
  - Tenure increment at step 11.5: workforce.tenure += 1 for ALL workers (D-06).
  - firm.output_per_worker: post-run attribute exposing the matrix for tests that need
    direct inspection (e.g., T-15 NaN-boundary assertions). Callers must not mutate it.

Phase 1.5 Stage 5 additions:
  - apply_firings_and_replace replaced by apply_firings (K may shrink; no auto-replace).
  - Step 2.5 (kernel-side clamp, D-06): AFTER step 2 prev_modes capture, BEFORE step 3
    compute_adj_cost. Demotes excess H/A tasks to T when n_HA > workforce.K * tpw.
    Ensures every downstream callsite (compute_adj_cost → count_workers_entering_a_first_time
    → task_to_worker_map) sees capacity-consistent modes. See D-13 for adj_cost semantics.
  - History gains K_active (already present) and NEW K_clamp_events column (int, 0 when no
    clamp; 1 when clamp fires this period).
"""
import math
from typing import Callable

import numpy as np
import pandas as pd

from firm_ai_abm.adjustment import adj_cost as compute_adj_cost
from firm_ai_abm.firm import Firm
from firm_ai_abm.production import compute_K, productivity_vec, cost_vec
from firm_ai_abm.review import apply_firings, firing_review
from firm_ai_abm.workers import task_to_worker_map


def run_simulation(firm: Firm, strategy: Callable, T: int | None = None) -> pd.DataFrame:
    """Run a single strategy for T periods and return the history as a DataFrame.

    Calls `firm.reset()` at entry so that strategies share the same firm instance
    without inheriting stale state from a prior run. Callers do NOT need to reset
    manually before calling this function.

    Per-period sequence (thirteen-step loop; Stage 3 additions noted):

      PRE-LOOP: Allocate output_per_worker matrix (T × K_max), NaN-filled.

      Step 0 (Stage 3, D-08 — START-OF-PERIOD):
        If t > 0 AND T_review is finite AND t % T_review == 0:
          - firing_review → fire_indices, c_train_lost_metric
          - apply_firings → new_workforce, new_output_per_worker (K may shrink)
          - period_review_fire_cost = c_fire * len(fire_indices)
        Else: period_review_fire_cost = 0.0; n_review_fired = 0; c_train_lost = 0.0

      Step 1: new_modes = strategy(firm, t)
      Step 2: prev_modes = firm.modes.copy()   (capture AFTER strategy, BEFORE install)
      Step 2b: assert prev_modes == firm.modes (eager contract: strategy must not mutate)
      Step 2.5 (Stage 5, D-06): kernel-side clamp — demote excess H/A tasks to T so
        that adj_cost (step 3) and all downstream task_to_worker_map calls see
        capacity-consistent modes. Records K_clamp_events=1 when active.
        See D-13 for adj_cost semantics on clamped modes.
      Step 3: period_adj = compute_adj_cost(prev_modes, new_modes, params, workforce)
      Step 4: firm.modes = new_modes           (install clamped modes)
      Step 5: K_modes already computed at step 2.5 (post-clamp recompute)
      Step 6: build theta_per_task; compute Y (productivity_vec); record per-worker output
              in output_per_worker[t, :workforce.K]
      Step 7: task_costs = cost_vec(...).sum()
      Step 8: wage_bill = sum of wages for ASSIGNED workers only (D-03)
      Step 9: C = task_costs + wage_bill + F + period_adj + period_review_fire_cost
      Step 10: pi = p * Y - C
      Step 11: append row to firm.history (includes n_review_fired, c_train_lost)
      Step 11.5 (Stage 3, D-06): workforce.tenure += 1 for ALL workers

      POST-LOOP: firm.output_per_worker = output_per_worker (test exposure seam)

    Returns:
        pd.DataFrame with columns {t, Y, C, pi, K, K_active, K_clamp_events, modes,
        adj_cost, wage_bill, mean_theta, mean_wage, n_a_trained, n_review_fired,
        c_train_lost}, one row per period.
        K records the post-clamp mode-derived headcount (compute_K), NOT workforce.K.
        K_clamp_events: 1 if the kernel-side clamp fired this period, else 0.
        New columns (wage_bill, mean_theta, mean_wage, n_a_trained) from Stages 1+2
        are additive; existing downstream code that reads pi and K is unaffected.
        n_a_trained: int count of workers with a_trained==True at end of the period
        (after adj_cost's in-place mutation, D-02).
        n_review_fired: int count of workers fired by the periodic review this period.
        c_train_lost: float value of trained capital lost due to review firings this
        period (diagnostic metric only — NOT included in C or pi, D-08).

    Side effect (Stage 3):
        After the loop completes, attaches `firm.output_per_worker` (shape T × K_max,
        float64). Exposed for tests that need direct inspection of NaN boundaries.
        Callers must not mutate this array. The DataFrame return value is unchanged.

    Risk citations:
        R-03: prev_modes captured AFTER strategy(firm, t) returns and BEFORE
              firm.modes = new_modes — the load-bearing timing rule for adj_cost.
        R-04: firm.modes.copy() prevents aliasing if a future strategy mutates firm.modes.
        R-09: eager assert (prev_modes == firm.modes).all() pins the contract.
        D-03: wage_bill from ASSIGNED workers only — all_T → no active workers → wage_bill=0,
              preserving byte-parity with Phase 1's w * K = 0 for all_T.
        D-08: period_review_fire_cost added to C; c_train_lost is metric-only.
        D-06: tenure incremented at step 11.5 for ALL workers including replacements.
    """
    if T is None:
        T = firm.params.T

    firm.reset()

    # Pre-loop: allocate output_per_worker matrix (D-03 Stage 3)
    K_max = firm.params.N // firm.params.tasks_per_worker
    output_per_worker = np.full((T, K_max), np.nan, dtype=np.float64)

    for t in range(T):
        # -----------------------------------------------------------------------
        # Step 0: periodic firing review (Stage 3, D-08 — START-OF-PERIOD)
        # -----------------------------------------------------------------------
        fire_indices, c_train_lost_period = firing_review(
            firm.workforce, t, output_per_worker, firm.params
        )
        n_review_fired_period = int(len(fire_indices))
        period_review_fire_cost = float(firm.params.c_fire) * n_review_fired_period
        if n_review_fired_period > 0:
            # Stage 5 D-03: apply_firings only — K may shrink; no auto-replace
            firm.workforce, output_per_worker = apply_firings(
                firm, fire_indices, t, output_per_worker
            )

        # -----------------------------------------------------------------------
        # Step 1: strategy proposes new modes (reads firm.modes as prev-period install)
        # -----------------------------------------------------------------------
        new_modes = strategy(firm, t)

        # Step 2: capture prev_modes AFTER strategy returns AND BEFORE install (R-03)
        prev_modes = firm.modes.copy()  # defensive copy (R-04)

        # Step 2b: eager contract assertion — strategy must not have mutated firm.modes (R-09)
        assert (prev_modes == firm.modes).all(), (
            f"firm.modes was mutated by strategy between strategy(firm, t) return "
            f"and prev_modes capture at t={t}. This violates the R-03 timing contract."
        )

        # -----------------------------------------------------------------------
        # Step 2.5 (Stage 5, D-06): kernel-side feasibility clamp.
        # PLACEMENT: AFTER step 2 (prev_modes capture), BEFORE step 3 (compute_adj_cost).
        # This ensures adj_cost → count_workers_entering_a_first_time → task_to_worker_map
        # all see capacity-consistent modes. See D-13 for adj_cost / a_trained semantics.
        # MIN-3 fix: copy new_modes to prevent strategy's cached array from being mutated.
        # -----------------------------------------------------------------------
        new_modes = new_modes.copy()
        params = firm.params
        n_HA = int(((new_modes == 0) | (new_modes == 1)).sum())  # H=0, A=1
        capacity = firm.workforce.K * params.tasks_per_worker
        if n_HA > capacity:
            ha_idx = np.where((new_modes == 0) | (new_modes == 1))[0]
            demote_idx = ha_idx[-(n_HA - capacity):]  # highest-index H/A tasks demoted
            new_modes[demote_idx] = 2  # T = 2 (Mode.T)
            clamp_event = 1
        else:
            clamp_event = 0
        # Post-clamp K_modes (replaces step 5 compute)
        K_modes = compute_K(new_modes, params)

        # -----------------------------------------------------------------------
        # Step 3: compute adjustment cost from prev to new (clamped) modes.
        # Stage 2: passes firm.workforce so adj_cost uses per-worker training memory
        # (D-02 side effect: workforce.a_trained mutated in-place inside adj_cost)
        # D-13: adj_cost reflects CLAMPED transitions; a_trained only fires for
        # workers actually assigned to A-tasks post-clamp.
        # -----------------------------------------------------------------------
        period_adj = compute_adj_cost(prev_modes, new_modes, firm.params, firm.workforce)

        # Step 4: install clamped modes
        firm.modes = new_modes

        # -----------------------------------------------------------------------
        # Step 6: build theta_per_task from workforce, compute output, and record
        # per-worker output in output_per_worker (Stage 3 D-02, D-03)
        # -----------------------------------------------------------------------
        t2w = task_to_worker_map(firm.modes, firm.workforce.K, firm.params.tasks_per_worker)
        # When workforce.K=0 (all tasks clamped to T), theta array is empty — guard indexing.
        # T-mode tasks never use theta, so 1.0 placeholder is correct.
        if firm.workforce.K > 0:
            theta_per_task = np.where(
                t2w >= 0,
                firm.workforce.theta[np.where(t2w >= 0, t2w, 0)],
                1.0,
            )
        else:
            theta_per_task = np.ones(firm.params.N, dtype=np.float64)
        prod_per_task = productivity_vec(
            firm.modes, firm.alpha, firm.beta, firm.params, theta_per_task=theta_per_task
        )
        Y = float(prod_per_task.sum())

        # Stage 3: per-worker output bookkeeping (D-02, D-03)
        output_this_period = np.full(firm.workforce.K, np.nan, dtype=np.float64)
        active_workers = np.unique(t2w[t2w >= 0])
        for k in active_workers:
            output_this_period[k] = float(prod_per_task[t2w == k].sum())
        output_per_worker[t, : firm.workforce.K] = output_this_period

        # Step 7: per-task variable costs (c_aug / c_auto; wages NOT included here)
        task_costs = float(cost_vec(firm.modes, firm.params).sum())

        # Step 8: wage bill from ASSIGNED workers only (D-03)
        # all_T → t2w is all-(-1) → active is empty → wage_bill = 0 (Phase 1 parity)
        active = np.unique(t2w[t2w >= 0])
        wage_bill = float(firm.workforce.wage[active].sum()) if active.size > 0 else 0.0

        # Step 9-10: total cost and profit (Stage 3: + period_review_fire_cost, D-08)
        C = task_costs + wage_bill + firm.params.F + period_adj + period_review_fire_cost
        pi = firm.params.p * Y - C

        # Step 11: append audit row
        # K_active: unique workers with at least one H/A task this period.
        # Differs from K_modes when H/A tasks are scattered across task slots
        # (e.g. large N) — K_modes uses the packed-ceiling formula which
        # underestimates actual paying headcount in that case.
        K_active = int(active.size)
        firm.history.append({
            "t": t,
            "Y": Y,
            "C": C,
            "pi": pi,
            "K": int(K_modes),             # packed-formula headcount: ceil(n_H_or_A / tpw)
            "K_active": K_active,           # actual unique workers assigned to H/A tasks
            "K_clamp_events": clamp_event,  # Stage 5: 1 if kernel clamp fired this period
            "modes": firm.modes.copy(),
            "adj_cost": period_adj,
            "wage_bill": wage_bill,
            "mean_theta": float(firm.workforce.theta.mean()) if firm.workforce.K > 0 else float("nan"),
            "mean_wage": float(firm.workforce.wage.mean()) if firm.workforce.K > 0 else float("nan"),
            "n_a_trained": int(firm.workforce.a_trained.sum()),
            "n_review_fired": n_review_fired_period,
            "c_train_lost": c_train_lost_period,
        })

        # -----------------------------------------------------------------------
        # Step 11.5: tenure increment for ALL workers (Stage 3, D-06)
        # Tenure increment fires even when workforce.K=0 (no-op: empty += 1 is empty).
        # -----------------------------------------------------------------------
        firm.workforce.tenure = firm.workforce.tenure + 1

    # Post-loop: expose output_per_worker for test inspection (T-05 seam; for tests only)
    firm.output_per_worker = output_per_worker  # type: ignore[attr-defined]

    df = pd.DataFrame(firm.history)
    return df
