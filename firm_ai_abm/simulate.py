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

Phase 1.5 Stage 6 additions:
  - Step 0.5 (opt-in hire-back): when enable_hiring=True and firings occurred, calls
    replace_to_target(firm, firm.K0, t, output_per_worker) to restore workforce.K to K0.
    Replacement workers are fresh draws (new theta/wage); period_hire_cost = c_hire * n_hired.
  - History gains n_hired column (int, 0 when enable_hiring=False).
  - firm.K0: initial headcount captured once in make_firm; never mutated by reset().
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
from firm_ai_abm.review import apply_firings, firing_review, replace_to_target
from firm_ai_abm.workers import task_to_worker_map


def run_horizon(firm: Firm, strategy: Callable, horizon: int) -> pd.DataFrame:
    """Run `horizon` periods IN-PLACE; return DataFrame of exactly `horizon` rows.

    Does NOT call firm.reset(). Does NOT touch firm.history.
    Caller MUST pass copy.deepcopy(firm) if the live firm must be preserved.
    firm.history is intentionally not written — rows accumulate only in a local list.

    Adds step 6 a_in_training_per_task gate and step 11.5 flip when
    firm.params.enable_training_delay is True.
    """
    import copy  # noqa: PLC0415 — local import avoids circular at module load

    params = firm.params

    # Pre-loop: allocate per-worker matrices (fresh for this call)
    K_max = params.N // params.tasks_per_worker
    output_per_worker = np.full((horizon, K_max), np.nan, dtype=np.float64)
    # aug_cost_per_worker mirrors output_per_worker shape/semantics.
    # Populated at Step 6 via a SEPARATE cost_vec call (D-07: two independent calls
    # per period; Step 7 call is unchanged to preserve byte-identity of task_costs,
    # C, and pi under dormant T_review=inf default). Two cost_vec calls per tick
    # by design — see decision-thirteen in review.py module docstring.
    aug_cost_per_worker = np.full((horizon, K_max), np.nan, dtype=np.float64)

    local_history: list[dict] = []

    for t in range(horizon):
        # -------------------------------------------------------------------
        # Step 0: periodic firing review (START-OF-PERIOD)
        # -------------------------------------------------------------------
        fire_indices, c_train_lost_period = firing_review(
            firm.workforce, t, output_per_worker, aug_cost_per_worker, params
        )
        n_review_fired_period = int(len(fire_indices))
        period_review_fire_cost = float(params.c_fire) * n_review_fired_period
        if n_review_fired_period > 0:
            firm.workforce, output_per_worker, aug_cost_per_worker = apply_firings(
                firm, fire_indices, t, output_per_worker, aug_cost_per_worker
            )

        # -------------------------------------------------------------------
        # Step 0.5: opt-in hire-back to K0 (same period, after apply_firings)
        # -------------------------------------------------------------------
        n_hired_period = 0
        period_hire_cost = 0.0
        if params.enable_hiring and n_review_fired_period > 0:
            # replace_to_target draws FRESH workers via sample_workforce (new theta and
            # wage draws from the same distribution as the initial workforce). Hired
            # workers are NOT the same individuals who were fired. Wage mean can drift
            # over many fire+rehire cycles per workers.py drift semantics.
            K_before_hire = firm.workforce.K
            firm.workforce, output_per_worker, aug_cost_per_worker = replace_to_target(
                firm, firm.K0, t, output_per_worker, aug_cost_per_worker
            )
            n_hired_period = firm.workforce.K - K_before_hire
            period_hire_cost = float(params.c_hire) * n_hired_period

        # -------------------------------------------------------------------
        # Step 1: strategy proposes new modes
        # -------------------------------------------------------------------
        new_modes = strategy(firm, t)

        # Step 2: capture prev_modes AFTER strategy, BEFORE install
        prev_modes = firm.modes.copy()

        # Step 2b: eager contract — strategy must not mutate firm.modes
        assert (prev_modes == firm.modes).all(), (
            f"firm.modes was mutated by strategy at t={t}."
        )

        # -------------------------------------------------------------------
        # Step 2.5: kernel-side feasibility clamp
        # -------------------------------------------------------------------
        new_modes = new_modes.copy()
        n_HA = int(((new_modes == 0) | (new_modes == 1)).sum())
        capacity = firm.workforce.K * params.tasks_per_worker
        if n_HA > capacity:
            ha_idx = np.where((new_modes == 0) | (new_modes == 1))[0]
            demote_idx = ha_idx[-(n_HA - capacity):]
            new_modes[demote_idx] = 2
            clamp_event = 1
        else:
            clamp_event = 0
        K_modes = compute_K(new_modes, params)

        # -------------------------------------------------------------------
        # Step 3: adjustment cost
        # -------------------------------------------------------------------
        period_adj = compute_adj_cost(prev_modes, new_modes, params, firm.workforce)

        # Step 4: install modes
        firm.modes = new_modes

        # -------------------------------------------------------------------
        # Step 6: build theta_per_task, a_in_training_per_task, compute Y
        # -------------------------------------------------------------------
        t2w = task_to_worker_map(firm.modes, firm.workforce.K, params.tasks_per_worker)
        if firm.workforce.K > 0:
            theta_per_task = np.where(
                t2w >= 0,
                firm.workforce.theta[np.where(t2w >= 0, t2w, 0)],
                1.0,
            )
        else:
            theta_per_task = np.ones(params.N, dtype=np.float64)

        if params.enable_training_delay and firm.workforce.K > 0:
            a_itp = np.zeros(params.N, dtype=bool)
            ha_mask = t2w >= 0
            a_itp[ha_mask] = firm.workforce.a_training_in_progress[t2w[ha_mask]]
        else:
            a_itp = None

        prod_per_task = productivity_vec(
            firm.modes, firm.alpha, firm.beta, params,
            theta_per_task=theta_per_task,
            a_in_training_per_task=a_itp,
        )
        Y = float(prod_per_task.sum())

        # Per-worker output bookkeeping
        output_this_period = np.full(firm.workforce.K, np.nan, dtype=np.float64)
        active_workers = np.unique(t2w[t2w >= 0])
        for k in active_workers:
            output_this_period[k] = float(prod_per_task[t2w == k].sum())
        output_per_worker[t, : firm.workforce.K] = output_this_period

        # Per-worker aug-cost bookkeeping (D-07: SEPARATE cost_vec call; Step 7 unchanged)
        # T-mode tasks have t2w == -1 → excluded by worker_mask automatically.
        # Training-period A tasks: cost_vec returns 0 → aug_cost recorded as 0.0 (Q-02 / MAJ-5).
        cost_per_task_aug = cost_vec(firm.modes, params, a_in_training_per_task=a_itp)
        aug_cost_this_period = np.full(firm.workforce.K, np.nan, dtype=np.float64)
        for k in active_workers:
            worker_mask = (t2w == k)
            aug_cost_this_period[k] = float(cost_per_task_aug[worker_mask].sum())
        aug_cost_per_worker[t, : firm.workforce.K] = aug_cost_this_period

        # Step 7: per-task variable costs (UNCHANGED — D-07: independent call preserves
        # byte-identity of task_costs, C, and pi under dormant T_review=inf default)
        task_costs = float(cost_vec(firm.modes, params, a_in_training_per_task=a_itp).sum())

        # Step 8: wage bill from ASSIGNED workers only
        active = np.unique(t2w[t2w >= 0])
        wage_bill = float(firm.workforce.wage[active].sum()) if active.size > 0 else 0.0

        # Steps 9-10: total cost and profit
        C = task_costs + wage_bill + params.F + period_adj + period_review_fire_cost + period_hire_cost
        pi = params.p * Y - C

        # Step 11: append to local history (NOT firm.history)
        K_active = int(active.size)
        local_history.append({
            "t": t,
            "Y": Y,
            "C": C,
            "pi": pi,
            "K": int(K_modes),
            "K_active": K_active,
            "K_clamp_events": clamp_event,
            "modes": firm.modes.copy(),
            "adj_cost": period_adj,
            "wage_bill": wage_bill,
            "mean_theta": float(firm.workforce.theta.mean()) if firm.workforce.K > 0 else float("nan"),
            "mean_wage": float(firm.workforce.wage.mean()) if firm.workforce.K > 0 else float("nan"),
            "n_a_trained": int(firm.workforce.a_trained.sum()),
            "n_review_fired": n_review_fired_period,
            "c_train_lost": c_train_lost_period,
            "n_hired": n_hired_period,
        })

        # -------------------------------------------------------------------
        # Step 11.5: tenure increment + training-delay flag flip
        # -------------------------------------------------------------------
        firm.workforce.tenure = firm.workforce.tenure + 1

        if params.enable_training_delay:
            flip_mask = firm.workforce.a_training_in_progress.copy()
            firm.workforce.a_trained[flip_mask] = True
            firm.workforce.a_training_in_progress[:] = False

    firm.output_per_worker = output_per_worker  # type: ignore[attr-defined]
    firm.aug_cost_per_worker = aug_cost_per_worker  # type: ignore[attr-defined]
    return pd.DataFrame(local_history)


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

      Step 0.5 (Stage 6, opt-in):
        If enable_hiring AND n_review_fired > 0:
          - replace_to_target(firm, firm.K0, t, output_per_worker) — restores workforce.K
            to K0 using fresh worker draws (new theta/wage from same distribution).
          - period_hire_cost = c_hire * n_hired
        Else: period_hire_cost = 0.0; n_hired = 0
        NOTE: replacement workers are NOT the same individuals fired; wage mean can drift
        over many fire+rehire cycles.

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
      Step 9: C = task_costs + wage_bill + F + period_adj + period_review_fire_cost + period_hire_cost
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
        n_hired: int count of workers hired back this period (0 when enable_hiring=False).

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
    return run_horizon(firm, strategy, T)
