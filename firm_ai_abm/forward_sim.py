"""Shared forward simulator for rolling-horizon strategies (D-01).

Provides `Action` NamedTuple and `forward_simulate_action_path` — the single
canonical forward-sim consumed by both dp_optimizer and margin_optimizer.

Key design points:
  - Uses TRUE per-task theta (not theta-flat) via firm.workforce.theta. Closes F-02.
  - Reads alpha_hat/beta_hat (posteriors) for unobserved tasks; reads firm.alpha/beta
    for tasks already in T/A mode. Same information set as the live kernel.
  - Fire cost: c_fire * |threshold_set ∪ planned_set|. Closes F-03.
  - Hire cost: charged at drain period (s + hire_delay_periods), not at queue time.
    Per review.py:417.
  - Works under both enable_hiring=True and enable_replenish_hiring=True. When
    action.n_hire > 0, workers are sampled and appended to f_workforce at drain time
    so subsequent steps see the updated workforce size and theta distribution.

Imports from dp_optimizer are LAZY (function-level) to avoid circular imports
at module load time: forward_sim ← dp_optimizer ← forward_sim would be circular
at module level.
"""
from __future__ import annotations

import copy
import math
from typing import NamedTuple, TYPE_CHECKING

import numpy as np

from firm_ai_abm.production import Mode, productivity_vec, cost_vec
from firm_ai_abm.workers import task_to_worker_map, Workforce, sample_workforce
from firm_ai_abm.review import firing_review

if TYPE_CHECKING:
    from firm_ai_abm.firm import Firm
    from firm_ai_abm.config import FirmParams


class Action(NamedTuple):
    """Unified action for one planning step: (n_fire, n_aug, n_hire)."""
    n_fire: int
    n_aug: int
    n_hire: int


def _replay_firing_review(
    firm: "Firm",
    step_t: int,
    opw_so_far: np.ndarray,
    acpw_so_far: np.ndarray,
    f_workforce,
) -> tuple[np.ndarray, float]:
    """Approximate firing_review output using accumulated history.

    At s=0 both arrays contain only real history → exact match with kernel's
    firing_review call. At s>0, synthetic rows accumulate approximation error.

    math.isinf(params.T_review) is checked FIRST (before any int cast) — mirrors
    firing_review's own guard and avoids OverflowError (lesson 2026-05-09).

    Returns (threshold_fire_indices, c_train_lost).
    """
    params = firm.params
    # isinf MUST be first check (int(math.inf) raises OverflowError)
    if math.isinf(params.T_review):
        return np.array([], dtype=int), 0.0
    return firing_review(f_workforce, step_t, opw_so_far, acpw_so_far, params)


def forward_simulate_action_path(
    firm: "Firm",
    t: int,
    action_path: list[Action],
    horizon: int,
) -> float:
    """Forward-simulate horizon steps along action_path; return cumulative profit.

    Hire-cost timing: c_hire is charged at the projected drain step
    (s + hire_delay_periods), not at queue time. Matches review.py:417.
    Supports both enable_hiring=True and enable_replenish_hiring=True; when
    action.n_hire > 0 workers are sampled and appended to the workforce copy.

    Uses function-level imports from dp_optimizer to avoid circular imports.
    """
    # Lazy imports to avoid circular: forward_sim ← dp_optimizer ← forward_sim
    from firm_ai_abm.dp_optimizer import (
        _apply_action_to_modes,
        _apply_firings_on_workforce,
        _is_review_period,
        _sort_fireable_workers_by_cost_effectiveness,
    )

    params = firm.params
    K_max_orig = params.N // params.tasks_per_worker

    # Isolate planning rng from live kernel state — prevents rng leakage across
    # planning calls (each planning path that samples workers would otherwise
    # advance firm.rng, breaking reproducibility for identical-seed runs).
    plan_rng = copy.deepcopy(firm.rng)

    # Deep-copy workforce once (immutable view of the rest)
    f_workforce = copy.deepcopy(firm.workforce)
    f_modes = firm.modes.copy()
    alpha_hat = firm.alpha_hat.copy()
    beta_hat = firm.beta_hat.copy()

    # Real history rows for the replay helper.
    # Both are kernel-set seams (not initialized by firm.reset()); use getattr fallback.
    opw_real = getattr(firm, "_output_per_worker_so_far", None)
    if opw_real is not None and opw_real.shape[0] >= t:
        opw_rows: list = list(opw_real[:t])
    else:
        opw_rows = []

    acpw_real = getattr(firm, "_aug_cost_per_worker_so_far", None)
    if acpw_real is not None and acpw_real.shape[0] >= t:
        acpw_rows: list = list(acpw_real[:t])
    else:
        acpw_rows = []

    # Pending hires: shallow copy of (period_eligible, n) tuples
    pending = list(getattr(firm, "pending_hires", []))

    path_pi = 0.0

    for s, action in enumerate(action_path):
        step_t = t + s
        prev_modes = f_modes.copy()

        # ------------------------------------------------------------------
        # Step 0: Threshold-rule replay (exact at s=0 with real history)
        # ------------------------------------------------------------------
        if math.isinf(params.T_review):
            threshold_set = np.array([], dtype=int)
        else:
            opw_so_far = np.array(opw_rows) if opw_rows else np.zeros((0, K_max_orig), dtype=np.float64)
            acpw_so_far = np.array(acpw_rows) if acpw_rows else np.zeros((0, K_max_orig), dtype=np.float64)
            threshold_set, _ = _replay_firing_review(
                firm, step_t, opw_so_far, acpw_so_far, f_workforce
            )

        # ------------------------------------------------------------------
        # Step 1: Strategy's planned firing — calendar-gated
        # ------------------------------------------------------------------
        if _is_review_period(step_t, params.T_review) and action.n_fire > 0:
            opw_hist = getattr(firm, "_output_per_worker_so_far", None)
            if opw_hist is None:
                opw_hist = np.zeros((1, max(firm.workforce.K, f_workforce.K)), dtype=np.float64)
            # _sort_fireable_workers_by_cost_effectiveness uses firm.workforce.K as K.
            # After firings f_workforce.K may be smaller — pad wages to orig K with inf
            # so removed worker slots are never chosen, then filter back to f_workforce bounds.
            fw_K = f_workforce.K
            orig_K = firm.workforce.K
            if fw_K < orig_K:
                wages_for_sort = np.concatenate([
                    f_workforce.wage,
                    np.full(orig_K - fw_K, np.inf, dtype=np.float64),
                ])
            else:
                wages_for_sort = f_workforce.wage
            planned_order = _sort_fireable_workers_by_cost_effectiveness(
                firm, opw_hist, wages_for_sort, step_t
            )
            # Keep only indices within current f_workforce bounds
            planned_set = planned_order[planned_order < fw_K][:action.n_fire].astype(int)
            merged = np.unique(np.concatenate([threshold_set, planned_set]))
            path_pi -= params.c_fire * len(merged)
            f_workforce, _, _ = _apply_firings_on_workforce(f_workforce, merged, step_t)
        elif len(threshold_set) > 0:
            path_pi -= params.c_fire * len(threshold_set)
            f_workforce, _, _ = _apply_firings_on_workforce(f_workforce, threshold_set, step_t)

        # ------------------------------------------------------------------
        # Step 2: Strategy's planned hiring (n_fire == 0 guard per D-04)
        # Works under both enable_replenish_hiring and enable_hiring.
        # Cost charged at drain period, not queue time — per review.py:417
        # ------------------------------------------------------------------
        hiring_enabled = params.enable_replenish_hiring or params.enable_hiring
        if action.n_hire > 0 and action.n_fire == 0 and hiring_enabled:
            pending.append((step_t + params.hire_delay_periods, action.n_hire))

        # ------------------------------------------------------------------
        # Step 3: Drain pending hires arriving this step — charge c_hire now
        # and materialize new workers in f_workforce (extends theta/wage SoA)
        # ------------------------------------------------------------------
        n_arr = sum(n for (per, n) in pending if per <= step_t)
        if n_arr > 0:
            n_new = min(n_arr, K_max_orig - f_workforce.K)
            if n_new > 0:
                path_pi -= params.c_hire * n_new
                new_w = sample_workforce(n_new, params, plan_rng, current_t=step_t)
                combined_aip = (
                    np.concatenate([f_workforce.a_training_in_progress, new_w.a_training_in_progress])
                    if f_workforce.a_training_in_progress is not None and new_w.a_training_in_progress is not None
                    else None
                )
                f_workforce = Workforce(
                    theta=np.concatenate([f_workforce.theta, new_w.theta]),
                    wage=np.concatenate([f_workforce.wage, new_w.wage]),
                    a_trained=np.concatenate([f_workforce.a_trained, new_w.a_trained]),
                    tenure=np.concatenate([f_workforce.tenure, new_w.tenure]),
                    hire_t=np.concatenate([f_workforce.hire_t, new_w.hire_t]),
                    cum_wage=np.concatenate([f_workforce.cum_wage, new_w.cum_wage]),
                    a_training_in_progress=combined_aip,
                )
        pending = [(per, n) for (per, n) in pending if per > step_t]  # keep only future

        # ------------------------------------------------------------------
        # Step 4: Apply mode change (promote top n_aug H tasks by beta_hat → A)
        # ------------------------------------------------------------------
        f_modes = _apply_action_to_modes(f_modes, alpha_hat, beta_hat, action.n_aug, params)

        # ------------------------------------------------------------------
        # Step 5: Compute period productivity with TRUE theta — closes F-02
        # ------------------------------------------------------------------
        # Clamp modes to capacity after firings (mirrors kernel Step 2.5).
        # K may have shrunk due to firings while f_modes still has more H/A tasks
        # than K * tasks_per_worker. task_to_worker_map asserts n_HA <= K * tpw.
        n_HA_fwd = int(((f_modes == int(Mode.H)) | (f_modes == int(Mode.A))).sum())
        capacity_fwd = f_workforce.K * params.tasks_per_worker
        if n_HA_fwd > capacity_fwd:
            ha_idx_fwd = np.where((f_modes == int(Mode.H)) | (f_modes == int(Mode.A)))[0]
            demote_fwd = ha_idx_fwd[-(n_HA_fwd - capacity_fwd):]
            f_modes = f_modes.copy()
            f_modes[demote_fwd] = int(Mode.T)

        t2w_proj = task_to_worker_map(f_modes, f_workforce.K, params.tasks_per_worker)
        if f_workforce.K > 0:
            theta_per_task = np.where(
                t2w_proj >= 0,
                f_workforce.theta[np.where(t2w_proj >= 0, t2w_proj, 0)],
                1.0,
            )
        else:
            theta_per_task = np.ones(params.N, dtype=np.float64)

        prod = productivity_vec(
            f_modes, alpha=alpha_hat, beta=beta_hat, params=params,
            theta_per_task=theta_per_task,
        )

        # ------------------------------------------------------------------
        # Step 6: Cost and profit
        # Wage bill: kernel-symmetric (D-03). Step 0 firings above already updated
        # f_workforce before we reach this line — f_workforce.K is post-fire here,
        # identical ordering to the kernel (kernel Step 0 fires BEFORE Step 8).
        # Under finite T_review: pay ALL K workers (employment liability).
        # Under T_review=inf: assigned-only wage bill (byte-identical to pre-fix).
        # ------------------------------------------------------------------
        if math.isfinite(params.T_review):
            _wage_f = float(f_workforce.wage.sum()) if f_workforce.K > 0 else 0.0
        else:
            _n_HA = int(((f_modes == int(Mode.H)) | (f_modes == int(Mode.A))).sum())
            _K_active = min(_n_HA // params.tasks_per_worker, f_workforce.K)
            _wage_f = float(f_workforce.wage[:_K_active].sum()) if _K_active > 0 else 0.0

        from firm_ai_abm.adjustment import adj_cost
        cost = (
            cost_vec(f_modes, alpha=alpha_hat, params=params).sum()
            + _wage_f
            + params.F
        )
        path_pi += (
            params.p * prod.sum() - cost
            - adj_cost(prev_modes, f_modes, params, workforce=None)
        )

        # ------------------------------------------------------------------
        # Step 7: Append synthetic rows for the next step's replay
        # ------------------------------------------------------------------
        opw_synthetic = np.full(K_max_orig, np.nan, dtype=np.float64)
        active_w = np.unique(t2w_proj[t2w_proj >= 0])
        for w in active_w:
            if w < K_max_orig:
                opw_synthetic[w] = float(prod[t2w_proj == w].sum())
        opw_rows.append(opw_synthetic)

        cost_per_task_aug = cost_vec(f_modes, alpha=alpha_hat, params=params)
        acpw_synthetic = np.full(K_max_orig, np.nan, dtype=np.float64)
        for w in active_w:
            if w < K_max_orig:
                worker_mask = (t2w_proj == w)
                acpw_synthetic[w] = float(cost_per_task_aug[worker_mask].sum())
        acpw_rows.append(acpw_synthetic)

    return path_pi
