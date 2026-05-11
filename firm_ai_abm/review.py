"""Periodic firing review for Phase 1.5 Stage 3 / Stage 5.

Every T_review periods (default math.inf — opt-in), the firm computes per-worker
mean surplus over the trailing T_review periods and fires workers whose surplus
falls below firing_threshold.

Key decisions implemented here:

D-01 — Surplus window is rolling [t - T_review, t), NaN-aware per-column mask.
D-02 — output_per_worker[t, k] is the SUM of per-task productivity for worker k
        at period t. NaN for T-mode-only workers and pre-hire periods.
D-05 — Workforce re-indexing after fire preserves longest-tenure-first in the low
        slots (stable argsort by -tenure).
D-06 (Stage 3) — workforce.tenure incremented at step 11.5 for ALL workers.
        Replacements have tenure=0 immediately after apply_firings returns; they
        end the period with tenure=1 after the step-11.5 increment.
D-08 — Review timing is START-OF-PERIOD (step 0), BEFORE strategy(firm, t), at
        periods t where t > 0 AND t % T_review == 0. c_fire per fired worker is
        added to the period's C so that pi reflects the review cost.
D-09 (R-10) — Greedy gaming via deliberate T-mode switching to reduce a worker's
        window output is possible in principle but does not occur with myopic
        strategies.
D-10 — c_fire per review-fire is paid SEPARATELY from adj_cost's lumpy delta-K
        hire/fire path. adj_cost is not edited in Stage 3.
D-11 — T_review: float = math.inf default in FirmParams. The math.isinf
        short-circuit is the FIRST check in firing_review (before any state read).
        c_train_lost is a HISTORY METRIC ONLY — it is NOT charged into C/pi (D-08).

Stage 5 API change (D-03):
  apply_firings_and_replace is REMOVED.
  New API:
    apply_firings(firm, fire_indices, t, output_per_worker) — drop-only; K SHRINKS.
    replace_to_target(firm, K_target, t, output_per_worker) — opt-in; appends
        K_target - wf.K replacements. Call AFTER apply_firings if hire-back is
        desired. run_simulation calls ONLY apply_firings (no auto-replace).

decision-twelve (Stage 6, see firing-logic-and-wage-viz plan): surplus = p · mean_output − wage.
  Pre-fix the formula omitted the price factor p, making surplus dimensionally
  inconsistent (output units ≠ wage units when p ≠ 1). Post-fix:
    surplus[k] = params.p * mean_output[k] - workforce.wage[k]
  Numeraire invariance: under (w, c_*, F, p) × 2, surplus × 2, fire_mask unchanged.
  The kernel-side threshold scaling at app.py:281 (firing_threshold_kernel =
  firing_threshold_ui * tasks_per_worker) remains correct pre- and post-fix: the
  UI threshold is in per-task surplus units, and the kernel compares the per-worker
  surplus to threshold * tpw — dimensional consistency preserved on both sides.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from firm_ai_abm.workers import Workforce, sample_workforce

if TYPE_CHECKING:
    from firm_ai_abm.config import FirmParams
    from firm_ai_abm.firm import Firm


def firing_review(
    workforce: Workforce,
    t: int,
    output_per_worker: np.ndarray,  # shape (T_max, K_max), NaN-filled for inactive
    params: "FirmParams",
) -> tuple[np.ndarray, float]:
    """Compute which workers to fire based on mean surplus over the trailing window.

    Returns (fire_indices, c_train_lost_metric).

    c_train_lost is a diagnostic metric (D-08, D-11): it records the value of
    trained capital destroyed by the firings. It is NOT charged into pi.

    Args:
        workforce: Current Workforce instance (read-only; not mutated).
        t: Current period index (0-based).
        output_per_worker: shape (T_max, K_max), float64. Entry [s, k] is the
            summed output of worker k at period s, or NaN if the worker was
            inactive (T-mode only) or not yet hired at period s.
        params: FirmParams with T_review and firing_threshold fields.

    Returns:
        (fire_indices, c_train_lost_metric) where:
          - fire_indices: np.ndarray of int, indices of workers to fire (may be empty).
            surplus is price-scaled revenue per worker minus wage:
            surplus[k] = params.p * mean_output[k] - workforce.wage[k]
          - c_train_lost_metric: float, value of trained capital lost (metric only).
        Returns (empty array, 0.0) when T_review=inf, t=0, or t is not a review period.
    """
    # D-11: math.isinf short-circuit — FIRST, before any state read
    if math.isinf(params.T_review):
        return np.array([], dtype=int), 0.0

    T_review_int = int(params.T_review)
    if t == 0 or t % T_review_int != 0:
        return np.array([], dtype=int), 0.0

    # D-01: rolling window [t - T_review, t)
    lo = max(0, t - T_review_int)
    window = output_per_worker[lo:t, : workforce.K]  # shape (<= T_review, K)

    # Fast-path for empty window (should not occur in practice but guards edge cases)
    if window.size == 0:
        return np.array([], dtype=int), 0.0

    # CRIT-1 (round 3): per-column mask to avoid RuntimeWarning from np.nanmean
    # on all-NaN columns. np.nanmean emits RuntimeWarning: Mean of empty slice
    # when the input slice is entirely NaN — even if other columns have data.
    # The per-column mask restricts nanmean to columns that have at least one
    # non-NaN entry, completely suppressing the warning.
    col_has_data = ~np.all(np.isnan(window), axis=0)  # True for cols with >= 1 non-NaN
    mean_output = np.full(workforce.K, np.nan, dtype=np.float64)
    if col_has_data.any():
        mean_output[col_has_data] = np.nanmean(window[:, col_has_data], axis=0)
    # Workers with col_has_data[k]=False have mean_output[k]=NaN → excluded from
    # firing candidates (D-01 "insufficient evidence" — e.g., T-mode-only workers,
    # or workers hired between the prior review and this one with zero observations).

    surplus = np.full(workforce.K, np.nan, dtype=np.float64)
    surplus[col_has_data] = (
        params.p * mean_output[col_has_data] - workforce.wage[col_has_data]
    )

    fire_mask = (~np.isnan(surplus)) & (surplus < params.firing_threshold)
    fire_indices = np.where(fire_mask)[0].astype(int)

    # c_train_lost: diagnostic metric only — NOT charged into C/pi (D-08, D-11)
    c_train_lost = float(workforce.a_trained[fire_indices].sum()) * float(params.c_train)

    return fire_indices, c_train_lost


def apply_firings(
    firm: "Firm",
    fire_indices: np.ndarray,
    t: int,
    output_per_worker: np.ndarray,
) -> tuple[Workforce, np.ndarray]:
    """Remove fired workers; K SHRINKS by len(fire_indices). No auto-replacement.

    Returns (new_workforce, new_output_per_worker). When fire_indices is empty,
    returns SAME objects (identity, not copies) — callers can rely on this for
    no-op fast-path detection.

    Stage 5 D-03: apply_firings_and_replace is replaced by this drop-only function.
    run_simulation calls only apply_firings; hire-back is opt-in via replace_to_target.

    After firing, output_per_worker[:, new_wf.K : K_max] is all-NaN (inactive slots).

    Args:
        firm: Firm instance (reads firm.workforce).
        fire_indices: np.ndarray of int, worker indices to fire.
        t: Current period (unused here; present for API symmetry with replace_to_target).
        output_per_worker: shape (T_max, K_max), float64. Not mutated; new array returned.

    Returns:
        (new_workforce, new_output_per_worker); new_workforce.K == wf.K - len(fire_indices).
    """
    # Stage 5 D-03: no-op returns SAME objects (identity — not copies)
    if fire_indices.size == 0:
        return firm.workforce, output_per_worker

    wf = firm.workforce
    keep_mask = np.ones(wf.K, dtype=bool)
    keep_mask[fire_indices] = False

    surv_aip = (
        wf.a_training_in_progress[keep_mask].copy()
        if wf.a_training_in_progress is not None
        else None
    )
    surv = Workforce(
        theta=wf.theta[keep_mask].copy(),
        wage=wf.wage[keep_mask].copy(),
        a_trained=wf.a_trained[keep_mask].copy(),
        tenure=wf.tenure[keep_mask].copy(),
        hire_t=wf.hire_t[keep_mask].copy(),
        a_training_in_progress=surv_aip,
    )

    order = np.argsort(-surv.tenure, kind="stable")
    new_aip = surv.a_training_in_progress[order] if surv.a_training_in_progress is not None else None
    new_wf = Workforce(
        theta=surv.theta[order],
        wage=surv.wage[order],
        a_trained=surv.a_trained[order],
        tenure=surv.tenure[order],
        hire_t=surv.hire_t[order],
        a_training_in_progress=new_aip,
    )

    # Drop fired columns; survivor active columns reordered; trailing slots → NaN
    active_cols = output_per_worker[:, : wf.K]
    surv_cols = active_cols[:, keep_mask]
    reordered = surv_cols[:, order]
    new_opw = np.full_like(output_per_worker, np.nan)
    new_opw[:, : new_wf.K] = reordered

    return new_wf, new_opw


def replace_to_target(
    firm: "Firm",
    K_target: int,
    t: int,
    output_per_worker: np.ndarray,
) -> tuple[Workforce, np.ndarray]:
    """Opt-in hire-back: append K_target - wf.K replacements. No-op if wf.K >= K_target.

    Call AFTER apply_firings if hire-back is desired. run_simulation does NOT call this.
    Phase 2 implements strategy-driven hiring via this function.

    Returns (new_workforce, new_output_per_worker). New replacement workers get
    output_per_worker columns initialized to NaN (no history yet).

    Args:
        firm: Firm instance (reads firm.workforce, firm.params, firm.rng).
        K_target: Target workforce headcount after replacement.
        t: Current period (passed to sample_workforce as hire_t for new workers).
        output_per_worker: shape (T_max, K_max), float64. Not mutated; new array returned.

    Returns:
        (new_workforce, new_output_per_worker); new_workforce.K == K_target.
    """
    wf = firm.workforce
    if wf.K >= K_target:
        return wf, output_per_worker

    n_repl = K_target - wf.K
    repl = sample_workforce(n_repl, firm.params, firm.rng, current_t=t)

    combined_aip = (
        np.concatenate([wf.a_training_in_progress, repl.a_training_in_progress])
        if wf.a_training_in_progress is not None and repl.a_training_in_progress is not None
        else None
    )
    combined = Workforce(
        theta=np.concatenate([wf.theta, repl.theta]),
        wage=np.concatenate([wf.wage, repl.wage]),
        a_trained=np.concatenate([wf.a_trained, repl.a_trained]),
        tenure=np.concatenate([wf.tenure, repl.tenure]),
        hire_t=np.concatenate([wf.hire_t, repl.hire_t]),
        a_training_in_progress=combined_aip,
    )

    order = np.argsort(-combined.tenure, kind="stable")
    new_aip = combined.a_training_in_progress[order] if combined.a_training_in_progress is not None else None
    new_wf = Workforce(
        theta=combined.theta[order],
        wage=combined.wage[order],
        a_trained=combined.a_trained[order],
        tenure=combined.tenure[order],
        hire_t=combined.hire_t[order],
        a_training_in_progress=new_aip,
    )

    active_cols = output_per_worker[:, : wf.K]
    new_repl_cols = np.full(
        (output_per_worker.shape[0], n_repl), np.nan, dtype=np.float64
    )
    combined_active = np.concatenate([active_cols, new_repl_cols], axis=1)
    reordered = combined_active[:, order]
    new_opw = np.full_like(output_per_worker, np.nan)
    new_opw[:, : K_target] = reordered

    return new_wf, new_opw
