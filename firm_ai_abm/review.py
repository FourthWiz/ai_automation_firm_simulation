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

decision-thirteen (adaptive-firing-surplus): effective_surplus = p * mean_output - wage
  - mean_aug_cost - F/K_review. The four-term full-cost surplus:
    (1) revenue: params.p * mean_output[k] — price-scaled mean output over the review window
    (2) wage:    workforce.wage[k]        — per-worker wage charge
    (3) aug:     mean_aug_cost[k]         — mean per-worker augmentation tool cost over the
                                            same review window (zero for H-mode, c_auto excluded
                                            as firm-level overhead handled by F_share)
    (4) overhead: F / K_review            — each worker carries a share of the fixed overhead F;
                                            K_review = workforce.K captured BEFORE any firing mask
                                            (stale-K cascade damping, D-01 below).
  Modeling claim: the firm makes its firing decision using allocated full-cost surplus. F is a
  firm-level lump sum in P&L; F_share enters only this decision rule (not the cost ledger).
  This is the "full-cost accounting heuristic" and is what makes the coordination-failure
  mechanism (CLAUDE.md pre-committed finding #1) work: each worker is judged against an
  overhead allocation that grows as headcount shrinks, even though the firm's actual overhead
  is unchanged.
"""
from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np

from firm_ai_abm.workers import Workforce, sample_workforce

if TYPE_CHECKING:
    from firm_ai_abm.config import FirmParams
    from firm_ai_abm.firm import Firm


def firing_review(
    workforce: Workforce,
    t: int,
    output_per_worker: np.ndarray,      # shape (T_max, K_max), NaN-filled for inactive
    aug_cost_per_worker: np.ndarray,    # shape (T_max, K_max), NaN-filled for inactive
    params: "FirmParams",
) -> tuple[np.ndarray, float]:
    """Compute which workers to fire based on mean full-cost surplus over the trailing window.

    Returns (fire_indices, c_train_lost_metric).

    c_train_lost is a diagnostic metric (D-08, D-11): it records the value of
    trained capital destroyed by the firings. It is NOT charged into pi.

    Surplus formula (decision-thirteen, four terms):
        effective_surplus[k] = params.p * mean_output[k]
                                - workforce.wage[k]
                                - mean_aug_cost[k]
                                - F / K_review
    where K_review = workforce.K captured BEFORE the fire mask (stale-K, D-01).

    Args:
        workforce: Current Workforce instance (read-only; not mutated).
        t: Current period index (0-based).
        output_per_worker: shape (T_max, K_max), float64. Entry [s, k] is the
            summed output of worker k at period s, or NaN if inactive.
        aug_cost_per_worker: shape (T_max, K_max), float64. Entry [s, k] is the
            per-worker augmentation cost at period s (0.0 for H-mode, 0.0 during
            training delay, NaN for inactive). Ignored when T_review=inf.
        params: FirmParams with T_review, firing_threshold, F fields.

    Returns:
        (fire_indices, c_train_lost_metric) where:
          - fire_indices: np.ndarray of int, indices of workers to fire (may be empty).
          - c_train_lost_metric: float, value of trained capital lost (metric only).
        Returns (empty array, 0.0) when T_review=inf, t=0, or t is not a review period.
    """
    # D-11: math.isinf short-circuit — FIRST, before any state read (load-bearing:
    # int(math.inf) raises OverflowError; this guard must remain the first check)
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

    # D-01 (cascade damping): K_review captures workforce.K ONCE at the start of this
    # review evaluation, BEFORE any fire_mask is computed. F_share = F / K_review uses
    # this stale K for the entire review tick. Even if N workers are fired in this tick,
    # F_share does not re-balloon mid-tick — preventing per-tick recursive cascade.
    #
    # Cross-period cascades remain possible by design: at the NEXT review tick,
    # K_review will be smaller (post-firing), F_share will be larger, and another
    # round of firings can occur. This is the coordination-failure mechanism from
    # CLAUDE.md pre-committed finding #1 and is intentional for Phase 1.5 research.
    #
    # Modeling claim (full-cost accounting): the firm makes its firing decision using
    # allocated full-cost surplus — revenue minus marginal labor cost minus per-worker
    # share of fixed overhead. The P&L records F as a firm-level lump sum; F_share
    # enters only this decision rule and never the cost ledger. This mirrors standard
    # managerial cost accounting and is what makes the coordination-failure mechanism
    # work: each worker is judged against an overhead allocation that grows as headcount
    # shrinks, even though the firm's actual overhead is unchanged.
    K_review = workforce.K
    F_share = (params.F / K_review) if K_review > 0 else 0.0

    # mean_aug_cost: same col_has_data mask as output (D-06: workers fill aug_cost and
    # output in the same Step 6 loop — NaN positions are identical by construction)
    window_aug = aug_cost_per_worker[lo:t, : workforce.K]
    mean_aug_cost = np.full(workforce.K, np.nan, dtype=np.float64)
    if col_has_data.any():
        mean_aug_cost[col_has_data] = np.nanmean(window_aug[:, col_has_data], axis=0)

    surplus = np.full(workforce.K, np.nan, dtype=np.float64)
    surplus[col_has_data] = (
        params.p * mean_output[col_has_data]
        - workforce.wage[col_has_data]
        - mean_aug_cost[col_has_data]
        - F_share
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
    aug_cost_per_worker: np.ndarray,
) -> tuple[Workforce, np.ndarray, np.ndarray]:
    """Remove fired workers; K SHRINKS by len(fire_indices). No auto-replacement.

    Returns (new_workforce, new_output_per_worker, new_aug_cost_per_worker). When
    fire_indices is empty, returns SAME objects (identity, not copies) — callers
    can rely on this for no-op fast-path detection.

    Stage 5 D-03: apply_firings_and_replace is replaced by this drop-only function.
    run_simulation calls only apply_firings; hire-back is opt-in via replace_to_target.

    After firing, output_per_worker[:, new_wf.K : K_max] and
    aug_cost_per_worker[:, new_wf.K : K_max] are all-NaN (inactive slots).

    Args:
        firm: Firm instance (reads firm.workforce).
        fire_indices: np.ndarray of int, worker indices to fire.
        t: Current period (unused here; present for API symmetry with replace_to_target).
        output_per_worker: shape (T_max, K_max), float64. Not mutated; new array returned.
        aug_cost_per_worker: shape (T_max, K_max), float64. Not mutated; new array returned.

    Returns:
        (new_workforce, new_output_per_worker, new_aug_cost_per_worker);
        new_workforce.K == wf.K - len(fire_indices).
    """
    # Stage 5 D-03: no-op returns SAME objects (identity — not copies)
    if fire_indices.size == 0:
        return firm.workforce, output_per_worker, aug_cost_per_worker

    wf = firm.workforce

    # Snapshot fired workers' cumulative wages BEFORE shrinking the SoA arrays.
    firm.closed_worker_wages.extend(wf.cum_wage[fire_indices].tolist())

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
        cum_wage=wf.cum_wage[keep_mask].copy(),
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
        cum_wage=surv.cum_wage[order],
        a_training_in_progress=new_aip,
    )

    # Drop fired columns; survivor active columns reordered; trailing slots → NaN
    # Apply identical reindex to both output and aug_cost arrays (D-06 alignment)
    active_cols = output_per_worker[:, : wf.K]
    surv_cols = active_cols[:, keep_mask]
    reordered = surv_cols[:, order]
    new_opw = np.full_like(output_per_worker, np.nan)
    new_opw[:, : new_wf.K] = reordered

    aug_active_cols = aug_cost_per_worker[:, : wf.K]
    aug_surv_cols = aug_active_cols[:, keep_mask]
    aug_reordered = aug_surv_cols[:, order]
    new_acpw = np.full_like(aug_cost_per_worker, np.nan)
    new_acpw[:, : new_wf.K] = aug_reordered

    return new_wf, new_opw, new_acpw


def replace_to_target(
    firm: "Firm",
    K_target: int,
    t: int,
    output_per_worker: np.ndarray,
    aug_cost_per_worker: np.ndarray,
) -> tuple[Workforce, np.ndarray, np.ndarray]:
    """Opt-in hire-back: append K_target - wf.K replacements. No-op if wf.K >= K_target.

    Call AFTER apply_firings if hire-back is desired. run_simulation does NOT call this.
    Phase 2 implements strategy-driven hiring via this function.

    Returns (new_workforce, new_output_per_worker, new_aug_cost_per_worker). New
    replacement workers get NaN-initialized columns in both arrays (no history yet).
    When wf.K >= K_target, returns SAME objects (identity — no-op).

    Args:
        firm: Firm instance (reads firm.workforce, firm.params, firm.rng).
        K_target: Target workforce headcount after replacement.
        t: Current period (passed to sample_workforce as hire_t for new workers).
        output_per_worker: shape (T_max, K_max), float64. Not mutated; new array returned.
        aug_cost_per_worker: shape (T_max, K_max), float64. Not mutated; new array returned.

    Returns:
        (new_workforce, new_output_per_worker, new_aug_cost_per_worker);
        new_workforce.K == K_target.
    """
    wf = firm.workforce
    if wf.K >= K_target:
        return wf, output_per_worker, aug_cost_per_worker

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
        # New hires (repl) get cum_wage=0 via Workforce.__post_init__; concatenate preserves that
        cum_wage=np.concatenate([wf.cum_wage, repl.cum_wage]),
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
        cum_wage=combined.cum_wage[order],
        a_training_in_progress=new_aip,
    )

    # Append NaN columns for replacement workers; reindex both arrays identically (D-06)
    nan_cols = np.full((output_per_worker.shape[0], n_repl), np.nan, dtype=np.float64)

    active_cols = output_per_worker[:, : wf.K]
    combined_active = np.concatenate([active_cols, nan_cols], axis=1)
    reordered = combined_active[:, order]
    new_opw = np.full_like(output_per_worker, np.nan)
    new_opw[:, : K_target] = reordered

    aug_active_cols = aug_cost_per_worker[:, : wf.K]
    aug_combined_active = np.concatenate([aug_active_cols, nan_cols], axis=1)
    aug_reordered = aug_combined_active[:, order]
    new_acpw = np.full_like(aug_cost_per_worker, np.nan)
    new_acpw[:, : K_target] = aug_reordered

    return new_wf, new_opw, new_acpw


def replenish_hire_step(
    firm: "Firm",
    t: int,
    output_per_worker: np.ndarray,
    aug_cost_per_worker: np.ndarray,
) -> "tuple[Workforce, np.ndarray, np.ndarray, int, float]":
    """Drain due entries from firm.pending_hires and hire back workers via replace_to_target.

    Called from simulate.run_horizon Step 0.5b when enable_replenish_hiring=True.
    REUSES replace_to_target — no parallel hiring code path.

    Returns:
        (new_workforce, new_output_per_worker, new_aug_cost_per_worker, n_hired, hire_cost)
    When no hires occur (empty backlog, no due entries, or K >= K_max), returns the input
    objects with SAME identity (mirrors apply_firings no-op semantics) and n_hired=0, hire_cost=0.0.

    FIFO drain (D-06): oldest tuple consumed first. Global per-period cap (D-07): max_hire_period
    is applied to the total due count, not per-firing. K_max floor (D-09): never exceeds physical
    capacity N // tasks_per_worker.
    """
    wf = firm.workforce
    K_max = firm.params.N // firm.params.tasks_per_worker

    if not firm.pending_hires:
        return wf, output_per_worker, aug_cost_per_worker, 0, 0.0
    if wf.K >= K_max:
        return wf, output_per_worker, aug_cost_per_worker, 0, 0.0

    # Sum all due entries (period_eligible <= t)
    total_due = sum(n for (period_eligible, n) in firm.pending_hires if period_eligible <= t)
    if total_due == 0:
        return wf, output_per_worker, aug_cost_per_worker, 0, 0.0

    # Apply K_max capacity floor (D-09)
    capacity_remaining = K_max - wf.K
    total_due = min(total_due, capacity_remaining)

    # Apply per-period cap: 0 means no cap (drain all)
    if firm.params.max_hire_period == 0:
        n_hire = total_due
    else:
        n_hire = min(total_due, firm.params.max_hire_period)

    # FIFO drain (D-06): consume oldest tuples first
    remaining_to_drain = n_hire
    new_backlog = []
    for (period_eligible, n) in firm.pending_hires:
        if period_eligible > t or remaining_to_drain == 0:
            new_backlog.append((period_eligible, n))
            continue
        if n <= remaining_to_drain:
            remaining_to_drain -= n
        else:
            new_backlog.append((period_eligible, n - remaining_to_drain))
            remaining_to_drain = 0
    firm.pending_hires = new_backlog

    # Delegate to existing replace_to_target (REUSES Stage 6 hiring helper)
    new_wf, new_opw, new_acpw = replace_to_target(
        firm, wf.K + n_hire, t, output_per_worker, aug_cost_per_worker
    )
    hire_cost = float(firm.params.c_hire) * n_hire
    return new_wf, new_opw, new_acpw, n_hire, hire_cost


def optimal_hire_target(
    firm: "Firm",
    t: int,
    output_per_worker: np.ndarray,
    aug_cost_per_worker: np.ndarray,
    params: "FirmParams",
) -> int:
    """Compute optimal hire target K* for the opt-in hire-back path.

    Returns the smallest integer K such that the expected surplus of an average
    new hire at headcount K meets firing_threshold, capped at the physical
    capacity K_max = N // tasks_per_worker, and floored at the current K
    (never fires via this path — only hires up).

    Expected-surplus model for a fresh draw at headcount K:
        E[surplus(K)] = p * E[output] - E[wage] - E[aug_cost] - F / K
    where:
        E[output]   = nanmean of survivors' output_per_worker over trailing window
        E[wage]     = params.w (population mean; new hires are fresh draws)
        E[aug_cost] = nanmean of survivors' aug_cost_per_worker over trailing window
                      (0.0 when no valid entries)

    Setting E[surplus(K*)] = firing_threshold and solving:
        K* = F / (p * E[output] - E[wage] - E[aug_cost] - firing_threshold)

    Fallback to wf.K (no hire) when:
        - T_review is inf (hire path is dormant anyway; defensive guard)
        - t == 0 (no trailing data yet)
        - trailing window has no non-NaN output entries
        - denominator <= 0 (new hires unprofitable at any K)

    Args:
        firm: Firm instance (reads firm.workforce).
        t: Current period (0-based). Trailing window is [max(0, t - T_review), t).
        output_per_worker: shape (T_max, K_max), NaN for inactive slots.
        aug_cost_per_worker: shape (T_max, K_max), NaN for inactive slots.
        params: FirmParams (reads T_review, p, w, F, firing_threshold,
            N, tasks_per_worker).

    Returns:
        int K_target in [wf.K, K_max].
    """
    wf = firm.workforce
    K_max = params.N // params.tasks_per_worker

    # math.isinf guard MUST come first — int(math.inf) raises OverflowError (load-bearing)
    if math.isinf(params.T_review):
        return wf.K
    if t == 0:
        return wf.K

    T_review_int = int(params.T_review)
    lo = max(0, t - T_review_int)
    window_out = output_per_worker[lo:t, : wf.K]
    window_aug = aug_cost_per_worker[lo:t, : wf.K]

    if window_out.size == 0 or np.all(np.isnan(window_out)):
        return wf.K

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        e_output = float(np.nanmean(window_out))
        e_aug = 0.0 if np.all(np.isnan(window_aug)) else float(np.nanmean(window_aug))

    denom = params.p * e_output - float(params.w) - e_aug - float(params.firing_threshold)

    if denom <= 0.0:
        return wf.K

    K_star = math.ceil(params.F / denom)
    return max(wf.K, min(K_star, K_max))
