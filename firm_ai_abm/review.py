"""Periodic firing review for Phase 1.5 Stage 3.

Every T_review periods (default math.inf — opt-in), the firm computes per-worker
mean surplus over the trailing T_review periods and fires workers whose surplus
falls below firing_threshold.

Key decisions implemented here:

D-01 — Surplus window is rolling [t - T_review, t), NaN-aware per-column mask.
D-02 — output_per_worker[t, k] is the SUM of per-task productivity for worker k
        at period t. NaN for T-mode-only workers and pre-hire periods.
D-04 — Replacement hire happens AFTER firing on the SAME period's tail (within
        step 0.5 in run_simulation), NOT next period.
D-05 — Workforce re-indexing after fire+replacement preserves longest-tenure-first
        in the low slots (stable argsort by -tenure).
D-06 — workforce.tenure incremented at step 11.5 for ALL workers (including
        replacements hired in the same period). Replacements have tenure=0
        immediately after apply_firings_and_replace returns; they end the period
        with tenure=1 after the step-11.5 increment.
D-08 — Review timing is START-OF-PERIOD (step 0), BEFORE strategy(firm, t), at
        periods t where t > 0 AND t % T_review == 0. c_fire per fired worker is
        added to the period's C so that pi reflects the review cost.
D-09 (R-10) — Greedy gaming via deliberate T-mode switching to reduce a worker's
        window output is possible in principle but does not occur with myopic
        strategies. Coincidental gaming (greedy's optimum coincidentally aligning
        with wage-shedding) is empirically ruled out by smoke test T-17
        (greedy_with_switching vs greedy_profit firing count comparison). Lookahead
        strategies (Phase 2+) may need mitigation; Stage 3 does not preempt.
D-10 — c_fire per review-fire is paid SEPARATELY from adj_cost's lumpy delta-K
        hire/fire path. adj_cost is not edited in Stage 3.
D-11 — T_review: float = math.inf default in FirmParams. The math.isinf
        short-circuit is the FIRST check in firing_review (before any state read).
        c_train_lost is a HISTORY METRIC ONLY — it is NOT charged into C/pi (D-08).
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
    surplus[col_has_data] = mean_output[col_has_data] - workforce.wage[col_has_data]

    fire_mask = (~np.isnan(surplus)) & (surplus < params.firing_threshold)
    fire_indices = np.where(fire_mask)[0].astype(int)

    # c_train_lost: diagnostic metric only — NOT charged into C/pi (D-08, D-11)
    c_train_lost = float(workforce.a_trained[fire_indices].sum()) * float(params.c_train)

    return fire_indices, c_train_lost


def apply_firings_and_replace(
    firm: "Firm",
    fire_indices: np.ndarray,
    t: int,
    output_per_worker: np.ndarray,
) -> tuple[Workforce, np.ndarray]:
    """Remove fired workers, hire replacements, re-index by descending tenure.

    Returns (new_workforce, new_output_per_worker). Does NOT mutate anything in
    the caller's scope — returns new objects (Workforce is a dataclass; arrays
    are new via concatenate / indexing).

    K-target invariant: new_workforce.K == firm.workforce.K (original K before firing).

    D-05 tenure-reindex: all workforce arrays are permuted by argsort(-tenure,
    stable) so survivors stay at low indices (long tenure) and replacements
    (tenure=0) move to high indices. The same permutation is applied to
    output_per_worker columns for consistency.

    D-03 column-reuse spec (CRIT-1 revised):
      - Fired-worker columns are DROPPED via keep_mask (not reused or wiped in-place).
      - Fresh all-NaN columns are appended for each replacement hire.
      - The combined matrix (survivor columns + NaN replacement columns) is then
        permuted by the same `order` array as the Workforce.
      - After this call, output_per_worker[:, k] is NaN for ALL rows before
        workforce.hire_t[k] for every k (pre-hire rows are guaranteed NaN).

    Args:
        firm: Firm instance (reads firm.workforce, firm.params, firm.rng).
        fire_indices: np.ndarray of int, worker indices to remove (from firing_review).
        t: Current period (passed to sample_workforce as current_t for hire_t).
        output_per_worker: shape (T_max, K_max), float64. Will NOT be mutated;
            a new array is returned.

    Returns:
        (new_workforce, new_output_per_worker) with K_target restored.
    """
    if fire_indices.size == 0:
        return firm.workforce, output_per_worker

    wf = firm.workforce
    keep_mask = np.ones(wf.K, dtype=bool)
    keep_mask[fire_indices] = False

    # 1. Survivors (those not fired)
    surv = Workforce(
        theta=wf.theta[keep_mask].copy(),
        wage=wf.wage[keep_mask].copy(),
        a_trained=wf.a_trained[keep_mask].copy(),
        tenure=wf.tenure[keep_mask].copy(),
        hire_t=wf.hire_t[keep_mask].copy(),
    )

    # 2. Replacements (consumes firm.rng — D-06 RNG decision)
    n_repl = int(fire_indices.size)
    repl = sample_workforce(n_repl, firm.params, firm.rng, current_t=t)

    # 3. Concatenate survivors and replacements
    combined = Workforce(
        theta=np.concatenate([surv.theta, repl.theta]),
        wage=np.concatenate([surv.wage, repl.wage]),
        a_trained=np.concatenate([surv.a_trained, repl.a_trained]),
        tenure=np.concatenate([surv.tenure, repl.tenure]),
        hire_t=np.concatenate([surv.hire_t, repl.hire_t]),
    )

    # 4. Re-index by descending tenure (D-05): survivors (high tenure) → low slots,
    #    replacements (tenure=0) → high slots. Stable sort preserves relative order
    #    among workers with equal tenure (e.g., multiple replacements all at tenure=0).
    order = np.argsort(-combined.tenure, kind="stable")
    new_wf = Workforce(
        theta=combined.theta[order],
        wage=combined.wage[order],
        a_trained=combined.a_trained[order],
        tenure=combined.tenure[order],
        hire_t=combined.hire_t[order],
    )

    # 5. Apply the same column permutation to output_per_worker:
    #    - output_per_worker has K_max columns (pre-allocated at simulation start).
    #      We only operate on the first wf.K columns (the active workforce slots).
    #    - Survivor columns: keep_mask indexes into [0 : wf.K] (active workforce only)
    #    - Replacement columns: all-NaN (replacements have no history)
    #    - Apply the same `order` permutation to columns
    #    - K_max - wf.K trailing columns (never-used slots) are preserved as-is (all NaN)
    active_cols = output_per_worker[:, : wf.K]  # shape (T_max, wf.K) — active worker columns
    surv_cols = active_cols[:, keep_mask]        # shape (T_max, n_surv)
    new_repl_cols = np.full(
        (output_per_worker.shape[0], n_repl), np.nan, dtype=np.float64
    )
    combined_active = np.concatenate([surv_cols, new_repl_cols], axis=1)  # (T_max, wf.K)
    reordered_active = combined_active[:, order]                           # (T_max, wf.K)
    # Rebuild the full K_max matrix: reordered active columns + trailing all-NaN padding
    K_max = output_per_worker.shape[1]
    new_opw = np.full_like(output_per_worker, np.nan)
    new_opw[:, : wf.K] = reordered_active

    return new_wf, new_opw
