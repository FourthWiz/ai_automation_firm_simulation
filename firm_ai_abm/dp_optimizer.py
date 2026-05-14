"""Rolling-horizon DP optimizer for FirmBehavior (F-03).

Implements a rolling horizon DP over a 5×5 (fire%, aug%) action grid.
Max lookahead depth: 5 steps. Worst-case path count: 25^5 = 9,765,625 (all 5
steps are review periods); typical dashboard (T_review=5, horizon=3): 125–625.

Posteriors are run-state: alpha_hat[i] and beta_hat[i] are updated in-place
as tasks are observed running in modes T and A respectively. They are cleared
to params.dp_prior_alpha / params.dp_prior_beta by Firm.reset() on each fresh
run_simulation call.

Forward simulation: _forward_simulate is a thin adapter over the shared
forward_simulate_action_path from firm_ai_abm.forward_sim. It translates
the DP's (n_fire, n_aug) 2-tuples into Action(n_fire, n_aug, n_hire=0) objects
and delegates to the canonical implementation (unify-strategy-controls, T-02).

Intent protocol: dp_rolling_horizon_strategy writes firm._fire_intent (via the
_dp_optimizer_n_fire property alias) at the end of each call — consumed by
simulate.py Step 0 of the NEXT period. The alias is defined in firm.py.

Usage:
    from firm_ai_abm.dp_optimizer import dp_rolling_horizon_strategy
    df = run_simulation(firm, dp_rolling_horizon_strategy)

CAUTION: posteriors are run-state. External callers that drive the strategy
directly without calling firm.reset() first inherit stale state — consistent
with how greedy strategies treat firm.modes.
"""
import math

import numpy as np

from firm_ai_abm.config import FirmParams
from firm_ai_abm.production import Mode, productivity_vec, cost_vec
from firm_ai_abm.adjustment import adj_cost

# ---------------------------------------------------------------------------
# Module-level constants (single source of truth)
# ---------------------------------------------------------------------------

_DP_HORIZON_MAX: int = 5
"""Hard cap on lookahead depth (D-03). UI margin_horizon slider is clamped to this.
Also applies to horizon_brute_strategy in margin_optimizer.py (T-01 — capped via
`min(_DP_HORIZON_MAX, firm.params.margin_horizon)` at planning time)."""

_DP_GRID_LEVELS: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
"""5 fractional levels for both fire% and aug% at each planning step."""

_DP_PRIOR_MEAN: float = 0.5
"""Bayesian prior mean fallback (used only when firm.params is unavailable — unreachable in production).
Prefer firm.params.dp_prior_alpha / dp_prior_beta — this constant exists solely for backward compat.
Value 0.5 matches the new dp_prior_alpha default; dp_prior_beta defaults to 0.7 in production."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dp_rolling_horizon_strategy(firm, t: int) -> np.ndarray:
    """Rolling-horizon DP optimizer with Bayesian per-task alpha/beta learning.

    Plans min(_DP_HORIZON_MAX, firm.params.margin_horizon) steps ahead,
    enumerates all paths (worst case 25^3 = 15,625), forward-simulates each
    under posterior beliefs, picks the path with the highest cumulative profit,
    and EXECUTES STEP 1 only (modes returned for this period).

    Bayesian update of alpha_hat/beta_hat happens AS A SIDE EFFECT before
    planning: any task currently in mode T has its alpha_hat[i] set to the
    true alpha[i]; any task in mode A has beta_hat[i] set to true beta[i].

    Calendar-aware: planning only places n_fire > 0 on steps that land on
    review periods (t' where t' % T_review == 0 and t' > 0); n_aug is
    unconstrained per D-02. When T_review is math.inf, n_fire is fixed at 0
    for all planning steps.

    Two-source firing merge: the optimizer's n_fire-step-1 decision is
    RECORDED on firm._dp_optimizer_n_fire for the kernel (firing_review in
    simulate.py Step 0) to consume via the two-source UNION merge (T-10).

    Args:
        firm: Live Firm instance at period t (must be constructed via make_firm).
        t: Current simulation period.

    Returns:
        Fresh np.ndarray of shape (N,), dtype int — the new modes array
        produced by the best step-1 (n_fire, n_aug) action.

    Raises:
        ValueError: if firm was constructed directly (not via make_firm), i.e.,
            firm.workforce is None or firm.modes is None. Fails before any
            AttributeError to surface the misuse clearly (CRIT-10).
    """
    # CRIT-10 fail-loud guard: workforce and modes MUST be initialized.
    # Dataclass defaults (firm.py) leave both as None when Firm is constructed
    # directly bypassing make_firm. Fail BEFORE _update_posteriors or K=0
    # short-circuit to surface the API contract immediately.
    if firm.workforce is None or firm.modes is None:
        raise ValueError(
            "dp_rolling_horizon_strategy requires a Firm constructed via make_firm "
            "(workforce and modes must be initialized); direct Firm(params=..., "
            "alpha=..., beta=...) construction is unsupported. Got "
            f"workforce={firm.workforce!r}, modes={firm.modes!r}."
        )

    # MAJ-8 / CRIT-10: defensive init for posterior arrays only (well-defined
    # initial value = params.dp_prior_alpha / dp_prior_beta; mirrors what reset() does).
    if firm.alpha_hat is None:
        firm.alpha_hat = np.full(firm.params.N, firm.params.dp_prior_alpha, dtype=np.float64)
    if firm.beta_hat is None:
        firm.beta_hat = np.full(firm.params.N, firm.params.dp_prior_beta, dtype=np.float64)

    # Bayesian update: observe true alpha/beta for tasks in T/A mode respectively.
    _update_posteriors(firm)

    # Edge case: K=0 (D-09; byte-symmetric to simulate.py:188–196 kernel clamp).
    # Safe to read firm.workforce.K here — fail-loud guard above proved workforce is not None.
    if firm.workforce.K == 0:
        firm._dp_optimizer_n_fire = 0  # type: ignore[attr-defined]
        return np.full(firm.params.N, int(Mode.T), dtype=int)

    horizon = min(_DP_HORIZON_MAX, firm.params.margin_horizon)

    # Enumerate all action paths depth-first (flat iteration; CRIT-3 per-step recompute).
    # Per MAJ-13: _forward_simulate does its own per-path deepcopy — one per leaf.
    all_paths = _build_all_paths(firm, t, horizon, firm.modes.copy(), firm.workforce.K)

    best_pi = -math.inf
    best_path = all_paths[0]  # fallback (all_paths non-empty because K>0 guard passed)

    for path in all_paths:
        pi = _forward_simulate(firm, t, path, horizon)
        if pi > best_pi:
            best_pi = pi
            best_path = path

    n_fire_step1, n_aug_step1 = best_path[0]
    # Compute modes after step-1 action (reuse _apply_action_to_modes)
    modes_after_step1 = _apply_action_to_modes(
        firm.modes.copy(), firm.alpha_hat, firm.beta_hat, n_aug_step1, firm.params
    )

    # Record optimizer's fire-hint for the kernel's two-source merge (T-10).
    # MAJ-15 calendar-alignment fix: the hint written HERE at period t is
    # CONSUMED at period t+1's Step 0 (simulate.py runs firing_review BEFORE
    # the strategy call). Therefore the gate checks whether period t+1 is a
    # review period, NOT period t.
    firm._dp_optimizer_n_fire = (  # type: ignore[attr-defined]
        n_fire_step1 if _is_review_period(t + 1, firm.params.T_review) else 0
    )

    # Return step-1 mode projection. firm.modes is unchanged (caller asserts
    # at simulate.py L180-182).
    return modes_after_step1


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _update_posteriors(firm) -> None:
    """Update per-task posterior means in-place from current firm.modes.

    Called as the FIRST action in dp_rolling_horizon_strategy (after guards).
    Observes the CURRENT period's modes (installed in the prior period's
    strategy call). At t=0 all modes are H (from reset) — no observations.
    """
    t_mask = (firm.modes == int(Mode.T))
    firm.alpha_hat[t_mask] = firm.alpha[t_mask]
    a_mask = (firm.modes == int(Mode.A))
    firm.beta_hat[a_mask] = firm.beta[a_mask]


def _is_review_period(t: int, T_review: float) -> bool:
    """Return True iff period t is a firing-review period.

    Mirrors the kernel's firing_review entry condition at review.py:121.
    The math.isinf guard MUST appear before any integer cast — calling
    int(math.inf) raises OverflowError (lesson 2026-05-09).
    """
    if math.isinf(T_review):
        return False
    # Fail loud if T_review is fractional (future-proof against config drift).
    assert T_review == int(T_review), (
        f"T_review must be an integer or math.inf; got {T_review!r}"
    )
    return t > 0 and (t % int(T_review)) == 0


def _apply_action_to_modes(
    modes_in: np.ndarray,
    alpha_hat: np.ndarray,
    beta_hat: np.ndarray,
    n_aug: int,
    params: "FirmParams",
) -> np.ndarray:
    """Apply (n_aug) H→A promotion rule to modes_in; return new modes array.

    Selects the top n_aug H-mode tasks ranked by beta_hat[i] DESCENDING
    (tasks expected to benefit most from augmentation go first). T-mode tasks
    are untouched. n_aug is a TASK count (D-08 — NOT a worker count).

    If n_aug > number of H tasks, clamp to available H tasks (D-09 edge case).
    """
    modes = modes_in.copy()
    h_indices = np.where(modes == int(Mode.H))[0]
    n_available = len(h_indices)
    if n_available == 0 or n_aug == 0:
        return modes

    # Clamp to available H tasks (D-09 edge case)
    n_promote = min(n_aug, n_available)

    # Rank H-mode tasks by beta_hat DESCENDING; promote top n_promote
    h_beta = beta_hat[h_indices]
    ranked = np.argsort(-h_beta, kind="stable")  # descending
    promote_idx = h_indices[ranked[:n_promote]]
    modes[promote_idx] = int(Mode.A)
    return modes


def _sort_fireable_workers_by_cost_effectiveness(
    firm,
    output_per_worker: np.ndarray,
    wages: np.ndarray,
    t: int,
) -> np.ndarray:
    """Return worker indices sorted ascending by output/wage ratio (least cost-effective first).

    MAJ-14 (round-3) shape contract: takes the FULL 2D output history array
    (shape (T_max, K_max)), NOT a pre-sliced 1D mean. Does its own windowing
    internally via params.T_review to bound the lookback window.

    Args:
        firm: Firm instance (reads firm.params.T_review, firm.workforce.K,
              and firm.modes to identify fireable workers).
        output_per_worker: Full 2D array, shape (T_max, K_max), dtype float.
            NaN-filled for inactive slots. Same array simulate.py carries through
            run_horizon.
        wages: 1D array, shape (K_max,), dtype float — firm.workforce.wage.
            NOT mutated; read-only.
        t: Current period (used to compute windowed lookback).

    Returns:
        np.ndarray of int worker indices (length ≤ firm.workforce.K),
        sorted ascending by cost-effectiveness ratio (least effective first).
        Stable tiebreak: descending wage (fire the more expensive worker first).
    """
    K = firm.workforce.K
    params = firm.params

    # Internal windowing: mirror firing_review's lookback at review.py:138
    if math.isinf(params.T_review):
        lo = 0
    else:
        lo = max(0, t - int(params.T_review))
    window = output_per_worker[lo:t, :K]  # shape (window_len, K)

    # Compute mean output per worker over the window
    if window.shape[0] > 0:
        # NaN-safe mean; workers with no history get 0.0
        with np.errstate(all="ignore"):
            output_contrib = np.where(
                np.isnan(window).all(axis=0),
                0.0,
                np.nanmean(window, axis=0),
            )
    else:
        output_contrib = np.zeros(K, dtype=np.float64)

    # Identify fireable workers: those covering at least one H or A task
    # (workers whose t2w mapping indicates non-T tasks). Simpler proxy:
    # fireable = not pure-T (i.e., some tasks in H or A mode belong to them).
    # We approximate using worker-level mode coverage.
    # Workers with any H/A task are fireable; pure-T workers are not employed.
    # Use modes to count H+A tasks, map to workers via tasks_per_worker.
    tpw = params.tasks_per_worker
    n_ha = int(((firm.modes == int(Mode.H)) | (firm.modes == int(Mode.A))).sum())
    # Approximate: all K workers that are still active are considered fireable
    # (the kernel's firing_review considers all active workers as candidates).
    # This matches the kernel's behaviour under the K*-ceiling invariant.
    fireable = np.arange(K, dtype=int)

    wage_w = wages[:K]
    output_w = output_contrib[:K]

    # Cost-effectiveness ratio: output / wage (higher = more cost-effective)
    # Avoid division by zero (wages are always > 0 per sample_workforce invariant)
    ratio = output_w / wage_w

    # Sort ascending by ratio (least cost-effective first); stable tiebreak by
    # descending wage (fire the more expensive worker first — deterministic).
    # Use lexsort: primary key = ratio (ascending), secondary key = -wage (ascending = desc wage).
    sort_order = np.lexsort((wage_w * -1, ratio))  # np.lexsort uses last key as primary
    # lexsort sorts by last key first (primary), so: primary=ratio (asc), secondary=-wage (asc=desc)
    return fireable[sort_order]


def _apply_firings_on_workforce(workforce_copy, fire_indices: np.ndarray, t: int):
    """Apply firings to a LOCAL deepcopy of workforce (forward-sim only).

    Does NOT touch firm.history, firm.closed_worker_wages, or any firm-level
    arrays. Shrinks workforce_copy.K by len(fire_indices).

    Returns: (updated_workforce_copy, _, _) — the trailing two elements are
    placeholders for API compatibility with the kernel's apply_firings signature;
    they are None here since we don't carry output matrices in the forward sim.
    """
    if len(fire_indices) == 0:
        return workforce_copy, None, None

    wf = workforce_copy
    keep_mask = np.ones(wf.K, dtype=bool)
    keep_mask[fire_indices] = False
    keep_idx = np.where(keep_mask)[0]

    # Shrink all SoA arrays by removing fired workers.
    # NOTE: wf.K is a read-only property (len(wf.theta)), so we MUST shrink
    # the theta array (and all other SoA arrays) rather than setting wf.K directly.
    wf.wage = wf.wage[keep_idx]
    wf.theta = wf.theta[keep_idx]
    wf.a_trained = wf.a_trained[keep_idx]
    wf.a_training_in_progress = wf.a_training_in_progress[keep_idx]
    wf.tenure = wf.tenure[keep_idx]
    wf.cum_wage = wf.cum_wage[keep_idx]
    # wf.K is now updated implicitly via len(wf.theta) — no explicit setter needed.
    return wf, None, None


def _forward_simulate(firm, t: int, path: list, horizon: int) -> float:
    """Thin compatibility shim — translates 2-tuple paths to Action(n_fire, n_aug, n_hire=0).

    Delegates to forward_simulate_action_path. Do not add new call sites; use Action directly.
    Preserves backward compatibility with all 7 existing call sites (lines 414, 522, 551,
    569, 577, 578, 601) and 3 test imports (lines 49, 51, 326).

    Args:
        firm: Live Firm instance.
        t: Current simulation period.
        path: List of (n_fire, n_aug) 2-tuples, length == horizon.
        horizon: Number of steps to project.

    Returns:
        Cumulative profit over the horizon steps (float).
    """
    from firm_ai_abm.forward_sim import Action, forward_simulate_action_path
    action_path = [Action(n_fire=nf, n_aug=na, n_hire=0) for nf, na in path]
    return forward_simulate_action_path(firm, t, action_path, horizon)


def _candidates_at_step(modes_s: np.ndarray, K_s: int, t_s: int, params: "FirmParams") -> tuple:
    """Compute (fire_candidates, aug_candidates) for a planning step.

    CRIT-3 fix: n_aug candidates derived from n_H_tasks_at_step (task count, not worker count).
    Both lists are recomputed from projected modes/K at each step (NOT captured once at step 0).
    Candidate lists are deduplicated via set→sorted idiom.
    """
    n_H = int((modes_s == int(Mode.H)).sum())
    # Fireable workers: workers covering H or A tasks. Approximated as
    # (H+A task count) // tasks_per_worker.
    n_HA = int(((modes_s == int(Mode.H)) | (modes_s == int(Mode.A))).sum())
    K_fireable = n_HA // params.tasks_per_worker

    # Fire candidates: fractional levels of fireable workers (review-period only)
    if _is_review_period(t_s, params.T_review):
        fire_cands = sorted({int(round(p * K_fireable)) for p in _DP_GRID_LEVELS})
    else:
        fire_cands = [0]

    # Aug candidates: fractional levels of H tasks (task count per D-08)
    aug_cands = sorted({int(round(p * n_H)) for p in _DP_GRID_LEVELS})
    if not aug_cands:
        aug_cands = [0]

    return fire_cands, aug_cands



def _build_all_paths(firm, t: int, horizon: int, modes_0: np.ndarray, K_0: int) -> list:
    """Enumerate all action paths via iterative depth-first expansion.

    Returns a list of action paths, where each path is a list of (n_fire, n_aug)
    tuples of length `horizon`. Action candidates are recomputed per step from
    the projected modes/K along that branch (CRIT-3 fix).

    This is the flat enumeration approach that provides the full path to
    _forward_simulate without requiring recursive path accumulation.
    """
    # Stack entries: (path_so_far, modes_at_step, K_at_step)
    stack = [([], modes_0.copy(), K_0)]
    complete_paths = []

    while stack:
        path_so_far, modes_s, K_s = stack.pop()
        s = len(path_so_far)
        t_s = t + s

        fire_cands, aug_cands = _candidates_at_step(modes_s, K_s, t_s, firm.params)

        for n_fire in fire_cands:
            for n_aug in aug_cands:
                # Project modes and K for the next step
                new_modes = _apply_action_to_modes(
                    modes_s, firm.alpha_hat, firm.beta_hat, n_aug, firm.params
                )
                new_K = max(0, K_s - n_fire) if (
                    _is_review_period(t_s, firm.params.T_review) and n_fire > 0
                ) else K_s

                new_path = path_so_far + [(n_fire, n_aug)]

                if len(new_path) == horizon:
                    complete_paths.append(new_path)
                else:
                    stack.append((new_path, new_modes, new_K))

    return complete_paths
