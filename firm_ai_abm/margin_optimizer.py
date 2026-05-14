"""Margin-optimizer strategy for Phase 1.5 Stage 6.

Brute-grid over the 5 existing strategies; picks whichever candidate
meets or exceeds firm.params.target_margin over a margin_horizon window.
Falls back to the candidate with the highest realized margin if none qualify.

Uses run_horizon (not run_simulation) so projections:
  - start from the live firm state (deepcopy carries modes/workforce forward)
  - return exactly horizon rows (not t + horizon rows)
  - do NOT touch firm.history

When enable_horizon_brute_action_grid=True (dormant default False), the strategy
additionally searches the full (n_fire, n_aug, n_hire) action grid per D-07 using
the shared forward_simulate_action_path. The 5-candidate legacy path remains the
default (byte-parity preserved).

Caching: memoizes by (t, id(params)) via firm._margin_cache.
Cache lifetime: cleared when firm.reset() is called (D-09).
Cache-key invalidation when enable_horizon_brute_action_grid toggles: because
a flag flip creates a new FirmParams instance, id(firm.params) changes, causing
a natural cache miss (D-09 note).
"""
import copy
import math

import numpy as np

from firm_ai_abm.firm import Firm
from firm_ai_abm.simulate import run_horizon
from firm_ai_abm.strategy import all_H, all_A, all_T, greedy_profit, greedy_with_switching

_CANDIDATES = [all_H, all_A, all_T, greedy_profit, greedy_with_switching]


def _params_hash(firm: Firm, t: int) -> tuple:
    """Lightweight cache key: (t, id(params)) — sufficient within a single run.

    Cache-miss when enable_horizon_brute_action_grid is toggled: a flag flip
    creates a new FirmParams instance with a new id(). No explicit flag-hash
    needed (closed D-09 / MAJ-3 in unify-strategy-controls plan).
    """
    return (t, id(firm.params))


def _build_action_grid_paths(firm: Firm, t: int, horizon: int) -> list:
    """Build (n_fire, n_aug, n_hire) action paths for the full brute action grid.

    Per D-07: hire-level dedup gives at most 5 distinct levels from _DP_GRID_LEVELS.
    When max_hire_per_step==0, hire-axis degenerates to {0} (byte-parity).
    """
    from firm_ai_abm.dp_optimizer import (
        _DP_GRID_LEVELS, _is_review_period, _candidates_at_step, _apply_action_to_modes,
    )
    from firm_ai_abm.forward_sim import Action
    params = firm.params

    # Hire-level dedup per D-07
    if params.max_hire_per_step > 0 and (params.enable_replenish_hiring or params.enable_hiring):
        hire_levels = sorted({int(round(lvl * params.max_hire_per_step))
                              for lvl in _DP_GRID_LEVELS})
    else:
        hire_levels = [0]

    # Walk horizon steps; for each step pick fire/aug from _candidates_at_step,
    # combine with hire levels. Build complete paths via iterative DFS.
    stack = [([], firm.modes.copy(), firm.workforce.K)]
    complete_paths: list = []

    while stack:
        path_so_far, modes_s, K_s = stack.pop()
        s = len(path_so_far)
        t_s = t + s

        fire_cands, aug_cands = _candidates_at_step(modes_s, K_s, t_s, params)

        for n_fire in fire_cands:
            for n_aug in aug_cands:
                for n_hire in (hire_levels if n_fire == 0 else [0]):
                    new_modes = _apply_action_to_modes(
                        modes_s, firm.alpha_hat, firm.beta_hat, n_aug, params
                    )
                    new_K = max(0, K_s - n_fire) if (
                        _is_review_period(t_s, params.T_review) and n_fire > 0
                    ) else K_s

                    new_path = path_so_far + [Action(n_fire=n_fire, n_aug=n_aug, n_hire=n_hire)]
                    if len(new_path) == horizon:
                        complete_paths.append(new_path)
                    else:
                        stack.append((new_path, new_modes, new_K))

    return complete_paths


def horizon_brute_strategy(firm: Firm, t: int) -> np.ndarray:
    """Return modes array from the candidate with the best objective for the current scenario.

    Branches on firm.params.scenario_mode:
    - "price" mode: maximizes cumulative profit sum(pi) over margin_horizon periods.
      Picks the candidate with the highest sum of per-period profit (p*Y - C).
    - "margin" mode (default): maximizes realized margin (revenue - cost) / revenue
      over margin_horizon periods. Picks the candidate with the highest realized margin.

    Projection uses copy.deepcopy(firm) + run_horizon — the deepcopy carries
    live modes, a_trained, a_training_in_progress, theta, wage, and tenure
    forward correctly without mutating the live firm.

    Args:
        firm: live Firm instance at period t.
        t: current simulation period (used for cache key and strategy calls).

    Returns:
        Fresh np.ndarray of shape (N,), dtype int — the selected strategy's
        modes evaluated on the LIVE firm (not the deepcopy).
    """
    cache = getattr(firm, "_margin_cache", None)
    if cache is None:
        firm._margin_cache = {}  # type: ignore[attr-defined]
        cache = firm._margin_cache

    key = _params_hash(firm, t)
    if key in cache:
        return cache[key].copy()

    horizon = firm.params.margin_horizon
    scenario_mode = firm.params.scenario_mode

    if firm.params.enable_horizon_brute_action_grid:
        if scenario_mode == "margin":
            raise ValueError(
                "scenario_mode='margin' is not supported with enable_horizon_brute_action_grid=True"
            )

        # Full (n_fire, n_aug, n_hire) action-grid search via shared forward_simulate_action_path
        from firm_ai_abm.forward_sim import forward_simulate_action_path
        from firm_ai_abm.dp_optimizer import _apply_action_to_modes
        action_paths = _build_action_grid_paths(firm, t, horizon)

        best_objective = -math.inf
        best_modes = firm.modes.copy()  # no-side-effect seed (MINOR-4: don't call a candidate)
        best_path = action_paths[0] if action_paths else None

        for path in action_paths:
            pi = forward_simulate_action_path(firm, t, path, horizon)
            if pi >= best_objective:
                best_objective = pi
                best_path = path

        # Derive modes from the winning path's step-1 n_aug; write fire/hire intents (D-02)
        if best_path is not None:
            best_modes = _apply_action_to_modes(
                firm.modes.copy(), firm.alpha_hat, firm.beta_hat,
                best_path[0].n_aug, firm.params
            )
            firm._fire_intent = int(best_path[0].n_fire)  # type: ignore[attr-defined]
            firm._hire_intent = int(best_path[0].n_hire)  # type: ignore[attr-defined]
    else:
        # Legacy 5-candidate run_horizon path (byte-identical to pre-change behavior)
        # Seed with first candidate so result is always non-None, even when all
        # candidates project revenue == 0 (objective = -inf for every candidate).
        best_objective = -math.inf
        best_modes = _CANDIDATES[0](firm, t)

        for cand in _CANDIDATES:
            firm_copy = copy.deepcopy(firm)
            proj_df = run_horizon(firm_copy, cand, horizon, use_posteriors=True)
            assert len(proj_df) == horizon, (
                f"run_horizon returned {len(proj_df)} rows, expected {horizon}"
            )

            if scenario_mode == "price":
                objective = float(proj_df["pi"].sum())
            else:
                revenue = firm.params.p * float(proj_df["Y"].sum())
                cost = float(proj_df["C"].sum())
                objective = (revenue - cost) / revenue if revenue > 0 else -math.inf

            # >= gives last-wins-on-ties semantics (stable across _CANDIDATES order).
            if objective >= best_objective:
                best_objective = objective
                best_modes = cand(firm, t)

    result = best_modes
    cache[key] = result.copy()
    return result.copy()


# D-07: backward-compat alias — preserves tests/test_margin_optimizer.py
# and any notebook code that imports target_margin_strategy by name.
target_margin_strategy = horizon_brute_strategy
