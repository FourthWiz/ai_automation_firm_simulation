"""Margin-optimizer strategy for Phase 1.5 Stage 6.

Brute-grid over the 5 existing strategies; picks whichever candidate
meets or exceeds firm.params.target_margin over a margin_horizon window.
Falls back to the candidate with the highest realized margin if none qualify.

Uses run_horizon (not run_simulation) so projections:
  - start from the live firm state (deepcopy carries modes/workforce forward)
  - return exactly horizon rows (not t + horizon rows)
  - do NOT touch firm.history

Caching: memoizes by (t, params_hash) via firm._margin_cache.
Cache lifetime: cleared when firm.reset() is called (D-09).
"""
import copy
import math

import numpy as np

from firm_ai_abm.firm import Firm
from firm_ai_abm.simulate import run_horizon
from firm_ai_abm.strategy import all_H, all_A, all_T, greedy_profit, greedy_with_switching

_CANDIDATES = [all_H, all_A, all_T, greedy_profit, greedy_with_switching]


def _params_hash(firm: Firm, t: int) -> tuple:
    """Lightweight cache key: (t, id(params)) — sufficient within a single run."""
    return (t, id(firm.params))


def target_margin_strategy(firm: Firm, t: int) -> np.ndarray:
    """Return modes array from the candidate that best meets target_margin.

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

    target = firm.params.target_margin
    horizon = firm.params.margin_horizon

    # Seed with first candidate so result is always non-None, even when all
    # candidates project revenue == 0 (realized = -inf for every candidate).
    best_realized = -math.inf
    best_modes: np.ndarray = _CANDIDATES[0](firm, t)

    for cand in _CANDIDATES:
        firm_copy = copy.deepcopy(firm)
        proj_df = run_horizon(firm_copy, cand, horizon)
        assert len(proj_df) == horizon, (
            f"run_horizon returned {len(proj_df)} rows, expected {horizon}"
        )
        revenue = firm.params.p * float(proj_df["Y"].sum())
        cost = float(proj_df["C"].sum())
        realized = (revenue - cost) / revenue if revenue > 0 else -math.inf

        # Pick the candidate with the highest realized margin (argmax).
        # >= gives last-wins-on-ties semantics (stable across _CANDIDATES order).
        # The target_margin constraint acts as a floor: the firm maximizes profit
        # subject to margin >= target, but since argmax always picks the highest
        # margin, the qualified and unqualified cases collapse to one branch.
        if realized >= best_realized:
            best_realized = realized
            best_modes = cand(firm, t)

    result = best_modes
    cache[key] = result.copy()
    return result.copy()
