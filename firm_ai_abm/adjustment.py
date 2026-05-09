"""Adjustment costs for mode transitions between periods.

Cost model contract (architecture D-02, R-03):
  - c_train per H->A task transition ONLY (per-task training cost). H->T, A->T,
    T->H, T->A, and A->H do NOT incur c_train. Training is a one-way investment;
    un-training has no economic cost in Phase 1.
  - Lumpy hire/fire: c_fire * max(0, K_prev - K_new) + c_hire * max(0, K_new - K_prev)
    where K = compute_K(modes, params) from production.py. The firm pays this lumpy
    hire/fire even though the greedy decision rule in strategy.greedy_with_switching
    estimates per-task amortized hire/fire as c_fire/tasks_per_worker/n_amortize
    (smooth approximation per the parent architecture's smooth-vs-lumpy asymmetry
    risk, used in strategy.py, NOT here).
  - Both costs paid in the period of change only (architecture R-03).
  - R-09 (workforce-boundary masking): when a few tasks switch H->T but n_HA stays
    in the same K-bucket, K_prev == K_new and lumpy hire/fire is 0.0. This is
    intentional per design doc section 3 last paragraph and architecture R-09.
"""
import numpy as np

from firm_ai_abm.config import FirmParams
from firm_ai_abm.production import Mode, compute_K


def adj_cost(
    prev_modes: np.ndarray,
    new_modes: np.ndarray,
    params: FirmParams,
) -> float:
    """Compute total adjustment cost for transitioning from prev_modes to new_modes.

    Args:
        prev_modes: Previous period's installed mode vector, shape (N,), dtype int.
        new_modes: Proposed new mode vector, shape (N,), dtype int.
        params: FirmParams instance.

    Returns:
        Total adjustment cost as a Python float (not np.float64). Returns 0.0
        when prev_modes == new_modes (no transitions).

    Risk citations:
        R-03: adjustment cost paid only in the transition period. Passing
              prev_modes == new_modes returns 0.0.
        R-09: workforce-boundary masking — if a small number of tasks switch
              H->T but K does not change (stays within the same ceiling bucket),
              lumpy hire/fire is 0.0. Intentional per architecture.
    """
    assert prev_modes.dtype.kind == "i", (
        f"prev_modes must be integer dtype, got {prev_modes.dtype}"
    )
    assert new_modes.dtype.kind == "i", (
        f"new_modes must be integer dtype, got {new_modes.dtype}"
    )
    assert ((prev_modes >= 0) & (prev_modes <= 2)).all(), (
        "prev_modes values must be in {0, 1, 2}"
    )
    assert ((new_modes >= 0) & (new_modes <= 2)).all(), (
        "new_modes values must be in {0, 1, 2}"
    )
    assert prev_modes.shape == (params.N,), (
        f"prev_modes shape must be ({params.N},), got {prev_modes.shape}"
    )
    assert new_modes.shape == (params.N,), (
        f"new_modes shape must be ({params.N},), got {new_modes.shape}"
    )

    if params.tasks_per_worker == 0:
        return 0.0

    # Training cost: c_train per H->A task transition only
    n_HA_trained = int(
        ((prev_modes == int(Mode.H)) & (new_modes == int(Mode.A))).sum()
    )
    training_cost = params.c_train * n_HA_trained

    # Lumpy hire/fire based on K change (architecture D-02)
    K_prev = compute_K(prev_modes, params)
    K_new = compute_K(new_modes, params)
    fire_cost = params.c_fire * max(0, K_prev - K_new)
    hire_cost = params.c_hire * max(0, K_new - K_prev)

    return float(training_cost + fire_cost + hire_cost)
