"""Adjustment costs for mode transitions between periods.

Cost model contract (architecture D-02, R-03):
  - Stage 2+: c_train per WORKER entering A-mode for the first time (per-worker training
    cost, gated by workforce.a_trained). H->T, A->T, T->H, T->A, and A->H do NOT incur
    c_train. Training is a one-way investment; un-training has no economic cost.
    Side effect: adj_cost mutates workforce.a_trained in-place (D-02). This is
    intentional — the mutation is co-located with the cost computation so no caller
    can forget to flip the flag. Document callers accordingly.
  - Legacy fallback (workforce=None): c_train per H->A TASK transition (Phase 1
    per-task counting). Preserved byte-identically for callers that omit workforce
    (D-04). The simulate.py call site ALWAYS passes workforce — no production code
    uses the fallback path.
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
  - Stage 2 D-01: greedy strategies do NOT consult workforce.a_trained when scoring
    modes. They use smooth c_train amortization (c_train/n_amortize per task). This
    decision-vs-payment asymmetry is conservative (greedy over-estimates switching
    cost) and consistent with Stage 1 D-04.
  - Stage 3 hook: prev_workforce parameter is reserved for detecting "trained workers
    fired this period → capital lost" (c_train_lost history column). Stage 2 body
    ignores prev_workforce; Stage 3 body will use it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from firm_ai_abm.config import FirmParams
from firm_ai_abm.production import Mode, compute_K
from firm_ai_abm.workers import task_to_worker_map

if TYPE_CHECKING:
    from firm_ai_abm.workers import Workforce


def count_workers_entering_a_first_time(
    prev_modes: np.ndarray,  # noqa: ARG001 — reserved for future use
    new_modes: np.ndarray,
    workforce: "Workforce",
    params: FirmParams,
) -> tuple[int, np.ndarray]:
    """Return (count, indices) of workers entering A-mode for the first time.

    "Entering A for the first time" means:
      - The worker is assigned to at least one A-task in new_modes, AND
      - workforce.a_trained[k] is False for that worker.

    Uses task_to_worker_map to determine which worker covers each task slot.
    T-mode tasks (mapped to -1) are ignored.

    Args:
        prev_modes: Previous mode vector, shape (N,), int. Reserved — not used
                    in Stage 2 body but kept for forward compatibility.
        new_modes: Proposed new mode vector, shape (N,), int.
        workforce: Workforce instance with K workers and a_trained array.
        params: FirmParams with tasks_per_worker field.

    Returns:
        (n_first_a, newly_trained_indices) where:
          - n_first_a: int, count of workers entering A for the first time.
          - newly_trained_indices: np.ndarray of dtype int, sorted worker indices.
          Returns (0, empty array) when workforce.K == 0 or no new A-entries.
    """
    if workforce.K == 0:
        return 0, np.array([], dtype=int)

    t2w_new = task_to_worker_map(new_modes, workforce.K, params.tasks_per_worker)
    a_mask = new_modes == int(Mode.A)
    # Workers that have at least one A-task assigned in new_modes
    workers_with_a_task = np.unique(t2w_new[a_mask & (t2w_new >= 0)])
    if workers_with_a_task.size == 0:
        return 0, np.array([], dtype=int)
    untrained = ~workforce.a_trained[workers_with_a_task]
    newly_trained = workers_with_a_task[untrained]
    return int(len(newly_trained)), newly_trained


def adj_cost(
    prev_modes: np.ndarray,
    new_modes: np.ndarray,
    params: FirmParams,
    workforce: "Workforce | None" = None,
    prev_workforce: "Workforce | None" = None,
) -> float:
    """Compute total adjustment cost for transitioning from prev_modes to new_modes.

    SIDE EFFECT (Stage 2, D-02): when workforce is not None, this function mutates
    workforce.a_trained in-place — setting newly-trained worker entries to True
    AFTER computing the training cost and BEFORE returning. This is intentional:
    the mutation is co-located with the cost so no caller can forget the flip.
    Callers that invoke adj_cost multiple times with the same workforce instance
    will observe 0 training cost on subsequent calls for already-trained workers
    (idempotent once all relevant workers are trained).

    Args:
        prev_modes: Previous period's installed mode vector, shape (N,), dtype int.
        new_modes: Proposed new mode vector, shape (N,), dtype int.
        params: FirmParams instance.
        workforce: Workforce instance (Stage 2+). When provided, training cost is
                   charged per worker entering A for the first time (per-worker
                   semantics, gated by workforce.a_trained). When None, falls back
                   to Phase 1 per-task H→A counting (D-04 backward compat).
        prev_workforce: Reserved for Stage 3 (firing wastes trained capital).
                        Ignored in Stage 2 body. Defaults to None.

    Returns:
        Total adjustment cost as a Python float (not np.float64). Returns 0.0
        when prev_modes == new_modes (no transitions) and no lumpy K change.

    Risk citations:
        R-03: adjustment cost paid only in the transition period. Passing
              prev_modes == new_modes returns 0.0.
        R-09: workforce-boundary masking — if a small number of tasks switch
              H->T but K does not change (stays within the same ceiling bucket),
              lumpy hire/fire is 0.0. Intentional per architecture.
        D-04: workforce=None falls back to Phase 1 per-task counting byte-identically.
        D-02: workforce.a_trained is mutated in-place after cost computed.
        D-03: prev_workforce is accepted but unused in Stage 2.
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

    if workforce is None:
        # D-04 backward-compat: per-task H->A counting (Phase 1 behavior)
        n_HA_trained = int(
            ((prev_modes == int(Mode.H)) & (new_modes == int(Mode.A))).sum()
        )
        training_cost = params.c_train * n_HA_trained
    else:
        # Stage 2+: per-worker training cost, gated by workforce.a_trained
        n_first_a, newly_trained = count_workers_entering_a_first_time(
            prev_modes, new_modes, workforce, params
        )
        training_cost = params.c_train * n_first_a
        # D-02: flip AFTER cost computed, BEFORE return
        if len(newly_trained) > 0:
            workforce.a_trained[newly_trained] = True

    # Lumpy hire/fire based on K change (architecture D-02)
    K_prev = compute_K(prev_modes, params)
    K_new = compute_K(new_modes, params)
    fire_cost = params.c_fire * max(0, K_prev - K_new)
    hire_cost = params.c_hire * max(0, K_new - K_prev)

    return float(training_cost + fire_cost + hire_cost)
