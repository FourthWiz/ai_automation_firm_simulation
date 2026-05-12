"""Production kernel: Mode enum, workforce sizing, and vectorized per-task functions.

compute_K, productivity_vec, cost_vec are the hot-path kernel.
_compute_K_table_check is a private boundary-table helper invoked by the smoke driver
(not at import time, not in __all__).
"""
import math
from enum import IntEnum

import numpy as np

from firm_ai_abm.config import FirmParams


class Mode(IntEnum):
    H = 0  # human
    A = 1  # augmented human
    T = 2  # automated


def compute_K(modes: np.ndarray, params: FirmParams) -> int:
    """Return the number of workers required for the current mode mix.

    Workers cover only H and A tasks; automated tasks need no human labour.
    Uses ceiling division: each partial group of tasks_per_worker tasks still
    requires a full worker (lumpy hire/fire model per architecture D-02).

    Edge case: if all tasks are automated (n_HA == 0), returns 0 exactly.
    Return type is plain int (not np.int64) so downstream arithmetic stays clean.
    """
    n_HA = int(((modes == Mode.H) | (modes == Mode.A)).sum())
    if n_HA == 0:
        return 0
    return int(math.ceil(n_HA / params.tasks_per_worker))


def productivity_vec(
    modes: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    params: FirmParams,
    theta_per_task: np.ndarray | None = None,
    a_in_training_per_task: np.ndarray | None = None,
) -> np.ndarray:
    """Return per-task productivity for each task, shape (N,), dtype float64.

    H  -> q_h * theta_per_task_i
    A  -> q_h * (1 + g * beta_i) * theta_per_task_i   (D-02: multiplicative)
    T  -> q_a * alpha_i                                 (T unaffected by theta)

    theta_per_task=None (default) reduces to Phase 1 behavior exactly (byte-identical).
    a_in_training_per_task=None (default) gives byte-identical output.
    When a_in_training_per_task[i]=True for an A-task: uses H formula (no aug bonus)
    for that period (worker is still in training — delay enabled path only).

    Kernel invariants (assert, not raise — these are call-site preconditions):
      - modes must have integer dtype
      - all values must be in {0, 1, 2}
      - theta_per_task must be float64 and same shape as modes (if provided)
    """
    assert modes.dtype.kind == "i", (
        f"modes must be integer dtype, got {modes.dtype}"
    )
    assert ((modes >= 0) & (modes <= 2)).all(), (
        "modes values must be in {0, 1, 2}"
    )
    if theta_per_task is None:
        theta_per_task = np.ones(len(modes), dtype=np.float64)
    assert theta_per_task.shape == modes.shape, (
        f"theta_per_task shape {theta_per_task.shape} != modes shape {modes.shape}"
    )
    assert theta_per_task.dtype.kind == "f", (
        f"theta_per_task must be float dtype, got {theta_per_task.dtype}"
    )
    p = np.where(modes == Mode.H, params.q_h * theta_per_task, 0.0)
    p = np.where(modes == Mode.A, params.q_h * (1.0 + params.g * beta) * theta_per_task, p)
    p = np.where(modes == Mode.T, params.q_a * alpha, p)
    if a_in_training_per_task is not None:
        # In-training A-tasks produce at H rate — no aug bonus yet
        training_mask = (modes == Mode.A) & a_in_training_per_task
        p = np.where(training_mask, params.q_h * theta_per_task, p)
    return p.astype(np.float64)


def cost_vec(
    modes: np.ndarray,
    alpha: np.ndarray,
    params: FirmParams,
    a_in_training_per_task: np.ndarray | None = None,
) -> np.ndarray:
    """Return per-task variable cost for each task, shape (N,), dtype float64.

    H  -> 0.0          (wages charged per worker in simulate.py, not per task)
    A  -> c_aug        (per-task augmentation tool cost)
    T  -> c_auto       (per-task automation infrastructure cost; see note below)

    a_in_training_per_task=None (default) gives byte-identical output.
    When a_in_training_per_task[i]=True for an A-task: c_aug is NOT charged
    (worker produces as H during training — no aug tool cost this period).
    c_train is still charged separately via adj_cost.

    NOTE: There is NO w/N term here. Per architecture D-01, wages are charged
    as w * K in simulate.run_simulation. Adding w/N here would double-count.

    Alpha-dependent T-cost (D-01, D-05):
      - When params.belief_alpha is None (default/dormant): T cost = params.c_auto (flat).
        This path is byte-identical to the pre-change formula for all inputs.
      - When params.belief_alpha is not None (engaged):
          c_auto_i = (w/tpw) * (c_auto_alpha_intercept - c_auto_alpha_slope * alpha_i)
          clamped at 0 via np.maximum(..., 0.0). No + params.c_auto additive term (D-01).
      T-mode entries in the aug-cost callsite (simulate.py Step 6) are filtered by
      worker_mask (t2w == -1 for T tasks), so aug_cost_per_worker remains unaffected
      by the engaged path (MAJ-1 note from plan T-03).

    Kernel invariants (assert, not raise — these are call-site preconditions):
      - modes must have integer dtype
      - all values must be in {0, 1, 2}
    """
    assert modes.dtype.kind == "i", (
        f"modes must be integer dtype, got {modes.dtype}"
    )
    assert ((modes >= 0) & (modes <= 2)).all(), (
        "modes values must be in {0, 1, 2}"
    )
    c = np.where(modes == Mode.H, 0.0, 0.0)
    c = np.where(modes == Mode.A, params.c_aug, c)
    if params.belief_alpha is None:
        # D-05 dormant sentinel: flat c_auto, byte-identical to pre-change formula
        c = np.where(modes == Mode.T, params.c_auto, c)
    else:
        # D-01 engaged path: linear alpha-dependent cost, no additive c_auto term
        wage_per_task = params.w / params.tasks_per_worker
        c_auto_per_task = (
            wage_per_task
            * (params.c_auto_alpha_intercept - params.c_auto_alpha_slope * alpha)
        )
        c_auto_per_task = np.maximum(c_auto_per_task, 0.0)
        c = np.where(modes == Mode.T, c_auto_per_task, c)
    if a_in_training_per_task is not None:
        # In-training A-tasks: tool not meaningfully used → zero c_aug this period
        c = np.where((modes == Mode.A) & a_in_training_per_task, 0.0, c)
    return c.astype(np.float64)


def _compute_K_table_check() -> None:
    """Boundary-table regression for compute_K. Called by the smoke driver (T-05).

    Exercises n_HA in {0, 9, 10, 11, 100, 101} with tasks_per_worker=10.
    Expected K values: {0, 1, 1, 2, 10, 11}.

    Not called at import time. Not in __all__. Phase 4 will migrate this to
    a pytest module when tests/ is created (R-11).
    """
    params = FirmParams(tasks_per_worker=10, p=1.0)

    # (n_HA, expected_K) pairs
    cases = [(0, 0), (9, 1), (10, 1), (11, 2), (100, 10), (101, 11)]

    for n_HA, expected_K in cases:
        # Build a modes array: first n_HA slots are H (0), rest are T (2)
        total_tasks = max(n_HA, params.tasks_per_worker)  # at least 10 tasks
        if n_HA > total_tasks:
            total_tasks = n_HA + params.tasks_per_worker  # pad with T slots
        modes = np.full(total_tasks, int(Mode.T), dtype=int)
        modes[:n_HA] = int(Mode.H)
        K = compute_K(modes, params)
        assert K == expected_K, (
            f"compute_K boundary check failed: n_HA={n_HA}, "
            f"tasks_per_worker={params.tasks_per_worker}, "
            f"expected K={expected_K}, got K={K}"
        )
        assert isinstance(K, int), (
            f"compute_K must return plain int, got {type(K)} for n_HA={n_HA}"
        )
