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
) -> np.ndarray:
    """Return per-task productivity for each task, shape (N,), dtype float64.

    H  -> q_h
    A  -> q_h * (1 + g * beta_i)
    T  -> q_a * alpha_i

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
    p = np.where(modes == Mode.H, params.q_h, 0.0)
    p = np.where(modes == Mode.A, params.q_h * (1.0 + params.g * beta), p)
    p = np.where(modes == Mode.T, params.q_a * alpha, p)
    return p.astype(np.float64)


def cost_vec(modes: np.ndarray, params: FirmParams) -> np.ndarray:
    """Return per-task variable cost for each task, shape (N,), dtype float64.

    H  -> 0.0          (wages charged per worker in simulate.py, not per task)
    A  -> c_aug        (per-task augmentation tool cost)
    T  -> c_auto       (per-task automation infrastructure cost)

    NOTE: There is NO w/N term here. Per architecture D-01, wages are charged
    as w * K in simulate.run_simulation. Adding w/N here would double-count.

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
    c = np.where(modes == Mode.T, params.c_auto, c)
    return c.astype(np.float64)


def _compute_K_table_check() -> None:
    """Boundary-table regression for compute_K. Called by the smoke driver (T-05).

    Exercises n_HA in {0, 9, 10, 11, 100, 101} with tasks_per_worker=10.
    Expected K values: {0, 1, 1, 2, 10, 11}.

    Not called at import time. Not in __all__. Phase 4 will migrate this to
    a pytest module when tests/ is created (R-11).
    """
    params = FirmParams(tasks_per_worker=10)

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
