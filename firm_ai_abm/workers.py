"""Worker heterogeneity for Phase 1.5 Stage 1 / Stage 5.

Workforce is represented as a struct-of-arrays (SoA) for vectorized hot-path
performance. No list-of-Worker objects — all arrays are shape (K,).

Key design decisions:
  D-02: Augmentation × heterogeneity is multiplicative: theta * q_h * (1 + g * beta).
  D-03: Wage bill uses ASSIGNED workers only (deviation from architecture §5 — load-bearing
        for all_T byte-parity; Stage 3 must choose between "idle workers cost nothing" vs
        "idle workers cost wages").
  D-08: sigma==0 short-circuit: no rng draws when sigma_theta==0 OR sigma_w==0,
        eliminating numpy-version dependency on Generator.normal(scale=0) behavior.

Stage 5 wage-formula change (D-02 numeraire contract):
  Mean-preserving tilt: wage = w * (theta^c / mean(theta^c)) * exp(eps - sigma_w^2/2)
  where c = corr_w_theta, eps ~ Normal(0, sigma_w).
  Two normalizations ensure E[wage] = w:
    - theta-tilt divided by its SAMPLE mean → tilt mean is exactly 1.0 within the batch.
    - exp(eps - sigma_w^2/2) has E = 1.0 exactly.
  Degenerate path (sigma_theta=0, sigma_w=0): theta = ones(K), mean(theta^c) = 1.0,
  eps = 0 → wage = w * 1.0 * 1.0 = w. Byte-equal to Stage-1 sigma=0 contract.

  Note (MIN-4): varying theta_min/theta_max changes the clip range, which shifts
  theta^c.mean() and hence the wage tilt. This is intentional — theta_min/theta_max
  sliders affect wage scale through the normalization.

  Replacement-draw normalization: each call to sample_workforce normalizes by the
  REPLACEMENT batch's mean (sample-mean, not population-mean). Each call is mean-
  preserving for its own batch; combined workforce wage mean may drift slightly over
  many fire+replace cycles (bounded by sampling variance, documented in README).
"""
from dataclasses import dataclass

import numpy as np

from firm_ai_abm.config import FirmParams
from firm_ai_abm.production import Mode


@dataclass
class Workforce:
    """Struct-of-arrays representation of the firm's worker pool. All arrays shape (K,)."""
    theta: np.ndarray    # productivity multiplier, float64
    wage: np.ndarray     # individual wage, float64
    a_trained: np.ndarray  # True if worker has completed A-mode training, bool
    tenure: np.ndarray   # periods employed at this firm, int
    hire_t: np.ndarray   # period at which worker was hired, int

    @property
    def K(self) -> int:
        return int(len(self.theta))


def sample_workforce(
    K: int,
    params: FirmParams,
    rng: np.random.Generator,
    current_t: int = 0,
) -> Workforce:
    """Sample a new workforce of K workers.

    Stage 5 mean-preserving wage formula:
      wage = w * (theta^c / mean(theta^c)) * exp(eps - sigma_w^2/2)
      where c = corr_w_theta, eps ~ Normal(0, sigma_w).
    Degenerate path (sigma_theta==0 AND sigma_w==0): returns wage = w exactly.

    D-08 short-circuit: when sigma_theta==0, theta is exactly ones (no rng draw).
    When sigma_w==0 AND sigma_theta==0, wage is exactly w (no rng draw).
    When sigma_w==0 AND sigma_theta>0, wage = w * tilt (no eps draw).
    """
    if params.sigma_theta == 0.0:
        theta = np.ones(K, dtype=np.float64)
    else:
        theta = np.clip(
            rng.normal(1.0, params.sigma_theta, size=K),
            params.theta_min,
            params.theta_max,
        ).astype(np.float64)

    # Stage 5 D-02: mean-preserving wage formula
    if params.sigma_theta == 0.0 and params.sigma_w == 0.0:
        # Degenerate path: byte-equal to Stage-1 sigma=0 contract
        wage = np.full(K, params.w, dtype=np.float64)
    else:
        theta_pow = theta ** params.corr_w_theta
        tilt = theta_pow / theta_pow.mean()  # sample-mean normalization
        if params.sigma_w == 0.0:
            wage = params.w * tilt
        else:
            eps = rng.normal(0.0, params.sigma_w, size=K)
            wage = params.w * tilt * np.exp(eps - 0.5 * params.sigma_w ** 2)

    return Workforce(
        theta=theta,
        wage=wage.astype(np.float64),
        a_trained=np.zeros(K, dtype=bool),
        tenure=np.zeros(K, dtype=int),
        hire_t=np.full(K, current_t, dtype=int),
    )


def _make_initial_workforce(params: FirmParams, rng: np.random.Generator) -> Workforce:
    """Convenience factory used ONLY by make_firm (not by reset()).

    Computes initial K = N // tasks_per_worker and calls sample_workforce.
    CRIT-2: this is the SOLE call site for workforce sampling at firm construction.
    reset() must NOT call this — it would consume rng state and break check2_greedy_dominance.
    """
    K = params.N // params.tasks_per_worker
    return sample_workforce(K, params, rng, current_t=0)


def task_to_worker_map(
    modes: np.ndarray,
    K: int,
    tasks_per_worker: int,
) -> np.ndarray:
    """Map each task to its covering worker index, or -1 for T-mode tasks.

    Worker k covers task slots [k * tasks_per_worker, (k+1) * tasks_per_worker).
    T-mode tasks get -1 (no worker assigned). H/A tasks get the index of the
    worker whose slot covers that task position.

    Args:
        modes: shape (N,), integer dtype, values in {0, 1, 2}
        K: number of workers (>= 0)
        tasks_per_worker: tasks per worker slot

    Returns:
        shape (N,), int dtype. Entry i is worker index k in [0, K-1] if modes[i]
        in {H, A}, else -1.

    Raises:
        AssertionError: if K < 0, or if the number of H/A tasks exceeds K * tasks_per_worker.
    """
    N = len(modes)
    ha_mask = (modes == Mode.H) | (modes == Mode.A)
    raw = np.arange(N) // tasks_per_worker

    assert K >= 0, f"K must be non-negative, got {K}"
    if K == 0:
        assert not ha_mask.any(), (
            "K=0 but some tasks have H/A mode — inconsistent state"
        )
        return np.full(N, -1, dtype=int)

    assert int(ha_mask.sum()) <= K * tasks_per_worker, (
        f"K={K} insufficient: {int(ha_mask.sum())} HA tasks require at least "
        f"{int(ha_mask.sum()) // tasks_per_worker + (1 if ha_mask.sum() % tasks_per_worker else 0)} workers"
    )

    out = np.where(ha_mask, np.minimum(raw, K - 1), -1)
    return out.astype(int)
