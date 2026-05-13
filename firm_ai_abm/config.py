"""Phase 1 firm parameters. Defaults from 01_phase1_single_firm.md §7."""
import math
from dataclasses import dataclass


@dataclass
class FirmParams:
    # Counts
    N: int = 500                 # number of tasks
    T: int = 60                  # number of periods
    tasks_per_worker: int = 5    # how many tasks one worker covers

    # Productivity
    q_h: float = 1.0             # human productivity per task (numeraire)
    q_a: float = 1.2             # AI productivity ceiling (key dial)
    g: float = 0.5               # augmentation gain (key dial)

    # Costs
    w: float = 1.0               # wage per worker per period (numeraire)
    c_aug: float = 0.05          # per-task augmentation cost
    c_auto: float = 0.4          # per-task automation cost (key dial)
    c_fire: float = 2.0          # firing cost per worker (lumpy, K-based)
    c_hire: float = 0.5          # hiring cost per worker (lumpy, K-based)
    c_train: float = 0.1         # training cost per H->A task (per-task)
    F: float = 5.0               # fixed cost per period

    # Prices
    p: float = 0.22              # output price (recalibrated; tpw=5, p=0.22 → all-H baseline ≈ −3/period)

    # Strategy
    n_amortize: int = 6          # horizon for greedy-with-switching amortization

    # RNG
    seed: int | None = None      # if set, reproducible alpha/beta sampling

    # Worker heterogeneity (Phase 1.5 Stage 1)
    sigma_theta: float = 0.2     # std of log-normal productivity draws; 0 → homogeneous
    theta_min: float = 0.4       # lower clip for theta
    theta_max: float = 1.6       # upper clip for theta
    corr_w_theta: float = 0.3    # elasticity exponent in wage = w * (theta^c / mean(theta^c)) * exp(eps - sigma_w^2/2); NOT a Pearson r
    sigma_w: float = 0.05        # multiplicative log-noise std on individual wage

    # Phase 1.5 Stage 3 — periodic firing review
    T_review: float = math.inf   # review every T_review periods; math.inf (default) disables the path
                                  # entirely — set to e.g. 10.0 to enable.
                                  # FLOAT-typed to support math.inf (disabled path). int(T_review) is
                                  # used internally for modulo arithmetic. Default math.inf means the
                                  # periodic firing review is disabled by default — opt in by setting
                                  # T_review=10.0 (or any finite value).
    firing_threshold: float = 0.0  # fire workers with surplus < firing_threshold
                                    # (defaults to 0.0 = fire negative-surplus workers)

    # Phase 1.5 Stage 6 — opt-in post-firing rehire to K_target = K0
    enable_hiring: bool = False

    # Training delay (Phase 1.5 Stage 6 — dormant by default)
    enable_training_delay: bool = False  # when True, H->A workers produce as H for 1 period before aug kicks in

    # Margin scenario (Phase 1.5 Stage 6)
    scenario_mode: str = "price"   # "price" or "margin"
    target_margin: float = 0.05    # target (revenue - cost) / revenue when scenario_mode == "margin"
    margin_horizon: int = 5        # look-ahead periods for margin-optimizer brute grid

    # Phase 1.5 Stage X — opt-in replenishment hiring (augment-replenish-hiring)
    # Dormant by default (enable_replenish_hiring=False). Mutually exclusive with
    # enable_hiring — both True raises ValueError at make_firm (validated in firm.py).
    enable_replenish_hiring: bool = False  # when True, fired workers are queued for rehire after hire_delay_periods
    max_hire_period: int = 0               # per-period hire cap; 0 = drain entire backlog in one period (sentinel)
    hire_delay_periods: int = 1            # periods to wait before hiring back fired workers (>=1)
    max_hire_per_step: int = 0             # planning action-grid hire cap; 0 = hire-axis degenerates to {0} (byte-parity)
                                           # distinct from max_hire_period (kernel drain-cap) — this controls the planner grid
    enable_horizon_brute_action_grid: bool = False  # when True, horizon_brute_strategy searches full (n_fire,n_aug,n_hire) action grid
                                                     # when False (default), falls back to 5-candidate run_horizon path (byte-parity)

    # Alpha-dependent automation cost (D-01, D-02, D-05)
    # When belief_alpha is None (default), all three fields are dormant: cost_vec uses the flat
    # params.c_auto branch bit-for-bit, and strategies score T-mode with realized alpha.
    # When belief_alpha is a float (e.g. 0.5), the alpha-dependent formula is engaged:
    #   c_auto_i(alpha_i) = (w/tasks_per_worker) * (c_auto_alpha_intercept - c_auto_alpha_slope * alpha_i)
    # clamped at 0. belief_alpha lives on FirmParams (not Firm) to avoid strategy-signature churn.
    c_auto_alpha_slope: float = 0.0       # slope of linear alpha-dependence in per-task auto cost (D-01)
    c_auto_alpha_intercept: float = 0.0   # intercept of linear alpha-dependence (D-01)
    belief_alpha: float | None = None     # prior mean E[alpha] for T-mode scoring; None = dormant (D-02, D-05)

    # Task-attribute distributions (beta-dist-task-attrs)
    # alpha_i ~ Beta(a, b) with a = alpha_mean * alpha_concentration,
    #                         b = (1 - alpha_mean) * alpha_concentration.
    # Default (0.5, 2.0) is the Uniform(0,1) special case (Beta(1,1)) — the
    # sampler short-circuits to rng.uniform(0,1) on exact equality with these
    # defaults to preserve byte-identity with Phase 1 parquet fixtures.
    alpha_mean: float = 0.5
    alpha_concentration: float = 2.0
    beta_mean: float = 0.5
    beta_concentration: float = 2.0
