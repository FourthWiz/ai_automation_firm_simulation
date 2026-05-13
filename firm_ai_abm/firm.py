"""Firm dataclass + factory. reset() is R-07-compliant: never touches alpha/beta/workforce."""
from dataclasses import dataclass, field

import numpy as np

from firm_ai_abm.config import FirmParams
from firm_ai_abm.production import Mode
from firm_ai_abm.tasks import sample_alpha, sample_beta
from firm_ai_abm.workers import Workforce, _make_initial_workforce


@dataclass
class Firm:
    params: FirmParams
    alpha: np.ndarray
    beta: np.ndarray
    modes: np.ndarray | None = None
    workforce: Workforce | None = None
    history: list = field(default_factory=list)
    rng: np.random.Generator | None = None
    # pending_hires: list of (period_eligible, n_remaining) tuples for replenish path.
    # MUST use field(default_factory=list) — mutable default would be shared across instances.
    # This is run-state (cleared by reset), unlike alpha/beta/workforce (firm-identity state).
    pending_hires: list = field(default_factory=list)
    # closed_worker_wages: terminal cum_wage value for each worker who has been fired.
    # Appended in review.apply_firings / replace_to_target before the worker's SoA slot
    # is removed. Cleared on reset() (run-state, mirrors pending_hires). Never shrinks
    # mid-run — only appended. Combined with live workforce.cum_wage for the per-period
    # mean_accum_wage metric.
    closed_worker_wages: list = field(default_factory=list)

    @property
    def K(self) -> int:
        return self.workforce.K if self.workforce is not None else 0

    def reset(self) -> None:
        """Reset modes/history/pending_hires. NEVER touches alpha, beta, or workforce (R-07, CRIT-2).

        pending_hires IS cleared here (it is run-state, like firm.history). If it were not
        cleared, check2_greedy_dominance (which reuses one firm across multiple run_simulation
        calls via reset()) would leak a phantom backlog from one strategy trial into the next.
        This is the exact asymmetry with workforce: workforce persists (firm-identity state),
        pending_hires resets (run-state). See D-08 in augment-replenish-hiring plan.
        """
        N = self.params.N
        self.modes = np.zeros(N, dtype=int)  # all H = Mode.H = 0
        self.history = []
        self.pending_hires = []
        self.closed_worker_wages = []
        self._margin_cache: dict = {}  # type: ignore[attr-defined]
        # workforce is NOT re-sampled here — it persists for the firm's lifetime.
        # cum_wage IS zeroed because it is run-state (like pending_hires), not firm-identity state.
        if self.workforce is not None:
            self.workforce.cum_wage = np.zeros(self.workforce.K, dtype=np.float64)
        # K0 is intentionally NOT reset here — it is the initial headcount set once in make_firm


def validate_hiring_params(params: FirmParams) -> None:
    """Validate hiring params. No-op when neither hiring mode is enabled.

    Called from make_firm BEFORE workforce sampling (fail-fast at construction).
    Mutual exclusion: enable_hiring and enable_replenish_hiring cannot both be True.
    hire_delay_periods >= 1 and max_hire_period >= 0 are enforced whenever EITHER
    hiring mode is on — delay applies to both paths after D-02.
    If a user mutates firm.params post-construction (anti-pattern), this check does
    not re-fire — the bypass is on them (R-05). Add a defensive re-check at
    run_horizon entry if that risk materializes.
    """
    if not (params.enable_hiring or params.enable_replenish_hiring):
        return
    if params.enable_hiring and params.enable_replenish_hiring:
        raise ValueError(
            "enable_hiring and enable_replenish_hiring are mutually exclusive; "
            "pick exactly one hiring path"
        )
    if params.hire_delay_periods < 1:
        raise ValueError(
            f"hire_delay_periods must be >= 1; got {params.hire_delay_periods}"
        )
    if params.max_hire_period < 0:
        raise ValueError(
            f"max_hire_period must be >= 0 (0 = drain all per period); "
            f"got {params.max_hire_period}"
        )


def make_firm(params: FirmParams) -> Firm:
    """Construct a Firm: seed RNG, sample alpha/beta, sample workforce once, reset to all-H.

    Workforce is sampled here (CRIT-2 fix): reset() must NOT resample workforce,
    because check2_greedy_dominance reuses one firm across five run_simulation calls
    and relies on reset() not consuming rng state.
    """
    validate_hiring_params(params)
    rng = np.random.default_rng(params.seed)
    alpha = sample_alpha(params.N, rng, params.alpha_mean, params.alpha_concentration)
    beta = sample_beta(params.N, rng, params.beta_mean, params.beta_concentration)
    firm = Firm(params=params, alpha=alpha, beta=beta, rng=rng)
    # Sample workforce BEFORE reset() so rng state is stable across resets (CRIT-2)
    firm.workforce = _make_initial_workforce(params, rng)
    firm.K0 = firm.workforce.K  # set ONCE at construction; never reassigned by reset()
    firm.reset()
    return firm
