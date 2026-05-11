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

    @property
    def K(self) -> int:
        return self.workforce.K if self.workforce is not None else 0

    def reset(self) -> None:
        """Reset modes/history. NEVER touches alpha, beta, or workforce (R-07, CRIT-2)."""
        N = self.params.N
        self.modes = np.zeros(N, dtype=int)  # all H = Mode.H = 0
        self.history = []
        self._margin_cache: dict = {}  # type: ignore[attr-defined]
        # workforce is NOT re-sampled here — it persists for the firm's lifetime


def make_firm(params: FirmParams) -> Firm:
    """Construct a Firm: seed RNG, sample alpha/beta, sample workforce once, reset to all-H.

    Workforce is sampled here (CRIT-2 fix): reset() must NOT resample workforce,
    because check2_greedy_dominance reuses one firm across five run_simulation calls
    and relies on reset() not consuming rng state.
    """
    rng = np.random.default_rng(params.seed)
    alpha = sample_alpha(params.N, rng)
    beta = sample_beta(params.N, rng)
    firm = Firm(params=params, alpha=alpha, beta=beta, rng=rng)
    # Sample workforce BEFORE reset() so rng state is stable across resets (CRIT-2)
    firm.workforce = _make_initial_workforce(params, rng)
    firm.reset()
    return firm
