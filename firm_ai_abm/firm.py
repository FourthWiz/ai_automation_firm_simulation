"""Firm dataclass + factory. reset() is R-07-compliant: never touches alpha/beta."""
from dataclasses import dataclass, field
import numpy as np

from firm_ai_abm.config import FirmParams
from firm_ai_abm.production import Mode
from firm_ai_abm.tasks import sample_alpha, sample_beta


@dataclass
class Firm:
    params: FirmParams
    alpha: np.ndarray
    beta: np.ndarray
    modes: np.ndarray | None = None
    K: int = 0
    history: list = field(default_factory=list)
    rng: np.random.Generator | None = None

    def reset(self) -> None:
        """Reset modes/K/history. NEVER touches alpha or beta (R-07)."""
        N = self.params.N
        self.modes = np.zeros(N, dtype=int)  # all H = Mode.H = 0
        self.K = N // self.params.tasks_per_worker
        self.history = []


def make_firm(params: FirmParams) -> Firm:
    """Construct a Firm: seed RNG, sample alpha/beta once, reset to all-H."""
    rng = np.random.default_rng(params.seed)
    alpha = sample_alpha(params.N, rng)
    beta = sample_beta(params.N, rng)
    firm = Firm(params=params, alpha=alpha, beta=beta, rng=rng)
    firm.reset()
    return firm
