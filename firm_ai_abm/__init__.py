"""firm_ai_abm — agent-based simulation of firm responses to AI."""
from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import Firm, make_firm
from firm_ai_abm.production import Mode, compute_K, productivity_vec, cost_vec
from firm_ai_abm.adjustment import adj_cost
from firm_ai_abm.strategy import (
    all_H, all_A, all_T, greedy_profit, greedy_with_switching,
)
# PEP 562 lazy import — keep bare `import firm_ai_abm` cheap and DP-free for
# scripts that only need the kernel + strategies.
_LAZY = {
    "run_simulation":                "firm_ai_abm.simulate",
    "run_horizon":                   "firm_ai_abm.simulate",
    "Workforce":                     "firm_ai_abm.workers",
    "sample_workforce":              "firm_ai_abm.workers",
    "task_to_worker_map":            "firm_ai_abm.workers",
    "_make_initial_workforce":       "firm_ai_abm.workers",
    "firing_review":                 "firm_ai_abm.review",
    "apply_firings":                 "firm_ai_abm.review",
    "replace_to_target":             "firm_ai_abm.review",
    "dp_rolling_horizon_strategy":   "firm_ai_abm.dp_optimizer",
    "horizon_brute_strategy":        "firm_ai_abm.margin_optimizer",
}

def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name])
        value = getattr(mod, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'firm_ai_abm' has no attribute {name!r}")

__all__ = [
    "FirmParams",
    "Firm",
    "make_firm",
    "Mode",
    "compute_K",
    "productivity_vec",
    "cost_vec",
    "adj_cost",
    "all_H",
    "all_A",
    "all_T",
    "greedy_profit",
    "greedy_with_switching",
    "run_simulation",
    "run_horizon",
    # Phase 1.5 Stage 1: worker heterogeneity
    "Workforce",
    "sample_workforce",
    "task_to_worker_map",
    "_make_initial_workforce",
    # Phase 1.5 Stage 3 / Stage 5: periodic firing review
    "firing_review",
    "apply_firings",
    "replace_to_target",
    # F-03: DP rolling-horizon optimizer
    "dp_rolling_horizon_strategy",
    "horizon_brute_strategy",
]
