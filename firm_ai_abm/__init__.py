"""firm_ai_abm — agent-based simulation of firm responses to AI."""
from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import Firm, make_firm
from firm_ai_abm.production import Mode, compute_K, productivity_vec, cost_vec
from firm_ai_abm.adjustment import adj_cost
from firm_ai_abm.strategy import (
    all_H, all_A, all_T, greedy_profit, greedy_with_switching,
)
from firm_ai_abm.simulate import run_simulation
from firm_ai_abm.workers import Workforce, sample_workforce, task_to_worker_map, _make_initial_workforce
from firm_ai_abm.viz import fig1_primary_lines, fig2_small_multiples_q_a, fig3_mode_mix_greedy

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
    # Phase 1.5 Stage 1: worker heterogeneity
    "Workforce",
    "sample_workforce",
    "task_to_worker_map",
    "_make_initial_workforce",
    # Visualization
    "fig1_primary_lines",
    "fig2_small_multiples_q_a",
    "fig3_mode_mix_greedy",
]
