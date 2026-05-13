"""Phase 1.5 FirmBehavior Streamlit dashboard.

Single-page app that imports the simulation kernel directly and renders
13 Plotly charts across 5 tabs with primary controls in the main panel.

Architecture notes:
- @st.cache_data keyed on a 31-tuple of scalars (D-01): avoids passing
  the FirmParams dataclass directly and sidesteps hash_funcs API.
- Default seed=0 (D-02): prevents cache instability from seed=None.
- Run button gates the simulation (D-04): no auto-rerun on slider drag.
- try/except wraps run_cached (D-06): kernel exceptions become st.error banners.
- RUN_COUNTER (T-07): mutable cell that increments only on actual computation.
  The cached function returns a monotonic timestamp (time.monotonic_ns()) that
  is the SAME on cache hits (same cached 5-tuple). The app writes this timestamp
  to session_state["RUN_COUNTER_VAL_THIS_RUN"] so AppTest can detect cache hits
  by checking if the value changes between calls.
- N default: FirmParams().N = 100 (sidebar previously hardcoded 500 — corrected).

Run:
    .venv/bin/streamlit run app.py
"""

import dataclasses
import math
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import streamlit as st

from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import make_firm
from firm_ai_abm.simulate import run_simulation
from firm_ai_abm.strategy import (
    all_H,
    all_A,
    all_T,
    greedy_profit,
    greedy_with_switching,
)
from firm_ai_abm.margin_optimizer import target_margin_strategy
from firm_ai_abm.dashboard import (
    fig_pi_per_period_over_time,
    fig_pi_over_time,
    fig_K_over_time,
    fig_mode_mix_area,
    fig_wage_bill_over_time,
    fig_theta_histogram,
    fig_mean_theta_over_time,
    fig_firing_events,
    fig_trained_capital,
    fig_wage_histogram,
    fig_wage_vs_mean_output,
    fig_hiring_events,
    fig_mean_accum_wage_over_time,
)
from firm_ai_abm.production import Mode

# set_page_config MUST be the first Streamlit call at module level
st.set_page_config(
    page_title="Firm Behavior under AI — Simulator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PLAUSIBLE_DOMAIN = ""  # Set to live domain at deploy time (e.g. "firm-behavior.streamlit.app")
# Plausible is cookieless and GDPR-friendly; tracks page-view count and country only.
if PLAUSIBLE_DOMAIN:
    st.markdown(
        f'<script defer data-domain="{PLAUSIBLE_DOMAIN}" '
        f'src="https://plausible.io/js/script.js"></script>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# T-04: Strategy registry and params key helpers
# ---------------------------------------------------------------------------

_STRATEGY_REGISTRY = {
    "all_H": all_H,
    "all_A": all_A,
    "all_T": all_T,
    "greedy_profit": greedy_profit,
    "greedy_with_switching": greedy_with_switching,
    "target_margin": target_margin_strategy,
}

# 30 scalar FirmParams fields in order (excludes 'seed' which is appended separately)
# Indices: 0=N, 15=sigma_theta, 20=T_review, 21=firing_threshold, 22=scenario_mode,
#          23=target_margin, 24=margin_horizon, 25=enable_training_delay,
#          26=enable_hiring, 27=enable_replenish_hiring, 28=max_hire_period,
#          29=hire_delay_periods, -1=seed (position 30)
# NOTE: alpha-cost fields (c_auto_alpha_slope, c_auto_alpha_intercept, belief_alpha)
# are intentionally absent — no sidebar exposure.
_PARAM_FIELDS = (
    "N", "T", "tasks_per_worker",
    "q_h", "q_a", "g",
    "w", "c_aug", "c_auto", "c_fire", "c_hire", "c_train", "F",
    "p",
    "n_amortize",
    "sigma_theta", "theta_min", "theta_max", "corr_w_theta", "sigma_w",
    "T_review", "firing_threshold",
    "scenario_mode", "target_margin", "margin_horizon", "enable_training_delay",
    "enable_hiring",                # index 26
    "enable_replenish_hiring",      # index 27
    "max_hire_period",              # index 28
    "hire_delay_periods",           # index 29; seed moves to position 30 (still key[-1])
)

# Named index constant for sigma_theta (used in tab_het branch logic)
_SIGMA_THETA_IDX = _PARAM_FIELDS.index("sigma_theta")  # == 15

# Subset assertion: catch misspellings without requiring exhaustive coverage.
# Alpha-cost fields are intentionally excluded from _PARAM_FIELDS (no sidebar).
assert set(_PARAM_FIELDS + ("seed",)) <= {f.name for f in dataclasses.fields(FirmParams)}, (
    "Unknown field in _PARAM_FIELDS — check for misspellings"
)


def params_to_key(params: FirmParams, seed: int) -> tuple:
    """Build a 31-tuple cache key from a FirmParams instance and seed.

    The tuple contains only Python scalars (int, float, bool, str). math.inf is
    included as a float — hash(math.inf) is stable in CPython.
    seed is appended as the 31st element (position 30).

    Indices: 0=N, 15=sigma_theta, 20=T_review, 21=firing_threshold, 22=scenario_mode,
             23=target_margin, 24=margin_horizon, 25=enable_training_delay,
             26=enable_hiring, 27=enable_replenish_hiring, 28=max_hire_period,
             29=hire_delay_periods, -1=seed (position 30)
    """
    values = tuple(getattr(params, f) for f in _PARAM_FIELDS)
    return values + (seed,)


# All widget key strings used in _build_controls — for Reset button
# Net delta from prior version: +strategy, +strategy_adv, +hiring_mode, -enable_hiring,
# -enable_replenish_hiring → net +1; final count = 32.
ALL_WIDGET_KEYS = (
    "strategy", "strategy_adv",
    "N", "T", "seed",
    "q_a", "g", "c_auto", "w", "p", "target_margin",
    "scenario",
    "tasks_per_worker",
    "q_h", "c_aug", "c_fire", "c_hire", "c_train", "F",
    "n_amortize", "margin_horizon",
    "sigma_theta", "theta_min", "theta_max", "corr_w_theta", "sigma_w",
    "T_review", "firing_threshold",
    "hiring_mode",
    "hire_delay_periods", "max_hire_period",
    "enable_training_delay",
)

# ---------------------------------------------------------------------------
# T-07: Cache-hit observability counter
# ---------------------------------------------------------------------------

# Module-level mutable cell. @st.cache_data does NOT call the function on a
# cache hit, so this counter does not advance on hits.
# NOTE: RUN_COUNTER is not reliably shared between the test process and AppTest's
# internal execution context. The observable for cache-hit tests is the monotonic
# timestamp embedded in the cached return tuple: identical timestamps → cache hit.
RUN_COUNTER = [0]


# ---------------------------------------------------------------------------
# T-04 continued: Cached simulation runner
# ---------------------------------------------------------------------------

@st.cache_data
def run_cached(params_key: tuple, strategy_name: str) -> tuple:
    """Run the simulation for the given params key and strategy.

    Args:
        params_key: 31-tuple from params_to_key(). The last element is seed.
            Indices 15=sigma_theta, 20=T_review, 21=firing_threshold, 22=scenario_mode,
            23=target_margin, 24=margin_horizon, 25=enable_training_delay,
            26=enable_hiring, 27=enable_replenish_hiring, 28=max_hire_period,
            29=hire_delay_periods, -1=seed (position 30).
        strategy_name: key into _STRATEGY_REGISTRY.

    Returns:
        7-tuple: (df, theta_final, wages_final, K_max, call_id, output_per_worker, a_trained_final)
          df:               run_simulation DataFrame (T rows × 13+ columns)
          theta_final:      list[float] — firm.workforce.theta after final period
          wages_final:      list[float] — firm.workforce.wage after final period
          K_max:            int — N // tasks_per_worker (max possible workers)
          call_id:          int — RUN_COUNTER[0] at time of actual computation;
                            identical for cache hits (same cached value returned)
          output_per_worker: list[list[float]] — firm.output_per_worker.tolist()
          a_trained_final:  list[bool] — firm.workforce.a_trained.tolist()

    Cache-hit observability: a monotonic timestamp (time.monotonic_ns()) is
        captured at actual computation time and returned as the 5th element.
        A cache hit returns the SAME (old) timestamp. The caller writes this
        timestamp to st.session_state["RUN_COUNTER_VAL_THIS_RUN"]; AppTest
        tests detect cache hits by checking the timestamp does not change.
    """
    RUN_COUNTER[0] += 1
    computed_at = time.monotonic_ns()

    # Reconstruct FirmParams from the scalar tuple
    field_dict = dict(zip(_PARAM_FIELDS + ("seed",), params_key))
    params = FirmParams(**field_dict)

    firm = make_firm(params)
    df = run_simulation(firm, _STRATEGY_REGISTRY[strategy_name])

    theta_final = firm.workforce.theta.tolist()
    wages_final = firm.workforce.wage.tolist()
    K_max = params.N // params.tasks_per_worker
    output_per_worker = firm.output_per_worker.tolist()
    a_trained_final = firm.workforce.a_trained.tolist()
    return df, theta_final, wages_final, K_max, computed_at, output_per_worker, a_trained_final


# ---------------------------------------------------------------------------
# P1-4: Controls (primary bar + advanced expander)
# ---------------------------------------------------------------------------

def _build_controls() -> tuple:
    """Render primary controls in the main panel and return (strategy, params_key, draft_params).

    Primary controls are rendered inline (called before the run button).
    Advanced controls are inside an expander below the primary bar.

    Layout:
      Row A: strategy (non-adv radio) | T | N
      Row B: q_a | g | c_auto | w
      Row C: tasks_per_worker | T_review | hire_delay_periods | max_hire_period
      Hiring policy radio (D-01 mutex)
      Advanced expander (6 tabs):
        Costs | Strategy & pricing | Worker heterogeneity |
        Firing (advanced) | Productivity baseline | Reproducibility

    Returns:
        (strategy_name: str, params_key: tuple[31 scalars], draft_params: FirmParams)
    """
    # ------------------------------------------------------------------
    # Row A: strategy radio (simple) | T | N
    # D-02: non-advanced shows greedy_profit / greedy_with_switching only
    # ------------------------------------------------------------------
    col_a1, col_a2, col_a3 = st.columns([2, 1, 1])

    with col_a1:
        strategy = st.radio(
            "Strategy",
            ["greedy_profit", "greedy_with_switching"],
            index=0,
            key="strategy",
            horizontal=True,
        )

    with col_a2:
        T = st.number_input(
            "T", min_value=1, max_value=200,
            value=FirmParams().T, step=10,
            help="Simulation periods", key="T",
        )

    with col_a3:
        N = st.number_input(
            "Firm size in tasks (N)",
            min_value=1, max_value=1000,
            value=int(FirmParams().N),
            step=10,
            help="Number of tasks per firm (FirmParams default: 100)",
            key="N",
        )

    # ------------------------------------------------------------------
    # Row B: q_a | g | c_auto | w
    # ------------------------------------------------------------------
    col_b1, col_b2, col_b3, col_b4 = st.columns([1, 1, 1, 1])

    with col_b1:
        q_a = st.slider(
            "AI productivity ceiling (q_a)",
            0.0, 3.0, float(FirmParams().q_a), 0.05,
            help="Automation productivity multiplier",
            key="q_a",
        )

    with col_b2:
        g = st.slider(
            "Augmentation gain (g)",
            0.0, 3.0, float(FirmParams().g), 0.05,
            help="Augmentation gain parameter",
            key="g",
        )

    with col_b3:
        c_auto = st.slider(
            "Automation cost per task (c_auto)",
            0.0, 2.0, float(FirmParams().c_auto), 0.05,
            help="Automation cost per task",
            key="c_auto",
        )

    with col_b4:
        w = st.slider(
            "Wage rate (w)",
            0.0, 5.0, float(FirmParams().w), 0.1,
            help="Wage rate",
            key="w",
        )

    # ------------------------------------------------------------------
    # Row C: tasks_per_worker | T_review | hire_delay_periods | max_hire_period
    # D-04: reclaimed from p/target_margin (moved to Advanced).
    # NOTE: tasks_per_worker is defined here BEFORE the Advanced expander so
    # that firing_threshold_kernel conversion inside the expander can use it (R-01).
    # ------------------------------------------------------------------
    # D-01: hiring_mode radio must also be defined before Row C so that
    # hire_delay_periods and max_hire_period disabled-state can reference it.
    # Render hiring_mode first, then row C.

    hiring_mode = st.radio(
        "Hiring policy",
        options=["off", "enable_hiring", "enable_replenish_hiring"],
        index=0,
        key="hiring_mode",
        horizontal=True,
        help="off = no hiring after firing; enable_hiring = immediate refill to K*; "
             "enable_replenish_hiring = delayed backlog refill (mutually exclusive)",
    )

    col_c1, col_c2, col_c3, col_c4 = st.columns([1, 1, 1, 1])

    with col_c1:
        tasks_per_worker = st.number_input(
            "Tasks per worker", min_value=1, max_value=100,
            value=FirmParams().tasks_per_worker, step=1,
            help="K_workforce = N / tasks_per_worker", key="tasks_per_worker",
        )

    with col_c2:
        T_REVIEW_OPTIONS = [5, 10, 20, 30, "inf"]
        T_review_choice = st.select_slider(
            "T_review", options=T_REVIEW_OPTIONS,
            value=5,  # UI default 5 differs from FirmParams().T_review = math.inf (two-defaults seam)
            help="Periodic firing review interval ('inf' = disabled)", key="T_review",
        )
        T_review_value = math.inf if T_review_choice == "inf" else float(T_review_choice)

    with col_c3:
        hire_delay_periods_val = st.number_input(
            "hire_delay_periods", min_value=1, max_value=20,
            value=1, step=1, key="hire_delay_periods",
            help="Periods to wait before hiring back fired workers. Active only when enable_replenish_hiring=True.",
            disabled=(hiring_mode != "enable_replenish_hiring"),
        )

    with col_c4:
        max_hire_period_val = st.number_input(
            "max_hire_period", min_value=0, max_value=200,
            # UI default 5 diverges from kernel sentinel 0 (drain entire backlog); user confirmed
            value=5, step=1, key="max_hire_period",
            help="Per-period hire cap from backlog. 0 = drain entire backlog in one period.",
            disabled=(hiring_mode == "off"),
        )

    # ------------------------------------------------------------------
    # Advanced expander: 6 tabs
    # D-05: new tab list — Run length and Firing & hiring tabs replaced
    # ------------------------------------------------------------------
    with st.expander("Advanced parameters", expanded=False):
        adv_tabs = st.tabs([
            "Costs",
            "Strategy & pricing",
            "Worker heterogeneity",
            "Firing (advanced)",
            "Productivity baseline",
            "Reproducibility",
        ])

        # Costs (unchanged)
        with adv_tabs[0]:
            c_aug = st.slider("c_aug", 0.0, 1.0, float(FirmParams().c_aug), 0.01,
                              help="Augmentation cost per task", key="c_aug")
            c_fire = st.slider("c_fire", 0.0, 10.0, float(FirmParams().c_fire), 0.1,
                               help="Firing cost per worker", key="c_fire")
            c_hire = st.slider("c_hire", 0.0, 5.0, float(FirmParams().c_hire), 0.1,
                               help="Hiring cost per worker", key="c_hire")
            c_train = st.slider("c_train", 0.0, 2.0, float(FirmParams().c_train), 0.05,
                                help="Training cost per worker", key="c_train")
            F = st.slider("F", 0.0, 20.0, float(FirmParams().F), 0.5,
                          help="Fixed overhead cost per period", key="F")

        # Strategy & pricing (D-02: scenario + strategy_adv + p + target_margin + meta)
        with adv_tabs[1]:
            scenario = st.radio("Scenario", ["price", "margin"], index=0, key="scenario")
            if scenario == "margin":
                st.markdown("**Strategy:** target_margin *(auto)*")
                strategy_adv = "target_margin"
            else:
                strategy_adv = st.radio(
                    "Strategy (override)",
                    list(_STRATEGY_REGISTRY.keys()),
                    index=3,  # greedy_profit
                    key="strategy_adv",
                )
            p = st.slider(
                "Output price (p)", 0.1, 2.0, float(FirmParams().p), 0.05,
                help="Output price (inactive in margin scenario)",
                key="p",
                disabled=(scenario == "margin"),
            )
            target_margin_val = st.slider(
                "Target margin (target_margin)",
                0.0, 0.5, float(FirmParams().target_margin), 0.01,
                help="Target (revenue − cost) / revenue for the margin-optimizer strategy",
                key="target_margin",
                disabled=(scenario == "price"),
            )
            n_amortize = st.number_input(
                "n_amortize", min_value=1, max_value=24,
                value=FirmParams().n_amortize, step=1,
                help="Amortization periods for greedy_with_switching", key="n_amortize",
            )
            margin_horizon_val = st.number_input(
                "margin_horizon", min_value=1, max_value=20,
                value=FirmParams().margin_horizon, step=1,
                help="Look-ahead periods for margin-optimizer brute grid",
                key="margin_horizon",
            )

        # Worker heterogeneity (unchanged)
        with adv_tabs[2]:
            sigma_theta = st.slider(
                "sigma_theta", 0.0, 0.5, float(FirmParams().sigma_theta), 0.01,
                help="Std dev of theta (worker skill)", key="sigma_theta",
            )
            theta_min = st.slider(
                "theta_min", 0.1, 1.0, float(FirmParams().theta_min), 0.05,
                help="Minimum worker skill", key="theta_min",
            )
            theta_max = st.slider(
                "theta_max", 1.0, 2.0, float(FirmParams().theta_max), 0.05,
                help="Maximum worker skill", key="theta_max",
            )
            corr_w_theta = st.slider(
                "corr_w_theta", 0.0, 1.5, float(FirmParams().corr_w_theta), 0.05,
                help="Correlation between wage and theta", key="corr_w_theta",
            )
            sigma_w = st.slider(
                "sigma_w", 0.0, 0.3, float(FirmParams().sigma_w), 0.01,
                help="Wage noise std dev", key="sigma_w",
            )

        # Firing (advanced): only firing_threshold remains; T_review/hiring moved to non-adv
        with adv_tabs[3]:
            # firing_threshold_ui is in per-task-output units.
            # UI range [-1.0, 1.0] maps to kernel range [-tpw, tpw].
            firing_threshold_ui = st.slider(
                "firing_threshold", -1.0, 1.0, 0.0, 0.05,
                help=(
                    "Per-task surplus threshold. Workers fired when "
                    "(p · mean_output − wage) < threshold · tasks_per_worker. "
                    "0 = fire workers whose price-scaled output does not cover their wage."
                ),
                key="firing_threshold",
            )
            # Convert UI (per-task) → kernel (per-worker) by multiplying by tasks_per_worker.
            # tasks_per_worker is defined in non-advanced Row C above (R-01 ordering).
            firing_threshold_kernel = float(firing_threshold_ui) * int(tasks_per_worker)

        # Productivity baseline (unchanged)
        with adv_tabs[4]:
            q_h = st.slider(
                "q_h", 0.0, 3.0, float(FirmParams().q_h), 0.05,
                help="Human productivity per task", key="q_h",
            )
            enable_training_delay_val = st.checkbox(
                "enable_training_delay",
                value=True,
                help="When enabled, H→A workers produce at H-rate for 1 period before augmentation kicks in.",
                key="enable_training_delay",
            )

        # Reproducibility (D-03: seed moved here from non-advanced)
        with adv_tabs[5]:
            seed = st.number_input(
                "Seed (random)",
                min_value=0, value=0, step=1,
                help="seed=0 default; change to vary worker draws.",
                key="seed",
            )

    # ------------------------------------------------------------------
    # D-01: inline UI→kernel mapping for hiring_mode radio
    # ------------------------------------------------------------------
    enable_hiring_val = (hiring_mode == "enable_hiring")
    enable_replenish_hiring_val = (hiring_mode == "enable_replenish_hiring")

    # ------------------------------------------------------------------
    # D-02: merge active_strategy from both radios
    # Rule: scenario=margin → target_margin always.
    #       Advanced radio != default ("greedy_profit") → advanced wins.
    #       Otherwise non-advanced radio wins.
    # ------------------------------------------------------------------
    _ADV_DEFAULT = "greedy_profit"
    if scenario == "margin":
        active_strategy = "target_margin"
    elif strategy_adv != _ADV_DEFAULT:
        active_strategy = strategy_adv
    else:
        active_strategy = strategy  # non-advanced wins when advanced is at default

    # ------------------------------------------------------------------
    # Build FirmParams and cache key
    # ------------------------------------------------------------------
    draft_params = FirmParams(
        N=int(N),
        T=int(T),
        tasks_per_worker=int(tasks_per_worker),
        q_h=float(q_h),
        q_a=float(q_a),
        g=float(g),
        w=float(w),
        c_aug=float(c_aug),
        c_auto=float(c_auto),
        c_fire=float(c_fire),
        c_hire=float(c_hire),
        c_train=float(c_train),
        F=float(F),
        p=float(p),
        n_amortize=int(n_amortize),
        sigma_theta=float(sigma_theta),
        theta_min=float(theta_min),
        theta_max=float(theta_max),
        corr_w_theta=float(corr_w_theta),
        sigma_w=float(sigma_w),
        T_review=T_review_value,
        firing_threshold=firing_threshold_kernel,
        scenario_mode=scenario,
        target_margin=float(target_margin_val),
        margin_horizon=int(margin_horizon_val),
        enable_training_delay=bool(enable_training_delay_val),
        enable_hiring=bool(enable_hiring_val),
        enable_replenish_hiring=bool(enable_replenish_hiring_val),
        max_hire_period=int(max_hire_period_val),
        hire_delay_periods=int(hire_delay_periods_val),
        seed=int(seed),
    )
    key = params_to_key(draft_params, int(seed))

    # load-bearing for tests: AppTest reads DRAFT_PARAMS_DEBUG to verify inline UI→kernel mapping
    # do NOT remove — removing breaks test_replenish_hiring_toggle_changes_cache_key
    #                                  and test_hiring_mode_mutex_kernel_mapping
    st.session_state["DRAFT_PARAMS_DEBUG"] = draft_params

    # load-bearing for tests: AppTest reads LAST_STRATEGY_DEBUG to verify merge function
    # do NOT remove — removing breaks test_strategy_merge_function
    st.session_state["LAST_STRATEGY_DEBUG"] = active_strategy

    return active_strategy, key, draft_params


# ---------------------------------------------------------------------------
# T-06: Main panel
# ---------------------------------------------------------------------------

def main() -> None:
    """Main app entry point."""
    st.title("Firm Behavior under AI")
    st.caption(
        "An agent-based simulator of how firms choose between human, AI-augmented, "
        "and fully automated production. Adjust the dials, hit Run, and see how "
        "profit, workforce, and task mix evolve over 60 periods."
    )

    strategy, params_key, draft_params = _build_controls()

    # ------------------------------------------------------------------
    # P1-7: Run button + Reset button
    # ------------------------------------------------------------------
    run_col, reset_col = st.columns([3, 1])
    with run_col:
        run_clicked = st.button("▶ Run simulation", type="primary", key="run_button")
    with reset_col:
        reset_clicked = st.button(
            "Reset to defaults",
            key="reset_button",
            help="Restores all controls to default values. Click Run after to re-simulate.",
        )

    if reset_clicked:
        for k in list(st.session_state.keys()):
            if k in ALL_WIDGET_KEYS:
                del st.session_state[k]
        st.rerun()

    # Initialize last_run_key on first load so the page renders with defaults
    if "last_run_key" not in st.session_state:
        st.session_state["last_run_key"] = params_key
        st.session_state["last_strategy"] = strategy

    if run_clicked:
        st.session_state["last_run_key"] = params_key
        st.session_state["last_strategy"] = strategy

    active_key = st.session_state["last_run_key"]
    active_strategy = st.session_state["last_strategy"]

    # T-10: stale-run banner — warn when current params differ from last run
    if (params_key != st.session_state["last_run_key"]) or (strategy != st.session_state["last_strategy"]):
        st.warning("Params changed — press Run to refresh", icon="⚠️")

    st.caption(f"Strategy: {active_strategy} — seed: {active_key[-1]}")

    # Run simulation (or return cached result).
    # Spinner is only shown on explicit run clicks (not on initial load or cache hits).
    if run_clicked:
        spinner_ctx = st.spinner("Simulating…")
        spinner_ctx.__enter__()
    try:
        df, theta_final, wages_final, K_max, computed_at, output_per_worker_list, a_trained_final = run_cached(active_key, active_strategy)
    except Exception as e:
        if run_clicked:
            spinner_ctx.__exit__(None, None, None)
        st.error(f"Simulation failed: {type(e).__name__}: {e}")
        st.code(traceback.format_exc(), language="text")
        st.stop()
    if run_clicked:
        spinner_ctx.__exit__(None, None, None)

    # Write computed_at to session_state so AppTest cache-hit tests can observe it.
    st.session_state["RUN_COUNTER_VAL_THIS_RUN"] = computed_at

    # Stage 5 T-09: cascade warning when workforce drops >50% during run
    k_initial = int(df["K_active"].iloc[0])
    k_final = int(df["K_active"].iloc[-1])
    if k_initial > 0 and k_final < 0.5 * k_initial:
        st.warning(
            f"Workforce dropped >50% during run (K_initial={k_initial}, K_final={k_final}) — "
            "try a lower firing_threshold."
        )

    # ------------------------------------------------------------------
    # P1-5: KPI strip
    # ------------------------------------------------------------------
    cum_pi = float(df["pi"].cumsum().iloc[-1])
    final_K = int(df["K_active"].iloc[-1])
    modes_arr = np.stack(df["modes"].values)   # shape (T, N)
    shares = {
        "Human": float((modes_arr == int(Mode.H)).mean()),
        "Augmented": float((modes_arr == int(Mode.A)).mean()),
        "Automated": float((modes_arr == int(Mode.T)).mean()),
    }
    dom_mode = max(shares, key=shares.get)
    k1, k2, k3 = st.columns(3)
    k1.metric("Cumulative profit", f"{cum_pi:.2f}")
    k2.metric("Final workforce (K)", f"{final_K}")
    k3.metric("Dominant mode", dom_mode, f"{shares[dom_mode]*100:.0f}% of tasks")

    # ------------------------------------------------------------------
    # P1-6: 5-tab plot layout
    # Derive hiring flags from active_key (last-run state), not draft params
    # ------------------------------------------------------------------
    enable_hiring_active = bool(active_key[26])
    enable_replenish_active = bool(active_key[27])

    # Prepare wage-vs-output data (preserve existing slice logic verbatim)
    opw_arr = np.array(output_per_worker_list)
    K_active_final = int(df["K_active"].iloc[-1])
    if K_active_final > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_opw = np.nanmean(opw_arr[:, :K_active_final], axis=0)
        all_nan_mask = np.all(np.isnan(opw_arr[:, :K_active_final]), axis=0)
        mean_opw[all_nan_mask] = np.nan
        wages_slice = np.array(wages_final[:K_active_final])
        a_trained_slice = np.array(a_trained_final[:K_active_final], dtype=bool)
    else:
        mean_opw = np.array([])
        wages_slice = np.array([])
        a_trained_slice = np.array([], dtype=bool)

    tab_out, tab_work, tab_modes, tab_wages, tab_het = st.tabs([
        "Outcomes", "Workforce", "Tasks & modes", "Wages", "Worker heterogeneity"
    ])

    with tab_out:
        st.plotly_chart(fig_pi_per_period_over_time(df), width="stretch", key="fig_pi_period")
        st.plotly_chart(fig_pi_over_time(df), width="stretch", key="fig_pi_cumul")

    with tab_work:
        st.plotly_chart(fig_K_over_time(df), width="stretch", key="fig_K")
        st.plotly_chart(
            fig_hiring_events(df, enable_hiring=enable_hiring_active,
                              enable_replenish_hiring=enable_replenish_active),
            width="stretch", key="fig_hiring",
        )
        st.plotly_chart(
            fig_firing_events(df, T_review=float(active_key[20])),
            width="stretch", key="fig_firing",
        )

    with tab_modes:
        st.plotly_chart(fig_mode_mix_area(df, int(active_key[0])), width="stretch", key="fig_modes")
        st.plotly_chart(fig_trained_capital(df), width="stretch", key="fig_trained")

    with tab_wages:
        st.plotly_chart(fig_wage_histogram(np.array(wages_final)), width="stretch", key="fig_wage_hist")
        st.plotly_chart(
            fig_wage_vs_mean_output(wages_slice, mean_opw, a_trained_slice),
            width="stretch", key="fig_wage_scatter",
        )
        st.plotly_chart(fig_wage_bill_over_time(df), width="stretch", key="fig_wage_bill")
        st.plotly_chart(fig_mean_accum_wage_over_time(df), width="stretch", key="fig_accum_wage")

    with tab_het:
        if active_key[_SIGMA_THETA_IDX] == 0.0:
            # sigma_theta=0 → render charts with empty/degenerate inputs to preserve 13-plot count
            st.plotly_chart(fig_theta_histogram(np.array([])), width="stretch", key="fig_theta_hist")
            st.plotly_chart(fig_mean_theta_over_time(df), width="stretch", key="fig_mean_theta")
        else:
            st.plotly_chart(fig_theta_histogram(np.array(theta_final)), width="stretch", key="fig_theta_hist")
            st.plotly_chart(fig_mean_theta_over_time(df), width="stretch", key="fig_mean_theta")

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    theta_arr = np.array(theta_final)
    wages_arr = np.array(wages_final)
    total_firings = int(df["n_review_fired"].sum()) if "n_review_fired" in df.columns else 0
    total_hirings = int(df["n_hired"].sum()) if "n_hired" in df.columns else 0
    final_k_active = int(df["K_active"].iloc[-1]) if "K_active" in df.columns else 0
    final_pi = float(df["pi"].cumsum().iloc[-1])

    if len(theta_arr) > 1 and len(wages_arr) > 1 and np.all(wages_arr > 0):
        realized_corr = float(np.corrcoef(theta_arr, np.log(wages_arr))[0, 1])
    else:
        realized_corr = float("nan")

    # Determine hiring path label for caption
    if enable_hiring_active:
        hiring_path = "K*-target"
    elif enable_replenish_active:
        hiring_path = "replenish"
    else:
        hiring_path = "none"

    # Stage 5 T-09: NaN guard when K dropped to 0
    if math.isnan(realized_corr) and final_k_active == 0:
        st.caption(
            "corr = N/A (workforce dropped to K=0; consider reducing firing_threshold)"
            f"    total firings = {total_firings}"
            f"    Hires: {total_hirings} (path: {hiring_path})"
            f"    final K_active = {final_k_active}"
        )
    else:
        st.caption(
            f"corr(theta, log(wage)) = {realized_corr:.4f}"
            f"    final cumulative profit = {final_pi:.4f}"
            f"    total firings = {total_firings}"
            f"    Hires: {total_hirings} (path: {hiring_path})"
            f"    final K_active = {final_k_active}"
        )
    # Show cache-hit observability via RUN_COUNTER.
    st.caption(f"sim invocations this session: {RUN_COUNTER[0]}")


main()
