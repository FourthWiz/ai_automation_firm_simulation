"""Phase 1.5 FirmBehavior Streamlit dashboard.

Single-page app that imports the simulation kernel directly and renders
8 plots (4 rows × 2 columns) with a sidebar for all FirmParams controls.

Architecture notes:
- @st.cache_data keyed on a 23-tuple of scalars (D-01): avoids passing
  the FirmParams dataclass directly and sidesteps hash_funcs API.
- Default seed=0 (D-02): prevents cache instability from seed=None.
- Run button gates the simulation (D-04): no auto-rerun on slider drag.
- try/except wraps run_cached (D-06): kernel exceptions become st.error banners.
- RUN_COUNTER (T-07): mutable cell that increments only on actual computation.
  The cached function returns a monotonic timestamp (time.monotonic_ns()) that
  is the SAME on cache hits (same cached 5-tuple). The app writes this timestamp
  to session_state["RUN_COUNTER_VAL_THIS_RUN"] so AppTest can detect cache hits
  by checking if the value changes between calls.

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
# Indices: 0=N, 20=T_review, 21=firing_threshold, 22=scenario_mode,
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

    Indices: 0=N, 20=T_review, 21=firing_threshold, 22=scenario_mode,
             23=target_margin, 24=margin_horizon, 25=enable_training_delay,
             26=enable_hiring, 27=enable_replenish_hiring, 28=max_hire_period,
             29=hire_delay_periods, -1=seed (position 30)
    """
    values = tuple(getattr(params, f) for f in _PARAM_FIELDS)
    return values + (seed,)


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
            Indices 20=T_review, 21=firing_threshold, 22=scenario_mode,
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
# T-05: Sidebar
# ---------------------------------------------------------------------------

def _build_sidebar() -> tuple:
    """Render the sidebar controls and return (params_key, strategy_name).

    Returns:
        (strategy_name: str, params_key: tuple[23 scalars])
    """
    st.sidebar.title("FirmBehavior dashboard")

    # Scenario selector — comes before strategy (T-09)
    scenario = st.sidebar.radio("Scenario", ["price", "margin"], index=0, key="scenario")

    if scenario == "margin":
        # Hide strategy radio in margin mode; strategy is forced to target_margin
        st.sidebar.markdown("Strategy: target_margin (auto)")
        strategy = "target_margin"
    else:
        # Strategy selector — default to greedy_profit (index 3)
        strategy = st.sidebar.radio(
            "Strategy",
            [k for k in _STRATEGY_REGISTRY if k != "target_margin"],
            index=3,
        )

    # Counts — use exact field names as labels (required for test_sidebar_all_param_fields_present)
    with st.sidebar.expander("Counts", expanded=True):
        N = st.number_input(
            "N", min_value=1, max_value=1000, value=500, step=10,
            help="Number of tasks per firm", key="N",
        )
        T = st.number_input(
            "T", min_value=1, max_value=200, value=FirmParams().T, step=10,
            help="Simulation periods", key="T",
        )
        tasks_per_worker = st.number_input(
            "tasks_per_worker", min_value=1, max_value=100,
            value=FirmParams().tasks_per_worker, step=1,
            help="K_workforce = N / tasks_per_worker. Lowering this RAISES K.", key="tasks_per_worker",
        )

    # Productivity
    with st.sidebar.expander("Productivity", expanded=False):
        q_h = st.slider("q_h", 0.0, 3.0, float(FirmParams().q_h), 0.05,
                        help="Human productivity per task", key="q_h")
        q_a = st.slider("q_a", 0.0, 3.0, float(FirmParams().q_a), 0.05,
                        help="Automation productivity multiplier", key="q_a")
        g = st.slider("g", 0.0, 3.0, float(FirmParams().g), 0.05,
                      help="Augmentation gain parameter", key="g")

    # Costs
    with st.sidebar.expander("Costs", expanded=False):
        w = st.slider("w", 0.0, 5.0, float(FirmParams().w), 0.1,
                      help="Wage rate", key="w")
        c_aug = st.slider("c_aug", 0.0, 1.0, float(FirmParams().c_aug), 0.01,
                          help="Augmentation cost per task", key="c_aug")
        c_auto = st.slider("c_auto", 0.0, 2.0, float(FirmParams().c_auto), 0.05,
                           help="Automation cost per task", key="c_auto")
        c_fire = st.slider("c_fire", 0.0, 10.0, float(FirmParams().c_fire), 0.1,
                           help="Firing cost per worker", key="c_fire")
        c_hire = st.slider("c_hire", 0.0, 5.0, float(FirmParams().c_hire), 0.1,
                           help="Hiring cost per worker", key="c_hire")
        c_train = st.slider("c_train", 0.0, 2.0, float(FirmParams().c_train), 0.05,
                            help="Training cost per worker", key="c_train")
        F = st.slider("F", 0.0, 20.0, float(FirmParams().F), 0.5,
                      help="Fixed overhead cost per period", key="F")

    # Price
    with st.sidebar.expander("Price", expanded=False):
        p = st.slider("p", 0.1, 2.0, float(FirmParams().p), 0.05,
                      help="Output price", key="p")

    # Strategy meta
    with st.sidebar.expander("Strategy meta", expanded=False):
        n_amortize = st.number_input(
            "n_amortize", min_value=1, max_value=24,
            value=FirmParams().n_amortize, step=1,
            help="Amortization periods for greedy_with_switching", key="n_amortize",
        )

    # Heterogeneity
    with st.sidebar.expander("Heterogeneity", expanded=False):
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

    # Firing review
    with st.sidebar.expander("Firing review", expanded=False):
        T_REVIEW_OPTIONS = ["inf", 5, 10, 20, 30]
        T_review_choice = st.select_slider(
            "T_review", options=T_REVIEW_OPTIONS, value="inf",
            help="Review period (inf = disabled)", key="T_review",
        )
        T_review_value = math.inf if T_review_choice == "inf" else float(T_review_choice)
        # firing_threshold_ui is in per-task-output units (comparable to q_h ≈ 1.0).
        # The kernel's surplus is per-worker output minus per-worker wage; per-worker
        # output ≈ q_h × tasks_per_worker, so the kernel threshold must be scaled.
        # UI range [-1.0, 1.0] maps to kernel range [-tpw, tpw].
        # Default 0.0: fire workers whose per-task output < per-worker wage.
        firing_threshold_ui = st.slider(
            "firing_threshold", -1.0, 1.0, 0.0, 0.05,
            help=(
                "Per-task surplus threshold. Workers fired when "
                "(p · mean_output − wage) < threshold · tasks_per_worker, "
                "where mean_output is summed over the worker's tasks. "
                "Threshold expressed in per-task price-scaled surplus units. "
                "0 = fire workers whose price-scaled output does not cover their wage."
            ),
            key="firing_threshold",
        )
        # Convert UI (per-task) → kernel (per-worker) by multiplying by tasks_per_worker
        firing_threshold_kernel = float(firing_threshold_ui) * int(tasks_per_worker)
        # enable_hiring UI default: False (matches FirmParams default) — hiring incurs c_hire; must be opt-in
        enable_hiring_val = st.checkbox(
            "enable_hiring",
            value=False,
            help="When enabled, fired workers are immediately rehired toward K*. c_hire is charged per hire.",
            key="enable_hiring",
        )
        # Replenishment hiring — mutually exclusive with enable_hiring.
        # Server-side validation in make_firm is authoritative; UI warns but does not block.
        enable_replenish_hiring_val = st.checkbox(
            "enable_replenish_hiring",
            value=False,
            help="Augmentation-targeted backlog hiring. Mutually exclusive with enable_hiring.",
            key="enable_replenish_hiring",
        )
        if enable_hiring_val and enable_replenish_hiring_val:
            st.warning("enable_hiring and enable_replenish_hiring are mutually exclusive — disable one.", icon="⚠️")
        hire_delay_periods_val = st.number_input(
            "hire_delay_periods", min_value=1, max_value=20,
            value=FirmParams().hire_delay_periods, step=1,
            help="Periods to wait before hiring back fired workers (>=1). Active only when enable_replenish_hiring=True.",
            key="hire_delay_periods",
            disabled=not enable_replenish_hiring_val,
        )
        max_hire_period_val = st.number_input(
            "max_hire_period", min_value=0, max_value=200,
            value=FirmParams().max_hire_period, step=1,
            help="Per-period hire cap from backlog. 0 = drain entire backlog in one period.",
            key="max_hire_period",
        )

    # Margin scenario (T-09)
    with st.sidebar.expander("Margin scenario", expanded=False):
        target_margin_val = st.slider(
            "target_margin", 0.0, 0.5, float(FirmParams().target_margin), 0.01,
            help="Target (revenue − cost) / revenue for the margin-optimizer strategy",
            key="target_margin",
        )
        margin_horizon_val = st.number_input(
            "margin_horizon", min_value=1, max_value=20,
            value=FirmParams().margin_horizon, step=1,
            help="Look-ahead periods for margin-optimizer brute grid",
            key="margin_horizon",
        )

    # Training (T-09 — dashboard default True; FirmParams default False — deliberate seam D-01)
    with st.sidebar.expander("Training", expanded=False):
        enable_training_delay_val = st.checkbox(
            "enable_training_delay",
            value=True,
            help="When enabled, H→A workers produce at H-rate for 1 period before augmentation kicks in.",
            key="enable_training_delay",
        )

    # Seed (D-02: default 0, not None)
    seed = st.sidebar.number_input(
        "Seed", min_value=0, value=0, step=1,
        help="seed=0 default; change to vary worker draws.",
        key="seed",
    )

    # Build a temporary FirmParams to use params_to_key
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
    return strategy, key, draft_params


# ---------------------------------------------------------------------------
# T-06: Main panel
# ---------------------------------------------------------------------------

def main() -> None:
    """Main app entry point."""
    st.title("Phase 1.5 simulation")

    strategy, params_key, draft_params = _build_sidebar()

    # Run button — gate the simulation (D-04: no auto-rerun on slider change)
    run_clicked = st.button("Run", type="primary")

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
    # computed_at is a monotonic timestamp captured at actual computation time.
    # A cache hit returns the SAME (old) timestamp — this is the observable.
    try:
        df, theta_final, wages_final, K_max, computed_at, output_per_worker_list, a_trained_final = run_cached(active_key, active_strategy)
    except Exception as e:
        st.error(f"Simulation failed: {type(e).__name__}: {e}")
        st.code(traceback.format_exc(), language="text")
        st.stop()

    # Write computed_at to session_state so AppTest cache-hit tests can observe it.
    # Cache hit: same computed_at (cached 5-tuple returned, function body not re-run).
    # Cache miss: new computed_at (function body ran, new timestamp).
    st.session_state["RUN_COUNTER_VAL_THIS_RUN"] = computed_at

    # Stage 5 T-09: cascade warning when workforce drops >50% during run
    k_initial = int(df["K_active"].iloc[0])
    k_final = int(df["K_active"].iloc[-1])
    if k_initial > 0 and k_final < 0.5 * k_initial:
        st.warning(
            f"Workforce dropped >50% during run (K_initial={k_initial}, K_final={k_final}) — "
            "try a lower firing_threshold."
        )

    # 6 rows × 2 columns layout (D-01; row6_right is blank placeholder for Phase 2)
    row1_left, row1_right = st.columns(2)
    with row1_left:
        # Stage 5 D-10: per-period profit replaces cumulative in row1_left
        st.pyplot(fig_pi_per_period_over_time(df))
    with row1_right:
        st.pyplot(fig_K_over_time(df))

    row2_left, row2_right = st.columns(2)
    with row2_left:
        st.pyplot(fig_mode_mix_area(df, int(active_key[0])))  # active_key[0] = N
    with row2_right:
        st.pyplot(fig_wage_bill_over_time(df))

    row3_left, row3_right = st.columns(2)
    with row3_left:
        st.pyplot(fig_theta_histogram(np.array(theta_final)))
    with row3_right:
        st.pyplot(fig_mean_theta_over_time(df))

    row4_left, row4_right = st.columns(2)
    with row4_left:
        st.pyplot(fig_firing_events(df, T_review=float(active_key[20])))
    with row4_right:
        st.pyplot(fig_trained_capital(df))

    row5_left, row5_right = st.columns(2)
    with row5_left:
        st.pyplot(fig_wage_histogram(np.array(wages_final)))
    with row5_right:
        opw_arr = np.array(output_per_worker_list)
        K_active_final = int(df["K_active"].iloc[-1])
        if K_active_final > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_opw = np.nanmean(opw_arr[:, :K_active_final], axis=0)
            all_nan_mask = np.all(np.isnan(opw_arr[:, :K_active_final]), axis=0)
            mean_opw[all_nan_mask] = np.nan
        else:
            mean_opw = np.array([])
        a_trained_arr = np.array(a_trained_final[:K_active_final], dtype=bool) if K_active_final > 0 else np.array([], dtype=bool)
        st.pyplot(fig_wage_vs_mean_output(
            np.array(wages_final[:K_active_final]),
            mean_opw,
            a_trained_arr,
        ))

    row6_left, row6_right = st.columns(2)
    with row6_left:
        enable_hiring_active = bool(active_key[26])          # index 26 = enable_hiring (unchanged)
        enable_replenish_active = bool(active_key[27])       # index 27 = enable_replenish_hiring
        st.pyplot(fig_hiring_events(
            df,
            enable_hiring=enable_hiring_active,
            enable_replenish_hiring=enable_replenish_active,
        ))
    with row6_right:
        st.pyplot(fig_pi_over_time(df))

    # Footer
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
