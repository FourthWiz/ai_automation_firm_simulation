"""
Slim embed entrypoint for the FirmBehavior in-browser simulator.

Loaded by stlite (Streamlit-in-browser via Pyodide) at https://igorban.ai/sim.
Exposes the same 6 strategies as the full app plus the 3 hiring modes.
Widget keys are suffixed `_embed` to avoid colliding with test_app.py contracts.

Performance note: horizon_brute and horizon_optimizer (DP) run fine on a laptop
but are 5–30× slower under Pyodide (Python loops ≈10× slower than native).
A warning is shown when one of these is selected.
"""

# === MODULE IMPORTS (pure Python imports — not Streamlit commands) ===
import streamlit as st
from firm_ai_abm.theme import THEME
from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import make_firm
from firm_ai_abm.simulate import run_simulation
from firm_ai_abm.strategy import all_H, all_A, all_T, greedy_profit, greedy_with_switching
from firm_ai_abm.margin_optimizer import horizon_brute_strategy
from firm_ai_abm.dp_optimizer import dp_rolling_horizon_strategy
from firm_ai_abm.dashboard import (
    fig_pi_per_period_over_time,
    fig_mode_mix_area,
)

_STRATEGIES = {
    "greedy_with_switching": greedy_with_switching,
    "greedy_profit":         greedy_profit,
    "all_A":                 all_A,
    "all_H":                 all_H,
    "all_T":                 all_T,
    "horizon_brute":         horizon_brute_strategy,
    "horizon_optimizer":     dp_rolling_horizon_strategy,
}

_SLOW = {"horizon_brute", "horizon_optimizer"}

# === STREAMLIT PAGE CONFIG — MUST BE THE FIRST st.* CALL ===
st.set_page_config(page_title="Firm Behavior — slim sim",
                   layout="centered", initial_sidebar_state="collapsed")

# === THEME OVERRIDE (D-06) — runs AFTER set_page_config ===
qp = st.query_params
_theme = qp.get("theme", "light")
if _theme == "dark":
    THEME["layout_defaults"]["plot_bgcolor"]  = "#1b1d22"
    THEME["layout_defaults"]["paper_bgcolor"] = "#1b1d22"
    THEME["font"]["color"]                    = "#ecedef"
else:
    THEME["layout_defaults"]["plot_bgcolor"]  = "#fdfdfd"
    THEME["layout_defaults"]["paper_bgcolor"] = "#fdfdfd"
    THEME["font"]["color"]                    = "#1f2328"

st.markdown("### Firm Behavior — try the simulator")
st.caption("Adjust params. Hit Run. Compare strategies. "
           "[Open the full simulator ↗](https://<HF-SPACE-URL>)")

c1, c2 = st.columns(2)
with c1:
    q_a    = st.slider("AI productivity ceiling (q_a)", 0.0, 3.0, 1.2, 0.05, key="q_a_embed")
    c_auto = st.slider("Automation cost per task (c_auto)", 0.0, 2.0, 0.10, 0.05, key="c_auto_embed")
with c2:
    g = st.slider("Augmentation gain (g)", 0.0, 3.0, 0.5, 0.05, key="g_embed")
    w = st.slider("Wage rate (w)", 0.0, 5.0, 2.0, 0.1, key="w_embed")

strategy_name = st.radio(
    "Strategy",
    options=list(_STRATEGIES.keys()),
    index=0,
    horizontal=True,
    key="strategy_embed",
    captions=[
        "smart greedy (switching costs)",
        "myopic greedy (no switching cost)",
        "all augmented",
        "all human",
        "all automated",
        "brute horizon (slow in browser)",
        "DP optimizer (slow in browser)",
    ],
)

hiring_mode = st.radio(
    "Hiring mode",
    options=["off", "enable_hiring", "enable_replenish_hiring"],
    index=0,
    horizontal=True,
    key="hiring_mode_embed",
    captions=[
        "no rehiring after firing",
        "refill immediately to K*",
        "delayed backlog refill",
    ],
)

if strategy_name in _SLOW:
    st.warning(
        f"**{strategy_name}** uses search/DP loops that run 5–30× slower under "
        "Pyodide (browser WebAssembly). Expect 15–60 s. "
        "Use the [full simulator](https://<HF-SPACE-URL>) for repeated DP runs.",
        icon="⚠️",
    )

if st.button("Run", type="primary", key="run_embed"):
    params = FirmParams(
        q_a=q_a, g=g, c_auto=c_auto, w=w,
        T=60, N=500, seed=0,
        enable_hiring=(hiring_mode == "enable_hiring"),
        enable_replenish_hiring=(hiring_mode == "enable_replenish_hiring"),
    )
    firm = make_firm(params)
    df = run_simulation(firm, _STRATEGIES[strategy_name])

    tab1, tab2 = st.tabs(["Profit over time", "Task mode mix"])
    with tab1:
        st.plotly_chart(fig_pi_per_period_over_time(df), use_container_width=True)
    with tab2:
        st.plotly_chart(fig_mode_mix_area(df, int(params.N)),
                        use_container_width=True)

    cum = float(df["pi"].cumsum().iloc[-1])
    st.metric("Cumulative profit (60 periods)", f"{cum:.2f}")
