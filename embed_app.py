"""
Slim embed entrypoint for the FirmBehavior in-browser simulator.

This file is loaded by stlite (Streamlit-in-browser via Pyodide) at
https://igorban.ai/sim. It is NOT a replacement for app.py — it is a
deliberate subset: 4 sliders, 1 strategy radio, Run button, 2 charts.

Why separate from app.py:
  - app.py has 36 params, 15 charts, and all strategies including the DP
    optimizer. Running the DP optimizer under Pyodide would take 5–30 s per
    click (Pyodide is 2–4× slower for numpy, ~10× for Python loops).
  - This file exposes only greedy_with_switching / all_H / all_T — all O(N)
    per step, well under 2 s under Pyodide.
  - Widget keys are suffixed `_embed` so they cannot collide with
    tests/test_app.py's widget-key contract.

Why DP optimizer is excluded (D-03):
  dp_rolling_horizon_strategy enumerates 25^5 paths × T=60 periods × N=500
  numpy ops. That is too slow under Pyodide. The DP strategy remains
  available in the full Streamlit app on HF Spaces.
"""

# === MODULE IMPORTS (pure Python imports — not Streamlit commands) ===
import streamlit as st
from firm_ai_abm.theme import THEME
from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import make_firm
from firm_ai_abm.simulate import run_simulation
from firm_ai_abm.strategy import all_H, all_T, greedy_with_switching
from firm_ai_abm.dashboard import (
    fig_pi_per_period_over_time,
    fig_mode_mix_area,
)

# === STREAMLIT PAGE CONFIG — MUST BE THE FIRST st.* CALL (round-2 MAJ-10, round-3 CRIT-1) ===
# All st.* calls (including st.query_params reads) MUST follow st.set_page_config.
st.set_page_config(page_title="Firm Behavior — slim sim",
                   layout="centered", initial_sidebar_state="collapsed")

# === THEME OVERRIDE (D-06) — runs AFTER set_page_config ===
# Only mutate fields that apply_theme() reads: layout_defaults + font.
# THEME["colors"] is informational and NOT read by apply_theme — do not touch it.
qp = st.query_params
_theme = qp.get("theme", "light")
if _theme == "dark":
    THEME["layout_defaults"]["plot_bgcolor"]  = "#1b1d22"
    THEME["layout_defaults"]["paper_bgcolor"] = "#1b1d22"
    THEME["font"]["color"]                    = "#ecedef"
else:  # light (default)
    THEME["layout_defaults"]["plot_bgcolor"]  = "#fdfdfd"
    THEME["layout_defaults"]["paper_bgcolor"] = "#fdfdfd"
    THEME["font"]["color"]                    = "#1f2328"  # site --foreground light

# Optional round-2 MIN-6 cosmetic: match site body font (uncomment to enable)
# THEME["font"]["family"] = "IBM Plex Serif, serif"

st.markdown("### Firm Behavior — try a slim version")
st.caption("4 dials. Hit Run. See profit and task-mode mix evolve. "
           "[Open the full simulator ↗](https://<HF-SPACE-URL>)")

c1, c2 = st.columns(2)
with c1:
    q_a    = st.slider("AI productivity ceiling (q_a)", 0.0, 3.0, 1.2, 0.05, key="q_a_embed")
    c_auto = st.slider("Automation cost per task (c_auto)", 0.0, 2.0, 0.10, 0.05, key="c_auto_embed")
with c2:
    g  = st.slider("Augmentation gain (g)", 0.0, 3.0, 0.5, 0.05, key="g_embed")
    w  = st.slider("Wage rate (w)", 0.0, 5.0, 2.0, 0.1, key="w_embed")

# D-03: DP optimizer intentionally EXCLUDED (too slow under Pyodide).
strategy_name = st.radio("Strategy",
    options=["greedy_with_switching", "all_H", "all_T"],
    horizontal=True, index=0, key="strategy_embed",
    captions=["smart greedy", "all human", "all automated"])

if st.button("Run", type="primary", key="run_embed"):
    params = FirmParams(q_a=q_a, g=g, c_auto=c_auto, w=w,
                        T=60, N=500, seed=0)
    firm = make_firm(params)
    strat = {"greedy_with_switching": greedy_with_switching,
             "all_H": all_H, "all_T": all_T}[strategy_name]
    df = run_simulation(firm, strat)

    tab1, tab2 = st.tabs(["Profit over time", "Task mode mix"])
    with tab1:
        st.plotly_chart(fig_pi_per_period_over_time(df), use_container_width=True)
    with tab2:
        st.plotly_chart(fig_mode_mix_area(df, int(params.N)),
                        use_container_width=True)

    cum = float(df["pi"].cumsum().iloc[-1])
    st.metric("Cumulative profit (60 periods)", f"{cum:.2f}")

# Round-3 MAJ-4: postMessage live-theme bridge REMOVED. streamlit_javascript is dead
# (latest v0.1.5 published May 2022 — no releases since). v1 ships reload-only:
# initial theme via ?theme= read above; site toggle reloads /sim?theme=<new>.
