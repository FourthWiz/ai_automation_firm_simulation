"""Streamlit-native plot helpers for the Phase 1.5 dashboard.

Each function accepts a run DataFrame (produced by run_simulation) and returns
a plotly.graph_objects.Figure. NO savefig calls, NO disk I/O, NO mkdir. The
dashboard calls st.plotly_chart(fig) directly.

Color scheme for mode-mix plots:
  H (human):     #111111 (black)
  A (augmented): #00B4B4 (teal)
  T (automated): #E85A2B (orange-red)
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from firm_ai_abm.production import Mode
from firm_ai_abm.theme import THEME, apply_theme


def fig_pi_per_period_over_time(df: pd.DataFrame) -> go.Figure:
    """Per-period profit over time with mean line and optional first-firing annotation.

    Input:
        df: run_simulation DataFrame with columns 't', 'pi', and optionally
            'n_review_fired'. Shape: (T, 13+).
    Output:
        Figure with one trace: line plot of df['pi'] vs df['t'], horizontal dashed
        line at df['pi'].mean(), optional vertical dashed line at the first period
        where n_review_fired > 0 (if any). No disk I/O.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["t"], y=df["pi"],
        mode="lines",
        line=dict(color=THEME["colors"]["primary"], width=1.5),
        name="per-period π",
    ))
    mean_pi = float(df["pi"].mean())
    fig.add_hline(
        y=mean_pi,
        line=dict(color="gray", width=0.8, dash="dash"),
        annotation_text=f"mean={mean_pi:.2f}",
        annotation_position="top right",
    )

    fired_col = df.get("n_review_fired", pd.Series(0, index=df.index))
    first_fire_rows = df.loc[fired_col > 0, "t"]
    if len(first_fire_rows) > 0:
        first_fire_t = int(first_fire_rows.iloc[0])
        fig.add_vline(
            x=first_fire_t,
            line=dict(color=THEME["colors"]["T"], width=0.8, dash="dot"),
            annotation_text=f"first firing t={first_fire_t}",
            annotation_position="top right",
        )

    fig.update_layout(
        title="Per-Period Profit",
        xaxis_title="period",
        yaxis_title="profit",
    )
    return apply_theme(fig)


def fig_pi_over_time(df: pd.DataFrame) -> go.Figure:
    """Cumulative profit over time.

    Input:
        df: run_simulation DataFrame with columns 't' (int, 1-indexed) and
            'pi' (float, per-period profit). Shape: (T, 13+).
    Output:
        Figure with one trace: line plot of df['pi'].cumsum() vs df['t'],
        plus a dashed gray zero-reference line.
    No disk I/O — returns Figure only; caller calls st.plotly_chart(fig).
    """
    cum_pi = df["pi"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["t"], y=cum_pi,
        mode="lines",
        line=dict(color=THEME["colors"]["primary"], width=1.5),
        name="cumulative profit",
    ))
    fig.add_hline(y=0, line=dict(color="lightgray", width=0.8, dash="dash"))
    fig.update_layout(
        title="Cumulative Profit",
        xaxis_title="period",
        yaxis_title="cumulative profit",
    )
    return apply_theme(fig)


def fig_K_over_time(df: pd.DataFrame) -> go.Figure:
    """Worker headcount over time with firing-event overlay.

    K (packed formula) = ceil(n_H_or_A / tasks_per_worker).
    K_active = unique workers assigned to at least one H/A task.
    When firings occur, red bars on the right y-axis show workers fired per
    review period.

    Input:
        df: run_simulation DataFrame with columns 't', 'K', 'K_active',
            and optionally 'n_review_fired'.
    Output:
        Figure with two lines and, when firings > 0, a secondary y-axis with
        firing bars. No disk I/O.
    """
    fired_col = df.get("n_review_fired", pd.Series(0, index=df.index))
    has_firings = bool((fired_col > 0).any())

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df["t"], y=df["K_active"],
        mode="lines",
        line=dict(color=THEME["colors"]["A"], width=1.5),
        name="K_active (actual)",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["t"], y=df["K"],
        mode="lines",
        line=dict(color=THEME["colors"]["primary"], width=1.0, dash="dash"),
        name="K (packed formula)",
    ), secondary_y=False)

    if has_firings:
        fig.add_trace(go.Bar(
            x=df["t"], y=fired_col,
            name="firings",
            marker_color=THEME["colors"]["T"],
            opacity=0.4,
        ), secondary_y=True)
        fig.update_yaxes(title_text="workers fired", secondary_y=True,
                         tickfont=dict(color=THEME["colors"]["T"]))

    fig.update_layout(
        title="Active Workers Over Time",
        xaxis_title="period",
        yaxis_title="workers",
    )
    # Caption matching current text (asserted by test_K_active_caption_present)
    fig.add_annotation(
        text="K = workforce headcount = N / tasks_per_worker",
        x=0, y=-0.25,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=10, color="gray"),
    )
    return apply_theme(fig)


def fig_mode_mix_area(df: pd.DataFrame, N: int) -> go.Figure:
    """Stacked area chart of H/A/T mode fractions over time.

    Input:
        df: run_simulation DataFrame with columns 't' (int) and 'modes'
            (object column of numpy int arrays of length N). Shape: (T, 13+).
        N: number of tasks (used to normalize fractions to [0, 1]).
    Output:
        Figure: stacked area chart. Three layers (H, A, T) sum to 1.0 at each period.
    Colors: H=#111111, A=#00B4B4, T=#E85A2B.
    No disk I/O — returns Figure only.
    """
    periods = df["t"].values
    modes_series = df["modes"].values

    frac_H = np.array([(m == int(Mode.H)).sum() / N for m in modes_series])
    frac_A = np.array([(m == int(Mode.A)).sum() / N for m in modes_series])
    frac_T = np.array([(m == int(Mode.T)).sum() / N for m in modes_series])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=periods, y=frac_H,
        mode="lines",
        stackgroup="modes",
        fillcolor=THEME["colors"]["H"],
        line=dict(color=THEME["colors"]["H"], width=0),
        name="H (human)",
        opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        x=periods, y=frac_A,
        mode="lines",
        stackgroup="modes",
        fillcolor=THEME["colors"]["A"],
        line=dict(color=THEME["colors"]["A"], width=0),
        name="A (augmented)",
        opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        x=periods, y=frac_T,
        mode="lines",
        stackgroup="modes",
        fillcolor=THEME["colors"]["T"],
        line=dict(color=THEME["colors"]["T"], width=0),
        name="T (automated)",
        opacity=0.85,
    ))
    fig.update_layout(
        title="Mode Mix (stacked area)",
        xaxis_title="period",
        yaxis_title="fraction of tasks",
        yaxis=dict(range=[0, 1.0]),
    )
    return apply_theme(fig)


def fig_wage_bill_over_time(df: pd.DataFrame) -> go.Figure:
    """Total wage bill over time.

    Input:
        df: run_simulation DataFrame with columns 't' (int) and 'wage_bill'
            (float, sum of wages paid to all active workers). Shape: (T, 13+).
    Output:
        Figure: line plot of df['wage_bill'] vs df['t'].
    Color: neutral (#8C8C8C).
    No disk I/O — returns Figure only.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["t"], y=df["wage_bill"],
        mode="lines",
        line=dict(color=THEME["colors"]["neutral"], width=1.5),
        name="wage bill",
    ))
    fig.update_layout(
        title="Wage Bill Over Time",
        xaxis_title="period",
        yaxis_title="wage bill",
    )
    return apply_theme(fig)


def fig_theta_histogram(workforce_theta: np.ndarray) -> go.Figure:
    """Histogram of the final-period worker skill (theta) distribution.

    Input:
        workforce_theta: 1-D numpy array of worker theta values from the
            final period of the simulation (shape: (K,) where K = active
            worker count). Values in [theta_min, theta_max].
    Output:
        Figure: histogram with 20 bins. Dashed vertical line at mean theta.
    No disk I/O — returns Figure only.
    """
    theta = np.asarray(workforce_theta)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=theta,
        nbinsx=20,
        marker_color=THEME["colors"]["neutral"],
        opacity=0.8,
        name="theta",
    ))
    if len(theta) > 0:
        mean_theta = float(np.mean(theta))
        fig.add_vline(
            x=mean_theta,
            line=dict(color=THEME["colors"]["T"], width=1.2, dash="dash"),
            annotation_text=f"mean={mean_theta:.3f}",
            annotation_position="top right",
        )
    fig.update_layout(
        title="Final-Period Theta Distribution",
        xaxis_title="theta (worker skill)",
        yaxis_title="count",
    )
    return apply_theme(fig)


def fig_alpha_histogram(alpha: np.ndarray) -> go.Figure:
    """Histogram of the task automatability (alpha) distribution.

    Input:
        alpha: 1-D numpy array of task alpha values sampled at make_firm time.
            Values in [0, 1] (Beta distribution). Shape: (N,).
    Output:
        Figure: histogram with 20 bins, fixed x-range [0, 1]. Dashed vertical
        line at mean alpha (omitted for empty input).
    No disk I/O — returns Figure only.
    """
    alpha = np.asarray(alpha)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=alpha,
        nbinsx=20,
        marker_color=THEME["colors"]["T"],
        opacity=0.8,
        name="alpha",
    ))
    if len(alpha) > 0:
        mean_alpha = float(np.mean(alpha))
        fig.add_vline(
            x=mean_alpha,
            line=dict(color=THEME["colors"]["neutral"], width=1.2, dash="dash"),
            annotation_text=f"mean={mean_alpha:.3f}",
            annotation_position="top right",
        )
    fig.update_layout(
        title="Task Automatability (alpha) Distribution",
        xaxis_title="alpha (automatability)",
        yaxis_title="count",
    )
    fig.update_xaxes(range=[0, 1])
    return apply_theme(fig)


def fig_beta_histogram(beta: np.ndarray) -> go.Figure:
    """Histogram of the task augmentability (beta) distribution.

    Input:
        beta: 1-D numpy array of task beta values sampled at make_firm time.
            Values in [0, 1] (Beta distribution). Shape: (N,).
    Output:
        Figure: histogram with 20 bins, fixed x-range [0, 1]. Dashed vertical
        line at mean beta (omitted for empty input).
    No disk I/O — returns Figure only.
    """
    beta = np.asarray(beta)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=beta,
        nbinsx=20,
        marker_color=THEME["colors"]["A"],
        opacity=0.8,
        name="beta",
    ))
    if len(beta) > 0:
        mean_beta = float(np.mean(beta))
        fig.add_vline(
            x=mean_beta,
            line=dict(color=THEME["colors"]["neutral"], width=1.2, dash="dash"),
            annotation_text=f"mean={mean_beta:.3f}",
            annotation_position="top right",
        )
    fig.update_layout(
        title="Task Augmentability (beta) Distribution",
        xaxis_title="beta (augmentability)",
        yaxis_title="count",
    )
    fig.update_xaxes(range=[0, 1])
    return apply_theme(fig)


def fig_mean_theta_over_time(df: pd.DataFrame) -> go.Figure:
    """Mean worker skill (theta) over time.

    Input:
        df: run_simulation DataFrame with columns 't' (int) and 'mean_theta'
            (float, mean of active worker theta values). Shape: (T, 13+).
    Output:
        Figure: line plot of df['mean_theta'] vs df['t'].
    No disk I/O — returns Figure only.
    """
    col = df.get("mean_theta", pd.Series(np.nan, index=df.index))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["t"], y=col,
        mode="lines",
        line=dict(color="#CCB974", width=1.5),
        name="mean theta",
    ))
    fig.update_layout(
        title="Mean Worker Skill Over Time",
        xaxis_title="period",
        yaxis_title="mean theta",
    )
    return apply_theme(fig)


def fig_firing_events(df: pd.DataFrame, T_review: float = math.inf) -> go.Figure:
    """Firing events at review periods.

    Input:
        df: run_simulation DataFrame with columns 't' (int) and
            'n_review_fired' (int, workers fired at this period's review;
            0 for non-review periods). Shape: (T, 13+).
        T_review: the T_review parameter used for the run (default math.inf).
            Used only to distinguish "review disabled" from "review active
            but no workers met the firing threshold" in the fallback message.
    Output:
        Figure: scatter of t values where n_review_fired > 0, marker size
        proportional to count (8–24 px range). Baseline at y=0 via add_hline.
        Empty-state annotation when no firings occurred.
    Color: markers in #E85A2B (orange-red).
    No disk I/O — returns Figure only.
    """
    fired_col = df.get("n_review_fired", pd.Series(0, index=df.index))
    mask = fired_col > 0
    t_fired = df["t"][mask].values
    n_fired = fired_col[mask].values

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="lightgray", width=0.8, dash="dash"))

    if len(t_fired) > 0:
        max_n = float(n_fired.max())
        # Scale sizes 8–24 px
        sizes = [max(8, 8 + 16 * v / max_n) for v in n_fired]
        fig.add_trace(go.Scatter(
            x=t_fired, y=n_fired,
            mode="markers",
            marker=dict(size=sizes, color=THEME["colors"]["T"], opacity=0.8),
            name="workers fired",
        ))
    else:
        if math.isinf(T_review):
            msg = "review disabled (T_review=∞)"
        else:
            msg = f"no firings at this threshold (T_review={int(T_review)}, raise firing_threshold)"
        fig.add_annotation(
            text=msg,
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=9, color="gray"),
        )

    t_max = int(df["t"].max()) if len(df) > 0 else 1
    fig.update_layout(
        title="Firing Events at Review Periods",
        xaxis_title="period",
        yaxis_title="workers fired",
        xaxis=dict(range=[-0.5, t_max + 0.5]),
    )
    return apply_theme(fig)


def fig_trained_capital(df: pd.DataFrame) -> go.Figure:
    """Cumulative trained workers (n_a_trained) over time.

    Input:
        df: run_simulation DataFrame with columns 't' (int) and 'n_a_trained'
            (int, count of workers who have completed H→A training). Shape: (T, 13+).
    Output:
        Figure: line plot of df['n_a_trained'] vs df['t'].
    Color: #00B4B4 (teal/augmented).
    No disk I/O — returns Figure only.
    """
    col = df.get("n_a_trained", pd.Series(0, index=df.index))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["t"], y=col,
        mode="lines",
        line=dict(color=THEME["colors"]["A"], width=1.5),
        name="n_a_trained",
    ))
    fig.update_layout(
        title="Trained-Capital Workers Over Time",
        xaxis_title="period",
        yaxis_title="trained workers (n_a_trained)",
    )

    # Degenerate annotation: flat nonzero series means all workers trained early
    col_vals = col.values
    if len(col_vals) > 0 and col_vals.min() == col_vals.max() and col_vals.max() > 0:
        value = int(col_vals.max())
        first_t = int(df.loc[col == value, "t"].iloc[0])
        fig.add_annotation(
            text=f"all {value} workers trained by t={first_t}",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=9, color=THEME["colors"]["A"]),
            opacity=0.7,
        )

    return apply_theme(fig)


def fig_wage_histogram(workforce_wage: np.ndarray) -> go.Figure:
    """Histogram of the final-period worker wage distribution.

    Input:
        workforce_wage: 1-D numpy array of worker wage values from the
            final period (shape: (K,) where K = active worker count).
    Output:
        Figure: histogram with 20 bins, dashed vertical mean line.
    No disk I/O — returns Figure only.
    """
    wage = np.asarray(workforce_wage)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=wage,
        nbinsx=20,
        marker_color=THEME["colors"]["neutral"],
        opacity=0.8,
        name="wage",
    ))
    if len(wage) > 0:
        mean_wage = float(np.mean(wage))
        fig.add_vline(
            x=mean_wage,
            line=dict(color=THEME["colors"]["T"], width=1.2, dash="dash"),
            annotation_text=f"mean={mean_wage:.3f}",
            annotation_position="top right",
        )
    fig.update_layout(
        title="Final-Period Wage Distribution",
        xaxis_title="wage",
        yaxis_title="count",
    )
    return apply_theme(fig)


def fig_wage_vs_mean_output(
    wages_final: np.ndarray,
    mean_output_per_worker: np.ndarray,
    a_trained_final: np.ndarray,
) -> go.Figure:
    """Scatter of worker wage vs mean per-period output, colored by training status.

    Input:
        wages_final: 1-D array of final-period wages, shape (K,).
        mean_output_per_worker: 1-D array of NaN-averaged per-period output per worker,
            shape (K,). Workers where all output columns are NaN are excluded.
        a_trained_final: 1-D bool array of training status, shape (K,).
    Output:
        Figure with scatter: X=wage, Y=mean output. Two colors:
          trained (#00B4B4 teal), untrained (#111111 black). Legend shows counts.
        Workers where all output is NaN are excluded.
    Trace names are exactly: "trained (n=N)" and "untrained (n=N)" — preserved for
    test assertions.
    No disk I/O — returns Figure only.
    """
    wages = np.asarray(wages_final)
    mean_out = np.asarray(mean_output_per_worker)
    trained = np.asarray(a_trained_final, dtype=bool)

    valid = ~np.isnan(mean_out)
    wages = wages[valid]
    mean_out = mean_out[valid]
    trained = trained[valid]

    n_trained = int(trained.sum())
    n_untrained = int((~trained).sum())

    fig = go.Figure()

    if n_untrained > 0:
        fig.add_trace(go.Scatter(
            x=wages[~trained], y=mean_out[~trained],
            mode="markers",
            marker=dict(color=THEME["colors"]["H"], size=6, opacity=0.7),
            name=f"untrained (n={n_untrained})",
        ))
    if n_trained > 0:
        fig.add_trace(go.Scatter(
            x=wages[trained], y=mean_out[trained],
            mode="markers",
            marker=dict(color=THEME["colors"]["A"], size=6, opacity=0.7),
            name=f"trained (n={n_trained})",
        ))

    fig.update_layout(
        title="Wage vs. Mean Output",
        xaxis_title="wage",
        yaxis_title="mean output per period (NaN-averaged)",
    )
    return apply_theme(fig)


def fig_hiring_events(
    df: pd.DataFrame,
    enable_hiring: bool = False,
    enable_replenish_hiring: bool = False,
) -> go.Figure:
    """Hiring events timeline at review periods.

    Mirrors fig_firing_events: scatter of t values where n_hired > 0.
    Input:
        df: run_simulation DataFrame with columns 't' (int) and
            'n_hired' (int, workers hired this period; 0 when both flags False).
        enable_hiring: the enable_hiring flag used for the run (default False).
        enable_replenish_hiring: the enable_replenish_hiring flag (default False).
    Output:
        Figure: scatter of t values where n_hired > 0.
        Baseline at y=0. Empty-state message when no hires occurred.
    Color: markers in #00B4B4 (teal — opposite semantic of firing red).
    No disk I/O — returns Figure only.
    """
    hired_col = df.get("n_hired", pd.Series(0, index=df.index))
    mask = hired_col > 0
    t_hired = df["t"][mask].values
    n_hired = hired_col[mask].values

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="lightgray", width=0.8, dash="dash"))

    if len(t_hired) > 0:
        max_n = float(n_hired.max())
        # Scale sizes 8–24 px
        sizes = [max(8, 8 + 16 * v / max_n) for v in n_hired]
        fig.add_trace(go.Scatter(
            x=t_hired, y=n_hired,
            mode="markers",
            marker=dict(size=sizes, color=THEME["colors"]["A"], opacity=0.8),
            name="workers hired",
        ))
    else:
        if not enable_hiring and not enable_replenish_hiring:
            msg = "hiring disabled (both flags False)"
        else:
            msg = "no hiring events\n(no firings to backfill)"
        fig.add_annotation(
            text=msg,
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=9, color="gray"),
        )

    t_max = int(df["t"].max()) if len(df) > 0 else 1
    fig.update_layout(
        title="Hiring Events Over Time",
        xaxis_title="period",
        yaxis_title="workers hired",
        xaxis=dict(range=[-0.5, t_max + 0.5]),
    )
    return apply_theme(fig)


def fig_mean_accum_wage_over_time(df: pd.DataFrame) -> go.Figure:
    """Mean accumulated wages per ever-worked worker, with secondary axis for headcount.

    Left axis: mean_accum_wage over time (orange).
    Right axis: ever_worked_count (cumulative distinct workers who held a slot).
    Handles all-NaN gracefully (renders empty axes with explanatory note).

    Input:
        df: run_simulation DataFrame with columns 't', 'mean_accum_wage', 'ever_worked_count'.
    Output:
        Figure with optional secondary y-axis. No disk I/O.
    """
    col_wage = df.get("mean_accum_wage", pd.Series(float("nan"), index=df.index))
    col_count = df.get("ever_worked_count", pd.Series(0, index=df.index))

    all_nan = col_wage.isna().all()

    if all_nan:
        fig = go.Figure()
        fig.add_annotation(
            text="no workers ever active",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=9, color="gray"),
        )
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=df["t"], y=col_wage,
            mode="lines",
            line=dict(color="#DD8452", width=1.5),
            name="mean accum. wage",
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df["t"], y=col_count,
            mode="lines",
            line=dict(color=THEME["colors"]["neutral"], width=1.0, dash="dash"),
            opacity=0.7,
            name="ever worked",
        ), secondary_y=True)
        fig.update_yaxes(title_text="ever-worked count", secondary_y=True)

    fig.update_layout(
        title="Mean Accumulated Wages per Ever-Worked Worker",
        xaxis_title="period",
        yaxis_title="cumulative wages",
    )
    return apply_theme(fig)


# Strategy display names and fixed colors for multi-strategy comparison.
_STRATEGY_DISPLAY = {
    "all_H": "All Human",
    "all_A": "All Augmented",
    "all_T": "All Automated",
    "greedy_with_switching": "Greedy",
    "horizon_brute": "Horizon Brute",
    "horizon_optimizer": "Horizon Optimizer",
}
_STRATEGY_COLORS = {
    "all_H": THEME["colors"]["H"],
    "all_A": THEME["colors"]["A"],
    "all_T": THEME["colors"]["T"],
    "greedy_with_switching": "#8C5E00",   # amber — mixed/uncertain bet
    "horizon_brute": "#6B4ECC",           # violet — brute optimizer
    "horizon_optimizer": "#0077AA",       # blue — DP optimizer
}


def fig_multi_strategy_compare(histories: dict) -> go.Figure:
    """Cumulative profit comparison across strategies.

    Input:
        histories: dict mapping strategy name → run_simulation DataFrame.
                   Each DataFrame must have columns 't' and 'pi'.
    Output:
        Figure with one cumulative-profit line per strategy, zero reference line,
        and legend. Returns an empty figure with a note if histories is empty.
    """
    fig = go.Figure()
    if not histories:
        fig.add_annotation(text="No strategies to compare.", showarrow=False,
                           font=dict(size=14, color=THEME["colors"]["neutral"]))
        return apply_theme(fig)

    for strategy_name, df in histories.items():
        color = _STRATEGY_COLORS.get(strategy_name, THEME["colors"]["neutral"])
        label = _STRATEGY_DISPLAY.get(strategy_name, strategy_name)
        cum_pi = df["pi"].cumsum()
        fig.add_trace(go.Scatter(
            x=df["t"], y=cum_pi,
            mode="lines",
            line=dict(color=color, width=2.0),
            name=label,
        ))

    fig.add_hline(y=0, line=dict(color="lightgray", width=0.8, dash="dash"))
    fig.update_layout(
        title="Cumulative Profit by Strategy",
        xaxis_title="period",
        yaxis_title="cumulative profit",
    )
    return apply_theme(fig)
