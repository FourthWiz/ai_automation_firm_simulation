"""Plotly theme constants for FirmBehavior dashboard.

Import THEME for colors/fonts and call apply_theme(fig) on every figure
before returning it from a dashboard function.
"""

THEME = {
    "colors": {
        "H": "#111111",
        "A": "#00B4B4",
        "T": "#E85A2B",
        "primary": "#00B4B4",
        "neutral": "#8C8C8C",
        "bg": "#FAF7F2",
        "grid": "rgba(0,0,0,0.12)",
    },
    "font": {
        "family": "Inter, IBM Plex Sans, system-ui, sans-serif",
        "size": 12,
        "color": "#111111",
    },
    "layout_defaults": {
        "plot_bgcolor": "#FAF7F2",
        "paper_bgcolor": "#FAF7F2",
        "margin": dict(l=40, r=20, t=40, b=40),
        "xaxis": {
            "gridcolor": "rgba(0,0,0,0.12)",
            "zerolinecolor": "rgba(0,0,0,0.24)",
        },
        "yaxis": {
            "gridcolor": "rgba(0,0,0,0.12)",
            "zerolinecolor": "rgba(0,0,0,0.24)",
        },
        "legend": {"orientation": "h", "y": -0.18, "x": 0},
    },
}


def apply_theme(fig):
    """Apply FirmBehavior visual theme to a plotly Figure in-place and return it."""
    fig.update_layout(**THEME["layout_defaults"], font=THEME["font"])
    return fig
