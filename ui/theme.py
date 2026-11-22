"""Institutional dark-slate design tokens, CSS, Plotly layout factory and formatters."""

from __future__ import annotations

import math

import plotly.graph_objects as go
import streamlit as st

from core.schemas import LiquidityRegime

COLORS: dict[str, str] = {
    "bg": "#0B1120",
    "card": "#111827",
    "border": "#1E293B",
    "grid": "#1E293B",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "blue": "#38BDF8",
    "deep_blue": "#0284C7",
    "emerald": "#10B981",
    "amber": "#F59E0B",
    "orange": "#FB923C",
    "red": "#EF4444",
    "slate": "#64748B",
    "violet": "#A78BFA",
}

TONES: dict[str, tuple[str, str]] = {
    "blue": (COLORS["blue"], "rgba(56, 189, 248, 0.14)"),
    "emerald": (COLORS["emerald"], "rgba(16, 185, 129, 0.14)"),
    "amber": (COLORS["amber"], "rgba(245, 158, 11, 0.14)"),
    "orange": (COLORS["orange"], "rgba(251, 146, 60, 0.14)"),
    "red": (COLORS["red"], "rgba(239, 68, 68, 0.16)"),
    "slate": (COLORS["muted"], "rgba(148, 163, 184, 0.12)"),
    "violet": (COLORS["violet"], "rgba(167, 139, 250, 0.14)"),
}

COMMODITY_COLORS: dict[str, str] = {
    "GC=F": "#FBBF24",
    "HG=F": "#FB923C",
    "LBR=F": "#34D399",
    "CC=F": "#C084FC",
}

REGIME_STYLES: dict[LiquidityRegime, tuple[str, str, str, str]] = {
    LiquidityRegime.HEALTHY: ("HEALTHY", COLORS["emerald"], "rgba(16, 185, 129, 0.12)", "#047857"),
    LiquidityRegime.MARGIN_TRAP_WARNING: (
        "MARGIN_TRAP_WARNING", COLORS["amber"], "rgba(245, 158, 11, 0.12)", "#B45309",
    ),
    LiquidityRegime.BB_DEFICIENCY_WARNING: (
        "BB_DEFICIENCY_WARNING", COLORS["orange"], "rgba(251, 146, 60, 0.12)", "#C2410C",
    ),
    LiquidityRegime.CRITICAL_SQUEEZE: ("CRITICAL_SQUEEZE", COLORS["red"], "rgba(239, 68, 68, 0.14)", "#B91C1C"),
}

REGIME_TONES: dict[LiquidityRegime, str] = {
    LiquidityRegime.HEALTHY: "emerald",
    LiquidityRegime.MARGIN_TRAP_WARNING: "amber",
    LiquidityRegime.BB_DEFICIENCY_WARNING: "orange",
    LiquidityRegime.CRITICAL_SQUEEZE: "red",
}

PLOTLY_CONFIG: dict = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}

FONT_STACK = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif"


def apply_institutional_layout(
    fig: go.Figure,
    title: str,
    height: int = 440,
    yaxis_title: str = "",
    xaxis_title: str = "",
) -> go.Figure:
    """Apply the shared dark-slate layout (Plotly 6/7 compliant title dicts, hex/0-255 colors)."""
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=f"<b>{title}</b>", font=dict(size=15, color=COLORS["text"], family=FONT_STACK),
                   x=0.0, xanchor="left", y=0.98, yanchor="top"),
        height=height,
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(15, 23, 42, 0.35)",
        font=dict(family=FONT_STACK, size=12, color=COLORS["muted"]),
        margin=dict(l=60, r=30, t=64, b=95),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0F172A", bordercolor="#334155",
                        font=dict(color=COLORS["text"], family=FONT_STACK, size=12)),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0.0,
                    bgcolor="rgba(0, 0, 0, 0)", font=dict(color=COLORS["muted"])),
        bargap=0.25,
    )
    fig.update_xaxes(gridcolor="rgba(51, 65, 85, 0.45)", zerolinecolor="#334155", linecolor="#334155",
                     tickfont=dict(color=COLORS["muted"]), showline=True)
    fig.update_yaxes(gridcolor="rgba(51, 65, 85, 0.45)", zerolinecolor="#334155", linecolor="#334155",
                     tickfont=dict(color=COLORS["muted"]))
    if yaxis_title:
        fig.update_yaxes(title=dict(text=yaxis_title, font=dict(color=COLORS["muted"])))
    if xaxis_title:
        fig.update_xaxes(title=dict(text=xaxis_title, font=dict(color=COLORS["muted"])))
    return fig


def inject_custom_css() -> None:
    """Global polish: typography, gradient canvas, pill tabs, sidebar, KPI cards, banners and audit cards."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, .stApp, [data-testid="stSidebar"], .stMarkdown, button, input, textarea {{
            font-family: {FONT_STACK};
        }}
        .stApp {{
            background:
              radial-gradient(1100px 520px at 0% -10%, rgba(56, 189, 248, 0.10), transparent 60%),
              radial-gradient(900px 480px at 100% -5%, rgba(167, 139, 250, 0.08), transparent 60%),
              {COLORS["bg"]};
        }}
        header[data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stAppDeployButton"] {{ display: none; }}
        .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1560px; }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0F172A 0%, #0B1120 100%);
            border-right: 1px solid {COLORS["border"]};
        }}
        .side-brand {{
            display: flex; align-items: center; gap: 0.7rem;
            padding: 0.2rem 0 1rem 0; margin-bottom: 0.4rem; border-bottom: 1px solid {COLORS["border"]};
        }}
        .side-brand .logo {{
            width: 38px; height: 38px; border-radius: 10px; display: grid; place-items: center;
            background: linear-gradient(135deg, {COLORS["deep_blue"]}, {COLORS["violet"]});
            color: white; font-weight: 800; font-size: 0.95rem; box-shadow: 0 6px 18px rgba(2, 132, 199, 0.35);
        }}
        .side-brand .name {{ color: {COLORS["text"]}; font-weight: 700; font-size: 0.98rem; line-height: 1.15; }}
        .side-brand .tag {{ color: {COLORS["muted"]}; font-size: 0.74rem; }}
        .side-section {{
            display: flex; align-items: center; gap: 0.5rem;
            color: {COLORS["blue"]}; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.09em;
            text-transform: uppercase; margin: 1.3rem 0 0.5rem 0;
        }}
        .side-section .num {{
            width: 20px; height: 20px; border-radius: 6px; display: grid; place-items: center;
            background: rgba(56, 189, 248, 0.15); color: {COLORS["blue"]}; font-size: 0.7rem;
        }}
        .side-section::after {{ content: ""; flex: 1; height: 1px; background: {COLORS["border"]}; }}

        /* Hero */
        .hero {{
            position: relative; overflow: hidden; border-radius: 16px; padding: 1.05rem 1.4rem 1rem 1.4rem;
            margin-bottom: 0.9rem; border: 1px solid #1F2A3D;
            background:
              linear-gradient(120deg, rgba(2, 132, 199, 0.22) 0%, rgba(15, 23, 42, 0.0) 45%),
              linear-gradient(300deg, rgba(167, 139, 250, 0.16) 0%, rgba(15, 23, 42, 0.0) 40%),
              #0F172A;
        }}
        .hero .eyebrow {{
            color: {COLORS["blue"]}; font-size: 0.72rem; letter-spacing: 0.14em; font-weight: 700;
            text-transform: uppercase;
        }}
        .hero h1 {{
            color: {COLORS["text"]}; font-size: 1.55rem; font-weight: 800; letter-spacing: -0.02em;
            margin: 0.25rem 0 0.35rem 0; padding: 0; line-height: 1.15;
        }}
        .hero .subtitle {{ color: {COLORS["muted"]}; font-size: 0.86rem; max-width: 980px; }}
        .hero .pills {{ margin-top: 0.65rem; display: flex; flex-wrap: wrap; gap: 0.45rem; }}
        .source-pill {{
            display: inline-flex; align-items: center; gap: 0.4rem;
            border: 1px solid #263349; background: rgba(15, 23, 42, 0.75); color: #CBD5E1;
            border-radius: 999px; padding: 0.22rem 0.75rem; font-size: 0.78rem;
        }}
        .source-pill .dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}

        /* Regime banner */
        .regime-banner {{
            display: flex; align-items: center; justify-content: space-between; gap: 1.2rem;
            border-radius: 14px; padding: 0.95rem 1.2rem; margin: 0 0 1rem 0;
            font-variant-numeric: tabular-nums;
        }}
        .regime-banner .left {{ display: flex; align-items: flex-start; gap: 0.9rem; }}
        .regime-banner .pulse {{
            width: 12px; height: 12px; border-radius: 50%; margin-top: 0.35rem; flex-shrink: 0;
            box-shadow: 0 0 0 0 currentColor; animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.35); }}
            70% {{ box-shadow: 0 0 0 10px rgba(255, 255, 255, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(255, 255, 255, 0); }}
        }}
        .regime-banner .regime-label {{ font-weight: 800; letter-spacing: 0.06em; font-size: 0.95rem; }}
        .regime-banner .regime-mode {{ color: {COLORS["muted"]}; font-size: 0.8rem; margin-left: 0.5rem; }}
        .regime-banner .regime-detail {{ color: #E2E8F0; font-size: 0.88rem; margin-top: 0.2rem; }}
        .regime-banner .right {{ text-align: right; flex-shrink: 0; }}
        .regime-banner .nlb-label {{
            color: {COLORS["muted"]}; font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase;
        }}
        .regime-banner .nlb-value {{ font-size: 1.6rem; font-weight: 800; letter-spacing: -0.01em; }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.3rem; background: rgba(15, 23, 42, 0.7); padding: 0.35rem; border-radius: 12px;
            border: 1px solid {COLORS["border"]}; flex-wrap: wrap;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 9px; padding: 0.45rem 0.85rem; height: auto; color: {COLORS["muted"]};
        }}
        .stTabs [data-baseweb="tab"]:hover {{ color: {COLORS["text"]}; background: rgba(148, 163, 184, 0.08); }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(180deg, rgba(56, 189, 248, 0.20), rgba(56, 189, 248, 0.08));
            color: {COLORS["text"]} !important;
        }}
        .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
        .stTabs [data-baseweb="tab-panel"] {{ padding-top: 1.1rem; }}

        /* Section headings */
        .section-head {{ margin: 0.6rem 0 0.6rem 0; }}
        .section-head .title {{ color: {COLORS["text"]}; font-size: 1.08rem; font-weight: 700; }}
        .section-head .sub {{ color: {COLORS["muted"]}; font-size: 0.84rem; margin-top: 0.1rem; }}

        /* KPI cards */
        .kpi-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(138px, 1fr));
            gap: 0.65rem; margin: 0.2rem 0 1rem 0;
        }}
        .kpi-card {{
            position: relative; overflow: hidden; border-radius: 14px; padding: 0.85rem 0.9rem 0.8rem 0.9rem;
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.96), rgba(15, 23, 42, 0.96));
            border: 1px solid {COLORS["border"]}; min-height: 118px;
            font-variant-numeric: tabular-nums;
            transition: border-color 0.15s ease, transform 0.15s ease;
        }}
        .kpi-card:hover {{ border-color: #334155; transform: translateY(-1px); }}
        .kpi-card::before {{
            content: ""; position: absolute; inset: 0 0 auto 0; height: 3px; background: var(--accent);
        }}
        .kpi-card::after {{
            content: ""; position: absolute; width: 140px; height: 140px; right: -50px; top: -60px;
            background: radial-gradient(circle, var(--glow), transparent 70%);
        }}
        .kpi-label {{
            color: {COLORS["muted"]}; font-size: 0.66rem; letter-spacing: 0.07em; text-transform: uppercase;
            font-weight: 600;
        }}
        .kpi-value {{
            color: {COLORS["text"]}; font-size: 1.42rem; font-weight: 750; letter-spacing: -0.02em;
            margin-top: 0.3rem; line-height: 1.1;
        }}
        .kpi-sub {{ color: {COLORS["muted"]}; font-size: 0.78rem; margin-top: 0.35rem; }}
        .chip {{
            display: inline-block; border-radius: 999px; padding: 0.12rem 0.55rem; font-size: 0.68rem;
            font-weight: 700; letter-spacing: 0.02em; margin-top: 0.45rem; white-space: nowrap;
        }}
        .kpi-spark {{ margin-top: 0.55rem; line-height: 0; opacity: 0.95; }}

        /* Cards, insights, audits */
        .audit-card, .insight-card {{
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.96), rgba(15, 23, 42, 0.96));
            border: 1px solid {COLORS["border"]}; border-radius: 14px; padding: 1rem 1.15rem;
            font-variant-numeric: tabular-nums; color: {COLORS["text"]}; font-size: 0.88rem; line-height: 1.65;
        }}
        .audit-card .head, .insight-card .head {{
            font-weight: 700; font-size: 0.95rem; margin-bottom: 0.45rem; color: {COLORS["text"]};
        }}
        .audit-card .row {{
            display: flex; justify-content: space-between; gap: 1rem;
            border-bottom: 1px dashed rgba(51, 65, 85, 0.6); padding: 0.18rem 0;
        }}
        .audit-card .row:last-child {{ border-bottom: none; }}
        .audit-card .muted {{ color: {COLORS["muted"]}; }}
        .insight-item {{ display: flex; gap: 0.65rem; padding: 0.4rem 0; border-bottom: 1px solid rgba(30, 41, 59, 0.8); }}
        .insight-item:last-child {{ border-bottom: none; }}
        .insight-item .bar {{ width: 4px; border-radius: 4px; flex-shrink: 0; }}
        .insight-item .txt {{ color: #E2E8F0; font-size: 0.86rem; line-height: 1.45; }}
        .insight-item .txt b {{ color: {COLORS["text"]}; }}

        [data-testid="stMetric"] {{
            background: {COLORS["card"]}; border: 1px solid {COLORS["border"]}; border-radius: 12px;
            padding: 0.75rem 0.9rem;
        }}
        [data-testid="stDataFrame"], [data-testid="stTable"] {{ font-variant-numeric: tabular-nums; }}
        [data-testid="stExpander"] details {{
            border-radius: 12px; border-color: {COLORS["border"]}; background: rgba(15, 23, 42, 0.55);
        }}
        .stDownloadButton button, .stButton button {{ border-radius: 10px; font-weight: 600; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _is_bad(val: float | None) -> bool:
    return val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val)))


def fmt_millions(val: float | None) -> str:
    if _is_bad(val):
        return "n/a"
    sign = "-" if val < 0 else ""
    return f"{sign}${abs(val) / 1e6:,.2f}M"


def fmt_signed_millions(val: float | None) -> str:
    if _is_bad(val):
        return "n/a"
    sign = "-" if val < 0 else "+"
    return f"{sign}${abs(val) / 1e6:,.2f}M"


def fmt_currency(val: float | None) -> str:
    if _is_bad(val):
        return "n/a"
    sign = "-" if val < 0 else ""
    return f"{sign}${abs(val):,.2f}"


def fmt_pct(val: float | None, decimals: int = 2) -> str:
    if _is_bad(val):
        return "n/a"
    return f"{val * 100:.{decimals}f}%"
