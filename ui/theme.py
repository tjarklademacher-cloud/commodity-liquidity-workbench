"""Single source of truth for the workbench's visual system.

Tokens here mirror `.streamlit/config.toml`. Streamlit's native theme handles widgets, tabs, tables and
fonts; the CSS below only styles the few custom HTML elements (header, verdict panels, briefing,
audit rows) and the keyed `panel_*` containers that hold charts.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import plotly.graph_objects as go
import streamlit as st

from core.schemas import LiquidityRegime

COLORS: dict[str, str] = {
    "canvas": "#F6F7F9",
    "surface": "#FFFFFF",
    "sidebar": "#F0F2F5",
    "border": "#E2E6EC",
    "grid": "#ECEFF4",
    "text": "#0F172A",
    "text_secondary": "#475569",
    "muted": "#5B6577",
    "accent": "#1B4965",
    "accent_hover": "#2C6E91",
    "accent_soft": "#DCE7EF",
    "neutral_fill": "#CBD5E1",
    "neutral_light": "#E8ECF1",
    "reference": "#64748B",
}

# Semantic risk states: (text colour, background colour). Use only where the model defines the state.
STATUS: dict[str, tuple[str, str]] = {
    "healthy": ("#166534", "#ECFDF3"),
    "caution": ("#9A3412", "#FFF7ED"),
    "breach": ("#991B1B", "#FEF2F2"),
    "neutral": ("#475569", "#F1F5F9"),
}

REGIME_STATUS: dict[LiquidityRegime, str] = {
    LiquidityRegime.HEALTHY: "healthy",
    LiquidityRegime.MARGIN_TRAP_WARNING: "caution",
    LiquidityRegime.BB_DEFICIENCY_WARNING: "caution",
    LiquidityRegime.CRITICAL_SQUEEZE: "breach",
}

REGIME_LABELS: dict[LiquidityRegime, str] = {
    LiquidityRegime.HEALTHY: "Healthy",
    LiquidityRegime.MARGIN_TRAP_WARNING: "Margin trap",
    LiquidityRegime.BB_DEFICIENCY_WARNING: "BB deficiency",
    LiquidityRegime.CRITICAL_SQUEEZE: "Critical squeeze",
}

COMMODITY_COLORS: dict[str, str] = {
    "GC=F": "#9B8018",
    "HG=F": "#AA5C35",
    "LBR=F": "#386A52",
    "CC=F": "#6D4A3C",
    "KC=F": "#3E2C23",
    "SB=F": "#7C8DA6",
}
COMMODITY_DASH: dict[str, str] = {"GC=F": "solid", "HG=F": "dash", "LBR=F": "dot", "CC=F": "dashdot",
                                  "KC=F": "longdash", "SB=F": "longdashdot"}
PORTFOLIO_COLOR = "#0F172A"

HOUSE_NAME = "Lademacher Commodities"
AUTHOR = {
    "name": "Tjark Lademacher",
    "bio": "Commodity trade finance and risk analytics",
    "github": "https://github.com/tjarklademacher-cloud",
    "repo": "https://github.com/tjarklademacher-cloud/commodity-liquidity-workbench",
}

FONT_SANS = "IBM Plex Sans, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif"
FONT_MONO = "IBM Plex Mono, SFMono-Regular, Menlo, Consolas, monospace"

SPACE = {"xs": 4, "s": 8, "m": 12, "l": 16, "xl": 24, "xxl": 32}

PLOTLY_CONFIG: dict = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "zoomIn2d", "zoomOut2d"],
}


# Display names for the core waterfall step labels (the model's own labels are left unchanged).
BRIDGE_LABELS: dict[str, str] = {
    "Gross Inventory": "Gross inventory",
    "Inv. Haircuts": "Inventory haircuts",
    "Eligible Inventory": "Eligible inventory",
    "Gross AR": "Gross receivables",
    "AR Haircuts": "Receivable advance haircuts",
    "Conc. Cutback": "Concentration cutback",
    "Final Borrowing Base": "Borrowing base",
}


def bridge_label(label: str) -> str:
    if label.startswith("Aging >"):
        return f"Aged over {label.removeprefix('Aging >').removesuffix('d')} days"
    return BRIDGE_LABELS.get(label, label)


def regime_label(regime: LiquidityRegime | str) -> str:
    return REGIME_LABELS[LiquidityRegime(str(regime))]


def regime_status(regime: LiquidityRegime | str) -> str:
    return REGIME_STATUS[LiquidityRegime(str(regime))]


def apply_institutional_layout(
    fig: go.Figure,
    title: str = "",
    height: int = 380,
    yaxis_title: str = "",
    xaxis_title: str = "",
) -> go.Figure:
    """Shared Plotly styling. Titles normally live in the surrounding panel, so `title` defaults to none."""
    top = 44 if title else 12
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=14, color=COLORS["text"], family=FONT_SANS), x=0.0,
                   xanchor="left") if title else None,
        height=height,
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(family=FONT_SANS, size=12, color=COLORS["text_secondary"]),
        margin=dict(l=56, r=16, t=top, b=44),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=COLORS["surface"], bordercolor=COLORS["border"],
                        font=dict(color=COLORS["text"], family=FONT_SANS, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0,
                    bgcolor="rgba(0, 0, 0, 0)", font=dict(size=11, color=COLORS["text_secondary"]),
                    itemclick="toggleothers"),
        bargap=0.28,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=True, linecolor=COLORS["border"], ticks="outside",
                     tickcolor=COLORS["border"], ticklen=4, tickfont=dict(size=11, color=COLORS["muted"]))
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["grid"], zeroline=False, showline=False,
                     tickfont=dict(size=11, color=COLORS["muted"]))
    if yaxis_title:
        fig.update_yaxes(title=dict(text=yaxis_title, font=dict(size=11, color=COLORS["muted"]), standoff=8))
    if xaxis_title:
        fig.update_xaxes(title=dict(text=xaxis_title, font=dict(size=11, color=COLORS["muted"]), standoff=8))
    return fig


def inject_custom_css() -> None:
    """Minimal CSS for custom HTML elements and keyed panel containers. Widgets use the native theme."""
    c = COLORS
    st.markdown(
        f"""
<style>
.block-container {{ padding-top: 3.25rem; padding-bottom: 2rem; max-width: 1480px; }}
.num, .kpi .value, .ledger-num {{ font-family: {FONT_MONO}; font-variant-numeric: tabular-nums; }}
[data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}

[class*="st-key-panel_"] {{
  background: {c["surface"]}; border: 1px solid {c["border"]}; border-radius: 6px;
  padding: 16px 16px 8px 16px;
}}

.app-header {{
  display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; flex-wrap: wrap;
  padding-bottom: 12px; margin-bottom: 16px; border-bottom: 1px solid {c["border"]};
}}
.app-header .title {{ font-size: 21px; font-weight: 600; color: {c["text"]}; line-height: 1.2; }}
.app-header .subtitle {{ font-size: 13px; color: {c["text_secondary"]}; margin-top: 2px; }}
.app-header .meta {{
  display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 4px 16px;
  font-size: 12px; color: {c["muted"]};
}}
.app-header .meta span b {{ color: {c["text_secondary"]}; font-weight: 500; }}
.dot {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px;
  vertical-align: 1px; }}

.verdict {{
  display: grid; grid-template-columns: minmax(300px, 1.6fr) repeat(4, minmax(150px, 1fr));
  gap: 12px; margin-bottom: 16px;
}}
@media (max-width: 1180px) {{
  .verdict {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .verdict .primary {{ grid-column: 1 / -1; }}
}}
.kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px;
  margin-bottom: 16px; }}
@media (max-width: 640px) {{ .verdict, .kpi-row {{ grid-template-columns: 1fr; }} }}

.kpi {{
  background: {c["surface"]}; border: 1px solid {c["border"]}; border-radius: 6px; padding: 14px 16px;
  min-width: 0;
}}
.kpi.primary {{ border-left-width: 3px; padding: 16px 20px; }}
.kpi .eyebrow {{ font-size: 11px; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase;
  color: {c["muted"]}; }}
.kpi .value {{ font-size: clamp(18px, 1.55vw, 22px); font-weight: 500; color: {c["text"]}; margin-top: 6px;
  line-height: 1.15; overflow-wrap: anywhere; }}
.kpi.primary .value {{ font-size: clamp(24px, 2.1vw, 30px); }}
.kpi .context {{ font-size: 12px; color: {c["text_secondary"]}; margin-top: 6px; line-height: 1.45; }}
.kpi .kpi-top {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; }}
.kpi .status {{ display: inline-block; font-size: 11px; font-weight: 600; padding: 1px 7px;
  border-radius: 4px; white-space: nowrap; flex-shrink: 0; }}

.section-head {{ margin: 0 0 8px 0; }}
.section-head .title {{ font-size: 15px; font-weight: 600; color: {c["text"]}; }}
.section-head .sub {{ font-size: 12px; color: {c["muted"]}; margin-top: 2px; line-height: 1.45; }}

.side-section {{
  font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: {c["muted"]};
  border-top: 1px solid {c["border"]}; padding-top: 12px; margin: 16px 0 4px 0;
}}
.side-section.first {{ border-top: none; padding-top: 0; margin-top: 0; }}

.briefing-item {{ display: flex; gap: 10px; padding: 8px 0; border-top: 1px solid {c["grid"]};
  font-size: 13px; line-height: 1.5; color: {c["text"]}; }}
.briefing-item:first-child {{ border-top: none; padding-top: 0; }}
.briefing-item .marker {{ width: 3px; border-radius: 2px; flex-shrink: 0; }}
.briefing-item b {{ font-weight: 600; }}

.audit-row {{ display: flex; justify-content: space-between; gap: 16px; padding: 6px 0;
  border-top: 1px solid {c["grid"]}; font-size: 13px; }}
.audit-row:first-child {{ border-top: none; }}
.audit-row .k {{ color: {c["text_secondary"]}; }}
.audit-row .v {{ color: {c["text"]}; text-align: right; font-family: {FONT_MONO};
  font-variant-numeric: tabular-nums; }}
.audit-row .v.text {{ font-family: {FONT_SANS}; }}
.badge {{ display: inline-block; font-size: 11px; font-weight: 600; padding: 1px 7px; border-radius: 4px;
  font-family: {FONT_SANS}; }}
.provenance {{ font-size: 12px; color: {c["muted"]}; margin: -4px 0 12px 0; }}
.author {{ font-size: 12px; color: {c["text_secondary"]}; line-height: 1.5; }}
.author .name {{ font-weight: 600; color: {c["text"]}; font-size: 13px; }}
.author a {{ color: {c["accent_hover"]}; text-decoration: none; }}
</style>
        """,
        unsafe_allow_html=True,
    )


@dataclass(frozen=True)
class DisplayCurrency:
    """Reporting currency for outputs. The model and all inputs stay in USD; outputs divide by usd_per_unit."""

    code: str = "USD"
    symbol: str = "$"
    usd_per_unit: float = 1.0
    source: str = "model currency"
    as_of: str = ""


CURRENCY_SYMBOLS: dict[str, str] = {"USD": "$", "EUR": "\u20ac", "GBP": "\u00a3", "CHF": "CHF "}
_CURRENCY: ContextVar[DisplayCurrency] = ContextVar("display_currency", default=DisplayCurrency())


def set_display_currency(currency: DisplayCurrency) -> None:
    _CURRENCY.set(currency)


def display_currency() -> DisplayCurrency:
    return _CURRENCY.get()


@contextmanager
def using_currency(currency: DisplayCurrency) -> Iterator[None]:
    token = _CURRENCY.set(currency)
    try:
        yield
    finally:
        _CURRENCY.reset(token)


def money_divisor() -> float:
    """Divide a USD amount by this to get display-currency millions."""
    return 1e6 * display_currency().usd_per_unit


def money_unit() -> str:
    return f"{display_currency().code} M"


def localize_figure(fig: go.Figure) -> go.Figure:
    """Swap the USD symbol and unit labels in a figure's text for the active reporting currency."""
    cur = display_currency()
    if cur.code == "USD":
        return fig

    def swap(text):
        if isinstance(text, str):
            return text.replace("USD millions", f"{cur.code} millions").replace("USD M", f"{cur.code} M").replace(
                "$", cur.symbol)
        return text

    for trace in fig.data:
        for attr in ("hovertemplate", "texttemplate", "name"):
            if hasattr(trace, attr) and isinstance(getattr(trace, attr), str):
                setattr(trace, attr, swap(getattr(trace, attr)))
        if hasattr(trace, "text") and trace.text is not None:
            trace.text = swap(trace.text) if isinstance(trace.text, str) else [
                [swap(v) for v in row] if isinstance(row, (list, tuple)) else swap(row) for row in trace.text]
        if hasattr(trace, "customdata") and trace.customdata is not None and getattr(trace, "type", "") == "waterfall":
            trace.customdata = [swap(v) for v in trace.customdata]
        if getattr(trace, "colorbar", None) is not None and trace.colorbar.title.text:
            trace.colorbar.title.text = swap(trace.colorbar.title.text)
            if trace.colorbar.tickprefix:
                trace.colorbar.tickprefix = swap(trace.colorbar.tickprefix)
    for ann in fig.layout.annotations:
        ann.text = swap(ann.text)
    fig.for_each_xaxis(lambda ax: ax.update(tickprefix=swap(ax.tickprefix), title_text=swap(ax.title.text)))
    fig.for_each_yaxis(lambda ax: ax.update(tickprefix=swap(ax.tickprefix), title_text=swap(ax.title.text)))
    return fig


def _is_bad(val: float | None) -> bool:
    return val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val)))


def fmt_millions(val: float | None) -> str:
    """USD amount shown in the reporting currency, in millions."""
    if _is_bad(val):
        return "n/a"
    sign = "-" if val < 0 else ""
    return f"{sign}{display_currency().symbol}{abs(val) / money_divisor():,.2f}M"


def fmt_signed_millions(val: float | None) -> str:
    if _is_bad(val):
        return "n/a"
    sign = "-" if val < 0 else "+"
    return f"{sign}{display_currency().symbol}{abs(val) / money_divisor():,.2f}M"


def fmt_currency(val: float | None) -> str:
    if _is_bad(val):
        return "n/a"
    sign = "-" if val < 0 else ""
    cur = display_currency()
    return f"{sign}{cur.symbol}{abs(val) / cur.usd_per_unit:,.2f}"


def fmt_pct(val: float | None, decimals: int = 2) -> str:
    if _is_bad(val):
        return "n/a"
    return f"{val * 100:.{decimals}f}%"
