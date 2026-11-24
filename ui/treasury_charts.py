"""Plotly figures for the treasury, credit, hedge frontier and case study views.

Money in the workbench book is USD and shown in the reporting currency (`money_divisor`); the 2022 case study
is a separate hypothetical entity whose functional currency is EUR, so its figures are in EUR millions and are
never converted. Titles live in the surrounding panel; single y-axis per subplot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.theme import COLORS, STATUS, apply_institutional_layout, money_divisor

BREACH = STATUS["breach"][0]
HEALTHY = STATUS["healthy"][0]
CAUTION = STATUS["caution"][0]
SERIES = ["#1B4965", "#64748B", "#9A3412", "#166534", "#6D4A3C", "#7C8DA6", "#0F172A"]


def _usd(values, divisor: float | None = None):
    d = divisor if divisor is not None else money_divisor()
    return np.asarray(values, dtype=float) / d


def build_treasury_path_chart(ledger: pd.DataFrame, min_cash: float, divisor: float | None = None,
                              unit: str = "USD M") -> go.Figure:
    """Top: unrestricted cash and available liquidity with the minimum operating cash line.
    Bottom: reported borrowing base, permitted utilization (capped at the commitment) and revolver drawn."""
    x = pd.to_datetime(ledger["date"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.55, 0.45])
    fig.add_trace(go.Scatter(x=x, y=_usd(ledger["available_liquidity"], divisor), name="Available liquidity",
                             line=dict(color=COLORS["accent"], width=2),
                             hovertemplate=f"%{{x|%Y-%m-%d}}: %{{y:,.2f}} {unit}<extra>Available liquidity</extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=_usd(ledger["cash"], divisor), name="Unrestricted cash",
                             line=dict(color=COLORS["reference"], width=1.5, dash="dot"),
                             hovertemplate=f"%{{x|%Y-%m-%d}}: %{{y:,.2f}} {unit}<extra>Cash</extra>"), row=1, col=1)
    fig.add_hline(y=float(_usd([min_cash], divisor)[0]), line=dict(color=BREACH, dash="dash", width=1), row=1, col=1)
    short = ledger["shortfall"].to_numpy() > 1e-6
    if short.any():
        fig.add_trace(go.Bar(x=x[short], y=-_usd(ledger["shortfall"][short], divisor), name="Unfunded shortfall",
                             marker_color=BREACH, opacity=0.6,
                             hovertemplate=f"%{{x|%Y-%m-%d}}: %{{y:,.2f}} {unit}<extra>Shortfall</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=_usd(ledger["reported_borrowing_base"], divisor), name="Reported borrowing base",
                             line=dict(color=HEALTHY, width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=_usd(ledger["permitted"], divisor), name="Permitted (min of base, commitment)",
                             line=dict(color=COLORS["reference"], width=1, dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=_usd(ledger["revolver_drawn"], divisor), name="Revolver drawn",
                             line=dict(color=CAUTION, width=2)), row=2, col=1)
    apply_institutional_layout(fig, height=520)
    fig.update_yaxes(title_text=f"Liquidity ({unit})", row=1, col=1)
    fig.update_yaxes(title_text=f"Facility ({unit})", row=2, col=1)
    return fig


CASH_FLOW_GROUPS: dict[str, tuple[tuple[str, int], ...]] = {
    "Collections and claims": (("collections", 1), ("insurance_receipts", 1), ("recoveries", 1)),
    "Variation margin": (("vm_received", 1), ("vm_paid", -1)),
    "Initial margin": (("im_released", 1), ("im_call", -1)),
    "Supplier payments": (("supplier_payments", -1),),
    "Operating costs": (("storage_costs", -1), ("opex", -1), ("transaction_costs", -1)),
    "Interest and fees": (("interest", -1), ("commitment_fees", -1)),
    "Financing received": (("revolver_received", 1), ("lines_received", 1), ("emergency_received", 1)),
    "Repayments": (("revolver_repaid", -1), ("lines_repaid", -1), ("mandatory_repayment", -1)),
}


def cash_flow_decomposition(ledger: pd.DataFrame) -> list[tuple[str, float]]:
    return [(g, float(sum(sign * ledger[f].sum() for f, sign in fields))) for g, fields in CASH_FLOW_GROUPS.items()]


def build_cash_flow_waterfall(ledger: pd.DataFrame, divisor: float | None = None, unit: str = "USD M") -> go.Figure:
    """Opening cash to closing cash over the horizon by flow category (operating vs financing)."""
    parts = [p for p in cash_flow_decomposition(ledger) if abs(p[1]) > 1e-6]
    labels = ["Opening cash", *[p[0] for p in parts], "Closing cash"]
    vals = [float(_usd([ledger["cash"].iloc[0]], divisor)[0]), *[float(_usd([p[1]], divisor)[0]) for p in parts], None]
    measure = ["absolute", *["relative"] * len(parts), "total"]
    closing = float(_usd([ledger["cash"].iloc[-1]], divisor)[0])
    text = [f"{v:,.2f}" if v is not None else f"{closing:,.2f}" for v in vals]
    fig = go.Figure(go.Waterfall(x=labels, y=vals, measure=measure, text=text, textposition="outside",
                                 increasing=dict(marker=dict(color=HEALTHY)),
                                 decreasing=dict(marker=dict(color=BREACH)),
                                 totals=dict(marker=dict(color=COLORS["accent"])),
                                 connector=dict(line=dict(color=COLORS["border"], width=1)),
                                 hovertemplate=f"%{{x}}: %{{y:,.2f}} {unit}<extra></extra>"))
    apply_institutional_layout(fig, height=380, yaxis_title=unit)
    fig.update_layout(showlegend=False)
    return fig


def build_survival_chart(survival: Sequence[float]) -> go.Figure:
    days = list(range(0, len(survival) + 1))
    fig = go.Figure(go.Scatter(x=days, y=[100.0, *[s * 100 for s in survival]], mode="lines", line_shape="hv",
                               line=dict(color=COLORS["accent"], width=2),
                               hovertemplate="Day %{x}: %{y:.2f}% of paths without a breach<extra></extra>"))
    apply_institutional_layout(fig, height=320, yaxis_title="Paths without a breach (%)", xaxis_title="Business day")
    fig.update_layout(showlegend=False)
    fig.update_yaxes(ticksuffix="%", range=[min(survival + [1.0]) * 100 - 5, 101])
    return fig


def build_distribution_chart(samples: Sequence[float], marker: float | None, xlabel: str,
                             marker_label: str) -> go.Figure:
    x = _usd(samples)
    fig = go.Figure(go.Histogram(x=x, nbinsx=50, marker_color=COLORS["neutral_fill"],
                                 hovertemplate="%{x:,.2f}: %{y} paths<extra></extra>"))
    if marker is not None:
        fig.add_vline(x=float(_usd([marker])[0]), line=dict(color=BREACH, dash="dash", width=1.5),
                      annotation_text=marker_label, annotation_position="top")
    apply_institutional_layout(fig, height=320, yaxis_title="Paths", xaxis_title=xlabel)
    fig.update_layout(showlegend=False, bargap=0.05)
    return fig


def build_sensitivity_bars(df: pd.DataFrame) -> go.Figure:
    y = df["assumption"].tolist()[::-1]
    x = _usd(df["min_liquidity"]).tolist()[::-1]
    colors = [BREACH if v < 0 else COLORS["accent"] for v in x]
    fig = go.Figure(go.Bar(x=x, y=y, orientation="h", marker_color=colors, text=[f"{v:,.2f}" for v in x],
                           textposition="inside", insidetextanchor="middle", textfont=dict(color="#FFFFFF"),
                           hovertemplate="%{y}: %{x:,.2f} USD M<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=COLORS["reference"], width=1))
    apply_institutional_layout(fig, height=340, xaxis_title="Minimum available liquidity (USD M)")
    fig.update_layout(showlegend=False, margin=dict(l=250, t=12))
    fig.update_yaxes(automargin=True)
    return fig


def build_exposure_chart(table: pd.DataFrame) -> go.Figure:
    t = table.sort_values("exposure")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=t["counterparty"], x=_usd(t["uninsured"]), orientation="h", name="Uninsured",
                         marker_color=COLORS["accent"], hovertemplate="%{y}: %{x:,.2f} USD M<extra>Uninsured</extra>"))
    fig.add_trace(go.Bar(y=t["counterparty"], x=_usd(t["insured"]), orientation="h", name="Insured",
                         marker_color=COLORS["neutral_fill"], hovertemplate="%{y}: %{x:,.2f} USD M<extra>Insured</extra>"))
    apply_institutional_layout(fig, height=60 + 34 * len(t), xaxis_title="Receivable exposure (USD M)")
    fig.update_layout(barmode="stack", margin=dict(l=220))
    return fig


def build_pareto_frontier_chart(df: pd.DataFrame) -> go.Figure:
    """Economic downside (5% quantile loss) against the 95th percentile funding need, one point per ratio."""
    fig = go.Figure()
    for feas, pareto, name, color, symbol in ((True, True, "Pareto-efficient", COLORS["accent"], "circle"),
                                              (True, False, "Dominated", COLORS["reference"], "circle-open"),
                                              (False, False, "Infeasible", BREACH, "x")):
        sub = df[(df["feasible"] == feas) & (df["pareto"] == pareto)]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=_usd(sub["economic_downside"]), y=_usd(sub["funding_p95"]), mode="markers+text", name=name,
            marker=dict(color=color, size=10, symbol=symbol, line=dict(width=1.5, color=color)),
            text=[f"{r:.0%}" for r in sub["hedge_ratio"]], textposition="top center",
            customdata=sub[["hedge_ratio", "prob_breach", "label"]].to_numpy(),
            hovertemplate=("Hedge %{customdata[0]:.0%}<br>Economic downside %{x:,.2f} USD M<br>Funding p95 "
                           "%{y:,.2f} USD M<br>P(breach) %{customdata[1]:.1%}<br>%{customdata[2]}<extra></extra>")))
    apply_institutional_layout(fig, height=380, xaxis_title="Economic downside: 5% quantile loss (USD M)",
                               yaxis_title="Funding need, 95th percentile (USD M)")
    fig.update_layout(hovermode="closest")
    return fig


def build_case_price_chart(dates: pd.DatetimeIndex, prices: np.ndarray, basis: np.ndarray | None,
                           ledger: Mapping[str, np.ndarray]) -> go.Figure:
    """Row 1: observed TTF (EUR/MWh) and synthetic physical price. Row 2: cumulative VM and posted IM (EUR M)."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=dates, y=prices, name="TTF=F close (observed)", line=dict(color=COLORS["accent"], width=1.8),
                             hovertemplate="%{x|%Y-%m-%d}: %{y:,.2f} EUR/MWh<extra>Observed</extra>"), row=1, col=1)
    if basis is not None:
        fig.add_trace(go.Scatter(x=dates, y=prices * (1 + basis), name="Physical at storage (synthetic basis)",
                                 line=dict(color=COLORS["reference"], width=1, dash="dot"),
                                 hovertemplate="%{x|%Y-%m-%d}: %{y:,.2f} EUR/MWh<extra>Synthetic</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=np.asarray(ledger["cum_vm"]) / 1e6, name="Cumulative variation margin",
                             line=dict(color=BREACH, width=1.8),
                             hovertemplate="%{x|%Y-%m-%d}: %{y:,.2f} EUR M<extra>Cumulative VM</extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=np.asarray(ledger["im_posted"]) / 1e6, name="Initial margin posted",
                             line=dict(color=CAUTION, width=1.5),
                             hovertemplate="%{x|%Y-%m-%d}: %{y:,.2f} EUR M<extra>IM posted</extra>"), row=2, col=1)
    apply_institutional_layout(fig, height=480)
    fig.update_yaxes(title_text="EUR/MWh", row=1, col=1)
    fig.update_yaxes(title_text="EUR M", row=2, col=1)
    return fig


def build_case_liquidity_chart(dates: pd.DatetimeIndex, ledgers: Mapping[str, Mapping[str, np.ndarray]],
                               names: Sequence[str]) -> go.Figure:
    fig = go.Figure()
    for i, n in enumerate(names):
        fig.add_trace(go.Scatter(x=dates, y=np.asarray(ledgers[n]["cash"]) / 1e6, name=n,
                                 line=dict(color=SERIES[i % len(SERIES)], width=1.6),
                                 hovertemplate="%{x|%Y-%m-%d}: %{y:,.2f} EUR M<extra>" + n + "</extra>"))
    fig.add_hline(y=0, line=dict(color=BREACH, dash="dash", width=1))
    apply_institutional_layout(fig, height=400, yaxis_title="Unrestricted cash after same-day funding (EUR M)")
    return fig
