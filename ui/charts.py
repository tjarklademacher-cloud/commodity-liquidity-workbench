"""Plotly chart builders. Single y-axis per panel, no in-figure titles (panels carry the headings)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.schemas import (
    REQUIRED_TICKERS,
    BorrowingBaseResult,
    CommoditySpec,
    CrisisReplayResult,
    FacilityTerms,
    HedgeFrontierResult,
    HistoricalReplayResult,
    InventoryValuation,
    MonteCarloResult,
    RunwayResult,
    ShockGridResult,
    StressScenarioResult,
    StressSweepResult,
)
from ui.theme import (
    COLORS,
    COMMODITY_COLORS,
    COMMODITY_DASH,
    FONT_SANS,
    PORTFOLIO_COLOR,
    STATUS,
    apply_institutional_layout,
    bridge_label,
    fmt_millions,
    money_divisor,
    regime_label,
    regime_status,
)

BREACH = STATUS["breach"][0]
CAUTION = STATUS["caution"][0]
HEALTHY = STATUS["healthy"][0]


def _m(val: float) -> float:
    return val / money_divisor()


def _mtext(val: float) -> str:
    return fmt_millions(val)


def _wrap_label(name: str) -> str:
    words = name.split()
    if len(name) <= 14 or len(words) < 2:
        return name
    best = min(range(1, len(words)), key=lambda k: abs(len(" ".join(words[:k])) - len(" ".join(words[k:]))))
    return " ".join(words[:best]) + "<br>" + " ".join(words[best:])


def _note(fig: go.Figure, x: float, text: str, color: str, top: bool, align: str = "left", row: int | None = None,
          yref: str = "paper") -> None:
    """Compact text label pinned to the top or bottom edge of the plot at data x."""
    xanchor = "left" if align == "left" else "right"
    shift = 4 if align == "left" else -4
    kwargs = dict(x=x, y=1.0 if top else 0.0, yref=yref if row is None else f"y{'' if row == 1 else row} domain",
                  xref="x" if row in (None, 1) else f"x{row}", text=text, showarrow=False, xanchor=xanchor,
                  yanchor="top" if top else "bottom", xshift=shift, yshift=-2 if top else 2,
                  font=dict(size=11, color=color, family=FONT_SANS), bgcolor="rgba(255, 255, 255, 0.85)")
    fig.add_annotation(**kwargs)


def build_borrowing_base_waterfall(bb_result: BorrowingBaseResult) -> go.Figure:
    """Gross collateral to final borrowing base bridge, with facility limit and drawn balance."""
    wf = bb_result.waterfall
    fig = go.Figure(
        go.Waterfall(
            x=[_wrap_label(bridge_label(w.label)) for w in wf],
            measure=[w.measure for w in wf],
            y=[None if w.measure == "total" else _m(w.amount) for w in wf],
            text=[_mtext(w.amount) for w in wf],
            textposition="outside",
            textfont=dict(color=COLORS["text"], size=11),
            cliponaxis=False,
            connector=dict(line=dict(color=COLORS["neutral_fill"], width=1)),
            increasing=dict(marker=dict(color=COLORS["accent_hover"])),
            decreasing=dict(marker=dict(color="#94A3B8")),
            totals=dict(marker=dict(color=COLORS["accent"])),
            hovertemplate="%{x}<br>%{text}<extra></extra>",
            name="Borrowing base bridge",
        )
    )
    fig.add_hline(y=_m(bb_result.facility_limit), line=dict(color=COLORS["reference"], dash="dash", width=1.2))
    fig.add_hline(y=_m(bb_result.drawn_revolver), line=dict(color=COLORS["text"], dash="dot", width=1.2))
    fig.add_annotation(x=0, xref="paper", y=_m(bb_result.facility_limit), yref="y", xanchor="left", yanchor="bottom",
                       showarrow=False, text=f"Facility limit {_mtext(bb_result.facility_limit)}",
                       font=dict(size=11, color=COLORS["muted"]), bgcolor="rgba(255, 255, 255, 0.85)")
    fig.add_annotation(x=0, xref="paper", y=_m(bb_result.drawn_revolver), yref="y", xanchor="left", yanchor="top",
                       showarrow=False, text=f"Drawn {_mtext(bb_result.drawn_revolver)}",
                       font=dict(size=11, color=COLORS["text"]), bgcolor="rgba(255, 255, 255, 0.85)")
    peak = max(w.running_total for w in wf) if wf else 0.0
    apply_institutional_layout(fig, height=400, yaxis_title="USD millions")
    fig.update_layout(showlegend=False, hovermode="closest", margin=dict(t=24, b=56))
    fig.update_xaxes(tickangle=0, tickfont=dict(size=11, color=COLORS["text_secondary"]))
    fig.update_yaxes(range=[0.0, _m(max(peak, bb_result.facility_limit)) * 1.12], tickprefix="$")
    return fig


def build_haircut_decomposition_chart(inventory_items: Sequence[InventoryValuation]) -> go.Figure:
    """100% stacked advance rate vs. effective haircut components per commodity."""
    names = [i.name for i in inventory_items]
    custom = [
        [i.base_haircut * 100, i.vol_addon * 100, i.unhedged_penalty * 100, i.raw_haircut * 100,
         i.total_haircut * 100, i.realized_vol_30d * 100, i.hedge_ratio * 100]
        for i in inventory_items
    ]
    hover_tail = (
        "<br>Raw: base %{customdata[0]:.2f}%, vol add-on %{customdata[1]:.2f}%, unhedged %{customdata[2]:.2f}%"
        "<br>Haircut %{customdata[3]:.2f}% raw, %{customdata[4]:.2f}% after floor/cap"
        "<br>30d vol %{customdata[5]:.2f}%, hedge ratio %{customdata[6]:.1f}%<extra></extra>"
    )
    components = [
        ("Advance rate", [i.advance_rate for i in inventory_items], COLORS["accent"], "#FFFFFF"),
        ("Base haircut", [i.eff_base_haircut for i in inventory_items], "#94A3B8", COLORS["text"]),
        ("Volatility add-on", [i.eff_vol_addon for i in inventory_items], "#CBD5E1", COLORS["text"]),
        ("Unhedged penalty", [i.eff_unhedged_penalty for i in inventory_items], "#D9C9A8", COLORS["text"]),
    ]
    floor = [i.eff_floor_adjustment for i in inventory_items]
    if any(f > 0.0 for f in floor):
        components.append(("Minimum haircut floor", floor, "#E8ECF1", COLORS["text"]))
    fig = go.Figure()
    for label, values, color, text_color in components:
        fig.add_trace(
            go.Bar(
                y=names, x=[v * 100 for v in values], name=label, orientation="h",
                marker=dict(color=color, line=dict(color="#FFFFFF", width=1)), customdata=custom,
                text=[f"{v * 100:.1f}%" if v >= 0.04 else "" for v in values],
                textposition="inside", insidetextanchor="middle", textfont=dict(color=text_color, size=11),
                hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.2f}}% of market value" + hover_tail,
            )
        )
    apply_institutional_layout(fig, height=330, xaxis_title="Share of gross market value (%)")
    fig.update_layout(barmode="stack", hovermode="closest", margin=dict(l=72, t=36))
    fig.update_xaxes(range=[0, 100], ticksuffix="%", showgrid=True, gridcolor=COLORS["grid"])
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=12, color=COLORS["text_secondary"]))
    return fig


def build_volatility_history_chart(vol_df: pd.DataFrame, specs: Mapping[str, CommoditySpec]) -> go.Figure:
    """Rolling 30-day annualized realized volatility with each commodity's reference volatility."""
    fig = go.Figure()
    for ticker in vol_df.columns:
        series = vol_df[ticker].dropna()
        label = specs[ticker].name if ticker in specs else ticker
        fig.add_trace(
            go.Scatter(
                x=series.index, y=series.to_numpy() * 100, mode="lines", name=label,
                line=dict(color=COMMODITY_COLORS.get(ticker, PORTFOLIO_COLOR), width=1.8,
                          dash=COMMODITY_DASH.get(ticker, "solid")),
                hovertemplate=f"{label}: %{{y:.2f}}%<extra></extra>",
            )
        )
    refs: dict[float, list[str]] = {}
    for t in vol_df.columns:
        if t in specs:
            refs.setdefault(round(specs[t].ref_vol * 100, 4), []).append(specs[t].name)
    for level in sorted(refs):
        fig.add_hline(y=level, line=dict(color=COLORS["reference"], dash="dot", width=1))
    apply_institutional_layout(fig, height=330, yaxis_title="Annualized volatility (%)")
    fig.update_layout(margin=dict(t=36))
    fig.update_yaxes(ticksuffix="%", rangemode="tozero")
    return fig


def build_receivables_concentration_chart(bb_result: BorrowingBaseResult) -> go.Figure:
    """Pre-cap vs. capped eligible AR per buyer against the single-buyer cap."""
    items = bb_result.receivable_items
    buyers = [_wrap_label(r.buyer_name) for r in items]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=buyers, y=[_m(r.pre_cap_eligible) for r in items], name="Eligible before cap",
            marker=dict(color="#CBD5E1"), hovertemplate="%{x}<br>Before cap: $%{y:,.2f}M<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=buyers, y=[_m(r.final_eligible) for r in items], name="Eligible after cap",
            marker=dict(color=[CAUTION if r.is_capped else COLORS["accent"] for r in items]),
            text=[_mtext(r.final_eligible) if r.final_eligible > 0 else "" for r in items],
            textposition="outside", textangle=0, cliponaxis=False,
            textfont=dict(color=COLORS["text"], size=10),
            hovertemplate="%{x}<br>After cap: $%{y:,.2f}M<extra></extra>",
        )
    )
    for r in items:
        if not r.is_aging_eligible:
            fig.add_annotation(x=_wrap_label(r.buyer_name), y=0.0, text="Over 90 days,<br>ineligible",
                               showarrow=False, yanchor="bottom", yshift=4, font=dict(color=COLORS["muted"], size=10))
    cap_m = _m(bb_result.cap_dollar_limit)
    fig.add_hline(y=cap_m, line=dict(color=COLORS["reference"], dash="dash", width=1.2))
    fig.add_annotation(x=1.0, xref="paper", y=cap_m, yref="y", xanchor="right", yanchor="bottom", showarrow=False,
                       text=f"Cap {bb_result.concentration_cap_pct:.1%} of BB = ${cap_m:.2f}M",
                       font=dict(size=11, color=COLORS["muted"]))
    peak = max([_m(r.pre_cap_eligible) for r in items] + [cap_m, 1.0])
    apply_institutional_layout(fig, height=360, yaxis_title="USD millions")
    fig.update_layout(barmode="group", hovermode="closest", margin=dict(t=36, b=64),
                      uniformtext=dict(minsize=9, mode="show"))
    fig.update_xaxes(tickangle=0, tickfont=dict(size=10, color=COLORS["text_secondary"]))
    fig.update_yaxes(range=[0.0, peak * 1.22], tickprefix="$")
    return fig


def _regime_segments(sweep_result: StressSweepResult) -> list[tuple[float, float, str]]:
    pts = sweep_result.points
    if not pts:
        return []
    step = (pts[1].shock_pct - pts[0].shock_pct) * 100 if len(pts) > 1 else 1.0
    segments: list[tuple[float, float, str]] = []
    start = pts[0].shock_pct * 100 - step / 2
    current = pts[0].regime
    for prev, nxt in zip(pts[:-1], pts[1:]):
        if nxt.regime != current:
            mid = (prev.shock_pct + nxt.shock_pct) * 50
            segments.append((start, mid, current))
            start, current = mid, nxt.regime
    segments.append((start, pts[-1].shock_pct * 100 + step / 2, current))
    return segments


def _add_regime_bands(fig: go.Figure, sweep_result: StressSweepResult, row: int | None = None) -> None:
    for x0, x1, regime in _regime_segments(sweep_result):
        kwargs = dict(row=row, col=1) if row else {}
        fig.add_vrect(x0=x0, x1=x1, fillcolor=STATUS[regime_status(regime)][1], line_width=0, layer="below",
                      **kwargs)


def _buffer_trace(df: pd.DataFrame) -> go.Scatter:
    labels = [regime_label(r) for r in df["regime"]]
    return go.Scatter(
        x=df["shock_pct"] * 100, y=df["net_liquidity_buffer"] / money_divisor(), mode="lines+markers",
        name="Net liquidity buffer", line=dict(color=PORTFOLIO_COLOR, width=2), marker=dict(size=4),
        customdata=labels, hovertemplate="Shock %{x:+.0f}%<br>Buffer $%{y:,.2f}M<br>%{customdata}<extra></extra>",
    )


def build_buffer_profile_chart(sweep_result: StressSweepResult, active_shock_pct: float | None) -> go.Figure:
    """Overview: net liquidity buffer across parallel shocks with regime shading and breakevens."""
    df = sweep_result.to_dataframe()
    fig = go.Figure(_buffer_trace(df))
    _add_regime_bands(fig, sweep_result)
    fig.add_hline(y=0.0, line=dict(color=BREACH, dash="dash", width=1))
    if active_shock_pct is not None:
        fig.add_vline(x=active_shock_pct * 100, line=dict(color=COLORS["accent"], width=1.5))
        _note(fig, active_shock_pct * 100, f"Active {active_shock_pct:+.0%}", COLORS["accent"], top=True)
    if sweep_result.upside_squeeze_breakeven_pct is not None:
        be = sweep_result.upside_squeeze_breakeven_pct * 100
        fig.add_vline(x=be, line=dict(color=BREACH, dash="dot", width=1))
        _note(fig, be, f"Squeeze {be:+.2f}%", BREACH, top=False, align="right")
    if sweep_result.downside_deficiency_breakeven_pct is not None:
        be = sweep_result.downside_deficiency_breakeven_pct * 100
        fig.add_vline(x=be, line=dict(color=CAUTION, dash="dot", width=1))
        _note(fig, be, f"Deficiency {be:+.2f}%", CAUTION, top=False, align="right")
    apply_institutional_layout(fig, height=430, yaxis_title="Net liquidity buffer (USD M)",
                               xaxis_title="Parallel price shock, all commodities (%)")
    fig.update_layout(showlegend=False, hovermode="closest", margin=dict(t=12))
    fig.update_xaxes(ticksuffix="%", dtick=5)
    fig.update_yaxes(tickprefix="$")
    return fig


def build_stress_sweep_chart(
    sweep_result: StressSweepResult,
    active_shock_pct: float,
    is_custom_per_commodity: bool = False,
) -> go.Figure:
    """Two stacked panels: net buffer, then liquid resources vs. cash outflows, across parallel shocks."""
    df = sweep_result.to_dataframe()
    x = df["shock_pct"] * 100
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, row_heights=[0.56, 0.44],
        subplot_titles=("Net liquidity buffer (USD M)", "Liquid resources vs. cash outflows (USD M)"),
    )
    fig.add_trace(_buffer_trace(df), row=1, col=1)
    _add_regime_bands(fig, sweep_result, row=1)
    fig.add_hline(y=0.0, line=dict(color=BREACH, dash="dash", width=1), row=1, col=1)
    if not is_custom_per_commodity:
        fig.add_vline(x=active_shock_pct * 100, line=dict(color=COLORS["accent"], width=1.5), row=1, col=1)
        _note(fig, active_shock_pct * 100, f"Active {active_shock_pct:+.0%}", COLORS["accent"], top=True, row=1)
    if sweep_result.upside_squeeze_breakeven_pct is not None:
        be = sweep_result.upside_squeeze_breakeven_pct * 100
        fig.add_vline(x=be, line=dict(color=BREACH, dash="dot", width=1), row=1, col=1)
        _note(fig, be, f"Squeeze {be:+.2f}%", BREACH, top=False, align="right", row=1)
    if sweep_result.downside_deficiency_breakeven_pct is not None:
        be = sweep_result.downside_deficiency_breakeven_pct * 100
        fig.add_vline(x=be, line=dict(color=CAUTION, dash="dot", width=1), row=1, col=1)
        _note(fig, be, f"Deficiency {be:+.2f}%", CAUTION, top=False, align="right", row=1)
    fig.add_trace(
        go.Scatter(x=x, y=df["total_liquid_resources"] / money_divisor(), mode="lines", name="Liquid resources",
                   line=dict(color=HEALTHY, width=1.8),
                   hovertemplate="Resources $%{y:,.2f}M<extra></extra>"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=df["total_cash_outflows"] / money_divisor(), mode="lines", name="Cash outflows",
                   line=dict(color=BREACH, width=1.8), fill="tonexty", fillcolor="rgba(153, 27, 27, 0.06)",
                   hovertemplate="Outflows $%{y:,.2f}M<extra></extra>"),
        row=2, col=1,
    )
    apply_institutional_layout(fig, height=560)
    for subplot_title in fig.layout.annotations[:2]:
        subplot_title.font = dict(color=COLORS["text_secondary"], size=12, family=FONT_SANS)
        subplot_title.x = 0.0
        subplot_title.xanchor = "left"
    fig.update_layout(margin=dict(t=28, b=48), legend=dict(y=0.42, yanchor="bottom", x=1.0, xanchor="right"))
    fig.update_xaxes(ticksuffix="%", dtick=5)
    fig.update_xaxes(title=dict(text="Parallel price shock, all commodities (%)",
                                font=dict(size=11, color=COLORS["muted"])), row=2, col=1)
    fig.update_yaxes(tickprefix="$")
    fig.data[0].showlegend = False
    return fig


def build_active_shock_sources_uses_chart(stress_result: StressScenarioResult, terms: FacilityTerms) -> go.Figure:
    """Liquidity sources minus cash uses, bridging to the net liquidity buffer for the active shock."""
    steps = [
        ("Unrestricted cash", terms.unrestricted_cash),
        ("Shocked headroom", stress_result.shocked_headroom),
        ("Futures VM inflow", stress_result.vm_inflow),
        ("Futures VM call", -stress_result.vm_outflow),
        ("IM step-up call", -stress_result.im_addon_outflow),
        ("BB deficiency cure", -stress_result.bb_deficiency_cure),
    ]
    labels = [_wrap_label(s[0]) for s in steps] + ["Net liquidity<br>buffer"]
    amounts = [s[1] for s in steps] + [stress_result.net_liquidity_buffer]
    fig = go.Figure(
        go.Waterfall(
            x=labels,
            measure=["relative"] * len(steps) + ["total"],
            y=[_m(a) for a in amounts[:-1]] + [None],
            text=[_mtext(a) if (abs(a) >= 5e3 or i == len(amounts) - 1) else "" for i, a in enumerate(amounts)],
            textposition="outside", cliponaxis=False,
            textfont=dict(color=COLORS["text"], size=11),
            connector=dict(line=dict(color=COLORS["neutral_fill"], width=1)),
            increasing=dict(marker=dict(color=COLORS["accent_hover"])),
            decreasing=dict(marker=dict(color=BREACH)),
            totals=dict(marker=dict(
                color=COLORS["accent"] if stress_result.net_liquidity_buffer >= 0 else BREACH
            )),
            customdata=[_mtext(a) for a in amounts],
            hovertemplate="%{x}<br>%{customdata}<extra></extra>",
            name="Sources and uses",
        )
    )
    fig.add_hline(y=0.0, line=dict(color=COLORS["border"], width=1))
    apply_institutional_layout(fig, height=360, yaxis_title="USD millions")
    fig.update_layout(showlegend=False, hovermode="closest", margin=dict(t=24, b=56))
    fig.update_xaxes(tickangle=0, tickfont=dict(size=11, color=COLORS["text_secondary"]))
    fig.update_yaxes(tickprefix="$")
    return fig


def _adverse_scale(zmin: float, zmax: float) -> list[list]:
    """Neutral for comfortable buffers, sand to amber approaching zero, crimson below zero."""
    span = (zmax - zmin) or 1.0
    f0 = min(max((0.0 - zmin) / span, 0.0), 1.0)
    if f0 <= 0.0:
        return [[0.0, "#D97706"], [0.35, "#FDE68A"], [1.0, "#F8FAFC"]]
    if f0 >= 1.0:
        return [[0.0, "#7F1D1D"], [1.0, "#991B1B"]]
    upper = f0 + (1.0 - f0) * 0.4
    return [[0.0, "#7F1D1D"], [f0 * 0.5, "#991B1B"], [f0, "#D97706"], [upper, "#FDE68A"], [1.0, "#F8FAFC"]]


def build_shock_heatmap(grid: ShockGridResult, active_point: tuple[float, float] | None = None) -> go.Figure:
    """Metals vs. softs/lumber shock grid of net liquidity buffer with the $0 contour."""
    x = [v * 100 for v in grid.softs_axis]
    y = [v * 100 for v in grid.metals_axis]
    z = [[v / money_divisor() for v in row] for row in grid.net_buffer]
    custom = [[[regime_label(grid.regime[i][j]), grid.shocked_bb[i][j] / money_divisor()] for j in range(len(x))]
              for i in range(len(y))]
    zmin = min(min(r) for r in z)
    zmax = max(max(r) for r in z)
    fig = go.Figure(
        go.Heatmap(
            x=x, y=y, z=z, zmin=zmin, zmax=zmax, customdata=custom, xgap=1, ygap=1,
            colorscale=_adverse_scale(zmin, zmax),
            text=[[f"{v:+.1f}" for v in row] for row in z], texttemplate="%{text}",
            textfont=dict(size=10, family=FONT_SANS),
            colorbar=dict(title=dict(text="Net buffer<br>(USD M)", font=dict(size=11, color=COLORS["muted"])),
                          tickfont=dict(size=11, color=COLORS["muted"]), tickprefix="$", outlinewidth=0,
                          thickness=10, len=0.9),
            hovertemplate=("Gold & copper %{y:+.1f}%, lumber & cocoa %{x:+.1f}%<br>Net buffer $%{z:,.2f}M"
                           "<br>Shocked BB $%{customdata[1]:,.2f}M<br>%{customdata[0]}<extra></extra>"),
        )
    )
    fig.add_trace(
        go.Contour(
            x=x, y=y, z=z, showscale=False, hoverinfo="skip", name="$0 buffer",
            contours=dict(start=0.0, end=0.0, size=1.0, coloring="lines"),
            line=dict(color=COLORS["text"], width=1.5, dash="dash"),
        )
    )
    if active_point is not None:
        fig.add_trace(
            go.Scatter(
                x=[active_point[1] * 100], y=[active_point[0] * 100], mode="markers", name="Active scenario",
                marker=dict(size=15, color="rgba(0, 0, 0, 0)", line=dict(color=COLORS["accent"], width=2.5)),
                hoverinfo="skip",
            )
        )
    apply_institutional_layout(fig, height=600, yaxis_title="Gold & copper shock (%)",
                               xaxis_title="Lumber & cocoa shock (%)")
    fig.update_layout(hovermode="closest", showlegend=False, margin=dict(t=12, l=64))
    fig.update_xaxes(ticksuffix="%", dtick=5, showline=False, ticks="")
    fig.update_yaxes(ticksuffix="%", dtick=5, showgrid=False)
    return fig


def build_mc_distribution_chart(mc: MonteCarloResult) -> go.Figure:
    """Histogram of simulated net liquidity buffers; paths at or below the 5th percentile shaded as the tail."""
    vals = [v / money_divisor() for v in mc.net_buffer_samples]
    p05 = mc.p05_net_buffer / money_divisor()
    lo, hi = min(vals + [0.0]), max(vals)
    size = max((max(vals) - min(vals)) / 70.0, 0.01)
    bins = dict(start=min(vals) - size, end=hi + size, size=size)
    body = [v for v in vals if v > p05]
    tail = [v for v in vals if v <= p05]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=body, xbins=bins, name="Simulated paths", marker=dict(color=COLORS["accent"]),
                               opacity=0.85, hovertemplate="$%{x:.2f}M: %{y} paths<extra></extra>"))
    fig.add_trace(go.Histogram(x=tail, xbins=bins, name="Worst 5% of paths", marker=dict(color=BREACH),
                               opacity=0.75, hovertemplate="$%{x:.2f}M: %{y} paths<extra></extra>"))
    markers = [
        (mc.p01_net_buffer / money_divisor(), f"LaR 99%: {_mtext(mc.p01_net_buffer)}", BREACH, "dash", 1.0, "right"),
        (p05, f"LaR 95%: {_mtext(mc.p05_net_buffer)}", BREACH, "dash", 0.92, "left"),
        (mc.mean_net_buffer / money_divisor(), f"Mean: {_mtext(mc.mean_net_buffer)}", COLORS["muted"], "dot", 1.0, "right"),
        (0.0, "$0 buffer", COLORS["text"], "solid", 0.84, "left"),
    ]
    for xv, label, color, dash, ypos, side in markers:
        fig.add_vline(x=xv, line=dict(color=color, dash=dash, width=1.2))
        fig.add_annotation(x=xv, y=ypos, xref="x", yref="paper", text=label, showarrow=False,
                           xanchor="left" if side == "left" else "right", xshift=4 if side == "left" else -4,
                           yanchor="top", font=dict(size=11, color=color), bgcolor="rgba(255, 255, 255, 0.85)")
    apply_institutional_layout(fig, height=400, yaxis_title="Number of paths",
                               xaxis_title=f"Net liquidity buffer after {mc.horizon_days} trading days (USD M)")
    fig.update_layout(barmode="overlay", hovermode="closest", margin=dict(t=36))
    fig.update_xaxes(tickprefix="$", ticksuffix="M", range=[lo - size * 2, hi + size * 2])
    return fig


def build_correlation_heatmap(mc: MonteCarloResult, specs: Mapping[str, CommoditySpec]) -> go.Figure:
    names = [specs[t].name for t in REQUIRED_TICKERS]
    z = [[mc.correlation[a][b] for b in REQUIRED_TICKERS] for a in REQUIRED_TICKERS]
    fig = go.Figure(
        go.Heatmap(
            x=names, y=names, z=z, zmin=-1, zmax=1, xgap=2, ygap=2,
            colorscale=[[0.0, "#9A3412"], [0.5, "#FFFFFF"], [1.0, COLORS["accent"]]],
            text=[[f"{v:+.2f}" for v in row] for row in z], texttemplate="%{text}",
            textfont=dict(size=11, family=FONT_SANS), showscale=False,
            hovertemplate="%{y} / %{x}: %{z:+.3f}<extra></extra>",
        )
    )
    apply_institutional_layout(fig, height=320)
    fig.update_layout(hovermode="closest", margin=dict(l=64, r=8, t=8, b=32))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showline=False, ticks="")
    return fig


def build_replay_chart(replay: HistoricalReplayResult, terms: FacilityTerms) -> go.Figure:
    """Three stacked panels: BB vs. limit/drawn, daily futures VM, 5-day stressed liquidity buffer."""
    df = replay.frame
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.42, 0.29, 0.29],
        subplot_titles=(
            "Borrowing base vs. facility limit and drawn balance (USD M)",
            "Daily futures variation margin (USD M, positive = inflow)",
            "5-day stressed liquidity buffer (USD M)",
        ),
    )
    bb_m = df["borrowing_base"] / money_divisor()
    y_lo = min(float(bb_m.min()), terms.drawn_revolver / money_divisor()) * 0.97
    y_hi = max(float(bb_m.max()), terms.facility_limit / money_divisor()) * 1.02
    fig.add_trace(
        go.Scatter(x=df.index, y=bb_m, name="Borrowing base", mode="lines",
                   line=dict(color=PORTFOLIO_COLOR, width=1.8), hovertemplate="BB $%{y:,.2f}M<extra></extra>"),
        row=1, col=1,
    )
    fig.add_hrect(y0=y_lo, y1=terms.drawn_revolver / money_divisor(), fillcolor=STATUS["breach"][1], line_width=0,
                  layer="below", row=1, col=1)
    fig.add_hline(y=terms.facility_limit / money_divisor(), line=dict(color=COLORS["reference"], dash="dash", width=1.2),
                  row=1, col=1)
    fig.add_hline(y=terms.drawn_revolver / money_divisor(), line=dict(color=BREACH, dash="dot", width=1.2), row=1, col=1)
    fig.add_annotation(x=0, xref="x domain", y=terms.facility_limit / money_divisor(), yref="y", xanchor="left",
                       yanchor="bottom", showarrow=False, text="Facility limit",
                       font=dict(size=10, color=COLORS["muted"]))
    fig.add_annotation(x=0, xref="x domain", y=terms.drawn_revolver / money_divisor(), yref="y", xanchor="left",
                       yanchor="top", showarrow=False, text="Drawn balance (deficiency below)",
                       font=dict(size=10, color=BREACH))
    vm = df["daily_vm"] / money_divisor()
    fig.add_trace(
        go.Bar(x=df.index, y=vm, name="Daily VM",
               marker=dict(color=[HEALTHY if v >= 0 else BREACH for v in vm], opacity=0.85),
               hovertemplate="VM $%{y:+,.2f}M<extra></extra>"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["rolling_vm"] / money_divisor(), name="5-day net VM", mode="lines",
                   line=dict(color=COLORS["muted"], width=1.2), hovertemplate="5-day VM $%{y:+,.2f}M<extra></extra>"),
        row=2, col=1,
    )
    buf = df["stressed_buffer"] / money_divisor()
    fig.add_trace(
        go.Scatter(x=df.index, y=buf.clip(lower=0.0), mode="lines", line=dict(width=0), fill="tozeroy",
                   fillcolor="rgba(22, 101, 52, 0.10)", showlegend=False, hoverinfo="skip"),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=buf.clip(upper=0.0), mode="lines", line=dict(width=0), fill="tozeroy",
                   fillcolor="rgba(153, 27, 27, 0.18)", showlegend=False, hoverinfo="skip"),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=buf, name="Stressed buffer", mode="lines", line=dict(color=PORTFOLIO_COLOR, width=1.4),
                   hovertemplate="Buffer $%{y:+,.2f}M<extra></extra>"),
        row=3, col=1,
    )
    fig.add_hline(y=0.0, line=dict(color=BREACH, width=1, dash="dash"), row=3, col=1)
    apply_institutional_layout(fig, height=760)
    for subplot_title in fig.layout.annotations[:3]:
        subplot_title.font = dict(color=COLORS["text_secondary"], size=12, family=FONT_SANS)
        subplot_title.x = 0.0
        subplot_title.xanchor = "left"
    fig.update_layout(hovermode="x unified", showlegend=False, margin=dict(t=28, b=32))
    fig.update_yaxes(range=[y_lo, y_hi], row=1, col=1)
    fig.update_yaxes(tickprefix="$")
    return fig


def build_hedge_frontier_chart(res: HedgeFrontierResult) -> go.Figure:
    """Two panels over hedge ratio: buffer in a +20% rally / -20% crash and worst case; unhedged crash loss."""
    x = [r * 100 for r in res.ratios]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, row_heights=[0.6, 0.4],
        subplot_titles=("Net liquidity buffer (USD M)", "Unhedged inventory value lost in a 20% price fall (USD M)"),
    )
    series = [
        ("Buffer in a +20% rally", res.buffer_up_20, BREACH, "solid"),
        ("Buffer in a -20% crash", res.buffer_down_20, CAUTION, "dash"),
        ("Worst case across -20% to +20%", res.min_buffer, PORTFOLIO_COLOR, "solid"),
    ]
    for name, ys, color, dash in series:
        fig.add_trace(go.Scatter(x=x, y=[v / money_divisor() for v in ys], mode="lines+markers", name=name,
                                 line=dict(color=color, width=2 if color == PORTFOLIO_COLOR else 1.6, dash=dash),
                                 marker=dict(size=4),
                                 hovertemplate=f"Hedge %{{x:.0f}}%<br>{name}: $%{{y:,.2f}}M<extra></extra>"),
                      row=1, col=1)
    fig.add_hline(y=0.0, line=dict(color=BREACH, dash="dot", width=1), row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=[v / money_divisor() for v in res.crash_value_loss], name="Unhedged loss",
                         marker=dict(color="#CBD5E1"), showlegend=False,
                         hovertemplate="Hedge %{x:.0f}%<br>Loss $%{y:,.2f}M<extra></extra>"),
                  row=2, col=1)
    for xv, label, color, top in [(res.current_ratio * 100, f"Current {res.current_ratio:.0%}", COLORS["muted"], True),
                                  (res.recommended_ratio * 100, f"Liquidity-optimal {res.recommended_ratio:.0%}",
                                   COLORS["accent"], False)]:
        fig.add_vline(x=xv, line=dict(color=color, width=1.5, dash="dash" if color == COLORS["muted"] else "solid"))
        _note(fig, xv, label, color, top=top, row=1)
    apply_institutional_layout(fig, height=520)
    for subplot_title in fig.layout.annotations[:2]:
        subplot_title.font = dict(color=COLORS["text_secondary"], size=12, family=FONT_SANS)
        subplot_title.x = 0.0
        subplot_title.xanchor = "left"
    fig.update_layout(hovermode="x unified", margin=dict(t=28, b=48),
                      legend=dict(y=1.06, yanchor="bottom", x=1.0, xanchor="right"))
    fig.update_xaxes(ticksuffix="%", range=[-2.5, 102.5], dtick=10)
    fig.update_xaxes(title=dict(text="Hedge ratio (futures as % of physical quantity)",
                                font=dict(size=11, color=COLORS["muted"])), row=2, col=1)
    fig.update_yaxes(tickprefix="$")
    return fig


def build_runway_fan_chart(rw: RunwayResult) -> go.Figure:
    """Percentile fan of the simulated net liquidity buffer by trading day."""
    days = rw.days
    p = {k: [v / money_divisor() for v in vals] for k, vals in rw.percentiles.items()}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=p["p95"], mode="lines", line=dict(width=0), hoverinfo="skip",
                             showlegend=False))
    fig.add_trace(go.Scatter(x=days, y=p["p5"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(27, 73, 101, 0.10)", name="5th to 95th percentile",
                             hovertemplate="Day %{x}: 5th pct $%{y:,.2f}M<extra></extra>"))
    fig.add_trace(go.Scatter(x=days, y=p["p75"], mode="lines", line=dict(width=0), hoverinfo="skip",
                             showlegend=False))
    fig.add_trace(go.Scatter(x=days, y=p["p25"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(27, 73, 101, 0.22)", name="25th to 75th percentile",
                             hovertemplate="Day %{x}: 25th pct $%{y:,.2f}M<extra></extra>"))
    fig.add_trace(go.Scatter(x=days, y=p["p50"], mode="lines+markers", line=dict(color=COLORS["accent"], width=2),
                             marker=dict(size=4), name="Median",
                             hovertemplate="Day %{x}: median $%{y:,.2f}M<extra></extra>"))
    fig.add_hline(y=0.0, line=dict(color=BREACH, dash="dash", width=1))
    apply_institutional_layout(fig, height=360, yaxis_title="Net liquidity buffer (USD M)",
                               xaxis_title="Trading days from today")
    fig.update_layout(hovermode="x unified", margin=dict(t=36))
    fig.update_xaxes(dtick=2)
    fig.update_yaxes(tickprefix="$")
    return fig


def build_runway_breach_chart(rw: RunwayResult) -> go.Figure:
    """Cumulative probability that the buffer has gone below zero by each trading day."""
    fig = go.Figure(go.Scatter(
        x=rw.days, y=[v * 100 for v in rw.prob_breached_by_day], mode="lines+markers", line_shape="hv",
        line=dict(color=BREACH, width=2), marker=dict(size=4), name="Probability",
        hovertemplate="By day %{x}: %{y:.2f}% of paths<extra></extra>",
    ))
    top = max([v * 100 for v in rw.prob_breached_by_day] + [1.0])
    apply_institutional_layout(fig, height=360, yaxis_title="Paths with a shortfall (%)",
                               xaxis_title="Trading days from today")
    fig.update_layout(showlegend=False, hovermode="x unified", margin=dict(t=12))
    fig.update_xaxes(dtick=2)
    fig.update_yaxes(ticksuffix="%", range=[0, top * 1.15])
    return fig


def build_crisis_replay_chart(res: CrisisReplayResult, specs: Mapping[str, CommoditySpec],
                              terms: FacilityTerms) -> go.Figure:
    """Observed price moves (top) and the modelled borrowing base and net liquidity buffer (below)."""
    df = res.frame
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.36, 0.32, 0.32],
        subplot_titles=(
            "Observed: futures price move since episode start (%)",
            "Modelled: borrowing base vs. facility limit and drawn balance (USD M)",
            "Modelled: net liquidity buffer (USD M)",
        ),
    )
    for t in res.covered_tickers:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[f"move_{t}"] * 100, mode="lines", name=specs[t].name,
            line=dict(color=COMMODITY_COLORS.get(t, PORTFOLIO_COLOR), width=1.6, dash=COMMODITY_DASH.get(t, "solid")),
            hovertemplate=f"{specs[t].name} %{{y:+.1f}}%<extra></extra>"), row=1, col=1)
    fig.add_hline(y=0.0, line=dict(color=COLORS["border"], width=1), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["shocked_bb"] / money_divisor(), mode="lines", name="Borrowing base",
                             line=dict(color=PORTFOLIO_COLOR, width=1.8), showlegend=False,
                             hovertemplate="BB $%{y:,.2f}M<extra></extra>"), row=2, col=1)
    fig.add_hline(y=terms.facility_limit / money_divisor(), line=dict(color=COLORS["reference"], dash="dash", width=1.2),
                  row=2, col=1)
    fig.add_hline(y=terms.drawn_revolver / money_divisor(), line=dict(color=BREACH, dash="dot", width=1.2), row=2, col=1)
    nlb = df["net_liquidity_buffer"] / money_divisor()
    fig.add_trace(go.Scatter(x=df.index, y=nlb.clip(lower=0.0), mode="lines", line=dict(width=0), fill="tozeroy",
                             fillcolor="rgba(22, 101, 52, 0.10)", showlegend=False, hoverinfo="skip"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=nlb.clip(upper=0.0), mode="lines", line=dict(width=0), fill="tozeroy",
                             fillcolor="rgba(153, 27, 27, 0.18)", showlegend=False, hoverinfo="skip"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=nlb, mode="lines", name="Net buffer", showlegend=False,
                             line=dict(color=PORTFOLIO_COLOR, width=1.6),
                             customdata=[regime_label(r) for r in df["regime"]],
                             hovertemplate="Buffer $%{y:+,.2f}M (%{customdata})<extra></extra>"), row=3, col=1)
    fig.add_hline(y=0.0, line=dict(color=BREACH, width=1, dash="dash"), row=3, col=1)
    apply_institutional_layout(fig, height=760)
    for subplot_title in fig.layout.annotations[:3]:
        subplot_title.font = dict(color=COLORS["text_secondary"], size=12, family=FONT_SANS)
        subplot_title.x = 0.0
        subplot_title.xanchor = "left"
    fig.update_layout(hovermode="x unified", margin=dict(t=28, b=32),
                      legend=dict(y=1.04, yanchor="bottom", x=1.0, xanchor="right"))
    fig.update_yaxes(ticksuffix="%", row=1, col=1)
    fig.update_yaxes(tickprefix="$", row=2, col=1)
    fig.update_yaxes(tickprefix="$", row=3, col=1)
    return fig
