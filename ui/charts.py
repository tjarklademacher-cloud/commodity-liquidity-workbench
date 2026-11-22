"""Plotly chart builders for the workbench (single-axis panels, Plotly 6/7 compliant)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.schemas import (
    REQUIRED_TICKERS,
    BorrowingBaseResult,
    CommoditySpec,
    FacilityTerms,
    HistoricalReplayResult,
    InventoryValuation,
    LiquidityRegime,
    MonteCarloResult,
    ShockGridResult,
    StressScenarioResult,
    StressSweepResult,
)
from ui.theme import COLORS, COMMODITY_COLORS, FONT_STACK, REGIME_STYLES, apply_institutional_layout


def _m(val: float) -> float:
    return val / 1e6


def _mtext(val: float) -> str:
    sign = "-" if val < 0 else ""
    return f"{sign}${abs(val) / 1e6:,.2f}M"


def build_borrowing_base_waterfall(bb_result: BorrowingBaseResult) -> go.Figure:
    """Chart 1: gross collateral to final borrowing base bridge."""
    wf = bb_result.waterfall
    measures = [w.measure for w in wf]
    y_vals = [None if w.measure == "total" else _m(w.amount) for w in wf]
    fig = go.Figure(
        go.Waterfall(
            x=[_wrap_label(w.label) for w in wf],
            measure=measures,
            y=y_vals,
            text=[_mtext(w.amount) for w in wf],
            textposition="outside",
            textfont=dict(color=COLORS["text"]),
            connector=dict(line=dict(color=COLORS["slate"], width=1, dash="dot")),
            increasing=dict(marker=dict(color=COLORS["emerald"])),
            decreasing=dict(marker=dict(color=COLORS["red"])),
            totals=dict(marker=dict(color=COLORS["deep_blue"])),
            hovertemplate="%{x}<br>%{text}<extra></extra>",
            name="Borrowing Base Bridge",
        )
    )
    fig.add_hline(
        y=_m(bb_result.facility_limit), line=dict(color=COLORS["blue"], dash="dash", width=1.5),
        annotation_text=f"Facility Limit {_mtext(bb_result.facility_limit)}",
        annotation_position="top left", annotation_font=dict(color=COLORS["blue"]),
    )
    fig.add_hline(
        y=_m(bb_result.drawn_revolver), line=dict(color=COLORS["amber"], dash="dash", width=1.5),
        annotation_text=f"Drawn Revolver {_mtext(bb_result.drawn_revolver)}",
        annotation_position="bottom left", annotation_font=dict(color=COLORS["amber"]),
    )
    peak = max(w.running_total for w in wf) if wf else 0.0
    apply_institutional_layout(fig, "Borrowing Base Waterfall Bridge ($M)", height=470, yaxis_title="$M")
    fig.update_layout(showlegend=False, hovermode="closest")
    fig.update_xaxes(tickangle=0)
    fig.update_yaxes(range=[0.0, _m(max(peak, bb_result.facility_limit)) * 1.15])
    return fig


def build_haircut_decomposition_chart(inventory_items: Sequence[InventoryValuation]) -> go.Figure:
    """Chart 2: 100% stacked advance rate vs. effective haircut components per commodity."""
    names = [f"{i.name} ({i.ticker})" for i in inventory_items]
    custom = [
        [i.base_haircut * 100, i.vol_addon * 100, i.unhedged_penalty * 100, i.raw_haircut * 100,
         i.total_haircut * 100, i.realized_vol_30d * 100, i.hedge_ratio * 100]
        for i in inventory_items
    ]
    hover_tail = (
        "<br>Raw base %{customdata[0]:.2f}% | vol add-on %{customdata[1]:.2f}% | "
        "unhedged %{customdata[2]:.2f}%<br>Raw haircut %{customdata[3]:.2f}% -> clamped %{customdata[4]:.2f}%"
        "<br>30d vol %{customdata[5]:.2f}% | hedge ratio %{customdata[6]:.1f}%<extra></extra>"
    )
    components = [
        ("Advance Rate", [i.advance_rate for i in inventory_items], COLORS["emerald"]),
        ("Base Haircut", [i.eff_base_haircut for i in inventory_items], COLORS["slate"]),
        ("Vol Add-On", [i.eff_vol_addon for i in inventory_items], COLORS["amber"]),
        ("Unhedged Penalty", [i.eff_unhedged_penalty for i in inventory_items], COLORS["orange"]),
    ]
    floor = [i.eff_floor_adjustment for i in inventory_items]
    if any(f > 0.0 for f in floor):
        components.append(("Min Haircut Floor Adj", floor, COLORS["deep_blue"]))
    fig = go.Figure()
    for label, values, color in components:
        fig.add_trace(
            go.Bar(
                y=names, x=[v * 100 for v in values], name=label, orientation="h",
                marker=dict(color=color), customdata=custom,
                text=[f"{v * 100:.1f}%" if v >= 0.03 else "" for v in values],
                textposition="inside", insidetextanchor="middle", textfont=dict(color=COLORS["text"]),
                hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.2f}}% of collateral value" + hover_tail,
            )
        )
    apply_institutional_layout(fig, "Advance Rate vs. Haircut Decomposition (% of Market Value)", height=420)
    fig.update_layout(barmode="stack", hovermode="closest")
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_yaxes(autorange="reversed")
    return fig


def build_volatility_history_chart(vol_df: pd.DataFrame, specs: Mapping[str, CommoditySpec]) -> go.Figure:
    """Chart 3: rolling 30-day annualized realized volatility history."""
    fig = go.Figure()
    for ticker in vol_df.columns:
        series = vol_df[ticker].dropna()
        label = specs[ticker].name if ticker in specs else ticker
        fig.add_trace(
            go.Scatter(
                x=series.index, y=series.to_numpy() * 100, mode="lines", name=f"{label} ({ticker})",
                line=dict(color=COMMODITY_COLORS.get(ticker, COLORS["blue"]), width=2),
                hovertemplate=f"{label}: %{{y:.2f}}%<extra></extra>",
            )
        )
    for level in (15, 20, 30):
        fig.add_hline(
            y=level, line=dict(color=COLORS["slate"], dash="dot", width=1),
            annotation_text=f"{level}%", annotation_position="right",
            annotation_font=dict(color=COLORS["muted"], size=10),
        )
    apply_institutional_layout(fig, "30-Day Realized Volatility (Annualized)", height=420, yaxis_title="Vol (%)")
    fig.update_yaxes(ticksuffix="%", rangemode="tozero")
    return fig


def _wrap_label(name: str) -> str:
    words = name.split()
    if len(name) <= 14 or len(words) < 2:
        return name
    best = min(range(1, len(words)), key=lambda k: abs(len(" ".join(words[:k])) - len(" ".join(words[k:]))))
    return " ".join(words[:best]) + "<br>" + " ".join(words[best:])


def build_receivables_concentration_chart(bb_result: BorrowingBaseResult) -> go.Figure:
    """Chart 4: pre-cap vs. capped eligible AR per buyer against the dynamic cap line."""
    items = bb_result.receivable_items
    buyers = [_wrap_label(r.buyer_name) for r in items]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=buyers, y=[_m(r.pre_cap_eligible) for r in items], name="Pre-Cap Eligible AR ($M)",
            marker=dict(color=COLORS["slate"]),
            hovertemplate="%{x}<br>Pre-cap: $%{y:,.2f}M<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=buyers, y=[_m(r.final_eligible) for r in items], name="Final Capped Eligible AR ($M)",
            marker=dict(color=[COLORS["red"] if r.is_capped else COLORS["blue"] for r in items]),
            text=[_mtext(r.final_eligible) if r.final_eligible > 0 else "" for r in items],
            textposition="outside", textangle=0, cliponaxis=False,
            textfont=dict(color=COLORS["text"], size=10),
            hovertemplate="%{x}<br>Final: $%{y:,.2f}M<extra></extra>",
        )
    )
    for r in items:
        if not r.is_aging_eligible:
            fig.add_annotation(
                x=_wrap_label(r.buyer_name), y=0.0, text=">90d<br>Ineligible", showarrow=False, yshift=24,
                font=dict(color=COLORS["orange"], size=11),
            )
    cap_m = _m(bb_result.cap_dollar_limit)
    fig.add_hline(
        y=cap_m, line=dict(color=COLORS["amber"], dash="dash", width=1.5),
        annotation_text=f"Dynamic Cap ({bb_result.concentration_cap_pct:.1%} × BB = ${cap_m:.2f}M)",
        annotation_position="top right", annotation_yshift=4, annotation_font=dict(color=COLORS["amber"]),
    )
    peak = max([_m(r.pre_cap_eligible) for r in items] + [cap_m, 1.0])
    apply_institutional_layout(fig, "Receivables Concentration: Pre-Cap vs. Capped Eligible AR", height=440,
                               yaxis_title="$M")
    fig.update_layout(barmode="group", hovermode="closest", margin=dict(b=110),
                      uniformtext=dict(minsize=9, mode="show"))
    fig.update_xaxes(tickangle=0, tickfont=dict(size=10))
    fig.update_yaxes(range=[0.0, peak * 1.25])
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


def build_stress_sweep_chart(
    sweep_result: StressSweepResult,
    active_shock_pct: float,
    is_custom_per_commodity: bool = False,
) -> go.Figure:
    """Chart 5: two stacked panels (net buffer; resources vs. outflows) across parallel shocks."""
    df = sweep_result.to_dataframe()
    x = df["shock_pct"] * 100
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09, row_heights=[0.58, 0.42],
        subplot_titles=(
            "Net Liquidity Buffer ($M) Across Parallel Price Shocks",
            "Available Liquid Resources vs. Total Cash Outflows ($M)",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=df["net_liquidity_buffer"] / 1e6, mode="lines+markers", name="Net Liquidity Buffer ($M)",
            line=dict(color=COLORS["blue"], width=2.5), marker=dict(size=5),
            customdata=df["regime"], hovertemplate="Net buffer: $%{y:,.2f}M (%{customdata})<extra></extra>",
        ),
        row=1, col=1,
    )
    for x0, x1, regime in _regime_segments(sweep_result):
        _, _, bg_rgba, _ = REGIME_STYLES[regime]
        fig.add_vrect(x0=x0, x1=x1, fillcolor=bg_rgba, line_width=0, layer="below", row=1, col=1)
    fig.add_hline(y=0.0, line=dict(color=COLORS["red"], dash="dash", width=1.5), row=1, col=1,
                  annotation_text="$0M liquidity exhaustion", annotation_position="bottom left",
                  annotation_font=dict(color=COLORS["red"], size=10))
    if not is_custom_per_commodity:
        fig.add_vline(x=active_shock_pct * 100, line=dict(color=COLORS["text"], width=1.5), row=1, col=1,
                      annotation_text=f"Active {active_shock_pct:+.0%}", annotation_position="top left",
                      annotation_font=dict(color=COLORS["text"], size=11))
    if sweep_result.upside_squeeze_breakeven_pct is not None:
        be = sweep_result.upside_squeeze_breakeven_pct * 100
        fig.add_vline(x=be, line=dict(color=COLORS["red"], dash="dot", width=1.2), row=1, col=1,
                      annotation_text=f"Squeeze BE {be:+.2f}%", annotation_position="bottom right",
                      annotation_font=dict(color=COLORS["red"], size=10))
    if sweep_result.downside_deficiency_breakeven_pct is not None:
        be = sweep_result.downside_deficiency_breakeven_pct * 100
        fig.add_vline(x=be, line=dict(color=COLORS["orange"], dash="dot", width=1.2), row=1, col=1,
                      annotation_text=f"Deficiency BE {be:+.2f}%", annotation_position="bottom left",
                      annotation_font=dict(color=COLORS["orange"], size=10))
    fig.add_trace(
        go.Scatter(
            x=x, y=df["total_liquid_resources"] / 1e6, mode="lines", name="Total Liquid Resources ($M)",
            line=dict(color=COLORS["emerald"], width=2),
            hovertemplate="Resources: $%{y:,.2f}M<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=df["total_cash_outflows"] / 1e6, mode="lines", name="Total Cash Outflows ($M)",
            line=dict(color=COLORS["red"], width=2), fill="tonexty", fillcolor="rgba(239, 68, 68, 0.10)",
            hovertemplate="Outflows: $%{y:,.2f}M<extra></extra>",
        ),
        row=2, col=1,
    )
    apply_institutional_layout(fig, "41-Point Margin-Trap Liquidity Stress Sweep", height=680)
    for subplot_title in fig.layout.annotations[:2]:
        subplot_title.font = dict(color=COLORS["text"], size=13)
    fig.update_xaxes(ticksuffix="%", row=2, col=1, title=dict(text="Parallel Price Shock (%)"))
    fig.update_xaxes(ticksuffix="%", row=1, col=1)
    fig.update_yaxes(title=dict(text="$M"), row=1, col=1)
    fig.update_yaxes(title=dict(text="$M"), row=2, col=1)
    return fig


def build_active_shock_sources_uses_chart(stress_result: StressScenarioResult, terms: FacilityTerms) -> go.Figure:
    """Chart 6: liquidity sources minus cash uses bridging to the net liquidity buffer."""
    steps = [
        ("Unrestricted Cash", terms.unrestricted_cash),
        ("Shocked Headroom", stress_result.shocked_headroom),
        ("Futures VM Inflow", stress_result.vm_inflow),
        ("Futures VM Call", -stress_result.vm_outflow),
        ("IM Step-Up Call", -stress_result.im_addon_outflow),
        ("BB Deficiency Cure", -stress_result.bb_deficiency_cure),
    ]
    labels = [_wrap_label(s[0]) for s in steps] + ["Net Liquidity<br>Buffer"]
    amounts = [s[1] for s in steps] + [stress_result.net_liquidity_buffer]
    fig = go.Figure(
        go.Waterfall(
            x=labels,
            measure=["relative"] * len(steps) + ["total"],
            y=[_m(a) for a in amounts[:-1]] + [None],
            text=[_mtext(a) if (abs(a) >= 5e3 or i == len(amounts) - 1) else "" for i, a in enumerate(amounts)],
            textposition="outside",
            textfont=dict(color=COLORS["text"]),
            connector=dict(line=dict(color=COLORS["slate"], width=1, dash="dot")),
            increasing=dict(marker=dict(color=COLORS["emerald"])),
            decreasing=dict(marker=dict(color=COLORS["red"])),
            totals=dict(marker=dict(
                color=COLORS["blue"] if stress_result.net_liquidity_buffer >= 0 else COLORS["red"]
            )),
            customdata=[_mtext(a) for a in amounts],
            hovertemplate="%{x}<br>%{customdata}<extra></extra>",
            name="Sources vs. Uses",
        )
    )
    fig.add_hline(y=0.0, line=dict(color=COLORS["slate"], width=1))
    apply_institutional_layout(
        fig, f"Active Shock Cash Sources vs. Uses ({str(stress_result.regime).replace('_', ' ').title()})",
        height=440, yaxis_title="$M",
    )
    fig.update_layout(showlegend=False, hovermode="closest")
    fig.update_xaxes(tickangle=0)
    return fig


REGIME_ORDER: tuple[LiquidityRegime, ...] = (
    LiquidityRegime.HEALTHY,
    LiquidityRegime.MARGIN_TRAP_WARNING,
    LiquidityRegime.BB_DEFICIENCY_WARNING,
    LiquidityRegime.CRITICAL_SQUEEZE,
)


def build_facility_gauge(bb_result: BorrowingBaseResult) -> go.Figure:
    """Availability usage gauge: drawn revolver as a share of min(facility limit, borrowing base)."""
    availability = min(bb_result.facility_limit, bb_result.final_borrowing_base)
    usage = bb_result.drawn_revolver / availability * 100 if availability > 0 else 100.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=usage,
            number=dict(suffix="%", valueformat=".1f", font=dict(size=40, color=COLORS["text"], family=FONT_STACK)),
            gauge=dict(
                axis=dict(range=[0, 110], tickcolor=COLORS["slate"], tickfont=dict(color=COLORS["muted"])),
                bar=dict(color=COLORS["blue"], thickness=0.28),
                bgcolor="rgba(15, 23, 42, 0.6)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 85], color="rgba(16, 185, 129, 0.22)"),
                    dict(range=[85, 95], color="rgba(245, 158, 11, 0.26)"),
                    dict(range=[95, 110], color="rgba(239, 68, 68, 0.28)"),
                ],
                threshold=dict(line=dict(color=COLORS["red"], width=3), thickness=0.8, value=100),
            ),
        )
    )
    apply_institutional_layout(fig, "Availability Usage", height=320)
    fig.update_layout(margin=dict(l=30, r=30, t=60, b=10))
    fig.add_annotation(
        x=0.5, y=-0.02, xref="paper", yref="paper", showarrow=False,
        text=f"{_mtext(bb_result.drawn_revolver)} drawn of {_mtext(availability)} available",
        font=dict(color=COLORS["muted"], size=12),
    )
    return fig


def build_collateral_mix_donut(bb_result: BorrowingBaseResult) -> go.Figure:
    """Eligible collateral composition: each inventory line plus capped eligible receivables."""
    labels = [f"{i.name}" for i in bb_result.inventory_items] + ["Receivables (capped)"]
    values = [i.eligible_collateral / 1e6 for i in bb_result.inventory_items] + [
        bb_result.final_eligible_receivables / 1e6
    ]
    colors = [COMMODITY_COLORS.get(i.ticker, COLORS["blue"]) for i in bb_result.inventory_items] + [COLORS["blue"]]
    fig = go.Figure(
        go.Pie(
            labels=labels, values=values, hole=0.62, sort=False,
            marker=dict(colors=colors, line=dict(color=COLORS["bg"], width=2)),
            textinfo="percent", textfont=dict(color=COLORS["bg"], size=11),
            hovertemplate="%{label}<br>$%{value:,.2f}M (%{percent})<extra></extra>",
        )
    )
    apply_institutional_layout(fig, "Eligible Collateral Mix", height=320)
    fig.update_layout(
        margin=dict(l=10, r=10, t=56, b=10), showlegend=True, hovermode="closest",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.02, yanchor="top", font=dict(size=11)),
        annotations=[dict(text=f"<b>{_mtext(bb_result.final_borrowing_base)}</b><br>"
                                f"<span style='font-size:11px;color:{COLORS['muted']}'>Borrowing Base</span>",
                          x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                          font=dict(size=17, color=COLORS["text"]))],
    )
    return fig


def build_regime_ribbon(sweep_result: StressSweepResult, active_shock_pct: float | None) -> go.Figure:
    """One-row regime map across the 41-point parallel sweep."""
    codes = {r: i for i, r in enumerate(REGIME_ORDER)}
    x = [p.shock_pct * 100 for p in sweep_result.points]
    z = [[codes[p.regime] for p in sweep_result.points]]
    text = [[f"{p.shock_pct:+.0%}: {p.regime} · buffer {_mtext(p.net_liquidity_buffer)}" for p in sweep_result.points]]
    scale = []
    for i, r in enumerate(REGIME_ORDER):
        color = REGIME_STYLES[r][1]
        scale += [[i / 4, color], [(i + 1) / 4, color]]
    fig = go.Figure(
        go.Heatmap(
            x=x, y=[""], z=z, zmin=-0.5, zmax=3.5, colorscale=scale, showscale=False, xgap=2,
            text=text, hovertemplate="%{text}<extra></extra>", opacity=0.85,
        )
    )
    if active_shock_pct is not None:
        fig.add_vline(x=active_shock_pct * 100, line=dict(color=COLORS["text"], width=2.5))
    apply_institutional_layout(fig, "Regime Map Across Parallel Price Shocks", height=170)
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=30), hovermode="closest")
    fig.update_xaxes(ticksuffix="%", dtick=5, showgrid=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)
    for r in REGIME_ORDER:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=str(r).replace("_", " ").title().replace("Bb ", "BB "),
                                 marker=dict(size=10, color=REGIME_STYLES[r][1], symbol="square")))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.25, yanchor="bottom", x=1.0,
                                                   xanchor="right", font=dict(size=11)))
    return fig


def build_shock_heatmap(grid: ShockGridResult, active_point: tuple[float, float] | None = None) -> go.Figure:
    """Metals vs. softs/lumber shock grid of net liquidity buffer with the $0 exhaustion contour."""
    x = [v * 100 for v in grid.softs_axis]
    y = [v * 100 for v in grid.metals_axis]
    z = [[v / 1e6 for v in row] for row in grid.net_buffer]
    custom = [[[grid.regime[i][j], grid.shocked_bb[i][j] / 1e6] for j in range(len(x))] for i in range(len(y))]
    bound = max(abs(min(min(r) for r in z)), abs(max(max(r) for r in z)), 1.0)
    fig = go.Figure(
        go.Heatmap(
            x=x, y=y, z=z, zmid=0.0, zmin=-bound, zmax=bound, customdata=custom, xgap=1, ygap=1,
            colorscale=[[0.0, "#991B1B"], [0.35, "#EF4444"], [0.5, "#1E293B"], [0.65, "#0EA5E9"], [1.0, "#10B981"]],
            text=[[f"{v:+.1f}" for v in row] for row in z], texttemplate="%{text}",
            textfont=dict(size=9, color="#E2E8F0"),
            colorbar=dict(title=dict(text="Net buffer $M", font=dict(color=COLORS["muted"])),
                          tickfont=dict(color=COLORS["muted"]), outlinewidth=0, thickness=12),
            hovertemplate=("Metals %{y:+.1f}% · Softs/Lumber %{x:+.1f}%<br>Net buffer $%{z:,.2f}M"
                           "<br>Shocked BB $%{customdata[1]:,.2f}M<br>%{customdata[0]}<extra></extra>"),
        )
    )
    fig.add_trace(
        go.Contour(
            x=x, y=y, z=z, showscale=False, hoverinfo="skip", name="$0 buffer",
            contours=dict(start=0.0, end=0.0, size=1.0, coloring="lines"),
            line=dict(color=COLORS["text"], width=2, dash="dash"),
        )
    )
    if active_point is not None:
        fig.add_trace(
            go.Scatter(
                x=[active_point[1] * 100], y=[active_point[0] * 100], mode="markers", name="Active scenario",
                marker=dict(size=16, color="rgba(0, 0, 0, 0)", line=dict(color=COLORS["text"], width=2.5),
                            symbol="circle"),
                hoverinfo="skip",
            )
        )
    apply_institutional_layout(fig, "Net Liquidity Buffer: Metals vs. Softs & Lumber Shocks ($M)", height=600,
                               yaxis_title="Gold & Copper shock (%)", xaxis_title="Lumber & Cocoa shock (%)")
    fig.update_layout(hovermode="closest", showlegend=False, plot_bgcolor="rgba(0, 0, 0, 0)")
    fig.update_xaxes(ticksuffix="%", showgrid=False, dtick=5)
    fig.update_yaxes(ticksuffix="%", showgrid=False, dtick=5)
    return fig


def build_mc_distribution_chart(mc: MonteCarloResult) -> go.Figure:
    """Histogram of simulated net liquidity buffers with LaR / ES / zero markers."""
    vals = [v / 1e6 for v in mc.net_buffer_samples]
    lo, hi = min(vals), max(vals)
    size = max((hi - lo) / 60.0, 0.01)
    bins = dict(start=lo - size, end=hi + size, size=size)
    neg = [v for v in vals if v < 0.0]
    pos = [v for v in vals if v >= 0.0]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pos, xbins=bins, name="Buffer ≥ $0", marker=dict(color=COLORS["blue"]),
                               opacity=0.85, hovertemplate="$%{x:.2f}M: %{y} paths<extra></extra>"))
    if neg:
        fig.add_trace(go.Histogram(x=neg, xbins=bins, name="Buffer < $0 (squeeze)", marker=dict(color=COLORS["red"]),
                                   opacity=0.9, hovertemplate="$%{x:.2f}M: %{y} paths<extra></extra>"))
    markers = [
        (0.0, "$0", COLORS["red"], "solid"),
        (mc.p05_net_buffer / 1e6, f"LaR 95% {_mtext(mc.p05_net_buffer)}", COLORS["amber"], "dash"),
        (mc.p01_net_buffer / 1e6, f"LaR 99% {_mtext(mc.p01_net_buffer)}", COLORS["orange"], "dot"),
        (mc.mean_net_buffer / 1e6, f"Mean {_mtext(mc.mean_net_buffer)}", COLORS["emerald"], "dash"),
    ]
    for i, (xv, label, color, dash) in enumerate(markers):
        fig.add_vline(x=xv, line=dict(color=color, dash=dash, width=1.6), annotation_text=label,
                      annotation_position="top right" if i % 2 == 0 else "top left",
                      annotation_font=dict(color=color, size=11), annotation_yshift=-18 * (i // 2))
    apply_institutional_layout(
        fig, f"Simulated {mc.horizon_days}-Day Net Liquidity Buffer ({mc.n_sims:,} correlated paths)",
        height=460, yaxis_title="Paths", xaxis_title="Net liquidity buffer ($M)",
    )
    fig.update_layout(barmode="overlay", hovermode="closest")
    fig.update_xaxes(tickprefix="$", ticksuffix="M")
    return fig


def build_correlation_heatmap(mc: MonteCarloResult, specs: Mapping[str, CommoditySpec]) -> go.Figure:
    names = [specs[t].name for t in REQUIRED_TICKERS]
    z = [[mc.correlation[a][b] for b in REQUIRED_TICKERS] for a in REQUIRED_TICKERS]
    fig = go.Figure(
        go.Heatmap(
            x=names, y=names, z=z, zmin=-1, zmax=1, xgap=2, ygap=2,
            colorscale=[[0.0, "#EF4444"], [0.5, "#1E293B"], [1.0, "#38BDF8"]],
            text=[[f"{v:+.2f}" for v in row] for row in z], texttemplate="%{text}",
            textfont=dict(color=COLORS["text"], size=13), showscale=False,
            hovertemplate="%{y} / %{x}: %{z:+.3f}<extra></extra>",
        )
    )
    apply_institutional_layout(fig, "Daily Return Correlation (Lookback)", height=360)
    fig.update_layout(hovermode="closest", margin=dict(l=70, r=20, t=60, b=40), plot_bgcolor="rgba(0, 0, 0, 0)")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, showline=False)
    return fig


def build_replay_chart(replay: HistoricalReplayResult, terms: FacilityTerms) -> go.Figure:
    """Three stacked panels: BB vs. limit/drawn, daily futures VM, 5-day stressed liquidity buffer."""
    df = replay.frame
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.42, 0.29, 0.29],
        subplot_titles=(
            "Borrowing Base vs. Facility Limit & Drawn Revolver ($M)",
            "Daily Futures Variation Margin ($M, + inflow / − call)",
            "5-Day Stressed Liquidity Buffer ($M)",
        ),
    )
    bb_m = df["borrowing_base"] / 1e6
    y_lo = min(float(bb_m.min()), terms.drawn_revolver / 1e6) * 0.97
    y_hi = max(float(bb_m.max()), terms.facility_limit / 1e6) * 1.02
    fig.add_trace(
        go.Scatter(x=df.index, y=bb_m, name="Borrowing Base", mode="lines",
                   line=dict(color=COLORS["blue"], width=2.2),
                   hovertemplate="BB $%{y:,.2f}M<extra></extra>"),
        row=1, col=1,
    )
    fig.add_hrect(y0=y_lo, y1=terms.drawn_revolver / 1e6, fillcolor="rgba(239, 68, 68, 0.10)", line_width=0,
                  layer="below", row=1, col=1)
    fig.add_hline(y=terms.facility_limit / 1e6, line=dict(color=COLORS["emerald"], dash="dash", width=1.3),
                  annotation_text="Facility limit", annotation_position="top left",
                  annotation_font=dict(color=COLORS["emerald"], size=10), row=1, col=1)
    fig.add_hline(y=terms.drawn_revolver / 1e6, line=dict(color=COLORS["amber"], dash="dash", width=1.3),
                  annotation_text="Drawn revolver (deficiency below)", annotation_position="bottom left",
                  annotation_font=dict(color=COLORS["amber"], size=10), row=1, col=1)
    vm = df["daily_vm"] / 1e6
    fig.add_trace(
        go.Bar(x=df.index, y=vm, name="Daily VM", marker=dict(color=[COLORS["emerald"] if v >= 0 else COLORS["red"]
                                                                     for v in vm]),
               hovertemplate="VM $%{y:+,.2f}M<extra></extra>"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["rolling_vm"] / 1e6, name="5-day net VM", mode="lines",
                   line=dict(color=COLORS["amber"], width=1.6), hovertemplate="5d VM $%{y:+,.2f}M<extra></extra>"),
        row=2, col=1,
    )
    buf = df["stressed_buffer"] / 1e6
    fig.add_trace(
        go.Scatter(x=df.index, y=buf.clip(lower=0.0), mode="lines", line=dict(width=0), fill="tozeroy",
                   fillcolor="rgba(16, 185, 129, 0.25)", showlegend=False, hoverinfo="skip"),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=buf.clip(upper=0.0), mode="lines", line=dict(width=0), fill="tozeroy",
                   fillcolor="rgba(239, 68, 68, 0.35)", showlegend=False, hoverinfo="skip"),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=buf, name="Stressed buffer", mode="lines", line=dict(color=COLORS["text"], width=1.6),
                   hovertemplate="Buffer $%{y:+,.2f}M<extra></extra>"),
        row=3, col=1,
    )
    fig.add_hline(y=0.0, line=dict(color=COLORS["red"], width=1, dash="dash"), row=3, col=1)
    apply_institutional_layout(fig, "Historical Replay of Today's Book", height=820)
    fig.update_yaxes(range=[y_lo, y_hi], row=1, col=1)
    for subplot_title in fig.layout.annotations[:3]:
        subplot_title.font = dict(color=COLORS["text"], size=13)
    fig.update_layout(hovermode="x unified", legend=dict(y=-0.07))
    return fig
