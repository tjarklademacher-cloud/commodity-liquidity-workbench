"""Self-contained HTML credit committee report built from the same results and charts as the app."""

from __future__ import annotations

import html
from datetime import datetime, timezone

import plotly.graph_objects as go

from core.schemas import (
    BorrowingBaseResult,
    ContingencyPlan,
    FacilityTerms,
    MarketLoadResult,
    MonteCarloResult,
    RunwayResult,
    StressScenarioResult,
    StressSweepResult,
)
from ui.charts import (
    build_borrowing_base_waterfall,
    build_mc_distribution_chart,
    build_runway_fan_chart,
    build_stress_sweep_chart,
)
from ui.theme import (
    AUTHOR,
    COLORS,
    FONT_MONO,
    FONT_SANS,
    HOUSE_NAME,
    STATUS,
    DisplayCurrency,
    bridge_label,
    fmt_millions,
    fmt_pct,
    fmt_signed_millions,
    localize_figure,
    regime_label,
    regime_status,
    using_currency,
)


def _table(headers: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    head = "".join(f"<th class='{'num' if i >= numeric_from else ''}'>{html.escape(h)}</th>"
                   for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(f"<td class='{'num' if i >= numeric_from else ''}'>{c}</td>" for i, c in enumerate(r)) + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _fig(fig: go.Figure, first: bool) -> str:
    return localize_figure(fig).to_html(full_html=False, include_plotlyjs=True if first else False,
                                        config={"displaylogo": False, "responsive": True})


def build_report_html(
    currency: DisplayCurrency,
    market: MarketLoadResult,
    terms: FacilityTerms,
    bb: BorrowingBaseResult,
    stress: StressScenarioResult,
    sweep: StressSweepResult,
    mc: MonteCarloResult,
    runway: RunwayResult,
    plan: ContingencyPlan,
    scenario: str,
    explanation: str,
) -> str:
    """Render the report in `currency`. Safe to call from a download-button thread (sets its own currency)."""
    with using_currency(currency):
        status = regime_status(stress.regime)
        color, bg = STATUS[status]
        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        fx_note = ("" if currency.code == "USD"
                   else f" Figures in {currency.code} at {currency.usd_per_unit:.4f} USD per {currency.code} "
                        f"({currency.source}, {currency.as_of}).")
        verdict_rows = [
            ["Net liquidity buffer, active stress", fmt_signed_millions(stress.net_liquidity_buffer)],
            ["Liquidity regime", regime_label(stress.regime)],
            ["Borrowing base", fmt_millions(bb.final_borrowing_base)],
            ["Facility limit / drawn", f"{fmt_millions(terms.facility_limit)} / {fmt_millions(terms.drawn_revolver)}"],
            ["Revolver headroom", fmt_millions(bb.available_headroom)],
            ["Margin financing line", fmt_millions(terms.margin_line_limit)],
            [f"LaR 95%, {mc.horizon_days}-day", fmt_signed_millions(mc.p05_net_buffer)],
            ["Squeeze / deficiency breakeven",
             f"{fmt_pct(sweep.upside_squeeze_breakeven_pct) if sweep.upside_squeeze_breakeven_pct is not None else 'none'}"
             f" / {fmt_pct(sweep.downside_deficiency_breakeven_pct) if sweep.downside_deficiency_breakeven_pct is not None else 'none'}"],
            ["Shortfall within 20 trading days", fmt_pct(runway.prob_breach_within_horizon, 2) + " of paths"],
        ]
        recon_rows = [[html.escape(bridge_label(w.label)), fmt_millions(w.amount), fmt_millions(w.running_total)]
                      for w in bb.waterfall]
        action_rows = [
            [str(i + 1), html.escape(a.label),
             fmt_millions(a.required_usd) if a.required_usd is not None else "Not sufficient alone",
             "Yes" if a.feasible else "No",
             html.escape(a.detail) + (f" Can contribute up to {fmt_millions(a.capacity_usd)}."
                                      if a.capacity_usd is not None and not a.feasible else "")]
            for i, a in enumerate(plan.actions)
        ]
        trigger_rows = [[f"{t.shock_pct:+.2%}", regime_label(t.regime), html.escape(t.description)]
                        for t in plan.triggers]
        sections = [
            ("Borrowing base bridge", _fig(build_borrowing_base_waterfall(bb), True)
             + _table(["Line", "Amount", "Running total"], recon_rows)),
            ("Margin-trap stress sweep", _fig(build_stress_sweep_chart(sweep, stress.shock_pct), False)),
            (f"Monte Carlo liquidity-at-risk ({mc.n_sims:,} paths, {mc.horizon_days}-day)",
             _fig(build_mc_distribution_chart(mc), False)
             + _table(["Measure", "Value"], [
                 ["LaR 95% (5th percentile buffer)", fmt_signed_millions(mc.p05_net_buffer)],
                 ["LaR 99% (1st percentile buffer)", fmt_signed_millions(mc.p01_net_buffer)],
                 ["Expected shortfall 95%", fmt_signed_millions(mc.es95_net_buffer)],
                 ["P(critical squeeze)", fmt_pct(mc.prob_critical, 2)],
             ])),
            ("Liquidity runway, 20 trading days", _fig(build_runway_fan_chart(runway), False)),
            (f"Contingency funding plan: {html.escape(plan.target_label)}",
             f"<p>Net buffer in this scenario {fmt_signed_millions(plan.target_net_buffer)}; "
             f"shortfall to close {fmt_millions(plan.shortfall)}.</p>"
             + _table(["#", "Action", "Size needed", "Sufficient alone", "Detail"], action_rows, numeric_from=2)
             + "<h3>Early-warning triggers (parallel moves)</h3>"
             + _table(["Move", "Regime", "Meaning"], trigger_rows, numeric_from=0)),
        ]
        body = "".join(f"<section><h2>{t}</h2>{c}</section>" for t, c in sections)
        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(HOUSE_NAME)}: credit committee liquidity report</title>
<style>
body {{ font-family: {FONT_SANS}; color: {COLORS['text']}; background: {COLORS['canvas']}; margin: 0; }}
main {{ max-width: 1080px; margin: 0 auto; padding: 32px 24px 48px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }} h2 {{ font-size: 16px; margin: 0 0 12px; }}
h3 {{ font-size: 14px; margin: 20px 0 8px; }}
.meta {{ color: {COLORS['muted']}; font-size: 12px; margin-bottom: 20px; }}
section {{ background: #fff; border: 1px solid {COLORS['border']}; border-radius: 6px; padding: 20px;
  margin-bottom: 16px; }}
.verdict {{ border-left: 3px solid {color}; }}
.badge {{ display: inline-block; color: {color}; background: {bg}; font-weight: 600; font-size: 12px;
  padding: 2px 8px; border-radius: 4px; margin-left: 8px; }}
.big {{ font-family: {FONT_MONO}; font-size: 30px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 8px; }}
th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid {COLORS['grid']}; vertical-align: top; }}
th {{ background: {COLORS['sidebar']}; font-weight: 600; }}
td.num, th.num {{ text-align: right; font-family: {FONT_MONO}; font-variant-numeric: tabular-nums; }}
p {{ font-size: 13px; line-height: 1.55; }}
footer {{ color: {COLORS['muted']}; font-size: 12px; line-height: 1.5; }}
</style></head><body><main>
<h1>{html.escape(HOUSE_NAME)}: liquidity and borrowing base report</h1>
<div class="meta">Prepared {generated} with the Commodity Liquidity Workbench. Prices as of {market.as_of_date}
({html.escape(market.source_label)}). Scenario: {html.escape(scenario)}.{html.escape(fx_note)}</div>
<section class="verdict"><h2>Verdict<span class="badge">{regime_label(stress.regime)}</span></h2>
<div class="big">{fmt_signed_millions(stress.net_liquidity_buffer)}</div>
<p>{html.escape(explanation)}</p>{_table(["Measure", "Value"], verdict_rows)}</section>
{body}
<footer>Illustrative book: positions, receivables and facility terms are not real. Model assumptions: quantities,
receivables and drawn balance held fixed; haircuts use 30-day realized volatility; margin settled in cash on the
shock day. Certificate exports remain in USD. Built by {html.escape(AUTHOR['name'])},
{html.escape(AUTHOR['bio'].lower())}. Source: {AUTHOR['repo']}</footer>
</main></body></html>"""
