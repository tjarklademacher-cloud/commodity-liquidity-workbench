"""Reporting currency, FX fallback, branding and the credit committee report (no network)."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import pytest

from core.contingency import build_contingency_plan
from core.data_feed import FX_FALLBACK, FX_FALLBACK_DATE, load_fx_rate
from core.haircuts import compute_portfolio_inventory
from core.borrowing_base import calculate_borrowing_base
from core.scenario_analytics import run_liquidity_monte_carlo, run_liquidity_runway
from core.stress_engine import build_bb_certificate_json, run_stress_scenario, run_stress_sweep
from ui.report import build_report_html
from ui.theme import (
    CURRENCY_SYMBOLS,
    HOUSE_NAME,
    DisplayCurrency,
    display_currency,
    fmt_millions,
    fmt_signed_millions,
    localize_figure,
    money_divisor,
    using_currency,
)

EUR = DisplayCurrency(code="EUR", symbol=CURRENCY_SYMBOLS["EUR"], usd_per_unit=1.25, source="test", as_of="2026-11-20")


def test_default_currency_is_usd():
    assert display_currency().code == "USD"
    assert fmt_millions(114.12e6) == "$114.12M"
    assert money_divisor() == 1e6


def test_currency_context_converts_and_restores():
    with using_currency(EUR):
        assert fmt_millions(125e6) == "€100.00M"
        assert fmt_signed_millions(-2.5e6) == "-€2.00M"
        assert money_divisor() == pytest.approx(1.25e6)
    assert display_currency().code == "USD"


def test_localize_figure_swaps_symbols_not_numbers():
    fig = go.Figure(go.Scatter(x=[1], y=[2.0], hovertemplate="BB $%{y:,.2f}M<extra></extra>", text=["$2.00M"]))
    fig.update_yaxes(tickprefix="$", title_text="Buffer (USD M)")
    fig.add_annotation(text="Cap $17.12M", x=1, y=1)
    with using_currency(EUR):
        localize_figure(fig)
    assert fig.data[0].hovertemplate.startswith("BB €")
    assert fig.data[0].y[0] == 2.0
    assert fig.layout.yaxis.tickprefix == "€"
    assert fig.layout.yaxis.title.text == "Buffer (EUR M)"
    assert fig.layout.annotations[0].text == "Cap €17.12M"


def test_fx_rate_usd_live_and_fallback():
    assert load_fx_rate("USD") == (1.0, "model currency", "")

    def boom(*a, **k):
        raise ConnectionError("offline")

    assert load_fx_rate("EUR", download=boom) == (FX_FALLBACK["EUR"], "bundled rate", FX_FALLBACK_DATE)

    def fake(*a, **k):
        idx = pd.bdate_range(end="2026-11-20", periods=3)
        return pd.DataFrame({("Close", "CHFUSD=X"): [1.18, 1.19, 1.2]}, index=idx)

    rate, source, as_of = load_fx_rate("CHF", download=fake)
    assert rate == pytest.approx(1.2) and source == "Yahoo Finance" and as_of == "2026-11-20"
    with pytest.raises(ValueError):
        load_fx_rate("JPY")


def test_certificate_carries_borrower_and_currency(market, default_positions, default_receivables, default_terms):
    items = compute_portfolio_inventory(default_positions, market.snapshot)
    bb = calculate_borrowing_base(items, default_receivables, default_terms)
    stress = run_stress_scenario(default_positions, default_receivables, market.snapshot, default_terms,
                                 shock_pct=0.14, baseline_bb_result=bb)
    payload = json.loads(build_bb_certificate_json(bb, stress, market, HOUSE_NAME))
    assert payload["certificate"]["borrower"] == "Lademacher Commodities"
    assert payload["certificate"]["currency"] == "USD"
    assert payload["active_stress"]["margin_line_draw"] == 0.0


def test_report_html(market, default_positions, default_receivables, default_terms):
    items = compute_portfolio_inventory(default_positions, market.snapshot)
    bb = calculate_borrowing_base(items, default_receivables, default_terms)
    stress = run_stress_scenario(default_positions, default_receivables, market.snapshot, default_terms,
                                 shock_pct=0.14, baseline_bb_result=bb)
    sweep = run_stress_sweep(default_positions, default_receivables, market.snapshot, default_terms)
    mc = run_liquidity_monte_carlo(default_positions, default_receivables, market.snapshot, default_terms,
                                   market.prices_df, n_sims=500)
    runway = run_liquidity_runway(default_positions, default_receivables, market.snapshot, default_terms,
                                  market.prices_df, n_paths=300)
    plan = build_contingency_plan(default_positions, default_receivables, market.snapshot, default_terms, sweep,
                                  stress.shocks_map, "Active scenario")
    usd = build_report_html(DisplayCurrency(), market, default_terms, bb, stress, sweep, mc, runway, plan,
                            "Parallel +14%", "test explanation")
    assert usd.startswith("<!DOCTYPE html>")
    assert "Lademacher Commodities" in usd and "-$1.79M" in usd and "Critical squeeze" in usd
    assert usd.count("plotly.js v") <= 2 and "Plotly.newPlot" in usd
    eur = build_report_html(EUR, market, default_terms, bb, stress, sweep, mc, runway, plan, "Parallel +14%", "x")
    assert "-€1.43M" in eur and "1.2500 USD per EUR" in eur
    assert display_currency().code == "USD"
