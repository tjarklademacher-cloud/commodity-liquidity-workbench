"""Dynamic inventory haircut engine."""

from __future__ import annotations

import pytest

from core.haircuts import compute_inventory_valuation, compute_portfolio_inventory, inventory_valuations_to_dataframe
from core.schemas import COMMODITY_SPECS, InventoryPosition, MarketQuote

EXPECTED = {
    "GC=F": (0.1425, 0.8575, 24.39e6, 20.91e6),
    "HG=F": (0.2200, 0.7800, 23.925e6, 18.66e6),
    "LBR=F": (0.35025, 0.64975, 18.08e6, 11.75e6),
    "CC=F": (0.3925, 0.6075, 23.635e6, 14.36e6),
}


def _quote(ticker: str, spot: float, vol: float) -> MarketQuote:
    return MarketQuote(
        ticker=ticker, spot_price=spot, prev_price=spot, ret_1d=0.0, ret_30d=0.0,
        realized_vol_30d=vol, vol_30d_min=vol, vol_30d_max=vol, as_of_date="2026-11-20",
    )


def _components_sum(v) -> float:
    return v.advance_rate + v.eff_base_haircut + v.eff_vol_addon + v.eff_unhedged_penalty + v.eff_floor_adjustment


@pytest.mark.parametrize("ticker", list(EXPECTED))
def test_baseline_haircuts(baseline_items, ticker):
    item = next(i for i in baseline_items if i.ticker == ticker)
    hc, adv, gmv, elig = EXPECTED[ticker]
    assert item.total_haircut == pytest.approx(hc, abs=1e-4)
    assert item.advance_rate == pytest.approx(adv, abs=1e-4)
    assert item.gross_market_value == pytest.approx(gmv, abs=0.01e6)
    assert item.eligible_collateral == pytest.approx(elig, abs=0.01e6)
    assert _components_sum(item) == pytest.approx(1.0, abs=1e-12)


def test_portfolio_totals(baseline_items):
    assert sum(i.gross_market_value for i in baseline_items) == pytest.approx(90.03e6, abs=0.01e6)
    assert sum(i.haircut_dollars for i in baseline_items) == pytest.approx(24.35e6, abs=0.01e6)
    assert sum(i.eligible_collateral for i in baseline_items) == pytest.approx(65.68e6, abs=0.01e6)
    assert sum(i.hedged_notional for i in baseline_items) == pytest.approx(76.85e6, abs=0.01e6)


def test_contract_counts(baseline_items):
    contracts = {i.ticker: i.futures_contracts for i in baseline_items}
    assert contracts["GC=F"] == pytest.approx(90.0)
    assert contracts["HG=F"] == pytest.approx(187.0)
    assert contracts["LBR=F"] == pytest.approx(698.18, abs=0.01)
    assert contracts["CC=F"] == pytest.approx(261.0)


def test_over_hedging_clips_hedge_ratio():
    pos = InventoryPosition(ticker="HG=F", physical_qty=1_000_000.0, hedged_qty=1_500_000.0)
    v = compute_inventory_valuation(pos, _quote("HG=F", 4.0, 0.20), COMMODITY_SPECS["HG=F"])
    assert v.hedge_ratio == 1.0
    assert v.unhedged_penalty == 0.0
    assert v.hedged_notional == pytest.approx(6_000_000.0)
    assert v.futures_contracts == pytest.approx(60.0)


def test_min_clamp_adds_floor_adjustment():
    spec = COMMODITY_SPECS["GC=F"].model_copy(update={"base_haircut": 0.02})
    pos = InventoryPosition(ticker="GC=F", physical_qty=100.0, hedged_qty=100.0)
    v = compute_inventory_valuation(pos, _quote("GC=F", 2000.0, 0.05), spec)
    assert v.raw_haircut == pytest.approx(0.02)
    assert v.total_haircut == pytest.approx(0.10)
    assert v.was_clamped_min and not v.was_clamped_max
    assert v.eff_floor_adjustment == pytest.approx(0.08)
    assert _components_sum(v) == pytest.approx(1.0, abs=1e-12)


def test_extreme_vol_max_clamp_preserves_100pct():
    pos = InventoryPosition(ticker="CC=F", physical_qty=1000.0, hedged_qty=0.0)
    v = compute_inventory_valuation(pos, _quote("CC=F", 8000.0, 1.20), COMMODITY_SPECS["CC=F"])
    assert v.raw_haircut == pytest.approx(0.28 + 0.45 * 0.90 + 0.18)
    assert v.total_haircut == pytest.approx(0.65)
    assert v.was_clamped_max
    assert v.eff_base_haircut == pytest.approx(0.28)
    assert v.eff_vol_addon == pytest.approx(0.37)
    assert v.eff_unhedged_penalty == 0.0
    assert v.eff_floor_adjustment == 0.0
    assert _components_sum(v) == pytest.approx(1.0, abs=1e-12)


def test_zero_inventory_is_safe():
    pos = InventoryPosition(ticker="LBR=F", physical_qty=0.0, hedged_qty=500.0)
    v = compute_inventory_valuation(pos, _quote("LBR=F", 500.0, 0.30), COMMODITY_SPECS["LBR=F"])
    assert v.hedge_ratio == 0.0
    assert v.gross_market_value == 0.0
    assert v.eligible_collateral == 0.0
    assert v.hedged_notional == pytest.approx(250_000.0)
    assert _components_sum(v) == pytest.approx(1.0, abs=1e-12)


def test_shock_moves_price_and_floors_at_zero():
    pos = InventoryPosition(ticker="GC=F", physical_qty=10.0, hedged_qty=10.0)
    up = compute_inventory_valuation(pos, _quote("GC=F", 2000.0, 0.15), COMMODITY_SPECS["GC=F"], 0.10)
    crash = compute_inventory_valuation(pos, _quote("GC=F", 2000.0, 0.15), COMMODITY_SPECS["GC=F"], -1.5)
    assert up.shocked_price == pytest.approx(2200.0)
    assert crash.shocked_price == 0.0


def test_portfolio_shocks_map(positions, snapshot):
    items = compute_portfolio_inventory(positions, snapshot, shocks_map={"CC=F": 0.30})
    cc = next(i for i in items if i.ticker == "CC=F")
    gc = next(i for i in items if i.ticker == "GC=F")
    assert cc.shocked_price == pytest.approx(8150.0 * 1.3)
    assert gc.shock_pct == 0.0


def test_dataframe_export(baseline_items):
    df = inventory_valuations_to_dataframe(baseline_items)
    assert len(df) == 4
    assert {"ticker", "total_haircut", "eligible_collateral"} <= set(df.columns)
