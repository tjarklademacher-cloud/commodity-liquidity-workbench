"""Ground-truth checkpoints for the current six-commodity default book (adds coffee and sugar)."""

from __future__ import annotations

import pytest

from core.borrowing_base import calculate_borrowing_base, compare_concentration_cap_tiers
from core.haircuts import compute_portfolio_inventory
from core.schemas import COMMODITY_SPECS, REQUIRED_TICKERS, LiquidityRegime
from core.stress_engine import run_stress_scenario, run_stress_sweep

M = 0.01e6


@pytest.fixture()
def items(default_positions, snapshot):
    return compute_portfolio_inventory(default_positions, snapshot)


def test_six_commodities_present(default_positions):
    assert [p.ticker for p in default_positions] == list(REQUIRED_TICKERS)
    assert COMMODITY_SPECS["KC=F"].contract_multiplier == 37_500.0
    assert COMMODITY_SPECS["SB=F"].contract_multiplier == 112_000.0


@pytest.mark.parametrize(
    ("ticker", "haircut", "gmv", "eligible", "contracts"),
    [("KC=F", 0.3140, 12.375e6, 8.489e6, 102.0), ("SB=F", 0.2650, 7.661e6, 5.631e6, 320.0)],
)
def test_new_commodity_valuations(items, ticker, haircut, gmv, eligible, contracts):
    item = next(i for i in items if i.ticker == ticker)
    assert item.total_haircut == pytest.approx(haircut, abs=1e-4)
    assert item.gross_market_value == pytest.approx(gmv, abs=M)
    assert item.eligible_collateral == pytest.approx(eligible, abs=M)
    assert item.futures_contracts == pytest.approx(contracts)


def test_default_borrowing_base(items, default_receivables, default_terms):
    bb = calculate_borrowing_base(items, default_receivables, default_terms)
    assert bb.gross_inventory == pytest.approx(110.07e6, abs=M)
    assert bb.eligible_inventory == pytest.approx(79.80e6, abs=M)
    assert bb.total_hedged_notional == pytest.approx(93.49e6, abs=M)
    assert bb.gross_receivables == pytest.approx(83.00e6, abs=M)
    assert bb.pre_cap_eligible_receivables == pytest.approx(67.20e6, abs=M)
    assert bb.final_borrowing_base == pytest.approx(144.94e6, abs=M)
    assert bb.concentration_cutback_dollars == pytest.approx(2.06e6, abs=M)
    assert bb.capped_buyer_names == ["Barry Callebaut AG"]
    assert bb.available_headroom == pytest.approx(7.00e6, abs=M)
    tiers = compare_concentration_cap_tiers(items, default_receivables, default_terms)
    assert [round(t.final_borrowing_base / 1e6, 2) for t in tiers] == [144.94, 147.00, 147.00]


@pytest.mark.parametrize(
    ("shock", "nlb", "regime"),
    [
        (0.0, 12.50e6, LiquidityRegime.HEALTHY),
        (0.01, 11.57e6, LiquidityRegime.MARGIN_TRAP_WARNING),
        (0.14, -1.79e6, LiquidityRegime.CRITICAL_SQUEEZE),
        (-0.20, 16.16e6, LiquidityRegime.BB_DEFICIENCY_WARNING),
    ],
)
def test_default_stress_points(default_positions, default_receivables, snapshot, default_terms, shock, nlb, regime):
    r = run_stress_scenario(default_positions, default_receivables, snapshot, default_terms, shock_pct=shock)
    assert r.net_liquidity_buffer == pytest.approx(nlb, abs=M)
    assert r.regime == regime


def test_default_sweep(default_positions, default_receivables, snapshot, default_terms):
    sweep = run_stress_sweep(default_positions, default_receivables, snapshot, default_terms)
    assert sweep.regime_counts == {
        "HEALTHY": 13, "MARGIN_TRAP_WARNING": 12, "BB_DEFICIENCY_WARNING": 8, "CRITICAL_SQUEEZE": 8,
    }
    assert sweep.upside_squeeze_breakeven_pct == pytest.approx(0.1209, abs=0.0005)
    assert sweep.downside_deficiency_breakeven_pct == pytest.approx(-0.1272, abs=0.0005)


def test_minus_20_cure(default_positions, default_receivables, snapshot, default_terms):
    r = run_stress_scenario(default_positions, default_receivables, snapshot, default_terms, shock_pct=-0.20)
    assert r.bb_deficiency_cure == pytest.approx(6.83e6, abs=M)
    assert r.vm_inflow == pytest.approx(18.70e6, abs=M)
