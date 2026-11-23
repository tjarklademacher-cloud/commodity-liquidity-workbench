"""Advanced sweep, grid and hedge frontier: neutral settings reproduce the legacy engine; defaults are pinned."""

from __future__ import annotations

import pytest

from core.advanced_analytics import run_advanced_hedge_frontier, run_advanced_shock_grid, run_advanced_sweep
from core.advanced_scenario import default_advanced_settings, neutral_settings
from core.advanced_schemas import ScenarioSpec
from core.hedge_optimizer import run_hedge_frontier
from core.scenario_analytics import run_shock_grid
from core.stress_engine import run_stress_sweep

TOL = 1e-6


@pytest.fixture()
def neutral():
    return neutral_settings(default_advanced_settings())


@pytest.fixture()
def base_spec():
    return ScenarioSpec(name="Active scenario")


def test_neutral_advanced_sweep_equals_legacy(default_positions, default_receivables, default_terms, snapshot,
                                              neutral, base_spec):
    legacy = run_stress_sweep(default_positions, default_receivables, snapshot, default_terms)
    adv = run_advanced_sweep(default_positions, default_receivables, snapshot, default_terms, neutral, base_spec,
                             apply_trigger_rule=False)
    assert [p.shock_pct for p in adv.points] == [p.shock_pct for p in legacy.points]
    for a, b in zip(adv.points, legacy.points):
        assert a.net_liquidity_buffer == pytest.approx(b.net_liquidity_buffer, abs=TOL)
        assert a.shocked_bb == pytest.approx(b.shocked_bb, abs=TOL)
        assert a.regime == b.regime
    assert adv.upside_squeeze_breakeven_pct == pytest.approx(legacy.upside_squeeze_breakeven_pct, abs=1e-9)
    assert adv.downside_deficiency_breakeven_pct == pytest.approx(legacy.downside_deficiency_breakeven_pct, abs=1e-9)
    assert adv.regime_counts == legacy.regime_counts


def test_neutral_advanced_grid_equals_legacy(default_positions, default_receivables, default_terms, snapshot,
                                             neutral, base_spec):
    legacy = run_shock_grid(default_positions, default_receivables, snapshot, default_terms)
    adv = run_advanced_shock_grid(default_positions, default_receivables, snapshot, default_terms, neutral,
                                  base_spec, apply_trigger_rule=False)
    for row_a, row_b in zip(adv.net_buffer, legacy.net_buffer):
        assert row_a == pytest.approx(row_b, abs=TOL)
    assert adv.regime == legacy.regime
    assert adv.squeeze_share == pytest.approx(legacy.squeeze_share)


def test_neutral_advanced_frontier_equals_legacy(default_positions, default_receivables, default_terms, snapshot,
                                                 neutral, base_spec):
    legacy = run_hedge_frontier(default_positions, default_receivables, snapshot, default_terms)
    adv = run_advanced_hedge_frontier(default_positions, default_receivables, snapshot, default_terms, neutral,
                                      base_spec, apply_trigger_rule=False)
    assert adv.min_buffer == pytest.approx(legacy.min_buffer, abs=TOL)
    assert adv.positive_share == pytest.approx(legacy.positive_share)
    assert adv.max_margin_need == pytest.approx(legacy.max_margin_need, abs=TOL)
    assert adv.baseline_bb == pytest.approx(legacy.baseline_bb, abs=TOL)
    assert adv.recommended_ratio == legacy.recommended_ratio


def test_default_advanced_sweep_matches_active_scenario(default_positions, default_receivables, default_terms,
                                                       snapshot, base_spec):
    from core.advanced_scenario import auto_im_multipliers, run_advanced_scenario
    from core.schemas import REQUIRED_TICKERS

    settings = default_advanced_settings()
    sweep = run_advanced_sweep(default_positions, default_receivables, snapshot, default_terms, settings, base_spec)
    point = next(p for p in sweep.points if p.shock_pct == 0.14)
    shocks = {t: 0.14 for t in REQUIRED_TICKERS}
    spec = ScenarioSpec(name="x", futures_shocks=shocks,
                        im_multipliers=auto_im_multipliers(shocks, default_terms, {}, True))
    direct = run_advanced_scenario(default_positions, default_receivables, snapshot, default_terms, settings, spec)
    assert point.net_liquidity_buffer == pytest.approx(direct.net_liquidity_buffer, abs=TOL)
    assert point.net_liquidity_buffer == pytest.approx(-2.61e6, abs=0.01e6)
    identity = point.total_liquid_resources - point.total_cash_outflows
    assert identity == pytest.approx(point.net_liquidity_buffer, abs=TOL)
