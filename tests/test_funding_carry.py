"""Funding cost, carry and dilution reserve: zero by default, consistent across every engine when set."""

from __future__ import annotations

import numpy as np
import pytest

from core.advanced_scenario import (
    default_advanced_settings,
    legacy_equivalent_scenario,
    neutral_settings,
    run_advanced_scenario,
)
from core.borrowing_base import calculate_borrowing_base
from core.haircuts import compute_portfolio_inventory
from core.scenario_analytics import REGIME_CODES, evaluate_shock_matrix
from core.schemas import DEFAULT_FACILITY_TERMS, REQUIRED_TICKERS, FacilityTerms
from core.stress_engine import run_stress_scenario

COSTLY = {"revolver_interest_rate": 0.08, "inventory_carry_rate": 0.03, "carry_horizon_days": 30,
          "ar_dilution_reserve_pct": 0.05}


@pytest.fixture()
def costly_terms() -> FacilityTerms:
    return FacilityTerms.model_validate({**DEFAULT_FACILITY_TERMS.model_dump(), **COSTLY})


def _inventory_value(positions, snapshot) -> float:
    return sum(p.physical_qty * snapshot[p.ticker].spot_price for p in positions)


def test_defaults_are_zero_and_cost_nothing():
    t = DEFAULT_FACILITY_TERMS
    assert (t.revolver_interest_rate, t.inventory_carry_rate, t.ar_dilution_reserve_pct) == (0.0, 0.0, 0.0)
    assert t.funding_carry_cost(1e9) == 0.0


def test_carry_formula_actual_360(costly_terms, default_positions, snapshot):
    value = _inventory_value(default_positions, snapshot)
    expected = (133e6 * 0.08 + value * 0.03) * 30 / 360
    assert costly_terms.funding_carry_cost(value) == pytest.approx(expected)
    assert 133e6 * 0.08 * 30 / 360 == pytest.approx(886_666.67, abs=0.01)


def test_carry_lowers_buffer_by_exactly_its_cost(costly_terms, default_positions, default_receivables, snapshot):
    no_dilution = costly_terms.model_copy(update={"ar_dilution_reserve_pct": 0.0})
    base = run_stress_scenario(default_positions, default_receivables, snapshot, DEFAULT_FACILITY_TERMS,
                               shock_pct=0.14)
    costly = run_stress_scenario(default_positions, default_receivables, snapshot, no_dilution, shock_pct=0.14)
    carry = no_dilution.funding_carry_cost(_inventory_value(default_positions, snapshot))
    assert costly.funding_carry_outflow == pytest.approx(carry)
    assert base.net_liquidity_buffer - costly.net_liquidity_buffer == pytest.approx(carry, abs=1e-6)
    assert costly.total_liquid_resources - costly.total_cash_outflows == pytest.approx(costly.net_liquidity_buffer)


def test_dilution_reserve_reduces_receivables_and_shows_in_waterfall(costly_terms, default_positions,
                                                                    default_receivables, snapshot):
    items = compute_portfolio_inventory(default_positions, snapshot)
    base = calculate_borrowing_base(items, default_receivables, DEFAULT_FACILITY_TERMS)
    diluted = calculate_borrowing_base(items, default_receivables, costly_terms)
    eligible_gross = sum(r.eligible_gross_amount for r in base.receivable_items)
    assert diluted.dilution_reserve_dollars == pytest.approx(0.05 * eligible_gross)
    assert diluted.pre_cap_eligible_receivables == pytest.approx(base.pre_cap_eligible_receivables * 0.95)
    labels = [w.label for w in diluted.waterfall]
    assert "Dilution Reserve" in labels and "Dilution Reserve" not in [w.label for w in base.waterfall]
    assert diluted.waterfall[-1].running_total == pytest.approx(diluted.final_borrowing_base)
    running = [w for w in diluted.waterfall if w.label == "Conc. Cutback"][0].running_total
    assert running == pytest.approx(diluted.final_borrowing_base)


def test_vectorized_engine_matches_exact_engine_with_costs(costly_terms, default_positions, default_receivables,
                                                          snapshot):
    rng = np.random.default_rng(23)
    shocks = rng.uniform(-0.3, 0.3, size=(80, len(REQUIRED_TICKERS)))
    out = evaluate_shock_matrix(default_positions, default_receivables, snapshot, costly_terms, shocks)
    for i, row in enumerate(shocks):
        exact = run_stress_scenario(default_positions, default_receivables, snapshot, costly_terms,
                                    per_commodity_shocks=dict(zip(REQUIRED_TICKERS, row)))
        assert out["net_liquidity_buffer"][i] == pytest.approx(exact.net_liquidity_buffer, abs=1e-4)
        assert out["shocked_bb"][i] == pytest.approx(exact.shocked_bb, abs=1e-4)
        assert str(REGIME_CODES[out["regime_code"][i]]) == str(exact.regime)


def test_neutral_advanced_matches_legacy_with_costs(costly_terms, default_positions, default_receivables, snapshot):
    neutral = neutral_settings(default_advanced_settings())
    rng = np.random.default_rng(5)
    for _ in range(25):
        shocks = dict(zip(REQUIRED_TICKERS, rng.uniform(-0.3, 0.3, len(REQUIRED_TICKERS))))
        adv = run_advanced_scenario(default_positions, default_receivables, snapshot, costly_terms, neutral,
                                    legacy_equivalent_scenario(shocks))
        leg = run_stress_scenario(default_positions, default_receivables, snapshot, costly_terms,
                                  per_commodity_shocks=shocks)
        assert adv.net_liquidity_buffer == pytest.approx(leg.net_liquidity_buffer, abs=1e-4)
        assert adv.funding_carry_outflow == pytest.approx(leg.funding_carry_outflow)
        assert str(adv.regime) == str(leg.regime)
