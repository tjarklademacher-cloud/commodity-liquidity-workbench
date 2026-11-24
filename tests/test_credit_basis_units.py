"""Counterparty credit, units and currency, futures curve and roll accounting. Expected values by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.credit_risk import (
    NEVER,
    CreditScenario,
    InvoiceSchedule,
    default_settlement,
    expected_credit_loss,
    horizon_pd,
    horizon_transition,
    illustrative_transition_matrix,
    simulate_credit_outcomes,
)
from core.futures_curve import build_market_paths, business_dates, ewma_vol_path, ou_basis_paths, roll_economics
from core.treasury_engine import default_credit_outcomes, ledger_frame, run_treasury
from core.treasury_schemas import Counterparty, FuturesPosition, Invoice, TreasuryBook
from core.units import (
    UnitMismatchError,
    convert_currency,
    convert_price,
    convert_quantity,
    convert_with_factor,
    hedge_ratio_in_physical_units,
)
from tests.test_treasury_engine import X, _book, _cfg, _inst, _market

# --------------------------------------------------------------------------- credit


def test_horizon_pd_constant_hazard():
    assert horizon_pd(0.02, 365) == pytest.approx(0.02)
    assert horizon_pd(0.02, 0) == 0
    assert horizon_pd(0.02, 182.5) == pytest.approx(1 - 0.98 ** 0.5)
    with pytest.raises(ValueError):
        horizon_pd(1.2, 10)


def test_default_settlement_insured_and_recovery_sum_to_exposure():
    """E 1,000,000; coverage 90% after 100,000 deductible = 810,000 insured; uninsured 190,000; LGD 60% ->
    recovery 76,000, loss 114,000."""
    cp = Counterparty(id="B", name="B", insurance_coverage=0.9, insurance_deductible=100_000, lgd=0.6)
    s = default_settlement(1_000_000, cp)
    assert float(s["insured"]) == pytest.approx(810_000)
    assert float(s["recovery"]) == pytest.approx(76_000)
    assert float(s["loss"]) == pytest.approx(114_000)
    assert float(s["insured"] + s["recovery"] + s["loss"]) == pytest.approx(1_000_000)
    capped = default_settlement(1_000_000, cp.model_copy(update={"insurance_limit": 500_000}))
    assert float(capped["insured"]) == 500_000
    assert float(default_settlement(1_000_000, cp.model_copy(update={"insurance_excluded": True}))["insured"]) == 0


def test_ecl_formula_and_zero_pd():
    cp = Counterparty(id="A", name="A", annual_pd=0.03, lgd=0.5)
    assert expected_credit_loss(2_000_000, cp, 365) == pytest.approx(0.03 * 0.5 * 2_000_000)
    assert expected_credit_loss(2_000_000, cp.model_copy(update={"annual_pd": 0.0}), 365) == 0.0


def _sched(n_inv=1):
    return InvoiceSchedule(np.zeros(n_inv, dtype=int), np.zeros(n_inv, dtype=int), np.full(n_inv, 3))


def test_simulated_defaults_zero_and_certain():
    dates = business_dates("2026-11-20", 60)
    zero = simulate_credit_outcomes([Counterparty(id="A", name="A", annual_pd=0.0)], _sched(), dates, 2000, 1,
                                    extended_delay_prob=0.0, migration=False)
    assert (zero.default_day == NEVER).all()
    full = simulate_credit_outcomes([Counterparty(id="A", name="A", annual_pd=1.0)], _sched(), dates, 500, 1,
                                    migration=False)
    assert (full.default_day < NEVER).all()


@pytest.mark.parametrize("rho,beta", [(0.0, 0.0), (0.3, 0.0), (0.3, 0.8)])
def test_simulated_default_frequency_matches_target(rho, beta):
    dates = business_dates("2026-11-20", 250)
    pd_h = horizon_pd(0.10, (dates[-1] - dates[0]).days)
    n = 40_000
    factor = np.random.default_rng(9).standard_normal(n)
    cp = Counterparty(id="A", name="A", annual_pd=0.10, commodity_link="C", commodity_loading=beta)
    out = simulate_credit_outcomes([cp], _sched(), dates, n, 3, rho, {"C": factor}, migration=False)
    freq = (out.default_day[:, 0] < NEVER).mean()
    assert abs(freq - pd_h) < 4 * np.sqrt(pd_h * (1 - pd_h) / n)


def test_correlation_fattens_the_default_count_distribution():
    dates = business_dates("2026-11-20", 250)
    cps = [Counterparty(id=f"c{i}", name=str(i), annual_pd=0.05) for i in range(20)]
    sch = InvoiceSchedule(np.arange(20), np.zeros(20, dtype=int), np.full(20, 3))
    ind = simulate_credit_outcomes(cps, sch, dates, 5000, 1, 0.0, migration=False)
    cor = simulate_credit_outcomes(cps, sch, dates, 5000, 1, 0.5, migration=False)
    v_ind = (ind.default_day < NEVER).sum(axis=1).var()
    v_cor = (cor.default_day < NEVER).sum(axis=1).var()
    assert v_cor > 2 * v_ind


def test_wrong_way_loading_links_defaults_to_commodity_falls():
    dates = business_dates("2026-11-20", 250)
    n = 20_000
    f = np.random.default_rng(2).standard_normal(n)
    cp = Counterparty(id="A", name="A", annual_pd=0.10, commodity_link="C", commodity_loading=0.9)
    out = simulate_credit_outcomes([cp], _sched(), dates, n, 4, 0.5, {"C": f}, migration=False)
    d = out.default_day[:, 0] < NEVER
    assert d[f < -1].mean() > 2 * d[f > 1].mean()


def test_transition_matrix_rows_and_scaling():
    m = illustrative_transition_matrix()
    assert np.allclose(m.sum(axis=1), 1.0) and (m.to_numpy() >= 0).all()
    h = horizon_transition(m, 73)
    assert np.allclose(h.sum(axis=1), 1.0) and (h.to_numpy() >= 0).all()
    assert h.loc["BBB", "D"] == pytest.approx(0.2 * m.loc["BBB", "D"])


def _credit_book(cp):
    return TreasuryBook(instruments=_inst(), counterparties=[cp],
                        invoices=[Invoice(counterparty_id=cp.id, amount=100_000, due_in_days=4)])


def test_payment_delay_without_default_moves_cash_not_value():
    cp = Counterparty(id="A", name="A", rating="BBB")
    book = _credit_book(cp)
    cfg = _cfg(6, drawn=0)
    mk = _market([100] * 7)
    on_time = ledger_frame(run_treasury(book, cfg, mk))
    late = ledger_frame(run_treasury(book, cfg, mk, default_credit_outcomes(
        book, cfg, mk, CreditScenario(extra_delay_days={"A": 3}))))
    assert on_time["collections"].iloc[2] == pytest.approx(100_000)
    assert late["collections"].iloc[2] == 0 and late["collections"].iloc[5] == pytest.approx(100_000)
    assert late["credit_loss"].sum() == 0
    assert late["networth"].iloc[-1] == pytest.approx(on_time["networth"].iloc[-1])


def test_insured_default_is_a_claim_not_cash():
    """Default on day 1 of 100,000 insured 90% (no deductible), claim paid after 3 calendar days (Thu 26th = day
    4), LGD 100% on the uninsured 10,000 -> realized loss 10,000, claims receivable 90,000 until day 4."""
    cp = Counterparty(id="A", name="A", insurance_coverage=0.9, claim_delay_days=3, lgd=1.0)
    book = _credit_book(cp)
    cfg = _cfg(5, drawn=0)
    mk = _market([100] * 6)
    out = default_credit_outcomes(book, cfg, mk, CreditScenario(default_day={"A": 1}))
    f = ledger_frame(run_treasury(book, cfg, mk, out))
    assert f["credit_loss"].sum() == pytest.approx(10_000)
    assert f["claims_receivable"].tolist() == pytest.approx([0, 90_000, 90_000, 90_000, 0, 0])
    assert f["insurance_receipts"].tolist() == pytest.approx([0, 0, 0, 0, 90_000, 0])
    assert f["collections"].sum() == 0


def test_partial_recovery_uninsured():
    cp = Counterparty(id="A", name="A", lgd=0.6, recovery_delay_days=2)
    book = _credit_book(cp)
    cfg = _cfg(4, drawn=0)
    mk = _market([100] * 5)
    f = ledger_frame(run_treasury(book, cfg, mk, default_credit_outcomes(book, cfg, mk,
                                                                          CreditScenario(default_day={"A": 1}))))
    assert f["credit_loss"].sum() == pytest.approx(60_000)
    assert f["recoveries"].sum() == pytest.approx(40_000)
    assert f["recoveries"].sum() <= 100_000


def test_downgrade_makes_receivable_ineligible():
    cp = Counterparty(id="A", name="A", rating="BBB")
    book = TreasuryBook(instruments=_inst(), counterparties=[cp],
                        invoices=[Invoice(counterparty_id="A", amount=100_000, due_in_days=60)])
    cfg = _cfg(3, drawn=0)
    mk = _market([100] * 4)
    out = default_credit_outcomes(book, cfg, mk, CreditScenario(downgrade_day={"A": 2}))
    f = ledger_frame(run_treasury(book, cfg, mk, out))
    assert f["receivable_contribution"].tolist() == pytest.approx([85_000, 85_000, 0, 0])


def test_partial_payment_and_dispute():
    cp = Counterparty(id="A", name="A")
    book = _credit_book(cp)
    cfg = _cfg(6, drawn=0)
    mk = _market([100] * 7)
    out = default_credit_outcomes(book, cfg, mk, CreditScenario(partial_payment={"A": 0.7}, dispute_days=2))
    f = ledger_frame(run_treasury(book, cfg, mk, out))
    assert f["collections"].iloc[2] == pytest.approx(70_000)
    assert f["eligible_receivables"].iloc[2] == 0
    assert f["collections"].sum() == pytest.approx(100_000)


# --------------------------------------------------------------------------- units and currency

def test_units_and_mismatch():
    assert convert_quantity(1, "MT", "lbs") == pytest.approx(2204.6226218, rel=1e-9)
    assert convert_quantity(1, "bbl", "US gal") == pytest.approx(42.0)
    assert convert_quantity(1, "MWh", "MMBtu") == pytest.approx(3.412141633, rel=1e-6)
    assert convert_price(1.0, "lbs", "MT") == pytest.approx(2204.6226218, rel=1e-9)
    with pytest.raises(UnitMismatchError):
        convert_quantity(1, "bbl", "MT")
    with pytest.raises(ValueError):
        convert_quantity(1, "MT", "parsec")
    assert convert_with_factor(1_000, 7.33, "illustrative crude density") == pytest.approx(7_330)
    with pytest.raises(ValueError):
        convert_with_factor(1, 7.33, " ")


def test_hedge_ratio_in_physical_units_and_currency():
    """Copper: 100 contracts x 25,000 lbs = 2.5M lbs = 1,133.98 MT against 1,000 MT physical."""
    assert hedge_ratio_in_physical_units(1_000, "MT", 100, 25_000, "lbs") == pytest.approx(1.1339809, rel=1e-6)
    assert convert_currency(100, "EUR", "USD", {"EUR": 1.10}) == pytest.approx(110)
    assert convert_currency(110, "USD", "EUR", {"EUR": 1.10}) == pytest.approx(100)
    with pytest.raises(ValueError):
        convert_currency(1, "JPY", "USD", {})


# --------------------------------------------------------------------------- curve, maturity mismatch, roll

def test_roll_creates_no_cash_beyond_costs():
    """Short 10 in contract 0 expiring day 4, roll 2 days before (day 2). Curve contango 2% per month.
    VM continuity: days 1-2 on contract 0, days 3-4 on contract 1. Cost 5 per contract per leg -> 100."""
    h = 4
    dates = business_dates("2026-11-20", h)
    front = np.array([100, 101, 102, 103, 104], dtype=float)[None, :, None]
    mk = build_market_paths(dates, front, np.array([0.3]), n_contracts=2, monthly_slope=0.02,
                            expiry_day=np.array([[4, 25]]), roll_days_before_expiry=2)
    book = _book(qty=1_000, contracts=-10)
    cfg = _cfg(h, im=0.0)
    cfg = cfg.model_copy(update={"transaction_cost_per_contract": 5.0, "roll_days_before_expiry": 2})
    f = ledger_frame(run_treasury(book, cfg, mk))
    c0 = front[0, :, 0]
    c1 = c0 * 1.02
    vm = [-10 * 100 * (c0[1] - c0[0]), -10 * 100 * (c0[2] - c0[1]), -10 * 100 * (c1[3] - c1[2]),
          -10 * 100 * (c1[4] - c1[3])]
    assert f["vm_net"].iloc[1:].tolist() == pytest.approx(vm)
    assert f["transaction_costs"].tolist() == pytest.approx([0, 0, 100, 0, 0])
    econ = roll_economics(-10, 100, c0[2], c1[2], 5.0)
    assert econ["cash_flow"] == pytest.approx(-100)
    assert econ["spread_exposure_value"] == pytest.approx(10 * 100 * (c1[2] - c0[2]))


def test_contract_maturity_mismatch_steepening_creates_pnl():
    """Physical referenced to contract 0, hedge in contract 1. A steepening (slope 0 -> +5%) with flat front
    raises contract 1 by 5%: short 10 x 100 x 5 = -5,000 with no physical gain."""
    h = 1
    dates = business_dates("2026-11-20", h)
    mk = build_market_paths(dates, np.array([[100.0, 100.0]])[:, :, None], np.array([0.3]), n_contracts=2,
                            monthly_slope=0.0, slope_shock=np.array([[0.0], [0.05]]),
                            expiry_day=np.array([[30, 50]]))
    book = TreasuryBook(instruments=_inst(), physical=_book().physical,
                        futures=[FuturesPosition(instrument_id=X, contract_index=1, signed_contracts=-10)],
                        counterparties=[Counterparty(id="A", name="A")])
    f = ledger_frame(run_treasury(book, _cfg(h, im=0.0), mk))
    assert f["economic_pnl"].iloc[1] == pytest.approx(-5_000)


def test_ewma_vol_and_ou_basis():
    prices = np.array([100, 110, 99, 108.9])[None, :, None]
    v = ewma_vol_path(prices, np.array([0.2]), lam=0.9)
    r1 = np.log(1.1)
    assert v[0, 1, 0] == pytest.approx(np.sqrt(0.9 * 0.04 + 0.1 * 252 * r1 ** 2))
    with pytest.raises(ValueError):
        ewma_vol_path(prices, np.array([0.2]), lam=1.0)
    b = ou_basis_paths(4000, 250, np.array([0.05]), np.array([0.0]), np.array([5.0]), np.array([0.0]), 1)
    assert b[0, -1, 0] == pytest.approx(0.05 * (1 - 5 / 252) ** 250)
    noisy = ou_basis_paths(4000, 500, np.array([0.0]), np.array([0.0]), np.array([4.0]), np.array([0.2]), 1)
    stationary_sd = 0.2 / np.sqrt(2 * 4.0)
    assert noisy[:, -1, 0].std() == pytest.approx(stationary_sd, rel=0.08)


def test_calendar_dates_skip_weekends():
    d = business_dates("2026-11-20", 3)
    assert [x.strftime("%a") for x in d] == ["Fri", "Mon", "Tue", "Wed"]
    assert isinstance(d, pd.DatetimeIndex)
