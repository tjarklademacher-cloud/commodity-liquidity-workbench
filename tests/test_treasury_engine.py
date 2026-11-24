"""Daily treasury engine: hand-calculated benchmarks and accounting invariants.

Expected values in this file are computed by hand in the comments, not by calling the engine.
Calendar: start Friday 2026-11-20; business days 1, 2, 3 = Mon 23, Tue 24, Wed 25 November.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.collateral import cap_solve_matrix
from core.borrowing_base import solve_concentration_cap_fixed_point
from core.futures_curve import build_market_paths, business_dates
from core.treasury_engine import ledger_frame, run_treasury
from core.treasury_schemas import (
    Counterparty,
    FacilityConfig,
    FuturesPosition,
    HaircutPolicy,
    Instrument,
    Invoice,
    MarginPolicy,
    Payable,
    PhysicalPosition,
    PhysicalTrade,
    TreasuryBook,
    TreasuryConfig,
)

X = "X"


def _inst():
    return [Instrument(id=X, name="Test commodity", unit="MT", contract_size=100)]


def _cfg(h, cash=20_000.0, commitment=150_000.0, drawn=50_000.0, delay=0, rate=0.0, min_cash=0.0, im=1_000.0,
         base=0.20, dynamic=False, sweep=False, grace=2, **fac):
    return TreasuryConfig(
        start_date="2026-11-20", horizon_days=h, opening_cash=cash, min_operating_cash=min_cash,
        facility=FacilityConfig(commitment=commitment, opening_drawn=drawn, interest_rate=rate,
                                drawdown_delay_days=delay, sweep_surplus=sweep, cure_grace_days=grace, **fac),
        haircuts={X: HaircutPolicy(base=base, vol_coefficient=0.0, min_haircut=0.0, max_haircut=1.0, dynamic=dynamic)},
        margins={X: MarginPolicy(mode="fixed_per_contract", im_per_contract=im)})


def _market(prices, vol=0.3, basis=None):
    px = np.asarray(prices, dtype=float)
    dates = business_dates("2026-11-20", len(px) - 1)
    b = None if basis is None else np.asarray(basis, dtype=float)[None, :, None]
    return build_market_paths(dates, px[None, :, None], np.array([vol]), basis=b)


def _book(qty=1_000.0, contracts=-8.0, invoices=(), payables=(), trades=(), cps=None):
    cps = cps if cps is not None else [Counterparty(id="A", name="Buyer A", rating="BBB")]
    fut = [FuturesPosition(instrument_id=X, signed_contracts=contracts)] if contracts else []
    return TreasuryBook(instruments=_inst(), physical=[PhysicalPosition(instrument_id=X, quantity=qty)] if qty else [],
                        futures=fut, counterparties=cps, invoices=list(invoices), payables=list(payables),
                        trades=list(trades))


# --------------------------------------------------------------------------- worked example (section 52)

def test_multi_day_worked_example_matches_hand_calculation():
    """Inventory 1,000 MT at 100, short 8 x 100 MT, cash 20,000, revolver 150,000 with 50,000 drawn at 3.6%,
    receivable 30,000 due Tue, payable 30,000 due Wed, minimum cash 10,000, haircut 20% then +10 points lender
    stress on day 3, IM 1,000 per contract doubling on day 3. Prices 100, 110, 105, 120.

    Day 1 (Mon, 3 calendar days of interest): VM -8 x 100 x 10 = -8,000; interest 50,000 x 3.6% x 3/360 = 15.
              cash 20,000 - 8,000 - 15 = 11,985. BB = 1,000 x 110 x 0.8 + 30,000 x 0.85 = 113,500.
    Day 2: VM +4,000; collection 30,000; interest 5. cash 45,980. BB = 1,000 x 105 x 0.8 = 84,000.
    Day 3: VM -12,000; IM call 16,000 - 8,000 = 8,000; payable 30,000; interest 5. cash -4,025.
           BB = 1,000 x 120 x 0.7 = 84,000; available 84,000 - 50,000 = 34,000; same-day draw 14,025
           -> cash 10,000, debt 64,025, undrawn 19,975, available liquidity 29,975.
    Net worth: day 0 = 20,000 + 8,000 + 100,000 + 30,000 - 30,000 - 50,000 = 78,000;
               P&L = inventory +20,000, futures -16,000, interest -25 -> day 3 = 81,975."""
    book = _book(invoices=[Invoice(counterparty_id="A", amount=30_000, due_in_days=4)],
                 payables=[Payable(amount=30_000, due_in_days=5)])
    cfg = _cfg(3, rate=0.036, min_cash=10_000, dynamic=True)
    stress = np.array([[0.0], [0.0], [0.0], [0.10]])
    im_mult = np.array([[1.0], [1.0], [1.0], [2.0]])
    run = run_treasury(book, cfg, _market([100, 110, 105, 120]), lender_stress=stress, im_multiplier=im_mult)
    f = ledger_frame(run)
    assert f["cash"].tolist() == pytest.approx([20_000, 11_985, 45_980, 10_000], abs=1e-6)
    assert f["vm_net"].tolist() == pytest.approx([0, -8_000, 4_000, -12_000])
    assert f["interest"].tolist() == pytest.approx([0, 15, 5, 5])
    assert f["collections"].tolist() == pytest.approx([0, 0, 30_000, 0])
    assert f["supplier_payments"].tolist() == pytest.approx([0, 0, 0, 30_000])
    assert f["im_call"].tolist() == pytest.approx([0, 0, 0, 8_000])
    assert f["borrowing_base"].tolist() == pytest.approx([105_500, 113_500, 84_000, 84_000])
    assert f["revolver_drawn"].tolist() == pytest.approx([50_000, 50_000, 50_000, 64_025])
    assert f["revolver_undrawn"].iloc[-1] == pytest.approx(19_975)
    assert f["available_liquidity"].iloc[-1] == pytest.approx(29_975)
    assert f["shortfall"].max() == pytest.approx(0.0)
    assert f["networth"].tolist() == pytest.approx([78_000, 79_985, 78_980, 81_975])
    assert run.reconciliation_residual < 1e-6 and run.networth_residual < 1e-6


# --------------------------------------------------------------------------- treasury and cash flow

def test_zero_movement_zero_transaction_is_inert():
    run = run_treasury(_book(), _cfg(5), _market([100] * 6))
    f = ledger_frame(run)
    assert (f["cash"] == 20_000).all()
    assert (f["networth"] == f["networth"].iloc[0]).all()
    assert f["shortfall"].max() == 0 and f["vm_net"].abs().max() == 0


def test_delayed_drawdown_leaves_an_unfunded_day():
    """Same as the worked example's day 3 need but with a one-day revolver delay: day 3 cash -4,025 is a
    14,025 shortfall against the 10,000 minimum; the draw arrives on day 4."""
    book = _book(invoices=[Invoice(counterparty_id="A", amount=30_000, due_in_days=4)],
                 payables=[Payable(amount=30_000, due_in_days=5)])
    cfg = _cfg(4, rate=0.036, min_cash=10_000, dynamic=True, delay=1)
    stress = np.array([[0.0], [0.0], [0.0], [0.10], [0.10]])
    im_mult = np.array([[1.0], [1.0], [1.0], [2.0], [2.0]])
    f = ledger_frame(run_treasury(book, cfg, _market([100, 110, 105, 120, 120]), lender_stress=stress,
                                  im_multiplier=im_mult))
    assert f["shortfall"].iloc[3] == pytest.approx(14_025)
    assert f["revolver_requested"].iloc[3] == pytest.approx(14_025)
    assert f["revolver_received"].iloc[4] == pytest.approx(14_025)
    assert f["revolver_drawn"].iloc[3] == pytest.approx(50_000)


def test_unfunded_shortfall_when_capacity_is_exhausted():
    """Commitment 50,000 fully drawn: a 25,000 VM call on 5,000 cash leaves 20,000 unfunded."""
    run = run_treasury(_book(qty=0, contracts=-10), _cfg(1, cash=5_000, commitment=50_000, drawn=50_000),
                       _market([100, 125]))
    f = ledger_frame(run)
    assert f["cash"].iloc[1] == pytest.approx(-20_000)
    assert f["shortfall"].iloc[1] == pytest.approx(20_000)
    assert f["revolver_drawn"].iloc[1] == pytest.approx(50_000)


def test_sweep_repays_revolver_and_debt_falls():
    """Collection of 30,000 on day 2 with sweep on: surplus above the 20,000 opening cash repays the revolver."""
    book = _book(qty=1_000, contracts=0, invoices=[Invoice(counterparty_id="A", amount=30_000, due_in_days=4)])
    f = ledger_frame(run_treasury(book, _cfg(2, sweep=True), _market([100, 100, 100])))
    assert f["revolver_repaid"].iloc[2] == pytest.approx(30_000)
    assert f["revolver_drawn"].iloc[2] == pytest.approx(20_000)
    assert f["cash"].iloc[2] == pytest.approx(20_000)


def test_commitment_fee_on_undrawn():
    """Fee 0.5% on undrawn 100,000 for 3 calendar days at ACT/360 = 4.1667."""
    cfg = _cfg(1)
    cfg = cfg.model_copy(update={"facility": cfg.facility.model_copy(update={"commitment_fee": 0.005})})
    f = ledger_frame(run_treasury(_book(contracts=0), cfg, _market([100, 100])))
    assert f["commitment_fees"].iloc[1] == pytest.approx(100_000 * 0.005 * 3 / 360)


# --------------------------------------------------------------------------- derivatives

@pytest.mark.parametrize("contracts,move,expected", [(10, +5, 5_000), (-10, +5, -5_000), (10, -5, -5_000),
                                                     (-10, -5, 5_000), (0, +5, 0)])
def test_variation_margin_sign(contracts, move, expected):
    f = ledger_frame(run_treasury(_book(qty=0, contracts=contracts), _cfg(1, im=0.0), _market([100, 100 + move])))
    assert f["vm_net"].iloc[1] == pytest.approx(expected)
    assert f["cash"].iloc[1] - f["cash"].iloc[0] == pytest.approx(expected)


def test_initial_margin_call_and_release_with_lag_is_not_an_expense():
    """IM 1,000 x 8 = 8,000; x2 on day 1 -> call 8,000; back to x1 on day 2 -> 8,000 released on day 4
    (two-day lag). Net worth unchanged throughout (no price move)."""
    cfg = _cfg(4)
    cfg = cfg.model_copy(update={"margins": {X: MarginPolicy(mode="fixed_per_contract", im_per_contract=1_000,
                                                              release_lag_days=2)}})
    mult = np.array([[1.0], [2.0], [1.0], [1.0], [1.0]])
    f = ledger_frame(run_treasury(_book(), cfg, _market([100] * 5), im_multiplier=mult))
    assert f["im_call"].tolist() == pytest.approx([0, 8_000, 0, 0, 0])
    assert f["im_released"].tolist() == pytest.approx([0, 0, 0, 0, 8_000])
    assert f["cash"].tolist() == pytest.approx([20_000, 12_000, 12_000, 12_000, 20_000])
    assert f["restricted_cash"].tolist() == pytest.approx([8_000, 16_000, 16_000, 16_000, 8_000])
    assert (f["networth"] == f["networth"].iloc[0]).all()


def test_no_duplicate_derivative_cash_flow():
    """Cumulative VM equals the total futures P&L once: -8 x 100 x (130 - 100) = -24,000."""
    f = ledger_frame(run_treasury(_book(), _cfg(3, im=0.0), _market([100, 110, 120, 130])))
    assert f["vm_net"].sum() == pytest.approx(-24_000)
    assert f["cum_vm"].iloc[-1] == pytest.approx(-24_000)
    assert f["economic_pnl"].sum() == pytest.approx(30_000 - 24_000)


def test_volatility_scaled_margin_rises_with_volatility():
    cfg = _cfg(2)
    cfg = cfg.model_copy(update={"margins": {X: MarginPolicy(mode="vol_scaled", z_score=2.0, mpor_days=2)}})
    mk = _market([100, 100, 100])
    mk.vol[:, 2, 0] = 0.6
    f = ledger_frame(run_treasury(_book(), cfg, mk))
    rate0 = 2.0 * 0.3 * np.sqrt(2 / 252) * 100 * 100
    assert f["im_posted"].iloc[0] == pytest.approx(8 * rate0)
    assert f["im_posted"].iloc[2] == pytest.approx(16 * rate0)


# --------------------------------------------------------------------------- borrowing base

def test_zero_collateral_gives_zero_base():
    f = ledger_frame(run_treasury(_book(qty=0, contracts=0, cps=[]), _cfg(1, drawn=0), _market([100, 100])))
    assert f["borrowing_base"].max() == 0 and f["revolver_undrawn"].max() == 0


def test_commitment_caps_permitted_and_undrawn_is_never_negative():
    f = ledger_frame(run_treasury(_book(qty=10_000, contracts=0), _cfg(1, commitment=150_000, drawn=50_000),
                                  _market([100, 100])))
    assert f["borrowing_base"].iloc[0] == pytest.approx(800_000)
    assert f["permitted"].iloc[0] == pytest.approx(150_000)
    assert f["revolver_undrawn"].iloc[0] == pytest.approx(100_000)


def test_deficiency_is_recorded_and_cured_after_grace():
    """Unhedged 1,000 MT, drawn 70,000. Price 100 -> 80: BB 64,000, deficiency 6,000 on day 1 and day 2,
    cured on day 3 (grace 2 days) from cash, never by creating cash."""
    run = run_treasury(_book(contracts=0), _cfg(3, drawn=70_000, grace=2), _market([100, 80, 80, 80]))
    f = ledger_frame(run)
    assert f["deficiency"].tolist()[:3] == pytest.approx([0, 6_000, 6_000])
    assert f["mandatory_repayment"].tolist() == pytest.approx([0, 0, 0, 6_000])
    assert f["revolver_drawn"].iloc[3] == pytest.approx(64_000)
    assert f["cash"].iloc[3] == pytest.approx(14_000)
    assert (f["revolver_undrawn"] >= 0).all()


def test_dynamic_haircut_increase_reduces_base_and_contractual_limit_binds():
    cfg = _cfg(2, dynamic=True)
    stress = np.array([[0.0], [0.15], [0.15]])
    f = ledger_frame(run_treasury(_book(contracts=0), cfg, _market([100, 100, 100]), lender_stress=stress))
    assert f["borrowing_base"].tolist() == pytest.approx([80_000, 65_000, 65_000])
    lim = cfg.model_copy(update={"haircuts": {X: cfg.haircuts[X].model_copy(
        update={"max_increase_per_redetermination": 0.05})}})
    g = ledger_frame(run_treasury(_book(contracts=0), lim, _market([100, 100, 100]), lender_stress=stress))
    assert g["borrowing_base"].tolist() == pytest.approx([80_000, 75_000, 70_000])


def test_weekly_redetermination_holds_reported_base():
    cfg = _cfg(3, dynamic=True, redetermination_days=5)
    f = ledger_frame(run_treasury(_book(contracts=0), cfg, _market([100, 120, 130, 140])))
    assert f["reported_borrowing_base"].tolist() == pytest.approx([80_000] * 4)
    assert f["borrowing_base"].iloc[3] == pytest.approx(112_000)


def test_receivable_ineligible_past_aging_limit():
    """Invoice 91 days past due is ineligible from the start; 85 days past due becomes ineligible after
    6 calendar days (day 3 = Wed 25th is 5 days after the start, day 4 = Thu 26th is 6 days)."""
    inv = [Invoice(counterparty_id="A", amount=10_000, days_past_due=91, due_in_days=0),
           Invoice(counterparty_id="A", amount=10_000, days_past_due=85, due_in_days=0)]
    cp = [Counterparty(id="A", name="A", rating="BBB", expected_delay_days=300)]
    f = ledger_frame(run_treasury(_book(qty=0, contracts=0, invoices=inv, cps=cp), _cfg(4, drawn=0),
                                  _market([100] * 5)))
    assert f["eligible_receivables"].tolist() == pytest.approx([10_000, 10_000, 10_000, 10_000, 0])


def test_vectorized_cap_solver_matches_legacy_solver():
    rng = np.random.default_rng(0)
    for _ in range(200):
        c = int(rng.integers(1, 8))
        a = rng.uniform(0, 50, size=c)
        inv = float(rng.uniform(0, 100))
        cap = float(rng.choice([0.1, 0.15, 0.2, 0.3]))
        legacy, _, _ = solve_concentration_cap_fixed_point(inv, a.tolist(), cap)
        new, _ = cap_solve_matrix(np.array([inv]), a[None, :], cap)
        assert new[0] == pytest.approx(legacy, abs=1e-6)


def test_sale_moves_collateral_from_inventory_to_receivables_once():
    """Selling 400 MT at market (100) to A: inventory value falls by 40,000, receivables rise by 40,000;
    net worth unchanged; no asset counted in both."""
    trades = [PhysicalTrade(day=1, instrument_id=X, side="sell", quantity=400, payment_days=30, counterparty_id="A")]
    f = ledger_frame(run_treasury(_book(contracts=0, trades=trades), _cfg(1), _market([100, 100])))
    assert f["inventory_value"].tolist() == pytest.approx([100_000, 60_000])
    assert f["receivables"].tolist() == pytest.approx([0, 40_000])
    assert f["networth"].iloc[1] == pytest.approx(f["networth"].iloc[0])
    assert f["inventory_contribution"].iloc[1] == pytest.approx(60_000 * 0.8)
    assert f["receivable_contribution"].iloc[1] == pytest.approx(40_000 * 0.85)


def test_purchase_with_payment_terms_and_transit():
    """Buy 200 MT at market 100 on day 1, paid 2 calendar days later (Wed 25th = day 3), 1 day in transit."""
    trades = [PhysicalTrade(day=1, instrument_id=X, side="buy", quantity=200, payment_days=2, transit_days=1)]
    f = ledger_frame(run_treasury(_book(contracts=0, trades=trades), _cfg(3), _market([100] * 4)))
    assert f["payables"].tolist() == pytest.approx([0, 20_000, 20_000, 0])
    assert f["supplier_payments"].tolist() == pytest.approx([0, 0, 0, 20_000])
    assert f["inventory_eligible_value"].tolist() == pytest.approx([100_000, 100_000, 120_000, 120_000])


# --------------------------------------------------------------------------- invariants

def test_inventory_appreciation_does_not_create_cash():
    f = ledger_frame(run_treasury(_book(contracts=0), _cfg(2), _market([100, 150, 200])))
    assert (f["cash"] == 20_000).all()
    assert f["networth"].iloc[-1] - f["networth"].iloc[0] == pytest.approx(100_000)


def test_zero_exposure_zero_market_pnl():
    f = ledger_frame(run_treasury(_book(qty=0, contracts=0), _cfg(3), _market([100, 50, 200, 10])))
    assert f["economic_pnl"].abs().max() == 0


def test_perfect_hedge_constant_basis_zero_price_pnl():
    f = ledger_frame(run_treasury(_book(qty=1_000, contracts=-10), _cfg(3, im=0.0), _market([100, 140, 90, 120],
                                  basis=[0.02] * 4)))
    # physical 1,000 x 1.02 x 20 = 20,400; futures -10 x 100 x 20 = -20,000: the residual 400 is the
    # opening basis applied to the price change (quantity hedge is 1:1 in units, not in value)
    assert f["economic_pnl"].sum() == pytest.approx(400)
    g = ledger_frame(run_treasury(_book(qty=1_000, contracts=-10), _cfg(3, im=0.0), _market([100, 140, 90, 120])))
    assert g["economic_pnl"].abs().max() == pytest.approx(0, abs=1e-9)


def test_perfect_quantity_hedge_changing_basis_pnl_is_basis_change():
    """Basis 0 -> -5% with futures 100 -> 120: P&L = Q x F1 x delta basis = 1,000 x 120 x -0.05 = -6,000."""
    f = ledger_frame(run_treasury(_book(qty=1_000, contracts=-10), _cfg(1, im=0.0),
                                  _market([100, 120], basis=[0.0, -0.05])))
    assert f["economic_pnl"].sum() == pytest.approx(-6_000)


def test_partial_hedge_pnl():
    """Half hedged, futures +10: inventory +10,000, futures -5,000."""
    f = ledger_frame(run_treasury(_book(qty=1_000, contracts=-5), _cfg(1, im=0.0), _market([100, 110])))
    assert f["economic_pnl"].iloc[1] == pytest.approx(5_000)


def test_total_debt_never_exceeds_commitment_without_breach():
    rng = np.random.default_rng(1)
    for _ in range(20):
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.05, 15)))
        run = run_treasury(_book(qty=1_000, contracts=-8), _cfg(14, cash=1_000, commitment=90_000, drawn=60_000),
                           _market(np.concatenate([[100], px[:14]])))
        f = ledger_frame(run)
        assert (f["revolver_drawn"] <= 90_000 + 1e-6).all()
        assert (f["revolver_undrawn"] >= -1e-9).all()
        assert run.reconciliation_residual < 1e-6 and run.networth_residual < 1e-6


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        HaircutPolicy(min_haircut=0.5, max_haircut=0.4)
    with pytest.raises(ValueError):
        TreasuryBook(instruments=_inst(), physical=[PhysicalPosition(instrument_id="Y", quantity=1)])
    with pytest.raises(ValueError):
        PhysicalTrade(day=1, instrument_id=X, side="sell", quantity=1)
    with pytest.raises(ValueError):
        run_treasury(_book(contracts=0, trades=[PhysicalTrade(day=1, instrument_id=X, side="sell", quantity=5_000,
                                                              counterparty_id="A")]), _cfg(1), _market([100, 100]))
    with pytest.raises(ValueError):
        _market([100, -1])
