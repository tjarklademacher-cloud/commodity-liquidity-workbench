"""Advanced model: margin engine, basis risk, reconciliation, hedge Pareto, calibration, daily cash-flow
ledger (hand-calculable cases), cash-flow Monte Carlo, scenario comparison and saved configurations."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from core.advanced_scenario import (
    MARGIN_PRESETS,
    auto_im_multipliers,
    buffer_sensitivities,
    default_advanced_settings,
    legacy_equivalent_scenario,
    neutral_settings,
    reconcile_to_legacy,
    run_advanced_scenario,
)
from core.advanced_schemas import (
    AdvancedSettings,
    CashflowSettings,
    MarginProfile,
    MarginSettings,
    ScenarioSpec,
)
from core.basis_risk import BASIS_PRESETS, default_basis_assumptions, economic_pnl, scenario_prices
from core.cashflow_engine import (
    collection_schedule,
    deterministic_paths,
    ledger_frame,
    run_cashflow_ledger,
    run_cashflow_monte_carlo,
)
from core.hedge_optimizer import run_hedge_frontier
from core.margin_engine import (
    contracts_for_hedge,
    default_margin_settings,
    im_change,
    initial_margin_required,
    signed_short_contracts,
    variation_margin,
)
from core.market_calibration import (
    CALIBRATION_WINDOWS,
    backtest_margin_forecast,
    calibrate,
    generate_factor_sample,
    nearest_psd,
    percentile_interval,
    simulate_daily_log_returns,
)
from core.scenario_comparison import (
    ConfigError,
    ScenarioState,
    WorkbenchConfig,
    compare_scenarios,
    dump_config,
    load_config,
)
from core.schemas import (
    REQUIRED_TICKERS,
    FacilityTerms,
    InventoryPosition,
    MarketQuote,
    ReceivableItem,
)
from core.stress_engine import run_stress_scenario

M = 0.01e6


# ---- Margin engine ------------------------------------------------------------------------------------------

def test_contract_conversion_policies():
    assert contracts_for_hedge(19_200.0, 27.5, "continuous") == pytest.approx((698.1818, 698.1818), abs=1e-3)
    assert contracts_for_hedge(19_200.0, 27.5, "nearest")[1] == 698.0
    assert contracts_for_hedge(19_200.0, 27.5, "floor")[1] == 698.0
    assert contracts_for_hedge(19_200.0, 27.5, "ceil")[1] == 699.0
    assert contracts_for_hedge(9_000.0, 100.0, "ceil")[1] == 90.0
    assert contracts_for_hedge(0.0, 100.0, "nearest") == (0.0, 0.0)


def test_signed_contracts_and_residual():
    s = default_margin_settings()
    signed, exact, residual = signed_short_contracts(19_200.0, "LBR=F", s)
    assert signed == -698.0 and exact == pytest.approx(698.1818, abs=1e-3)
    assert residual == pytest.approx(19_200.0 - 698 * 27.5)


def test_variation_margin_signs_and_units():
    assert variation_margin(-10, 100.0, 50.0) == -50_000.0
    assert variation_margin(10, 100.0, 50.0) == 50_000.0
    assert variation_margin(-10, 100.0, 0.0) == 0.0
    s = default_margin_settings()
    signed, _, _ = signed_short_contracts(3_825_000.0, "KC=F", s)
    assert variation_margin(signed, 37_500.0, 0.10) == pytest.approx(-102 * 37_500 * 0.10)


def test_initial_margin_order_and_changes():
    prof = MarginProfile(ticker="CC=F", im_per_contract=10_000.0, fixed_addon_usd=5_000.0)
    assert initial_margin_required(-10, prof, 1.5) == 10 * 10_000 * 1.5 + 5_000
    assert initial_margin_required(0, prof, 2.0) == 0.0
    assert im_change(100.0, 150.0, True) == (50.0, 0.0)
    assert im_change(150.0, 100.0, True) == (0.0, 50.0)
    assert im_change(150.0, 100.0, False) == (0.0, 0.0)


def test_default_margin_profiles_are_labelled_illustrative():
    s = default_margin_settings()
    for t in REQUIRED_TICKERS:
        p = s.profiles[t]
        assert p.assumption_type == "illustrative" and "Illustrative" in p.source
        assert 0 < p.maintenance_per_contract <= p.im_per_contract


# ---- Advanced scenario and reconciliation -------------------------------------------------------------------

def test_neutral_settings_reproduce_legacy(default_positions, default_receivables, snapshot, default_terms):
    neutral = neutral_settings(default_advanced_settings())
    rng = np.random.default_rng(9)
    for _ in range(40):
        shocks = dict(zip(REQUIRED_TICKERS, rng.uniform(-0.3, 0.3, len(REQUIRED_TICKERS))))
        adv = run_advanced_scenario(default_positions, default_receivables, snapshot, default_terms, neutral,
                                    legacy_equivalent_scenario(shocks))
        leg = run_stress_scenario(default_positions, default_receivables, snapshot, default_terms,
                                  per_commodity_shocks=shocks)
        assert adv.net_liquidity_buffer == pytest.approx(leg.net_liquidity_buffer, abs=1e-4)
        assert str(adv.regime) == str(leg.regime)
    plus14 = run_advanced_scenario(default_positions, default_receivables, snapshot, default_terms, neutral,
                                   legacy_equivalent_scenario({t: 0.14 for t in REQUIRED_TICKERS}))
    assert plus14.net_liquidity_buffer == pytest.approx(-1.79e6, abs=M)


def test_reconciliation_sums_to_total(default_positions, default_receivables, snapshot, default_terms):
    settings = default_advanced_settings()
    settings = AdvancedSettings(margin=settings.margin, basis={
        t: b.model_copy(update={"opening_basis_pct": 0.02}) for t, b in settings.basis.items()})
    shocks = {t: 0.14 for t in REQUIRED_TICKERS}
    spec = ScenarioSpec(futures_shocks=shocks, im_multipliers=auto_im_multipliers(shocks, default_terms),
                        basis_shocks={"CC=F": -0.10}, liquidation_discounts={"GC=F": 0.05}, margin_line_cut_pct=0.5)
    terms = FacilityTerms.model_validate({**default_terms.model_dump(), "margin_line_limit": 4e6})
    steps = reconcile_to_legacy(default_positions, default_receivables, snapshot, terms, settings, spec)
    full = run_advanced_scenario(default_positions, default_receivables, snapshot, terms, settings, spec)
    legacy = run_stress_scenario(default_positions, default_receivables, snapshot, terms, shock_pct=0.14)
    assert steps[0].net_liquidity_buffer == pytest.approx(legacy.net_liquidity_buffer, abs=1e-4)
    assert steps[-1].net_liquidity_buffer == pytest.approx(full.net_liquidity_buffer, abs=1e-4)
    assert sum(s.change for s in steps) == pytest.approx(full.net_liquidity_buffer - legacy.net_liquidity_buffer,
                                                       abs=1e-4)


def test_no_double_counting_of_posted_im(default_positions, default_receivables, snapshot, default_terms):
    s = default_advanced_settings()
    flat = run_advanced_scenario(default_positions, default_receivables, snapshot, default_terms, s, ScenarioSpec())
    assert flat.im_posted_opening == pytest.approx(flat.im_required)
    assert flat.im_additional == 0.0 and flat.im_released == 0.0
    assert flat.net_liquidity_buffer == pytest.approx(default_terms.unrestricted_cash + flat.shocked_headroom)
    hike = run_advanced_scenario(default_positions, default_receivables, snapshot, default_terms, s,
                                 ScenarioSpec(im_multipliers={t: 1.5 for t in REQUIRED_TICKERS}))
    assert hike.im_additional == pytest.approx(0.5 * flat.im_required)
    assert hike.net_liquidity_buffer == pytest.approx(flat.net_liquidity_buffer - 0.5 * flat.im_required)
    cut = run_advanced_scenario(default_positions, default_receivables, snapshot, default_terms, s,
                                ScenarioSpec(im_multipliers={t: 0.5 for t in REQUIRED_TICKERS}))
    assert cut.im_released == pytest.approx(0.5 * flat.im_required)


def test_cocoa_rally_with_margin_hike(default_positions, default_receivables, snapshot, default_terms):
    s = default_advanced_settings()
    spec = ScenarioSpec(**{k: v for k, v in MARGIN_PRESETS["Cocoa squeeze (+25% cocoa, IM x1.6)"].items()})
    r = run_advanced_scenario(default_positions, default_receivables, snapshot, default_terms, s, spec)
    cc = next(line for line in r.lines if line.ticker == "CC=F")
    assert r.vm_payable == pytest.approx(261 * 10 * 8150 * 0.25)
    assert r.im_additional == pytest.approx(261 * cc.im_per_contract * 0.6)
    assert r.margin_cash_need == pytest.approx(r.vm_payable + r.im_additional)


def test_margin_line_capacity_and_financeability(default_positions, default_receivables, snapshot, default_terms):
    s = default_advanced_settings()
    terms = FacilityTerms.model_validate({**default_terms.model_dump(), "margin_line_limit": 3e6})
    spec = ScenarioSpec(futures_shocks={t: 0.14 for t in REQUIRED_TICKERS})
    r = run_advanced_scenario(default_positions, default_receivables, snapshot, terms, s, spec)
    assert r.margin_line_draw == pytest.approx(3e6)
    assert r.margin_funded_from_cash == pytest.approx(r.margin_cash_need - 3e6)
    cut = run_advanced_scenario(default_positions, default_receivables, snapshot, terms, s,
                                spec.model_copy(update={"margin_line_cut_pct": 0.5}))
    assert cut.margin_line_draw == pytest.approx(1.5e6)
    nofin = AdvancedSettings(margin=MarginSettings(profiles={
        t: p.model_copy(update={"financeable": False}) for t, p in s.margin.profiles.items()}), basis=s.basis)
    assert run_advanced_scenario(default_positions, default_receivables, snapshot, terms, nofin,
                                 spec).margin_line_draw == 0.0
    crash = run_advanced_scenario(default_positions, default_receivables, snapshot,
                                  FacilityTerms.model_validate({**terms.model_dump(), "margin_line_limit": 1e9}), s,
                                  ScenarioSpec(futures_shocks={t: -0.2 for t in REQUIRED_TICKERS}))
    assert crash.bb_deficiency_cure > 0 and crash.margin_line_draw == 0.0


# ---- Basis risk ---------------------------------------------------------------------------------------------

def test_basis_prices_and_pnl_identity():
    f0, f1, p0, p1, coll = scenario_prices(100.0, 0.10, 0.02, -0.05, 0.08)
    assert (f0, f1) == (100.0, pytest.approx(110.0))
    assert p0 == pytest.approx(102.0) and p1 == pytest.approx(110.0 * 0.97)
    assert coll == pytest.approx(p1 * 0.92)
    pnl = economic_pnl(1000.0, -10.0, 100.0, f0, f1, p0, p1, 0.02, 0.08)
    assert pnl["combined_pnl"] == pytest.approx(pnl["physical_pnl"] + pnl["futures_pnl"])
    assert pnl["combined_pnl"] == pytest.approx(pnl["residual_price_pnl"] + pnl["basis_pnl"])
    assert pnl["futures_pnl"] == pytest.approx(-10 * 100 * 10.0)


def test_fully_hedged_book_still_exposed_to_basis(default_receivables, snapshot, default_terms):
    pos = [InventoryPosition(ticker="GC=F", physical_qty=10_000.0, hedged_qty=10_000.0)]
    s = default_advanced_settings()
    stable = run_advanced_scenario(pos, default_receivables, snapshot, default_terms, s,
                                   ScenarioSpec(futures_shocks={"GC=F": 0.10}))
    weak = run_advanced_scenario(pos, default_receivables, snapshot, default_terms, s,
                                 ScenarioSpec(futures_shocks={"GC=F": 0.10}, basis_shocks={"GC=F": -0.05}))
    assert stable.combined_pnl == pytest.approx(0.0, abs=1e-6)
    assert weak.combined_pnl == pytest.approx(10_000 * 2710 * 1.10 * -0.05)
    assert weak.vm_payable == pytest.approx(stable.vm_payable)
    assert weak.shocked_bb < stable.shocked_bb


def test_basis_presets_are_valid_scenarios():
    for preset in BASIS_PRESETS.values():
        ScenarioSpec(**preset)
    assert all(b.opening_basis_pct == 0.0 for b in default_basis_assumptions().values())


# ---- Hedge frontier multi-objective ---------------------------------------------------------------------------

def test_hedge_frontier_pareto(default_positions, default_receivables, snapshot, default_terms):
    h = run_hedge_frontier(default_positions, default_receivables, snapshot, default_terms, "ALL")
    assert len(h.pareto_efficient) == len(h.ratios) and any(h.pareto_efficient)
    for i, eff in enumerate(h.pareto_efficient):
        dominated = any(h.min_buffer[j] > h.min_buffer[i] + 1e-6 and h.crash_value_loss[j] < h.crash_value_loss[i]
                        - 1e-6 for j in range(len(h.ratios)))
        if eff:
            assert not dominated
    assert h.basis_exposure[0] == 0.0 and h.basis_exposure[-1] > 0
    assert all(a <= b + 1e-6 for a, b in zip(h.max_margin_need, h.max_margin_need[1:]))


# ---- Calibration and simulation -----------------------------------------------------------------------------

def test_calibration_windows_and_coverage(market):
    for label, w in CALIBRATION_WINDOWS.items():
        c = calibrate(market.prices_df, label)
        assert c.n_obs == (w if w else len(market.prices_df) - 1)
        assert not c.psd_repaired and all(v == 1.0 for v in c.coverage.values())
    short = calibrate(market.prices_df.iloc[-10:], "250 days")
    assert short.used_reference_vols and "reference" in short.note


def test_psd_repair_reports_and_fixes():
    bad = np.array([[1.0, 0.99, 0.0], [0.99, 1.0, 0.99], [0.0, 0.99, 1.0]])
    fixed, repaired, min_eig = nearest_psd(bad)
    assert repaired and min_eig < 0
    assert np.linalg.eigvalsh(fixed).min() >= 0
    good, rep2, _ = nearest_psd(np.eye(3))
    assert not rep2 and np.allclose(good, np.eye(3))


@pytest.mark.parametrize("method", ["gaussian", "bootstrap", "stress"])
def test_simulation_reproducible_and_scaled(market, method):
    c = calibrate(market.prices_df, "250 days")
    a = simulate_daily_log_returns(c, market.prices_df, method, 10, 2000, 5)
    b = simulate_daily_log_returns(c, market.prices_df, method, 10, 2000, 5)
    assert np.array_equal(a, b) and a.shape == (2000, 10, len(REQUIRED_TICKERS))
    vol = a.std(axis=(0, 1)) * np.sqrt(252)
    target = np.array([c.annual_vols[t] for t in REQUIRED_TICKERS])
    scale = 1.5 if method == "stress" else 1.0
    assert np.allclose(vol, target * scale, rtol=0.08)


def test_factor_sample_is_correlated_and_matches_targets():
    f = generate_factor_sample()
    assert f.index[-1] == pd.Timestamp("2026-11-20") and f["GC=F"].iloc[-1] == pytest.approx(2710.0)
    c = calibrate(f, "Maximum")
    assert c.correlation["GC=F"]["HG=F"] > 0.4 and c.correlation["CC=F"]["KC=F"] > 0.4
    with pytest.raises(ValueError):
        generate_factor_sample(market_loading=0.6, sector_loading=0.5)


def test_percentile_interval_brackets_quantile():
    x = np.random.default_rng(1).normal(size=5000)
    lo, hi = percentile_interval(x, 0.05)
    assert lo <= np.quantile(x, 0.05) <= hi and hi - lo < 0.2


def test_backtest_runs_on_sample(market, default_positions):
    b = backtest_margin_forecast(market.prices_df, default_positions, market.snapshot)
    assert b.n_forecasts > 100 and 0.0 <= b.breach_rate <= 1.0 and b.expected_rate == pytest.approx(0.05)


# ---- Daily cash-flow ledger: hand-calculable cases ---------------------------------------------------------

def _quote(price: float) -> dict[str, MarketQuote]:
    q = MarketQuote(ticker="GC=F", spot_price=price, prev_price=price, ret_1d=0.0, ret_30d=0.0,
                    realized_vol_30d=0.15, vol_30d_min=0.15, vol_30d_max=0.15, as_of_date="2026-11-20")
    return {t: (q if t == "GC=F" else q.model_copy(update={"ticker": t})) for t in REQUIRED_TICKERS}


def _gold_case(path, **cf_kw):
    """1,000 oz gold fully hedged (10 contracts), IM 5,000/contract, no receivables, limit 5M, drawn 1.5M,
    cash 100k, zero rates unless given. Haircut 14% (vol at reference, fully hedged)."""
    pos = [InventoryPosition(ticker="GC=F", physical_qty=1000.0, hedged_qty=1000.0)]
    terms = FacilityTerms(facility_limit=5e6, drawn_revolver=1.5e6, unrestricted_cash=100_000.0,
                          baseline_initial_margin=50_000.0)
    s = default_advanced_settings()
    profiles = {t: p.model_copy(update={"im_per_contract": 5_000.0, "maintenance_per_contract": None})
                for t, p in s.margin.profiles.items()}
    settings = AdvancedSettings(margin=MarginSettings(profiles=profiles), basis=s.basis)
    trigger = bool(cf_kw.pop("trigger", False))
    base = dict(horizon_days=len(path) - 1, revolver_rate=0.0, margin_line_rate=0.0, cure_grace_days=2)
    base.update(cf_kw)
    cf = CashflowSettings(**base)
    futures = np.tile(np.array(path, dtype=float)[None, :, None], (1, 1, len(REQUIRED_TICKERS)))
    run = run_cashflow_ledger(pos, [], _quote(path[0]), terms, settings, cf, futures, apply_trigger_rule=trigger)
    return run, ledger_frame(run)


def test_ledger_flat_prices_zero_flows():
    run, df = _gold_case([2000.0, 2000.0, 2000.0, 2000.0])
    assert (df["cash"] == 100_000.0).all() and (df["revolver_drawn"] == 1.5e6).all()
    assert df["shortfall"].sum() == 0 and run.first_shortfall_day[0] == 0
    assert df["borrowing_base"].iloc[0] == pytest.approx(1000 * 2000 * 0.86)


def test_ledger_rally_with_delayed_funding_three_days():
    run, df = _gold_case([2000.0, 2200.0, 2200.0, 2000.0], revolver_delay_days=1)
    # Day 1: VM -200k, cash -100k, revolver request 100k (arrives day 2): unmet 100k
    assert df.loc[1, "vm_paid"] == pytest.approx(200_000.0)
    assert df.loc[1, "cash"] == pytest.approx(-100_000.0)
    assert df.loc[1, "revolver_requested"] == pytest.approx(100_000.0)
    assert df.loc[1, "shortfall"] == pytest.approx(100_000.0)
    assert run.first_shortfall_day[0] == 1
    # Day 2: funding arrives, debt 1.6M, cash 0, no shortfall
    assert df.loc[2, "revolver_arrival"] == pytest.approx(100_000.0)
    assert df.loc[2, "revolver_drawn"] == pytest.approx(1.6e6) and df.loc[2, "cash"] == pytest.approx(0.0)
    assert df.loc[2, "shortfall"] == 0.0
    # Day 3: VM +200k received; surplus above opening cash (100k) sweeps to the revolver
    assert df.loc[3, "vm_received"] == pytest.approx(200_000.0)
    assert df.loc[3, "revolver_repaid"] == pytest.approx(100_000.0)
    assert df.loc[3, "cash"] == pytest.approx(100_000.0) and df.loc[3, "revolver_drawn"] == pytest.approx(1.5e6)
    assert run.max_shortfall[0] == pytest.approx(100_000.0)


def test_ledger_same_day_funding_avoids_shortfall():
    run, df = _gold_case([2000.0, 2200.0, 2200.0, 2000.0], revolver_delay_days=0)
    assert run.first_shortfall_day[0] == 0 and df["shortfall"].sum() == 0
    assert df.loc[1, "revolver_drawn"] == pytest.approx(1.6e6) and df.loc[1, "cash"] == pytest.approx(0.0)


def test_ledger_financing_capacity_exhausted():
    _, df = _gold_case([2000.0, 2600.0, 2600.0], revolver_delay_days=0)
    # Day 1: VM -600k; cash -500k; availability = min(5M, 1000*2600*0.86 = 2.236M) - 1.5M = 736k: fully funded
    assert df.loc[1, "revolver_requested"] == pytest.approx(500_000.0)
    assert df.loc[1, "shortfall"] == 0.0
    # Day 2 jump 2,050 -> 5,000: VM -2.95M, cash 50k - 2.95M = -2.90M; availability 1000*5000*0.86 - 1.5M
    # = 2.80M, so 2.80M is drawn and 100k stays unpaid (VM grows 1:1 with price, collateral only 0.86:1)
    _, df3 = _gold_case([2000.0, 2050.0, 5000.0], revolver_delay_days=0)
    assert df3.loc[2, "vm_paid"] == pytest.approx(2_950_000.0)
    assert df3.loc[2, "revolver_requested"] == pytest.approx(2_800_000.0)
    assert df3.loc[2, "shortfall"] == pytest.approx(100_000.0)


def test_ledger_im_increase_and_release():
    _, df = _gold_case([2000.0, 2200.0, 2000.0], revolver_delay_days=0, trigger=True)
    # +10% reaches the default 10% trigger: IM 50k -> 60k (x1.2), posted 10k; persists after the move back
    assert df.loc[0, "im_required"] == pytest.approx(50_000.0)
    assert df.loc[1, "im_posted"] == pytest.approx(10_000.0) and df.loc[1, "im_required"] == pytest.approx(60_000.0)
    assert df.loc[2, "im_released"] == 0.0


def test_ledger_interest_day_count():
    run, df = _gold_case([2000.0, 2000.0, 2000.0], revolver_rate=0.072, day_count=360)
    assert df.loc[1, "interest"] == pytest.approx(1.5e6 * 0.072 / 360)
    assert run.total_interest[0] == pytest.approx(2 * 1.5e6 * 0.072 / 360)


def test_ledger_deficiency_cure_after_grace():
    _, df = _gold_case([2000.0, 1500.0, 1500.0, 1500.0], sweep_surplus_to_revolver=False, cure_grace_days=2)
    bb = 1000 * 1500 * 0.86
    assert df.loc[1, "deficiency"] == pytest.approx(1.5e6 - bb)
    assert df.loc[1, "cure_paid"] == 0.0 and df.loc[2, "cure_paid"] == 0.0
    assert df.loc[3, "cure_paid"] == pytest.approx(1.5e6 - bb)
    assert df.loc[3, "revolver_drawn"] == pytest.approx(bb)


def test_ledger_receivable_collection_leaves_borrowing_base(snapshot):
    pos = [InventoryPosition(ticker="GC=F", physical_qty=1000.0, hedged_qty=1000.0)]
    rec = [ReceivableItem(buyer_name="Buyer A", commodity="Gold", gross_amount=200_000.0, days_past_due=10,
                          is_investment_grade=True)]
    terms = FacilityTerms(facility_limit=5e6, drawn_revolver=1.5e6, unrestricted_cash=100_000.0,
                          concentration_cap_pct=1.0)
    cf = CashflowSettings(horizon_days=3, overdue_collection_delay_days=2, revolver_rate=0.0,
                          sweep_surplus_to_revolver=False)
    s = default_advanced_settings()
    assert collection_schedule(rec, terms, cf) == [2]
    futures = np.tile(np.full(4, 2000.0)[None, :, None], (1, 1, len(REQUIRED_TICKERS)))
    run = run_cashflow_ledger(pos, rec, _quote(2000.0), terms, s, cf, futures, apply_trigger_rule=False)
    df = ledger_frame(run)
    assert df.loc[2, "collections"] == pytest.approx(200_000.0)
    assert df.loc[2, "cash"] - df.loc[1, "cash"] == pytest.approx(200_000.0)
    assert df.loc[1, "borrowing_base"] - df.loc[2, "borrowing_base"] == pytest.approx(200_000.0 * 0.85)
    skip = CashflowSettings(horizon_days=3, overdue_collection_delay_days=2, noncollected_buyers=["Buyer A"])
    assert collection_schedule(rec, terms, skip) == [None]


def test_ledger_cash_and_debt_reconcile(default_positions, default_receivables, snapshot, default_terms):
    s = default_advanced_settings()
    cf = CashflowSettings()
    fp, bp = deterministic_paths(snapshot, s, ScenarioSpec(futures_shocks={t: 0.14 for t in REQUIRED_TICKERS}), cf)
    df = ledger_frame(run_cashflow_ledger(default_positions, default_receivables, snapshot, default_terms, s, cf,
                                          fp, bp))
    inflow = df[["vm_received", "im_released", "collections", "revolver_arrival", "margin_line_arrival"]].sum(axis=1)
    outflow = df[["vm_paid", "im_posted", "interest", "cure_paid", "margin_line_repaid", "revolver_repaid"]].sum(axis=1)
    same_day = df["revolver_requested"] * (cf.revolver_delay_days == 0) + df["margin_line_requested"] * (
        cf.margin_line_delay_days == 0)
    expected_cash = default_terms.unrestricted_cash + (inflow - outflow + same_day).iloc[1:].cumsum()
    assert np.allclose(df["cash"].iloc[1:], expected_cash, atol=1e-3)
    debt = default_terms.drawn_revolver + (df["revolver_arrival"] - df["cure_paid"] - df["revolver_repaid"]
                                           ).iloc[1:].cumsum()
    assert np.allclose(df["revolver_drawn"].iloc[1:], debt, atol=1e-3)
    assert (df["revolver_drawn"] <= default_terms.facility_limit + 1e-6).all()


def test_cashflow_monte_carlo(market, default_positions, default_receivables, default_terms):
    s = default_advanced_settings()
    cf = CashflowSettings(horizon_days=10)
    cal = calibrate(market.prices_df, "250 days")
    a = run_cashflow_monte_carlo(default_positions, default_receivables, market.snapshot, default_terms, s, cf, cal,
                                 market.prices_df, n_paths=500)
    b = run_cashflow_monte_carlo(default_positions, default_receivables, market.snapshot, default_terms, s, cf, cal,
                                 market.prices_df, n_paths=500)
    assert a.prob_shortfall_by_day == b.prob_shortfall_by_day
    assert all(x <= y + 1e-12 for x, y in zip(a.prob_shortfall_by_day, a.prob_shortfall_by_day[1:]))
    assert a.prob_shortfall_by_day[-1] == pytest.approx(a.prob_any_shortfall)
    assert sum(a.first_shortfall_counts) / 500 == pytest.approx(a.prob_any_shortfall)
    assert a.reserve_99 >= a.reserve_95 >= 0


# ---- Scenario comparison and saved configurations ----------------------------------------------------------

def test_compare_scenarios_legacy_and_advanced(default_positions, default_receivables, snapshot, default_terms):
    s = default_advanced_settings()
    specs = [ScenarioSpec(name="Baseline"), ScenarioSpec(name="Active",
                                                         futures_shocks={t: 0.14 for t in REQUIRED_TICKERS})]
    legacy = compare_scenarios(default_positions, default_receivables, snapshot, default_terms, s, "legacy", specs)
    assert legacy[0].metrics["net_liquidity_buffer"] == pytest.approx(12.50e6, abs=M)
    assert legacy[1].metrics["net_liquidity_buffer"] == pytest.approx(-1.79e6, abs=M)
    assert legacy[1].metrics["funding_shortfall"] is None
    adv = compare_scenarios(default_positions, default_receivables, snapshot, default_terms, s, "advanced", specs,
                            CashflowSettings())
    assert adv[0].metrics["funding_shortfall"] == 0.0
    assert adv[1].metrics["funding_shortfall"] > 0


def _config(default_positions, default_receivables, default_terms) -> WorkbenchConfig:
    s = default_advanced_settings()
    return WorkbenchConfig(positions=default_positions, receivables=default_receivables, terms=default_terms,
                           margin=s.margin, basis=s.basis, scenario=ScenarioState(parallel_shock_pct=14))


def test_config_round_trip(default_positions, default_receivables, default_terms):
    cfg = _config(default_positions, default_receivables, default_terms)
    back = load_config(dump_config(cfg).encode())
    assert back == cfg


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: "not json {", "valid JSON"),
        (lambda d: json.dumps({**d, "kind": "other"}), "not a Commodity"),
        (lambda d: json.dumps({**d, "version": 99}), "version"),
        (lambda d: json.dumps({**d, "positions": d["positions"][:2]}), "positions"),
        (lambda d: json.dumps({**d, "terms": {**d["terms"], "facility_limit": -5}}), "terms.facility_limit"),
    ],
)
def test_config_rejects_bad_files(default_positions, default_receivables, default_terms, mutate, message):
    data = json.loads(dump_config(_config(default_positions, default_receivables, default_terms)))
    with pytest.raises(ConfigError, match=message):
        load_config(mutate(data))


def test_buffer_sensitivities_one_at_a_time(default_positions, default_receivables, snapshot, default_terms):
    """Each row is relative to the active case; cash and margin line moves are additive one for one here
    (both feed liquid resources directly and nothing else depends on them at +14%)."""
    s = default_advanced_settings()
    shocks = {t: 0.14 for t in REQUIRED_TICKERS}
    spec = ScenarioSpec(name="A", futures_shocks=shocks,
                        im_multipliers=auto_im_multipliers(shocks, default_terms, None, True))
    rows = buffer_sensitivities(default_positions, default_receivables, snapshot, default_terms, s, spec)
    base = rows[0].net_liquidity_buffer
    assert rows[0].label == "Active scenario" and rows[0].change == 0.0
    by = {r.label: r for r in rows}
    assert by["Unrestricted cash +5M"].change == pytest.approx(5e6, abs=0.01)
    assert by["Margin financing line +10M"].change == pytest.approx(10e6, abs=0.01)
    assert by["Initial margin requirement x1.5 (all commodities)"].change < 0
    for r in rows[1:]:
        assert r.change == pytest.approx(r.net_liquidity_buffer - base, abs=1e-6)
