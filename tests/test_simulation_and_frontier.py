"""Return models, liquidity metrics, validation statistics, hedge frontier, adapter cross-check, case study."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from core.advanced_scenario import default_advanced_settings
from core.advanced_schemas import CashflowSettings, ScenarioSpec
from core.cashflow_engine import deterministic_paths, run_cashflow_ledger
from core.case_study_2022 import CASE_VARIANTS, align_to_engine_calendar, fetch_ttf_2022, run_case_study
from core.data_feed import load_sample_market_data
from core.hedge_frontier import HedgeConstraints, pareto_mask, run_hedge_frontier
from core.liquidity_metrics import compute_metrics, quantile_interval, wilson_interval
from core.margin_engine import default_margin_settings
from core.model_validation import christoffersen_independence, kupiec_pof, moment_comparison, var_backtest
from core.return_models import (
    calibrate_return_model,
    ewma_covariance,
    fit_garch11,
    prepare_prices,
    repair_correlation,
    simulate_returns,
)
from core.schemas import DEFAULT_INVENTORY_POSITIONS, DEFAULT_RECEIVABLES, REQUIRED_TICKERS, FacilityTerms
from core.treasury_analytics import TreasurySettings, haircut_sensitivity, run_book_monte_carlo, run_book_scenario
from core.treasury_engine import TreasuryRun


@pytest.fixture(scope="module")
def market():
    return load_sample_market_data()


@pytest.fixture(scope="module")
def prices(market):
    return market.prices_df[list(REQUIRED_TICKERS)]


# --------------------------------------------------------------------------- return models

def test_reproducible_with_identical_seed(prices):
    for m in ("gaussian", "student_t", "bootstrap", "ewma"):
        model = calibrate_return_model(prices, m)
        a, b = simulate_returns(model, 5, 100, 7), simulate_returns(model, 5, 100, 7)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, simulate_returns(model, 5, 100, 8))


def test_student_t_covariance_is_normalized_and_fat_tailed(prices):
    model = calibrate_return_model(prices, "student_t", nu=5.0)
    r = simulate_returns(model, 1, 200_000, 1)[:, 0, :]
    cov = np.cov(r.T)
    assert np.allclose(np.diag(cov), np.diag(model.cov_daily), rtol=0.05)
    assert pd.Series(r[:, 0]).kurt() > 2.0
    with pytest.raises(ValueError):
        calibrate_return_model(prices, "student_t", nu=2.0)


def test_simulated_correlation_matches_target():
    rng = np.random.default_rng(0)
    corr = np.array([[1, 0.6, 0.2], [0.6, 1, 0.4], [0.2, 0.4, 1]])
    z = rng.multivariate_normal(np.zeros(3), corr * 1e-4, size=600)
    px = pd.DataFrame(100 * np.exp(np.cumsum(z, axis=0)), index=pd.bdate_range("2020-01-01", periods=600),
                      columns=list("ABC"))
    model = calibrate_return_model(px, "gaussian", window=None)
    sim = simulate_returns(model, 1, 100_000, 2)[:, 0, :]
    assert np.allclose(np.corrcoef(sim.T), model.corr, atol=0.01)
    assert np.allclose(model.corr, corr, atol=0.1)


def test_bootstrap_preserves_synchronized_rows(prices):
    model = calibrate_return_model(prices, "bootstrap", block_days=5)
    sim = simulate_returns(model, 10, 50, 3)
    hist = {tuple(np.round(row, 12)) for row in model.history}
    assert all(tuple(np.round(row, 12)) in hist for row in sim.reshape(-1, sim.shape[2]))


def test_correlation_repair_is_valid_and_disclosed():
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    fixed, diag = repair_correlation(bad)
    assert diag.repaired and diag.min_eigenvalue < 0
    assert np.allclose(np.diag(fixed), 1) and np.allclose(fixed, fixed.T)
    assert np.linalg.eigvalsh(fixed).min() > -1e-12


def test_ewma_and_garch_recovery():
    rng = np.random.default_rng(4)
    omega, alpha, beta = 2e-6, 0.08, 0.90
    n = 3000
    h, x = omega / (1 - alpha - beta), np.empty(n)
    for i in range(n):
        x[i] = math.sqrt(h) * rng.standard_normal()
        h = omega + alpha * x[i] ** 2 + beta * h
    fit = fit_garch11(x)
    assert fit.valid and abs(fit.alpha - alpha) < 0.04 and abs(fit.beta - beta) < 0.05
    short = fit_garch11(x[:100])
    assert not short.valid and "Not estimated" in short.message
    with pytest.raises(ValueError):
        ewma_covariance(x[:, None], 1.0)


def test_prepare_prices_handles_duplicates_invalid_and_gaps_without_dropping_extremes():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    df = pd.DataFrame({"A": [10, 11, 11.5, -1, 30, 31], "B": [5, 5, 5, 5, np.nan, 6]}, index=idx)
    out, rep = prepare_prices(df, max_fill_days=1)
    assert rep.duplicates_removed == 1 and rep.invalid_prices_removed["A"] == 1
    assert out.loc["2024-01-04", "A"] == 30  # a genuine jump is kept
    assert out.loc["2024-01-04", "B"] == 5   # one-day gap forward filled


# --------------------------------------------------------------------------- metrics and validation

def _fake_run(liq: np.ndarray, short: np.ndarray) -> TreasuryRun:
    n, h1 = liq.shape
    z = np.zeros((n, h1))
    ledger = {k: z.copy() for k in ("cash", "revolver_drawn", "permitted", "cum_vm", "im_posted", "credit_loss",
                                    "reported_borrowing_base", "receivables", "economic_pnl", "emergency_drawn")}
    ledger["permitted"] += 1.0
    ledger["available_liquidity"] = liq
    ledger["shortfall"] = short
    return TreasuryRun(pd.bdate_range("2026-11-20", periods=h1), ledger, None, ["X"], 0.0, 0.0, 0.0, [])


def test_lar_percentiles_breach_probability_and_path_maximum():
    """Four paths, baseline 100. Minimum levels 90, 70, 100, 40 -> deterioration 10, 30, 0, 60.
    LaR 75% = 75th percentile of (0, 10, 30, 60) = 37.5 (linear); ELS = mean of {60} = 60.
    Path 2 breaches on day 1 and recovers: its peak funding (25) counts though the terminal value is 0."""
    liq = np.array([[100, 95, 90], [100, 70, 80], [100, 100, 100], [100, 60, 40]], dtype=float)
    short = np.array([[0, 0, 0], [0, 25, 0], [0, 0, 0], [0, 0, 5]], dtype=float)
    m = compute_metrics(_fake_run(liq, short), 0.75)
    assert m.lar == pytest.approx(37.5)
    assert m.els == pytest.approx(60.0)
    assert m.prob_breach == pytest.approx(0.5)
    assert m.expected_unfunded == pytest.approx((25 + 5) / 4)
    assert m.survival_curve == pytest.approx([0.75, 0.5])


def test_interval_helpers():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi and hi - lo == pytest.approx(0.19, abs=0.01)
    lo, hi = quantile_interval(np.arange(1000.0), 0.5)
    assert lo < 500 < hi


def test_kupiec_and_independence():
    lr, p = kupiec_pof(1000, 50, 0.95)
    assert lr == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(1.0)
    lr, p = kupiec_pof(1000, 100, 0.95)
    assert p < 0.001
    clustered = [True] * 10 + [False] * 190
    assert christoffersen_independence(clustered)[1] < 0.01


def test_var_backtest_has_no_lookahead_and_flags_small_samples(prices):
    exposures = np.full(6, 1e6)
    res = var_backtest(prices, exposures, "gaussian", window=200, step=5, n_sims=2000)
    assert res.n_forecasts > 0 and not res.meaningful
    assert (res.frame["date"].diff().dropna() > pd.Timedelta(0)).all()


def test_moment_comparison_runs(prices):
    out = moment_comparison(prices, "student_t", n_paths=5000)
    assert set(out["asset"]) == set(REQUIRED_TICKERS)
    assert out.attrs["max_correlation_gap"] < 0.2


# --------------------------------------------------------------------------- adapter and cross-check

def test_new_engine_reproduces_existing_ledger_where_features_overlap(market):
    """No collections inside the horizon, continuous contracts, no IM trigger, zero rates: the treasury engine
    and the existing advanced ledger must give the same cash, debt and borrowing base every day."""
    snap = market.snapshot
    terms = FacilityTerms()
    settings = default_advanced_settings()
    margin = settings.margin.model_copy(update={"contract_policy": "continuous"})
    settings = settings.model_copy(update={"margin": margin})
    cf = CashflowSettings(horizon_days=10, ramp_days=5, revolver_rate=0.0, margin_line_rate=0.0,
                          overdue_collection_delay_days=30, invoice_terms_days=60)
    spec = ScenarioSpec(futures_shocks={t: 0.14 for t in REQUIRED_TICKERS})
    fut, basis = deterministic_paths(snap, settings, spec, cf)
    old = run_cashflow_ledger(DEFAULT_INVENTORY_POSITIONS, DEFAULT_RECEIVABLES, snap, terms, settings, cf, fut, basis,
                              apply_trigger_rule=False)
    ts = TreasurySettings(horizon_days=10, ramp_days=5, dynamic_haircuts=False, apply_trigger_rule=False,
                          overdue_delay_days=30, invoice_terms_days=60, drawdown_delay_days=1, cure_grace_days=2)
    new = run_book_scenario(DEFAULT_INVENTORY_POSITIONS, DEFAULT_RECEIVABLES, snap, terms, margin, ts,
                            {t: 0.14 for t in REQUIRED_TICKERS})
    np.testing.assert_allclose(new.ledger["cash"][0], old.ledger["cash"][0], atol=1e-4)
    np.testing.assert_allclose(new.ledger["revolver_drawn"][0], old.ledger["revolver_drawn"][0], atol=1e-4)
    np.testing.assert_allclose(new.ledger["borrowing_base"][0], old.ledger["borrowing_base"][0], rtol=1e-12)
    np.testing.assert_allclose(new.ledger["im_posted"][0], old.ledger["im_required"][0], atol=1e-6)


def test_default_book_scenario_reconciles_and_matches_day3_breach(market):
    ts = TreasurySettings(horizon_days=20)
    run = run_book_scenario(DEFAULT_INVENTORY_POSITIONS, DEFAULT_RECEIVABLES, market.snapshot, FacilityTerms(),
                            default_margin_settings(), ts, {t: 0.14 for t in REQUIRED_TICKERS})
    assert run.reconciliation_residual < 1e-4 and run.networth_residual < 1e-4
    first = int(np.argmax(run.ledger["shortfall"][0] > 1e-6))
    assert first == 3
    assert run.ledger["borrowing_base"][0, 0] == pytest.approx(144_942_532, abs=1)


def test_haircut_sensitivity_static_first_and_severe_is_worse(market):
    df = haircut_sensitivity(DEFAULT_INVENTORY_POSITIONS, DEFAULT_RECEIVABLES, market.snapshot, FacilityTerms(),
                             default_margin_settings(), TreasurySettings(horizon_days=20),
                             {t: 0.14 for t in REQUIRED_TICKERS})
    assert df["change_vs_static"].iloc[0] == 0
    sev = df.set_index("assumption")
    assert sev.loc["Lender tightening +15 points", "min_liquidity"] < sev.loc[
        "Static haircuts", "min_liquidity"]
    assert sev.loc["+15 points, capped at +2 per redetermination", "min_liquidity"] >= sev.loc[
        "Lender tightening +15 points", "min_liquidity"]


def test_monte_carlo_is_reproducible_and_path_dependent(prices, market):
    model = calibrate_return_model(prices, "gaussian")
    args = (DEFAULT_INVENTORY_POSITIONS, DEFAULT_RECEIVABLES, market.snapshot, FacilityTerms(),
            default_margin_settings(), TreasurySettings(horizon_days=10), model)
    a = run_book_monte_carlo(*args, n_paths=200, seed=5)
    b = run_book_monte_carlo(*args, n_paths=200, seed=5)
    assert a.metrics == b.metrics
    assert a.metrics.funding_p95 >= 0 and 0 <= a.metrics.prob_breach <= 1
    c = run_book_monte_carlo(*args, n_paths=200, seed=5, credit_mode="simulated", wrong_way_loading=0.5)
    assert c.credit_mode == "simulated"


def test_pareto_and_frontier_labels(prices, market):
    x = np.array([3.0, 2.0, 1.0, 2.5])
    y = np.array([1.0, 2.0, 3.0, 2.5])
    assert pareto_mask(x, y, np.ones(4, dtype=bool)).tolist() == [True, True, True, False]
    model = calibrate_return_model(prices, "gaussian")
    df = run_hedge_frontier(DEFAULT_INVENTORY_POSITIONS, DEFAULT_RECEIVABLES, market.snapshot, FacilityTerms(),
                            default_margin_settings(), TreasurySettings(horizon_days=10), model,
                            ratios=[0.0, 0.5, 1.0], n_paths=150,
                            constraints=HedgeConstraints(max_breach_probability=1.0))
    assert set(df["label"]) - {""}
    assert df.loc[df["hedge_ratio"] == 1.0, "pnl_sd"].iloc[0] < df.loc[df["hedge_ratio"] == 0.0, "pnl_sd"].iloc[0]
    labelled = df[df["label"] != ""]
    assert labelled["pareto"].all()


# --------------------------------------------------------------------------- case study (test series only)

def _test_series() -> pd.Series:
    """A made-up rising and falling path used only to exercise the code; never shown as observed data."""
    idx = pd.bdate_range("2022-01-03", "2022-12-30")
    t = np.linspace(0, np.pi, len(idx))
    return pd.Series(80 + 200 * np.sin(t) ** 2, index=idx, name="TTF=F")


def test_case_study_runs_reconciles_and_hedge_is_economically_effective():
    res = run_case_study(_test_series())
    tab = res["table"]
    assert list(tab["variant"]) == list(CASE_VARIANTS)
    assert (tab["reconciliation_residual"] < 1e-3).all()
    static = tab.iloc[0]
    assert static["peak_funding"] > 0
    assert abs(static["economic_pnl"]) < 0.1 * static["peak_cum_vm_paid"]
    with_line = tab.iloc[-1]
    assert with_line["peak_funding"] <= tab.iloc[-2]["peak_funding"]


def test_fetch_ttf_validates_vendor_data():
    def empty(**_):
        return pd.DataFrame()
    with pytest.raises(ValueError):
        fetch_ttf_2022(download=empty)
    s = _test_series()

    def fake(**_):
        return pd.DataFrame({("Close", "TTF=F"): s.to_numpy()}, index=s.index)
    out = fetch_ttf_2022(download=fake)
    assert len(out) == len(s)
    dates, px, filled = align_to_engine_calendar(out.drop(out.index[5]))
    assert filled == 1 and px[5] == px[4]
