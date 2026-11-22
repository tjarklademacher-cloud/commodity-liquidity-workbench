"""Extended analytics: vectorized engine equivalence, 2D shock grid, Monte Carlo LaR, historical replay."""

from __future__ import annotations

import numpy as np
import pytest

from core.data_feed import compute_realized_volatility
from core.scenario_analytics import (
    REGIME_CODES,
    evaluate_shock_matrix,
    run_historical_replay,
    run_liquidity_monte_carlo,
    run_shock_grid,
)
from core.schemas import REQUIRED_TICKERS, FacilityTerms
from core.stress_engine import run_stress_scenario


def test_vectorized_engine_matches_exact_engine(positions, receivables, snapshot, terms):
    rng = np.random.default_rng(11)
    shocks = rng.uniform(-0.35, 0.35, size=(150, len(REQUIRED_TICKERS)))
    shocks[0] = 0.14
    shocks[1] = -0.20
    shocks[2] = 0.0
    out = evaluate_shock_matrix(positions, receivables, snapshot, terms, shocks)
    for i, row in enumerate(shocks):
        exact = run_stress_scenario(positions, receivables, snapshot, terms,
                                    per_commodity_shocks=dict(zip(REQUIRED_TICKERS, row)))
        assert out["net_liquidity_buffer"][i] == pytest.approx(exact.net_liquidity_buffer, abs=1e-4)
        assert out["shocked_bb"][i] == pytest.approx(exact.shocked_bb, abs=1e-4)
        assert str(REGIME_CODES[out["regime_code"][i]]) == str(exact.regime)
    assert out["net_liquidity_buffer"][0] == pytest.approx(-1.22e6, abs=0.01e6)


def test_vectorized_engine_alternate_cap_tier(positions, receivables, snapshot):
    terms = FacilityTerms.model_validate({"concentration_cap_pct": 0.175, "drawn_revolver": 112e6})
    shocks = np.linspace(-0.2, 0.2, 9)[:, None] * np.ones((1, len(REQUIRED_TICKERS)))
    out = evaluate_shock_matrix(positions, receivables, snapshot, terms, shocks)
    for i, s in enumerate(shocks[:, 0]):
        exact = run_stress_scenario(positions, receivables, snapshot, terms, shock_pct=float(s))
        assert out["net_liquidity_buffer"][i] == pytest.approx(exact.net_liquidity_buffer, abs=1e-4)


def test_shock_grid_diagonal_matches_parallel_sweep(positions, receivables, snapshot, terms, sweep):
    grid = run_shock_grid(positions, receivables, snapshot, terms)
    assert len(grid.metals_axis) == 17 and grid.metals_axis[0] == -0.2 and grid.metals_axis[-1] == 0.2
    by_shock = {p.shock_pct: p for p in sweep.points}
    for k, s in enumerate(grid.metals_axis):
        if s in by_shock:
            assert grid.net_buffer[k][k] == pytest.approx(by_shock[s].net_liquidity_buffer, abs=1e-4)
            assert grid.regime[k][k] == str(by_shock[s].regime)
    assert grid.worst_net_buffer == pytest.approx(min(min(r) for r in grid.net_buffer))
    assert 0.0 < grid.squeeze_share < 1.0


def test_monte_carlo_is_reproducible_and_consistent(market, positions, receivables, terms):
    a = run_liquidity_monte_carlo(positions, receivables, market.snapshot, terms, market.prices_df, n_sims=2000)
    b = run_liquidity_monte_carlo(positions, receivables, market.snapshot, terms, market.prices_df, n_sims=2000)
    assert a.net_buffer_samples == b.net_buffer_samples
    assert sum(a.regime_counts.values()) == 2000
    assert a.prob_critical + a.prob_margin_trap + a.prob_deficiency + a.prob_healthy == pytest.approx(1.0)
    assert a.p01_net_buffer <= a.p05_net_buffer <= a.median_net_buffer
    assert a.es95_net_buffer <= a.p05_net_buffer
    for t in REQUIRED_TICKERS:
        assert a.correlation[t][t] == pytest.approx(1.0)


def test_monte_carlo_horizon_scales_vol(market, positions, receivables, terms):
    one = run_liquidity_monte_carlo(positions, receivables, market.snapshot, terms, market.prices_df,
                                    horizon_days=1, n_sims=500)
    twenty = run_liquidity_monte_carlo(positions, receivables, market.snapshot, terms, market.prices_df,
                                       horizon_days=20, n_sims=500)
    for t in REQUIRED_TICKERS:
        assert twenty.horizon_vols[t] == pytest.approx(one.horizon_vols[t] * np.sqrt(20.0), rel=1e-9)
    assert twenty.p05_net_buffer < one.p05_net_buffer


def test_monte_carlo_short_history_uses_reference_vols(market, positions, receivables, terms):
    mc = run_liquidity_monte_carlo(positions, receivables, market.snapshot, terms, market.prices_df.iloc[-8:],
                                   horizon_days=1, n_sims=300)
    assert mc.horizon_vols["CC=F"] == pytest.approx(0.30 / np.sqrt(252.0))
    assert mc.correlation["GC=F"]["CC=F"] == 0.0


def test_historical_replay_invariants(market, positions, receivables, terms, baseline_bb):
    replay = run_historical_replay(positions, receivables, market.prices_df, market.vol_df, terms)
    df = replay.frame
    assert replay.end_date == "2026-11-20"
    assert len(df) == replay.n_days > 200
    assert df["borrowing_base"].iloc[-1] == pytest.approx(baseline_bb.final_borrowing_base, rel=1e-9)
    assert (df["headroom"] >= 0.0).all() and (df["bb_deficiency_cure"] >= 0.0).all()
    assert ((df["headroom"] > 0.0) & (df["bb_deficiency_cure"] > 0.0)).sum() == 0
    gold = market.prices_df["GC=F"]
    expected_last_vm = -(9_000.0 * (gold.iloc[-1] - gold.iloc[-2]) + 4_675_000.0 * (
        market.prices_df["HG=F"].iloc[-1] - market.prices_df["HG=F"].iloc[-2]) + 19_200.0 * (
        market.prices_df["LBR=F"].iloc[-1] - market.prices_df["LBR=F"].iloc[-2]) + 2_610.0 * (
        market.prices_df["CC=F"].iloc[-1] - market.prices_df["CC=F"].iloc[-2]))
    assert df["daily_vm"].iloc[-1] == pytest.approx(expected_last_vm, rel=1e-9)
    assert replay.deficiency_days == int((df["borrowing_base"] < terms.drawn_revolver).sum())
    stressed = terms.unrestricted_cash + df["headroom"] + df["rolling_vm"] - df["bb_deficiency_cure"]
    assert np.allclose(df["stressed_buffer"], stressed)


def test_historical_replay_requires_history(market, positions, receivables, terms):
    short = market.prices_df.iloc[-15:]
    with pytest.raises(ValueError):
        run_historical_replay(positions, receivables, short, compute_realized_volatility(short), terms)
