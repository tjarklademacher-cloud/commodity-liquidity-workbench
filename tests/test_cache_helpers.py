"""Cache wrappers, input serialization round trip, data-status labels and live-feed fallback through the cache."""

from __future__ import annotations

import pandas as pd
import pytest

import core.data_feed as data_feed
from core.scenario_analytics import run_historical_replay, run_liquidity_monte_carlo, run_shock_grid
from core.stress_engine import run_stress_sweep
from ui.cache import (
    _load_inputs,
    cached_live_market,
    cached_monte_carlo,
    cached_sample_market,
    cached_scenario_suite,
    market_data_status,
    model_inputs_json,
)


def test_model_inputs_round_trip(positions, receivables, terms, snapshot):
    inputs = model_inputs_json(positions, receivables, terms, snapshot)
    assert all(isinstance(x, str) for x in inputs)
    p, r, t, s = _load_inputs(*inputs)
    assert p == positions
    assert r == receivables
    assert t == terms
    assert s == dict(snapshot)
    assert model_inputs_json(p, r, t, s) == inputs


def test_cached_scenario_suite_matches_direct_engines(market, positions, receivables, terms):
    inputs = model_inputs_json(positions, receivables, terms, market.snapshot)
    sweep, grid, replay = cached_scenario_suite(*inputs, market.prices_df, market.vol_df)
    direct_sweep = run_stress_sweep(positions, receivables, market.snapshot, terms)
    assert sweep.regime_counts == direct_sweep.regime_counts
    assert sweep.upside_squeeze_breakeven_pct == pytest.approx(direct_sweep.upside_squeeze_breakeven_pct)
    assert grid.net_buffer == run_shock_grid(positions, receivables, market.snapshot, terms).net_buffer
    direct_replay = run_historical_replay(positions, receivables, market.prices_df, market.vol_df, terms)
    pd.testing.assert_frame_equal(replay.frame, direct_replay.frame)
    again = cached_scenario_suite(*inputs, market.prices_df, market.vol_df)
    assert again[0].regime_counts == sweep.regime_counts


def test_cached_scenario_suite_handles_short_history(market, positions, receivables, terms):
    short = market.prices_df.iloc[-15:]
    inputs = model_inputs_json(positions, receivables, terms, market.snapshot)
    _, _, replay = cached_scenario_suite(*inputs, short, data_feed.compute_realized_volatility(short))
    assert replay is None


def test_cached_monte_carlo_matches_direct(market, positions, receivables, terms):
    inputs = model_inputs_json(positions, receivables, terms, market.snapshot)
    mc = cached_monte_carlo(*inputs, market.prices_df, 5, 1000, True)
    direct = run_liquidity_monte_carlo(positions, receivables, market.snapshot, terms, market.prices_df,
                                       horizon_days=5, n_sims=1000, winsorize_sigma=5.0)
    assert mc.net_buffer_samples == direct.net_buffer_samples
    assert mc.p05_net_buffer == direct.p05_net_buffer


def test_market_data_status_labels(market):
    assert market_data_status(market) == ("Bundled sample data", "neutral")
    live = market.model_copy(update={"source_label": "Live Yahoo Finance (yfinance)"})
    assert market_data_status(live) == ("Live market data", "neutral")
    custom = market.model_copy(update={"source_label": "Custom CSV Upload"})
    assert market_data_status(custom)[0] == "Uploaded CSV data"
    fallback = market.model_copy(update={"used_fallback": True, "source_label": "Calibrated Sample (Live Fallback)"})
    assert market_data_status(fallback) == ("Bundled data (live feed unavailable)", "caution")


def test_cached_live_market_falls_back_offline(monkeypatch):
    def offline(*args, **kwargs):
        raise TimeoutError("simulated timeout")

    monkeypatch.setattr(data_feed.yf, "download", offline)
    cached_live_market.clear()
    result = cached_live_market("1y", True)
    cached_live_market.clear()
    assert result.used_fallback
    assert "simulated timeout" in result.status_message
    assert market_data_status(result)[1] == "caution"
    assert result.snapshot["GC=F"].spot_price == pytest.approx(2710.0)


def test_live_loader_passes_timeout(monkeypatch, sample_prices):
    seen = {}

    def fake_download(*args, **kwargs):
        seen.update(kwargs)
        cols = pd.MultiIndex.from_product([["Close"], list(sample_prices.columns)], names=["Price", "Ticker"])
        return pd.DataFrame(sample_prices.to_numpy(), index=sample_prices.index, columns=cols)

    monkeypatch.setattr(data_feed.yf, "download", fake_download)
    result = data_feed.load_live_market_data()
    assert not result.used_fallback
    assert seen["timeout"] == data_feed.LIVE_TIMEOUT_SECONDS
    assert seen["auto_adjust"] is True and seen["threads"] is False


def test_cached_sample_market_is_bundled():
    result = cached_sample_market(True)
    assert not result.used_fallback
    assert result.as_of_date == "2026-11-20"
