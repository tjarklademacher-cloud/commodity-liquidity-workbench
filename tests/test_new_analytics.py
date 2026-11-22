"""Hedge-ratio frontier, liquidity runway and real crisis replays (Yahoo calls are faked; no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.crisis_replay import CRISIS_EPISODES, fetch_episode_closes, run_crisis_replay
from core.hedge_optimizer import ALL_COMMODITIES, run_hedge_frontier
from core.scenario_analytics import run_liquidity_runway
from core.schemas import REQUIRED_TICKERS, InventoryPosition
from core.stress_engine import run_stress_scenario, run_stress_sweep


def test_hedge_frontier_all(default_positions, default_receivables, snapshot, default_terms):
    res = run_hedge_frontier(default_positions, default_receivables, snapshot, default_terms, ALL_COMMODITIES)
    assert res.ratios[0] == 0.0 and res.ratios[-1] == 1.0 and len(res.ratios) == 21
    assert all(a >= b for a, b in zip(res.crash_value_loss, res.crash_value_loss[1:]))
    assert res.crash_value_loss[-1] == pytest.approx(0.0)
    assert all(a >= b - 1e-6 for a, b in zip(res.buffer_up_20, res.buffer_up_20[1:]))
    assert 0.0 <= res.recommended_ratio <= 1.0
    best = res.recommended_index
    assert res.positive_share[best] == max(res.positive_share)
    phys = sum(p.physical_qty * snapshot[p.ticker].spot_price for p in default_positions)
    hedged = sum(p.hedged_qty * snapshot[p.ticker].spot_price for p in default_positions)
    assert res.current_ratio == pytest.approx(hedged / phys)


def test_hedge_frontier_matches_direct_sweep(default_positions, default_receivables, snapshot, default_terms):
    res = run_hedge_frontier(default_positions, default_receivables, snapshot, default_terms, "CC=F")
    i = res.ratios.index(0.9)
    sweep = run_stress_sweep(default_positions, default_receivables, snapshot, default_terms)
    nlbs = [p.net_liquidity_buffer for p in sweep.points]
    assert res.current_ratio == pytest.approx(0.9)
    assert res.min_buffer[i] == pytest.approx(min(nlbs), abs=1e-3)
    assert res.positive_share[i] == pytest.approx(np.mean([v >= 0 for v in nlbs]))
    assert res.buffer_up_20[i] == pytest.approx(nlbs[-1], abs=1e-3)


def test_hedge_frontier_rejects_unknown_scope(default_positions, default_receivables, snapshot, default_terms):
    with pytest.raises(ValueError):
        run_hedge_frontier(default_positions[:2], default_receivables, snapshot, default_terms, "SB=F")


def test_runway_consistency(market, default_positions, default_receivables, default_terms):
    a = run_liquidity_runway(default_positions, default_receivables, market.snapshot, default_terms,
                             market.prices_df, horizon_days=10, n_paths=500)
    b = run_liquidity_runway(default_positions, default_receivables, market.snapshot, default_terms,
                             market.prices_df, horizon_days=10, n_paths=500)
    assert a.percentiles == b.percentiles
    assert a.days == list(range(1, 11))
    assert all(x <= y + 1e-12 for x, y in zip(a.prob_breached_by_day, a.prob_breached_by_day[1:]))
    assert a.prob_breached_by_day[-1] == pytest.approx(a.prob_breach_within_horizon)
    assert sum(a.first_breach_counts) / 500 == pytest.approx(a.prob_breach_within_horizon)
    for d in range(10):
        assert a.percentiles["p5"][d] <= a.percentiles["p50"][d] <= a.percentiles["p95"][d]


def test_runway_breaches_when_cash_is_thin(market, default_positions, default_receivables, default_terms):
    tight = default_terms.model_validate({**default_terms.model_dump(), "unrestricted_cash": 0.0,
                                          "drawn_revolver": 139_500_000.0})
    r = run_liquidity_runway(default_positions, default_receivables, market.snapshot, tight, market.prices_df,
                             horizon_days=20, n_paths=1000)
    assert r.prob_breach_within_horizon > 0.2
    assert r.median_first_breach_day is not None and 1 <= r.median_first_breach_day <= 20


def _fake_closes(start: str, end: str, tickers: list[str]) -> pd.DataFrame:
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(3)
    base = {"GC=F": 1600.0, "HG=F": 2.8, "LBR=F": 500.0, "CC=F": 2500.0, "KC=F": 120.0, "SB=F": 14.0}
    data = {t: base[t] * np.exp(np.cumsum(rng.normal(0.002, 0.02, len(idx)))) for t in tickers}
    return pd.DataFrame(data, index=idx)


def _fake_download(tickers):
    def download(*args, **kwargs):
        closes = _fake_closes(kwargs["start"], kwargs["end"], tickers)
        cols = pd.MultiIndex.from_product([["Close"], closes.columns], names=["Price", "Ticker"])
        return pd.DataFrame(closes.to_numpy(), index=closes.index, columns=cols)
    return download


def test_fetch_episode_scales_cents_and_drops_missing():
    ep = CRISIS_EPISODES["brazil_frost_2021"]
    tickers = [t for t in REQUIRED_TICKERS if t != "LBR=F"]
    closes = fetch_episode_closes(ep, download=_fake_download(tickers))
    assert list(closes.columns) == tickers
    raw = _fake_closes((pd.Timestamp(ep.start) - pd.Timedelta(days=70)).strftime("%Y-%m-%d"),
                       (pd.Timestamp(ep.end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), tickers)
    assert closes["KC=F"].iloc[0] == pytest.approx(raw["KC=F"].iloc[0] * 0.01)
    assert closes["GC=F"].iloc[0] == pytest.approx(raw["GC=F"].iloc[0])
    assert closes.index[0] < pd.Timestamp(ep.start)


def test_fetch_episode_raises_on_empty():
    with pytest.raises(ValueError):
        fetch_episode_closes(CRISIS_EPISODES["covid_2020"], download=lambda *a, **k: pd.DataFrame())


def test_crisis_replay_matches_stress_engine(market, default_positions, default_receivables, default_terms):
    ep = CRISIS_EPISODES["cocoa_2024"]
    tickers = [t for t in REQUIRED_TICKERS if t != "LBR=F"]
    closes = fetch_episode_closes(ep, download=_fake_download(tickers))
    res = run_crisis_replay(default_positions, default_receivables, market.snapshot, default_terms, closes, ep)
    assert res.missing_tickers == ["LBR=F"]
    assert (res.frame["move_LBR=F"] == 0.0).all()
    window = closes.loc[ep.start:ep.end]
    date = res.frame.index[20]
    expected_move = window.loc[date, "CC=F"] / window["CC=F"].iloc[0] - 1.0
    assert res.frame.loc[date, "move_CC=F"] == pytest.approx(expected_move)
    assert res.frame["move_GC=F"].iloc[0] == pytest.approx(0.0)
    moves = {t: float(res.frame.loc[date, f"move_{t}"]) for t in REQUIRED_TICKERS}
    direct = run_stress_scenario(default_positions, default_receivables, market.snapshot, default_terms,
                                 per_commodity_shocks=moves)
    assert res.frame.loc[date, "vm_outflow"] == pytest.approx(direct.vm_outflow, rel=1e-9)
    assert res.frame.loc[date, "vm_inflow"] == pytest.approx(direct.vm_inflow, rel=1e-9)
    assert res.worst_net_buffer == pytest.approx(res.frame["net_liquidity_buffer"].min())
    assert res.critical_days == int((res.frame["regime"] == "CRITICAL_SQUEEZE").sum())


def test_crisis_replay_single_position(market, default_receivables, default_terms):
    ep = CRISIS_EPISODES["covid_2020"]
    closes = fetch_episode_closes(ep, download=_fake_download(["GC=F"]))
    pos = [InventoryPosition(ticker="GC=F", physical_qty=1000.0, hedged_qty=1000.0)]
    res = run_crisis_replay(pos, default_receivables, market.snapshot, default_terms, closes, ep)
    assert res.covered_tickers == ["GC=F"]
    last = res.frame.iloc[-1]
    expected_mtm = -1000.0 * market.snapshot["GC=F"].spot_price * last["move_GC=F"]
    assert max(last["vm_inflow"], last["vm_outflow"]) == pytest.approx(abs(expected_mtm), rel=1e-9)
