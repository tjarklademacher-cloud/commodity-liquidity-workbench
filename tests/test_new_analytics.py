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


# ---- Margin financing line ---------------------------------------------------------------------------------

def _with(terms, **kw):
    from core.schemas import FacilityTerms
    return FacilityTerms.model_validate({**terms.model_dump(), **kw})


def test_margin_line_adds_funding_for_margin_calls_only(default_positions, default_receivables, snapshot,
                                                        default_terms):
    base = run_stress_scenario(default_positions, default_receivables, snapshot, default_terms, shock_pct=0.14)
    lined = run_stress_scenario(default_positions, default_receivables, snapshot,
                                _with(default_terms, margin_line_limit=5e6), shock_pct=0.14)
    assert lined.margin_line_draw == pytest.approx(5e6)
    assert lined.net_liquidity_buffer == pytest.approx(base.net_liquidity_buffer + 5e6)
    huge = run_stress_scenario(default_positions, default_receivables, snapshot,
                               _with(default_terms, margin_line_limit=1e9), shock_pct=0.14)
    assert huge.margin_line_draw == pytest.approx(base.vm_outflow + base.im_addon_outflow)
    assert str(huge.regime) == "HEALTHY"
    crash = run_stress_scenario(default_positions, default_receivables, snapshot,
                                _with(default_terms, margin_line_limit=1e9), shock_pct=-0.20)
    assert crash.margin_line_draw == pytest.approx(crash.im_addon_outflow)
    assert crash.bb_deficiency_cure > 0


def test_margin_line_vectorized_matches_exact(default_positions, default_receivables, snapshot, default_terms):
    from core.scenario_analytics import REGIME_CODES, evaluate_shock_matrix
    terms = _with(default_terms, margin_line_limit=4e6)
    rng = np.random.default_rng(5)
    shocks = rng.uniform(-0.3, 0.3, size=(80, len(REQUIRED_TICKERS)))
    out = evaluate_shock_matrix(default_positions, default_receivables, snapshot, terms, shocks)
    for i, row in enumerate(shocks):
        exact = run_stress_scenario(default_positions, default_receivables, snapshot, terms,
                                    per_commodity_shocks=dict(zip(REQUIRED_TICKERS, row)))
        assert out["net_liquidity_buffer"][i] == pytest.approx(exact.net_liquidity_buffer, abs=1e-4)
        assert out["margin_line_draw"][i] == pytest.approx(exact.margin_line_draw, abs=1e-4)
        assert str(REGIME_CODES[out["regime_code"][i]]) == str(exact.regime)


# ---- Contingency funding plan -------------------------------------------------------------------------------

def test_contingency_actions_restore_the_buffer(default_positions, default_receivables, snapshot, default_terms):
    from core.contingency import build_contingency_plan
    sweep = run_stress_sweep(default_positions, default_receivables, snapshot, default_terms)
    target = {t: 0.14 for t in REQUIRED_TICKERS}
    plan = build_contingency_plan(default_positions, default_receivables, snapshot, default_terms, sweep, target,
                                  "Parallel +14%")
    assert plan.shortfall == pytest.approx(1.79e6, abs=0.01e6)
    by_key = {a.key: a for a in plan.actions}
    cash = run_stress_scenario(default_positions, default_receivables, snapshot,
                               _with(default_terms, unrestricted_cash=default_terms.unrestricted_cash + plan.shortfall),
                               shock_pct=0.14)
    assert cash.net_liquidity_buffer == pytest.approx(0.0, abs=1.0)
    line = run_stress_scenario(default_positions, default_receivables, snapshot,
                               _with(default_terms, margin_line_limit=by_key["margin_line"].required_usd),
                               shock_pct=0.14)
    assert line.net_liquidity_buffer >= -1.0
    fac = run_stress_scenario(default_positions, default_receivables, snapshot,
                              _with(default_terms, facility_limit=default_terms.facility_limit
                                    + by_key["facility"].required_usd), shock_pct=0.14)
    assert fac.net_liquidity_buffer >= -1.0
    hedged_value = sum(p.hedged_qty * snapshot[p.ticker].spot_price for p in default_positions)
    scale = 1.0 - by_key["hedges"].required_usd / hedged_value
    cut = [InventoryPosition(ticker=p.ticker, physical_qty=p.physical_qty, hedged_qty=p.hedged_qty * scale)
           for p in default_positions]
    assert run_stress_scenario(cut, default_receivables, snapshot, default_terms,
                               shock_pct=0.14).net_liquidity_buffer >= -1.0
    feasible = [a.required_usd for a in plan.actions if a.feasible]
    assert feasible == sorted(feasible)


def test_contingency_no_shortfall_and_triggers(default_positions, default_receivables, snapshot, default_terms):
    from core.contingency import build_contingency_plan
    sweep = run_stress_sweep(default_positions, default_receivables, snapshot, default_terms)
    plan = build_contingency_plan(default_positions, default_receivables, snapshot, default_terms, sweep,
                                  {t: 0.0 for t in REQUIRED_TICKERS}, "No shock")
    assert plan.shortfall == 0.0
    assert next(a for a in plan.actions if a.key == "cash").required_usd == 0.0
    levels = [t.shock_pct for t in plan.triggers]
    assert levels == sorted(levels)
    assert sweep.upside_squeeze_breakeven_pct in levels
    assert sweep.downside_deficiency_breakeven_pct in levels


def test_breaking_moves_bracket_the_crossing(default_positions, default_receivables, snapshot, default_terms):
    from core.contingency import breaking_moves
    moves = {m.ticker: m for m in breaking_moves(default_positions, default_receivables, snapshot, default_terms)}
    gold = moves["GC=F"].squeeze_rally_pct
    assert gold is not None
    just_below = run_stress_scenario(default_positions, default_receivables, snapshot, default_terms,
                                     per_commodity_shocks={"GC=F": gold - 0.01})
    just_above = run_stress_scenario(default_positions, default_receivables, snapshot, default_terms,
                                     per_commodity_shocks={"GC=F": gold + 0.01})
    assert just_below.net_liquidity_buffer >= 0 > just_above.net_liquidity_buffer
    fall = moves["GC=F"].deficiency_fall_pct
    r = run_stress_scenario(default_positions, default_receivables, snapshot, default_terms,
                            per_commodity_shocks={"GC=F": fall - 0.01})
    assert r.bb_deficiency_cure > 0
