"""Extended scenario analytics: vectorized liquidity engine, 2D shock grid, Monte Carlo LaR, replay.

The vectorized engine reproduces ``run_stress_scenario`` exactly for many scenarios at once: under a
price shock, haircuts are unchanged (volatility is held at the snapshot), so eligible inventory scales
linearly with ``max(0, 1 + shock)`` and only the concentration-cap fixed point must be re-solved.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from core.borrowing_base import _value_receivables, calculate_borrowing_base
from core.haircuts import compute_portfolio_inventory
from core.schemas import (
    COMMODITY_SPECS,
    METALS_TICKERS,
    REQUIRED_TICKERS,
    SOFTS_TICKERS,
    CommoditySpec,
    FacilityTerms,
    HistoricalReplayResult,
    InventoryPosition,
    LiquidityRegime,
    MarketQuote,
    MonteCarloResult,
    ReceivableItem,
    RunwayResult,
    ShockGridResult,
)

REGIME_CODES: tuple[LiquidityRegime, ...] = (
    LiquidityRegime.HEALTHY,
    LiquidityRegime.MARGIN_TRAP_WARNING,
    LiquidityRegime.BB_DEFICIENCY_WARNING,
    LiquidityRegime.CRITICAL_SQUEEZE,
)


def _vectorized_cap_solve(bb_inv: np.ndarray, pre_caps: Sequence[float], cap_pct: float) -> np.ndarray:
    """Sorted-suffix closed form of BB = BB_inv + sum_j min(a_j, cap * BB), vectorized over BB_inv."""
    a_sorted = np.sort(np.maximum(0.0, np.asarray(pre_caps, dtype=float)))[::-1]
    n = a_sorted.size
    suffix = np.concatenate([np.cumsum(a_sorted[::-1])[::-1], [0.0]]) if n else np.array([0.0])
    solved = np.zeros_like(bb_inv)
    found = np.zeros(bb_inv.shape, dtype=bool)
    for k in range(n + 1):
        denom = 1.0 - k * cap_pct
        if denom <= 1e-12:
            break
        cand = (bb_inv + suffix[k]) / denom
        lim = cap_pct * cand
        upper_ok = np.ones_like(found) if k == 0 else a_sorted[k - 1] >= lim - 1e-9
        lower_ok = np.ones_like(found) if k == n else a_sorted[k] <= lim + 1e-9
        hit = upper_ok & lower_ok & ~found
        solved = np.where(hit, np.maximum(0.0, cand), solved)
        found |= hit
    limit = cap_pct * solved
    capped_sum = np.minimum(a_sorted[None, :], limit[:, None]).sum(axis=1) if n else np.zeros_like(solved)
    return bb_inv + capped_sum


def evaluate_shock_matrix(
    positions: Sequence[InventoryPosition],
    receivables: Sequence[ReceivableItem],
    market_snapshot: Mapping[str, MarketQuote],
    terms: FacilityTerms,
    shock_matrix: np.ndarray,
    specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS,
) -> dict[str, np.ndarray]:
    """Liquidity metrics for many scenarios; ``shock_matrix`` is (n_scenarios, 4) in REQUIRED_TICKERS order."""
    shocks = np.atleast_2d(np.asarray(shock_matrix, dtype=float))
    col = {t: i for i, t in enumerate(REQUIRED_TICKERS)}
    base_items = compute_portfolio_inventory(positions, market_snapshot, specs)
    baseline_bb = calculate_borrowing_base(base_items, receivables, terms).final_borrowing_base

    bb_inv = np.zeros(shocks.shape[0])
    mtm = np.zeros(shocks.shape[0])
    for pos, item in zip(positions, base_items):
        s = shocks[:, col[pos.ticker]]
        bb_inv += item.eligible_collateral * np.maximum(0.0, 1.0 + s)
        mtm -= pos.hedged_qty * market_snapshot[pos.ticker].spot_price * s
    pre_caps = [r["pre_cap_eligible"] for r in _value_receivables(receivables, terms)]
    shocked_bb = _vectorized_cap_solve(bb_inv, pre_caps, terms.concentration_cap_pct)

    baseline_headroom = max(0.0, min(terms.facility_limit, baseline_bb) - terms.drawn_revolver)
    shocked_headroom = np.maximum(0.0, np.minimum(terms.facility_limit, shocked_bb) - terms.drawn_revolver)
    headroom_change = shocked_headroom - baseline_headroom
    vm_outflow = np.maximum(0.0, -mtm)
    vm_inflow = np.maximum(0.0, mtm)
    max_abs = np.abs(shocks).max(axis=1) if shocks.shape[1] else np.zeros(shocks.shape[0])
    im = np.where(
        max_abs >= terms.im_stepup_trigger - 1e-9, terms.im_stepup_fraction * terms.baseline_initial_margin, 0.0
    )
    cure = np.maximum(0.0, terms.drawn_revolver - shocked_bb)
    carry = terms.funding_carry_cost(sum(p.physical_qty * market_snapshot[p.ticker].spot_price for p in positions))
    outflows = vm_outflow + im + cure + carry
    margin_draw = np.minimum(terms.margin_line_limit, vm_outflow + im)
    resources = terms.unrestricted_cash + shocked_headroom + vm_inflow + margin_draw
    nlb = resources - outflows
    uncovered_vm = vm_outflow - np.minimum(terms.margin_line_limit, vm_outflow)
    regime = np.where(
        nlb < 0.0, 3, np.where((uncovered_vm > 0.0) & (uncovered_vm > np.maximum(0.0, headroom_change)), 1,
                               np.where(cure > 0.0, 2, 0))
    )
    return {
        "shocked_bb": shocked_bb,
        "shocked_headroom": shocked_headroom,
        "futures_mtm_pnl": mtm,
        "vm_outflow": vm_outflow,
        "vm_inflow": vm_inflow,
        "im_addon_outflow": im,
        "margin_line_draw": margin_draw,
        "bb_deficiency_cure": cure,
        "funding_carry_outflow": np.full(shocks.shape[0], carry),
        "net_liquidity_buffer": nlb,
        "regime_code": regime,
    }


def run_shock_grid(
    positions: Sequence[InventoryPosition],
    receivables: Sequence[ReceivableItem],
    market_snapshot: Mapping[str, MarketQuote],
    terms: FacilityTerms,
    specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS,
    shock_min: float = -0.20,
    shock_max: float = 0.20,
    steps: int = 17,
) -> ShockGridResult:
    """Metals (GC, HG) shock vs. softs/lumber (LBR, CC) shock grid of net liquidity buffers."""
    axis = np.round(np.linspace(shock_min, shock_max, steps), 6)
    metals_idx = [REQUIRED_TICKERS.index(t) for t in METALS_TICKERS]
    softs_idx = [REQUIRED_TICKERS.index(t) for t in SOFTS_TICKERS]
    mm, ss = np.meshgrid(axis, axis, indexing="ij")
    shocks = np.zeros((mm.size, len(REQUIRED_TICKERS)))
    shocks[:, metals_idx] = mm.reshape(-1, 1)
    shocks[:, softs_idx] = ss.reshape(-1, 1)
    out = evaluate_shock_matrix(positions, receivables, market_snapshot, terms, shocks, specs)
    nlb = out["net_liquidity_buffer"].reshape(steps, steps)
    bb = out["shocked_bb"].reshape(steps, steps)
    codes = out["regime_code"].reshape(steps, steps)
    wi, wj = np.unravel_index(int(np.argmin(nlb)), nlb.shape)
    return ShockGridResult(
        metals_axis=axis.tolist(),
        softs_axis=axis.tolist(),
        net_buffer=nlb.tolist(),
        shocked_bb=bb.tolist(),
        regime=[[str(REGIME_CODES[c]) for c in row] for row in codes],
        squeeze_share=float((codes == 3).mean()),
        worst_net_buffer=float(nlb[wi, wj]),
        worst_metals_shock=float(axis[wi]),
        worst_softs_shock=float(axis[wj]),
    )


def _winsorized_log_returns(prices_df: pd.DataFrame, lookback: int, winsorize_sigma: float | None) -> pd.DataFrame:
    rets = np.log(prices_df / prices_df.shift(1)).dropna(how="any").iloc[-lookback:]
    if winsorize_sigma is not None and winsorize_sigma > 0:
        for t in rets.columns:
            sd = rets[t].std(ddof=1)
            if pd.notna(sd) and sd > 0:
                rets.loc[:, t] = rets[t].clip(lower=-winsorize_sigma * sd, upper=winsorize_sigma * sd)
    return rets


def _daily_covariance(
    prices_df: pd.DataFrame, lookback_days: int, winsorize_sigma: float | None, specs: Mapping[str, CommoditySpec]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Daily log-return covariance and correlation (reference vols and zero correlation if history is short)."""
    rets = _winsorized_log_returns(prices_df.loc[:, list(REQUIRED_TICKERS)], lookback_days, winsorize_sigma)
    if len(rets) < 20:
        vols = np.array([specs[t].ref_vol / np.sqrt(252.0) for t in REQUIRED_TICKERS])
        return np.diag(vols**2), np.eye(len(REQUIRED_TICKERS)), int(len(rets))
    return rets.cov(ddof=1).to_numpy(), rets.corr().to_numpy(), int(len(rets))


def _cholesky(cov: np.ndarray) -> np.ndarray:
    jitter = 0.0
    while True:
        try:
            return np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))
        except np.linalg.LinAlgError:
            jitter = max(jitter * 10.0, 1e-12)


def run_liquidity_monte_carlo(
    positions: Sequence[InventoryPosition],
    receivables: Sequence[ReceivableItem],
    market_snapshot: Mapping[str, MarketQuote],
    terms: FacilityTerms,
    prices_df: pd.DataFrame,
    horizon_days: int = 5,
    n_sims: int = 5000,
    seed: int = 7,
    lookback_days: int = 250,
    winsorize_sigma: float | None = 5.0,
    specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS,
) -> MonteCarloResult:
    """Correlated lognormal price moves over a horizon (Cholesky of historical covariance) -> buffer distribution."""
    rets = _winsorized_log_returns(prices_df.loc[:, list(REQUIRED_TICKERS)], lookback_days, winsorize_sigma)
    if len(rets) < 20:
        vols = np.array([specs[t].ref_vol / np.sqrt(252.0) for t in REQUIRED_TICKERS])
        cov = np.diag(vols**2)
        corr = np.eye(len(REQUIRED_TICKERS))
    else:
        cov = rets.cov(ddof=1).to_numpy()
        corr = rets.corr().to_numpy()
    cov_h = cov * float(horizon_days)
    jitter = 0.0
    while True:
        try:
            chol = np.linalg.cholesky(cov_h + jitter * np.eye(cov_h.shape[0]))
            break
        except np.linalg.LinAlgError:
            jitter = max(jitter * 10.0, 1e-12)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_sims, cov_h.shape[0]))
    shocks = np.expm1(z @ chol.T)
    out = evaluate_shock_matrix(positions, receivables, market_snapshot, terms, shocks, specs)
    nlb = out["net_liquidity_buffer"]
    codes = out["regime_code"]
    p05 = float(np.quantile(nlb, 0.05))
    tail = nlb[nlb <= p05]
    worst = int(np.argmin(nlb))
    counts = {str(r): int((codes == i).sum()) for i, r in enumerate(REGIME_CODES)}
    return MonteCarloResult(
        horizon_days=int(horizon_days),
        n_sims=int(n_sims),
        seed=int(seed),
        lookback_days=int(len(rets)),
        net_buffer_samples=nlb.tolist(),
        mean_net_buffer=float(nlb.mean()),
        median_net_buffer=float(np.median(nlb)),
        p05_net_buffer=p05,
        p01_net_buffer=float(np.quantile(nlb, 0.01)),
        es95_net_buffer=float(tail.mean()) if tail.size else p05,
        prob_critical=float((codes == 3).mean()),
        prob_margin_trap=float((codes == 1).mean()),
        prob_deficiency=float((codes == 2).mean()),
        prob_healthy=float((codes == 0).mean()),
        regime_counts=counts,
        horizon_vols={t: float(np.sqrt(cov_h[i, i])) for i, t in enumerate(REQUIRED_TICKERS)},
        correlation={a: {b: float(corr[i, j]) for j, b in enumerate(REQUIRED_TICKERS)}
                     for i, a in enumerate(REQUIRED_TICKERS)},
        worst_scenario_shocks={t: float(shocks[worst, i]) for i, t in enumerate(REQUIRED_TICKERS)},
    )


def run_historical_replay(
    positions: Sequence[InventoryPosition],
    receivables: Sequence[ReceivableItem],
    prices_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    terms: FacilityTerms,
    specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS,
    vm_window: int = 5,
) -> HistoricalReplayResult:
    """Replay today's book and facility through history: daily BB, headroom, futures VM and stressed buffer.

    Quantities, AR ledger and drawn balance are held at today's values; prices and 30d vols move by date.
    """
    prices = prices_df.loc[:, list(REQUIRED_TICKERS)].ffill()
    vols = vol_df.reindex(prices.index).loc[:, list(REQUIRED_TICKERS)]
    hedged = {t: 0.0 for t in REQUIRED_TICKERS}
    for pos in positions:
        hedged[pos.ticker] += pos.hedged_qty
    daily_vm = -sum(hedged[t] * prices[t].diff() for t in REQUIRED_TICKERS)

    rows = []
    for i, date in enumerate(prices.index):
        if i == 0:
            continue
        vol_row = vols.loc[date]
        price_row = prices.loc[date]
        prev_row = prices.iloc[i - 1]
        if vol_row.isna().any() or price_row.isna().any() or prev_row.isna().any():
            continue
        snap = {
            t: MarketQuote(
                ticker=t, spot_price=float(price_row[t]), prev_price=float(prev_row[t]),
                ret_1d=float(np.log(price_row[t] / prev_row[t])), ret_30d=0.0,
                realized_vol_30d=float(vol_row[t]), vol_30d_min=float(vol_row[t]), vol_30d_max=float(vol_row[t]),
                as_of_date=date.strftime("%Y-%m-%d"),
            )
            for t in REQUIRED_TICKERS
        }
        bb = calculate_borrowing_base(compute_portfolio_inventory(positions, snap, specs), receivables, terms)
        rows.append(
            {
                "date": date,
                "gross_inventory": bb.gross_inventory,
                "eligible_inventory": bb.eligible_inventory,
                "final_eligible_receivables": bb.final_eligible_receivables,
                "borrowing_base": bb.final_borrowing_base,
                "headroom": max(0.0, min(terms.facility_limit, bb.final_borrowing_base) - terms.drawn_revolver),
                "bb_deficiency_cure": max(0.0, terms.drawn_revolver - bb.final_borrowing_base),
                "daily_vm": float(daily_vm.loc[date]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Not enough price and volatility history to run a historical replay.")
    frame = frame.set_index("date")
    frame = frame.assign(rolling_vm=frame["daily_vm"].rolling(vm_window, min_periods=1).sum())
    frame = frame.assign(
        stressed_buffer=terms.unrestricted_cash + frame["headroom"] + frame["rolling_vm"] - frame["bb_deficiency_cure"]
    )
    fmt = "%Y-%m-%d"
    return HistoricalReplayResult(
        frame=frame,
        start_date=frame.index[0].strftime(fmt),
        end_date=frame.index[-1].strftime(fmt),
        n_days=int(len(frame)),
        min_headroom=float(frame["headroom"].min()),
        min_headroom_date=frame["headroom"].idxmin().strftime(fmt),
        max_5d_vm_outflow=float(max(0.0, -frame["rolling_vm"].min())),
        max_5d_vm_outflow_date=frame["rolling_vm"].idxmin().strftime(fmt),
        deficiency_days=int((frame["bb_deficiency_cure"] > 0.0).sum()),
        min_stressed_buffer=float(frame["stressed_buffer"].min()),
        min_stressed_buffer_date=frame["stressed_buffer"].idxmin().strftime(fmt),
    )


def run_liquidity_runway(
    positions: Sequence[InventoryPosition],
    receivables: Sequence[ReceivableItem],
    market_snapshot: Mapping[str, MarketQuote],
    terms: FacilityTerms,
    prices_df: pd.DataFrame,
    horizon_days: int = 20,
    n_paths: int = 2000,
    seed: int = 11,
    lookback_days: int = 250,
    winsorize_sigma: float | None = 5.0,
    specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS,
) -> RunwayResult:
    """Simulate daily correlated price paths and evaluate the net liquidity buffer at each day's cumulative move.

    Each day is evaluated with the same engine as the stress sweep, using the cumulative price move since today.
    A path's runway ends on the first day its buffer is below zero (a critical squeeze).
    """
    cov, _, _ = _daily_covariance(prices_df, lookback_days, winsorize_sigma, specs)
    chol = _cholesky(cov)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, horizon_days, cov.shape[0]))
    cum_log = np.cumsum(z @ chol.T, axis=1)
    shocks = np.expm1(cum_log)
    nlb = np.empty((n_paths, horizon_days))
    for d in range(horizon_days):
        nlb[:, d] = evaluate_shock_matrix(positions, receivables, market_snapshot, terms, shocks[:, d, :],
                                          specs)["net_liquidity_buffer"]
    breached = nlb < 0.0
    ever = breached.any(axis=1)
    first_day = np.where(ever, breached.argmax(axis=1) + 1, 0)
    counts = [int((first_day == d).sum()) for d in range(1, horizon_days + 1)]
    by_day = np.cumsum(counts) / n_paths
    pct = {f"p{q}": np.percentile(nlb, q, axis=0).tolist() for q in (5, 25, 50, 75, 95)}
    breach_days = first_day[ever]
    return RunwayResult(
        horizon_days=int(horizon_days),
        n_paths=int(n_paths),
        seed=int(seed),
        days=list(range(1, horizon_days + 1)),
        percentiles=pct,
        prob_breached_by_day=by_day.tolist(),
        first_breach_counts=counts,
        prob_breach_within_horizon=float(ever.mean()),
        median_first_breach_day=float(np.median(breach_days)) if breach_days.size else None,
        p10_first_breach_day=float(np.percentile(first_day, 10)) if ever.mean() >= 0.10 else None,
    )
