"""Real crisis replays: apply observed futures price paths from historical stress episodes to today's book.

Each day's cumulative price move since the start of the episode is applied to today's spot prices, and
haircuts use the realized 30-day volatility observed on that day. The same stress engine as the stress
sweep turns each day into a borrowing base, margin flows and a net liquidity buffer. Commodities with no
history in the window (for example CME physical lumber before August 2022) are held at today's price and
reported as not covered.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from core.borrowing_base import calculate_borrowing_base
from core.data_feed import compute_realized_volatility
from core.haircuts import compute_portfolio_inventory
from core.schemas import (
    COMMODITY_SPECS,
    REQUIRED_TICKERS,
    CommoditySpec,
    CrisisEpisode,
    CrisisReplayResult,
    FacilityTerms,
    InventoryPosition,
    LiquidityRegime,
    MarketQuote,
    ReceivableItem,
)
from core.stress_engine import run_stress_scenario

VOL_LOOKBACK_DAYS = 70
FETCH_TIMEOUT_SECONDS = 15

CRISIS_EPISODES: dict[str, CrisisEpisode] = {
    e.key: e
    for e in (
        CrisisEpisode(key="covid_2020", name="Covid-19 crash (Feb to Apr 2020)", start="2020-02-03",
                      end="2020-04-30",
                      note="Global demand collapse after the Covid-19 outbreak; risk assets sold off while gold rose."),
        CrisisEpisode(key="brazil_frost_2021", name="Brazil coffee frost (Jul to Aug 2021)", start="2021-07-01",
                      end="2021-08-31",
                      note="Frosts in Brazil's coffee belt in July 2021 drove arabica futures sharply higher."),
        CrisisEpisode(key="ukraine_2022", name="Russia-Ukraine shock (Feb to Apr 2022)", start="2022-02-01",
                      end="2022-04-29",
                      note="Russia's invasion of Ukraine on 24 February 2022 set off a broad commodity price spike."),
        CrisisEpisode(key="cocoa_2024", name="Cocoa squeeze (Jan to May 2024)", start="2024-01-02",
                      end="2024-05-31",
                      note="Poor West African harvests pushed cocoa futures to record highs in 2024."),
    )
}


def fetch_episode_closes(
    episode: CrisisEpisode,
    download: Callable[..., Any] = yf.download,
) -> pd.DataFrame:
    """Daily closes for the episode plus a volatility lookback, in USD per physical unit.

    Columns are the subset of REQUIRED_TICKERS with any data. Raises ValueError if nothing usable came back.
    """
    start = (pd.Timestamp(episode.start) - pd.Timedelta(days=VOL_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(episode.end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = download(tickers=list(REQUIRED_TICKERS), start=start, end=end, interval="1d", auto_adjust=True,
                   progress=False, threads=False, timeout=FETCH_TIMEOUT_SECONDS)
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance returned no data for this episode.")
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"].copy()
    else:
        closes = raw.loc[:, [t for t in REQUIRED_TICKERS if t in raw.columns]].copy()
    closes.index = pd.DatetimeIndex(pd.to_datetime(closes.index), name="Date")
    if closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)
    keep = [t for t in REQUIRED_TICKERS if t in closes.columns]
    closes = closes.loc[:, keep].apply(pd.to_numeric, errors="coerce")
    closes = closes.where(closes > 0.0)
    closes = closes.loc[:, [t for t in keep if closes[t].notna().sum() >= 2]]
    if closes.empty:
        raise ValueError("No commodity in the book has price history for this episode.")
    for t in closes.columns:
        closes.loc[:, t] = closes[t] * COMMODITY_SPECS[t].quote_scale
    return closes.sort_index().ffill(limit=5)


def run_crisis_replay(
    positions: Sequence[InventoryPosition],
    receivables: Sequence[ReceivableItem],
    market_snapshot: Mapping[str, MarketQuote],
    terms: FacilityTerms,
    closes: pd.DataFrame,
    episode: CrisisEpisode,
    specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS,
    winsorize_sigma: float | None = 5.0,
) -> CrisisReplayResult:
    """Replay today's book through the episode's observed cumulative price moves and volatilities."""
    window = closes.loc[pd.Timestamp(episode.start):pd.Timestamp(episode.end)]
    covered = [t for t in REQUIRED_TICKERS if t in window.columns and window[t].notna().any()]
    missing = [t for t in REQUIRED_TICKERS if t not in covered]
    if not covered or window.empty:
        raise ValueError("No usable prices inside the episode window.")
    window = window.loc[window[covered].notna().any(axis=1)]
    base_prices = {t: float(window[t].dropna().iloc[0]) for t in covered}
    vols = compute_realized_volatility(closes.loc[:, covered], winsorize_sigma=winsorize_sigma)

    baseline_items = compute_portfolio_inventory(positions, market_snapshot, specs)
    baseline_bb = calculate_borrowing_base(baseline_items, receivables, terms)

    rows = []
    for date in window.index:
        moves: dict[str, float] = {}
        snap: dict[str, MarketQuote] = {}
        for t in REQUIRED_TICKERS:
            quote = market_snapshot[t]
            vol = quote.realized_vol_30d
            move = 0.0
            if t in covered:
                price = window.loc[:date, t].dropna()
                if not price.empty:
                    move = float(price.iloc[-1] / base_prices[t] - 1.0)
                hist_vol = vols.loc[:date, t].dropna()
                if not hist_vol.empty:
                    vol = float(hist_vol.iloc[-1])
            moves[t] = move
            snap[t] = MarketQuote(
                ticker=t, spot_price=quote.spot_price, prev_price=quote.prev_price, ret_1d=quote.ret_1d,
                ret_30d=quote.ret_30d, realized_vol_30d=vol, vol_30d_min=quote.vol_30d_min,
                vol_30d_max=quote.vol_30d_max, as_of_date=quote.as_of_date,
            )
        r = run_stress_scenario(positions, receivables, snap, terms, specs, per_commodity_shocks=moves,
                                baseline_bb_result=baseline_bb)
        rows.append({
            "date": date, **{f"move_{t}": moves[t] for t in REQUIRED_TICKERS},
            "shocked_bb": r.shocked_bb, "shocked_headroom": r.shocked_headroom, "vm_outflow": r.vm_outflow,
            "vm_inflow": r.vm_inflow, "im_addon_outflow": r.im_addon_outflow,
            "bb_deficiency_cure": r.bb_deficiency_cure, "net_liquidity_buffer": r.net_liquidity_buffer,
            "regime": str(r.regime),
        })
    frame = pd.DataFrame(rows).set_index("date")
    worst = frame["net_liquidity_buffer"].idxmin()
    peak = frame["vm_outflow"].idxmax()
    fmt = "%Y-%m-%d"
    return CrisisReplayResult(
        episode=episode,
        frame=frame,
        covered_tickers=covered,
        missing_tickers=missing,
        first_date=frame.index[0].strftime(fmt),
        last_date=frame.index[-1].strftime(fmt),
        worst_net_buffer=float(frame.loc[worst, "net_liquidity_buffer"]),
        worst_date=worst.strftime(fmt),
        worst_regime=str(frame.loc[worst, "regime"]),
        critical_days=int((frame["regime"] == str(LiquidityRegime.CRITICAL_SQUEEZE)).sum()),
        peak_vm_outflow=float(frame.loc[peak, "vm_outflow"]),
        peak_vm_date=peak.strftime(fmt),
        max_moves={t: float(frame[f"move_{t}"].iloc[np.argmax(np.abs(frame[f"move_{t}"].to_numpy()))])
                   for t in REQUIRED_TICKERS},
    )
