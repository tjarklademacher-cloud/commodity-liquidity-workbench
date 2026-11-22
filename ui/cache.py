"""Streamlit cache wrappers for market loads and heavy scenario analytics.

Cached functions take only hashable primitives (JSON strings, bytes, numbers) plus price frames, and
rebuild Pydantic models with ``model_validate`` on the way in, so the cache key is the full model input.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence

import pandas as pd
import streamlit as st

from core.data_feed import load_custom_csv_market_data, load_live_market_data, load_sample_market_data
from core.scenario_analytics import run_historical_replay, run_liquidity_monte_carlo, run_shock_grid
from core.schemas import (
    FacilityTerms,
    HistoricalReplayResult,
    InventoryPosition,
    MarketLoadResult,
    MarketQuote,
    MonteCarloResult,
    ReceivableItem,
    ShockGridResult,
    StressSweepResult,
)
from core.stress_engine import run_stress_sweep

LIVE_TTL_SECONDS = 900


def _winsor(winsorize: bool) -> float | None:
    return 5.0 if winsorize else None


@st.cache_data(ttl=LIVE_TTL_SECONDS, show_spinner=False)
def cached_live_market(period: str, winsorize: bool) -> MarketLoadResult:
    """Live Yahoo Finance load (falls back to the bundled sample internally); refreshed every 15 minutes."""
    return load_live_market_data(period=period, winsorize_sigma=_winsor(winsorize))


@st.cache_data(show_spinner=False)
def cached_sample_market(winsorize: bool) -> MarketLoadResult:
    return load_sample_market_data(winsorize_sigma=_winsor(winsorize))


@st.cache_data(show_spinner=False, max_entries=8)
def cached_custom_market(file_bytes: bytes, winsorize: bool) -> MarketLoadResult:
    return load_custom_csv_market_data(io.BytesIO(file_bytes), winsorize_sigma=_winsor(winsorize))


def model_inputs_json(
    positions: Sequence[InventoryPosition],
    receivables: Sequence[ReceivableItem],
    terms: FacilityTerms,
    snapshot: Mapping[str, MarketQuote],
) -> tuple[str, str, str, str]:
    """Serialize validated model inputs into stable JSON strings usable as cache keys."""
    return (
        json.dumps([p.model_dump() for p in positions], sort_keys=True),
        json.dumps([r.model_dump() for r in receivables], sort_keys=True),
        terms.model_dump_json(),
        json.dumps({t: q.model_dump() for t, q in sorted(snapshot.items())}, sort_keys=True),
    )


def _load_inputs(
    positions_json: str, receivables_json: str, terms_json: str, snapshot_json: str
) -> tuple[list[InventoryPosition], list[ReceivableItem], FacilityTerms, dict[str, MarketQuote]]:
    positions = [InventoryPosition.model_validate(d) for d in json.loads(positions_json)]
    receivables = [ReceivableItem.model_validate(d) for d in json.loads(receivables_json)]
    terms = FacilityTerms.model_validate_json(terms_json)
    snapshot = {t: MarketQuote.model_validate(q) for t, q in json.loads(snapshot_json).items()}
    return positions, receivables, terms, snapshot


@st.cache_data(show_spinner=False, max_entries=32)
def cached_scenario_suite(
    positions_json: str,
    receivables_json: str,
    terms_json: str,
    snapshot_json: str,
    prices_df: pd.DataFrame,
    vol_df: pd.DataFrame,
) -> tuple[StressSweepResult, ShockGridResult, HistoricalReplayResult | None]:
    """41-point sweep, metals x softs grid and historical replay; independent of the active shock slider."""
    positions, receivables, terms, snapshot = _load_inputs(positions_json, receivables_json, terms_json, snapshot_json)
    sweep = run_stress_sweep(positions, receivables, snapshot, terms)
    grid = run_shock_grid(positions, receivables, snapshot, terms)
    try:
        replay = run_historical_replay(positions, receivables, prices_df, vol_df, terms)
    except ValueError:
        replay = None
    return sweep, grid, replay


@st.cache_data(show_spinner=False, max_entries=32)
def cached_monte_carlo(
    positions_json: str,
    receivables_json: str,
    terms_json: str,
    snapshot_json: str,
    prices_df: pd.DataFrame,
    horizon_days: int,
    n_sims: int,
    winsorize: bool,
) -> MonteCarloResult:
    positions, receivables, terms, snapshot = _load_inputs(positions_json, receivables_json, terms_json, snapshot_json)
    return run_liquidity_monte_carlo(
        positions, receivables, snapshot, terms, prices_df,
        horizon_days=horizon_days, n_sims=n_sims, winsorize_sigma=_winsor(winsorize),
    )


def market_data_status(market: MarketLoadResult) -> tuple[str, str]:
    """Short label and semantic state (ui.theme.STATUS key) for the active market data source.

    Only the live-feed fallback is flagged as a caution; every other source is neutral.
    """
    if market.used_fallback:
        return "Bundled data (live feed unavailable)", "caution"
    if market.source_label.startswith("Live"):
        return "Live market data", "neutral"
    if market.source_label.startswith("Custom"):
        return "Uploaded CSV data", "neutral"
    return "Bundled sample data", "neutral"
