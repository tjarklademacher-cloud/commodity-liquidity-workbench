"""Streamlit cache wrappers for market loads and heavy scenario analytics.

Cached functions take only hashable primitives (JSON strings, bytes, numbers) plus price frames, and
rebuild Pydantic models with ``model_validate`` on the way in, so the cache key is the full model input.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from core.data_feed import (
    load_custom_csv_market_data,
    load_fx_rate,
    load_live_market_data,
    load_sample_market_data,
)
from core.advanced_analytics import (
    run_advanced_hedge_frontier,
    run_advanced_shock_grid,
    run_advanced_sweep,
)
from core.advanced_schemas import AdvancedSettings, ScenarioSpec
from core.crisis_replay import CRISIS_EPISODES, fetch_episode_closes, run_crisis_replay
from core.hedge_optimizer import run_hedge_frontier
from core.scenario_analytics import (
    run_historical_replay,
    run_liquidity_monte_carlo,
    run_liquidity_runway,
    run_shock_grid,
)
from core.schemas import (
    CrisisReplayResult,
    FacilityTerms,
    HedgeFrontierResult,
    HistoricalReplayResult,
    InventoryPosition,
    MarketLoadResult,
    MarketQuote,
    MonteCarloResult,
    ReceivableItem,
    RunwayResult,
    ShockGridResult,
    StressSweepResult,
)
from core.stress_engine import run_stress_sweep
from core.advanced_schemas import MarginSettings
from core.case_study_2022 import CaseStudyAssumptions, fetch_ttf_2022, run_case_study
from core.hedge_frontier import HedgeConstraints
from core.hedge_frontier import run_hedge_frontier as run_mo_frontier
from core.model_validation import moment_comparison, var_backtest, vol_forecast_evaluation
from core.return_models import ReturnModel, calibrate_return_model
from core.schemas import REQUIRED_TICKERS
from core.treasury_analytics import (
    MonteCarloOutput,
    TreasurySettings,
    credit_stress_table,
    haircut_sensitivity,
    run_book_monte_carlo,
    run_book_scenario,
)

LIVE_TTL_SECONDS = 900
HISTORY_TTL_SECONDS = 86_400


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


@st.cache_data(show_spinner=False, max_entries=32)
def cached_hedge_frontier(
    positions_json: str, receivables_json: str, terms_json: str, snapshot_json: str, scope: str
) -> HedgeFrontierResult:
    positions, receivables, terms, snapshot = _load_inputs(positions_json, receivables_json, terms_json, snapshot_json)
    return run_hedge_frontier(positions, receivables, snapshot, terms, scope)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_advanced_suite(
    positions_json: str, receivables_json: str, terms_json: str, snapshot_json: str,
    settings_json: str, base_spec_json: str, multipliers_json: str, trigger_rule: bool,
) -> tuple[StressSweepResult, ShockGridResult]:
    """41-point sweep and metals x softs grid through the advanced engine (advanced model mode only)."""
    positions, receivables, terms, snapshot = _load_inputs(positions_json, receivables_json, terms_json, snapshot_json)
    settings = AdvancedSettings.model_validate_json(settings_json)
    base = ScenarioSpec.model_validate_json(base_spec_json)
    mults = json.loads(multipliers_json)
    sweep = run_advanced_sweep(positions, receivables, snapshot, terms, settings, base, mults, trigger_rule)
    grid = run_advanced_shock_grid(positions, receivables, snapshot, terms, settings, base, mults, trigger_rule)
    return sweep, grid


@st.cache_data(show_spinner=False, max_entries=16)
def cached_advanced_frontier(
    positions_json: str, receivables_json: str, terms_json: str, snapshot_json: str,
    settings_json: str, base_spec_json: str, multipliers_json: str, trigger_rule: bool, scope: str,
) -> HedgeFrontierResult:
    positions, receivables, terms, snapshot = _load_inputs(positions_json, receivables_json, terms_json, snapshot_json)
    return run_advanced_hedge_frontier(
        positions, receivables, snapshot, terms, AdvancedSettings.model_validate_json(settings_json),
        ScenarioSpec.model_validate_json(base_spec_json), json.loads(multipliers_json), trigger_rule, scope,
    )


@st.cache_data(show_spinner=False, max_entries=32)
def cached_runway(
    positions_json: str,
    receivables_json: str,
    terms_json: str,
    snapshot_json: str,
    prices_df: pd.DataFrame,
    n_paths: int,
    winsorize: bool,
) -> RunwayResult:
    positions, receivables, terms, snapshot = _load_inputs(positions_json, receivables_json, terms_json, snapshot_json)
    return run_liquidity_runway(positions, receivables, snapshot, terms, prices_df, horizon_days=20,
                                n_paths=n_paths, winsorize_sigma=_winsor(winsorize))


@st.cache_data(ttl=HISTORY_TTL_SECONDS, show_spinner=False, max_entries=8)
def cached_episode_closes(episode_key: str) -> pd.DataFrame:
    """Observed closes for a crisis episode; history does not change, so it is cached for a day."""
    return fetch_episode_closes(CRISIS_EPISODES[episode_key])


@st.cache_data(show_spinner=False, max_entries=32)
def cached_crisis_replay(
    positions_json: str, receivables_json: str, terms_json: str, snapshot_json: str, episode_key: str,
    winsorize: bool,
) -> CrisisReplayResult:
    positions, receivables, terms, snapshot = _load_inputs(positions_json, receivables_json, terms_json, snapshot_json)
    closes = cached_episode_closes(episode_key)
    return run_crisis_replay(positions, receivables, snapshot, terms, closes, CRISIS_EPISODES[episode_key],
                             winsorize_sigma=_winsor(winsorize))


@st.cache_data(ttl=LIVE_TTL_SECONDS, show_spinner=False)
def cached_fx_rate(code: str) -> tuple[float, str, str]:
    """USD per unit of the reporting currency (live, else the bundled 2026-11-20 rate)."""
    return load_fx_rate(code)


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
    if market.source_label.startswith("Factor"):
        return "Synthetic factor sample", "neutral"
    return "Bundled sample data", "neutral"


# ---- Advanced model -----------------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_factor_market(winsorize: bool) -> MarketLoadResult:
    from core.market_calibration import load_factor_sample_market_data

    return load_factor_sample_market_data(winsorize_sigma=_winsor(winsorize))


@st.cache_data(show_spinner=False, max_entries=16)
def cached_calibration(prices_df: pd.DataFrame, window_label: str, winsorize: bool):
    from core.market_calibration import calibrate

    return calibrate(prices_df, window_label, _winsor(winsorize))


@st.cache_data(show_spinner=False, max_entries=16)
def cached_backtest(positions_json: str, snapshot_json: str, prices_df: pd.DataFrame, window_days: int,
                    winsorize: bool):
    from core.market_calibration import backtest_margin_forecast

    positions = [InventoryPosition.model_validate(d) for d in json.loads(positions_json)]
    snapshot = {t: MarketQuote.model_validate(q) for t, q in json.loads(snapshot_json).items()}
    return backtest_margin_forecast(prices_df, positions, snapshot, horizon_days=5, window_days=window_days,
                                    winsorize_sigma=_winsor(winsorize))


@st.cache_data(show_spinner=False, max_entries=16)
def cached_cashflow_monte_carlo(
    positions_json: str, receivables_json: str, terms_json: str, snapshot_json: str,
    settings_json: str, cashflow_json: str, prices_df: pd.DataFrame, window_label: str, method: str,
    n_paths: int, basis_vol: float, im_mults_json: str, apply_trigger_rule: bool, winsorize: bool,
):
    from core.advanced_schemas import AdvancedSettings, CashflowSettings
    from core.cashflow_engine import run_cashflow_monte_carlo

    positions, receivables, terms, snapshot = _load_inputs(positions_json, receivables_json, terms_json, snapshot_json)
    calibration = cached_calibration(prices_df, window_label, winsorize)
    return run_cashflow_monte_carlo(
        positions, receivables, snapshot, terms, AdvancedSettings.model_validate_json(settings_json),
        CashflowSettings.model_validate_json(cashflow_json), calibration, prices_df, method=method,
        n_paths=n_paths, basis_vol_annual=basis_vol, base_im_multipliers=json.loads(im_mults_json),
        apply_trigger_rule=apply_trigger_rule, winsorize_sigma=_winsor(winsorize),
    )


# --------------------------------------------------------------------------- treasury engine (new analytics)



@st.cache_data(show_spinner=False, max_entries=16)
def cached_return_model(prices_df: pd.DataFrame, method: str, window: int | None, nu: float, block_days: int,
                        ewma_lambda: float, correlation_mode: str) -> ReturnModel:
    return calibrate_return_model(prices_df[list(REQUIRED_TICKERS)], method, window=window, nu=nu,
                                  block_days=block_days, ewma_lambda=ewma_lambda, correlation_mode=correlation_mode)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_treasury_mc(inputs: tuple[str, str, str, str], margin_json: str, ts_json: str, prices_df: pd.DataFrame,
                       model_key: tuple, n_paths: int, seed: int, confidence: float, credit_mode: str,
                       wrong_way: float) -> MonteCarloOutput:
    positions, receivables, terms, snapshot = _load_inputs(*inputs)
    model = cached_return_model(prices_df, *model_key)
    return run_book_monte_carlo(positions, receivables, snapshot, terms, MarginSettings.model_validate_json(margin_json),
                                TreasurySettings.model_validate_json(ts_json), model, n_paths, seed, confidence,
                                credit_mode, 0.2, wrong_way)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_haircut_sensitivity(inputs: tuple[str, str, str, str], margin_json: str, ts_json: str,
                               shocks_json: str) -> pd.DataFrame:
    positions, receivables, terms, snapshot = _load_inputs(*inputs)
    return haircut_sensitivity(positions, receivables, snapshot, terms, MarginSettings.model_validate_json(margin_json),
                               TreasurySettings.model_validate_json(ts_json), json.loads(shocks_json))


@st.cache_data(show_spinner=False, max_entries=8)
def cached_mo_frontier(inputs: tuple[str, str, str, str], margin_json: str, ts_json: str, prices_df: pd.DataFrame,
                       model_key: tuple, scope: str, n_paths: int, seed: int, max_breach: float | None,
                       max_downside: float | None) -> pd.DataFrame:
    positions, receivables, terms, snapshot = _load_inputs(*inputs)
    model = cached_return_model(prices_df, *model_key)
    return run_mo_frontier(positions, receivables, snapshot, terms, MarginSettings.model_validate_json(margin_json),
                           TreasurySettings.model_validate_json(ts_json), model, scope,
                           constraints=HedgeConstraints(max_breach_probability=max_breach,
                                                        max_economic_downside=max_downside),
                           n_paths=n_paths, seed=seed)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_validation(prices_df: pd.DataFrame, positions_json: str, snapshot_json: str, method: str, window: int,
                      level: float) -> dict:
    """Out-of-sample one-day backtest of the hedge book's futures P&L, plus vol forecasts and moments."""
    pos = json.loads(positions_json)
    snap = json.loads(snapshot_json)
    px = prices_df[list(REQUIRED_TICKERS)]
    exposures = np.array([-sum(p["hedged_qty"] for p in pos if p["ticker"] == t) * snap[t]["spot_price"]
                          for t in REQUIRED_TICKERS])
    step = 5 if method == "garch" else 2
    bt = var_backtest(px, exposures, method, window=window, level=level, step=step, n_sims=2000)
    vols = vol_forecast_evaluation(px, ("gaussian", "ewma", "garch"), window=window, step=10)
    moments = moment_comparison(px, method, window=window, n_paths=10_000)
    return {"backtest": bt, "vol_eval": vols, "moments": moments,
            "corr_gap": moments.attrs.get("max_correlation_gap", float("nan"))}


@st.cache_data(ttl=HISTORY_TTL_SECONDS, show_spinner=False)
def cached_ttf_2022() -> pd.Series:
    return fetch_ttf_2022()


@st.cache_data(show_spinner=False, max_entries=4)
def cached_case_study(observed: pd.Series, alt_line: float, opening_cash: float, hedge_ratio: float) -> dict:
    a = CaseStudyAssumptions(alternative_line=alt_line, opening_cash=opening_cash, hedge_ratio=hedge_ratio)
    out = run_case_study(observed, a)
    out["ledgers"] = {name: {k: v[0].copy() for k, v in run.ledger.items()} for name, run in out.pop("runs").items()}
    return out


@st.cache_data(show_spinner=False, max_entries=32)
def cached_book_scenario(inputs: tuple[str, str, str, str], margin_json: str, ts_json: str, shocks_json: str):
    positions, receivables, terms, snapshot = _load_inputs(*inputs)
    return run_book_scenario(positions, receivables, snapshot, terms, MarginSettings.model_validate_json(margin_json),
                             TreasurySettings.model_validate_json(ts_json), json.loads(shocks_json))


@st.cache_data(show_spinner=False, max_entries=16)
def cached_credit_stress(inputs: tuple[str, str, str, str], margin_json: str, ts_json: str,
                         shocks_json: str) -> pd.DataFrame:
    positions, receivables, terms, snapshot = _load_inputs(*inputs)
    return credit_stress_table(positions, receivables, snapshot, terms, MarginSettings.model_validate_json(margin_json),
                               TreasurySettings.model_validate_json(ts_json), json.loads(shocks_json))
