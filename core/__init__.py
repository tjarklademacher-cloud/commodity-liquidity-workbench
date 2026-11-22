"""Pure quantitative core: schemas, market data, haircuts, borrowing base and stress engines.

Nothing in this package may import streamlit or plotly. Public names are resolved lazily (PEP 562)
so that ``python -m core.data_feed`` runs without a double-import warning.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "HistoricalReplayResult": "core.schemas",
    "METALS_TICKERS": "core.schemas",
    "MonteCarloResult": "core.schemas",
    "SOFTS_TICKERS": "core.schemas",
    "ShockGridResult": "core.schemas",
    "evaluate_shock_matrix": "core.scenario_analytics",
    "run_historical_replay": "core.scenario_analytics",
    "run_liquidity_monte_carlo": "core.scenario_analytics",
    "run_shock_grid": "core.scenario_analytics",
    "BorrowingBaseResult": "core.schemas",
    "CAP_TIER_OPTIONS": "core.schemas",
    "COMMODITY_SPECS": "core.schemas",
    "CapTierComparison": "core.schemas",
    "CommoditySpec": "core.schemas",
    "DEFAULT_FACILITY_TERMS": "core.schemas",
    "DEFAULT_INVENTORY_POSITIONS": "core.schemas",
    "DEFAULT_RECEIVABLES": "core.schemas",
    "DEFAULT_SAMPLE_CSV_PATH": "core.data_feed",
    "FacilityTerms": "core.schemas",
    "InventoryPosition": "core.schemas",
    "InventoryValuation": "core.schemas",
    "LiquidityRegime": "core.schemas",
    "MarketLoadResult": "core.schemas",
    "MarketQuote": "core.schemas",
    "REQUIRED_TICKERS": "core.schemas",
    "ReceivableItem": "core.schemas",
    "ReceivableValuation": "core.schemas",
    "STRESS_SCENARIO_PRESETS": "core.schemas",
    "StressScenarioResult": "core.schemas",
    "StressSweepResult": "core.schemas",
    "WaterfallLine": "core.schemas",
    "build_bb_certificate_csv": "core.stress_engine",
    "build_bb_certificate_json": "core.stress_engine",
    "build_market_snapshot": "core.data_feed",
    "calculate_borrowing_base": "core.borrowing_base",
    "classify_liquidity_regime": "core.stress_engine",
    "compare_concentration_cap_tiers": "core.borrowing_base",
    "compute_inventory_valuation": "core.haircuts",
    "compute_portfolio_inventory": "core.haircuts",
    "compute_realized_volatility": "core.data_feed",
    "generate_sample_csv": "core.data_feed",
    "inventory_valuations_to_dataframe": "core.haircuts",
    "load_custom_csv_market_data": "core.data_feed",
    "load_live_market_data": "core.data_feed",
    "load_sample_market_data": "core.data_feed",
    "receivable_valuations_to_dataframe": "core.borrowing_base",
    "run_stress_scenario": "core.stress_engine",
    "run_stress_sweep": "core.stress_engine",
    "solve_concentration_cap_fixed_point": "core.borrowing_base",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        value = getattr(importlib.import_module(_EXPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'core' has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__
