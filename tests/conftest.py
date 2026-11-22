"""Shared fixtures built from the calibrated seed=42 sample and default portfolio."""

from __future__ import annotations

import pandas as pd
import pytest

from core.borrowing_base import calculate_borrowing_base
from core.data_feed import DEFAULT_SAMPLE_CSV_PATH, load_sample_market_data
from core.haircuts import compute_portfolio_inventory
from core.schemas import (
    DEFAULT_FACILITY_TERMS,
    DEFAULT_INVENTORY_POSITIONS,
    DEFAULT_RECEIVABLES,
    BorrowingBaseResult,
    FacilityTerms,
    InventoryPosition,
    InventoryValuation,
    MarketLoadResult,
    MarketQuote,
    ReceivableItem,
    StressSweepResult,
)
from core.stress_engine import run_stress_sweep

MILLION_TOL = 0.01e6


@pytest.fixture(scope="session")
def sample_prices() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_SAMPLE_CSV_PATH, index_col="Date", parse_dates=True)


@pytest.fixture(scope="session")
def market() -> MarketLoadResult:
    return load_sample_market_data()


@pytest.fixture(scope="session")
def snapshot(market: MarketLoadResult) -> dict[str, MarketQuote]:
    return market.snapshot


@pytest.fixture()
def positions() -> list[InventoryPosition]:
    return list(DEFAULT_INVENTORY_POSITIONS)


@pytest.fixture()
def receivables() -> list[ReceivableItem]:
    return list(DEFAULT_RECEIVABLES)


@pytest.fixture()
def terms() -> FacilityTerms:
    return DEFAULT_FACILITY_TERMS


@pytest.fixture()
def baseline_items(positions, snapshot) -> list[InventoryValuation]:
    return compute_portfolio_inventory(positions, snapshot)


@pytest.fixture()
def baseline_bb(baseline_items, receivables, terms) -> BorrowingBaseResult:
    return calculate_borrowing_base(baseline_items, receivables, terms)


@pytest.fixture(scope="session")
def sweep(snapshot) -> StressSweepResult:
    return run_stress_sweep(
        list(DEFAULT_INVENTORY_POSITIONS), list(DEFAULT_RECEIVABLES), snapshot, DEFAULT_FACILITY_TERMS
    )
