"""Shared fixtures built from the calibrated seed=42 sample.

`positions`, `receivables`, `terms` and `sweep` pin the original four-commodity book (gold, copper, lumber,
cocoa) and its facility, so the original audited checkpoints keep guarding the engines. The current
six-commodity defaults are covered by the `default_*` fixtures and tests/test_default_book.py.
"""

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


LEGACY_TICKERS = ("GC=F", "HG=F", "LBR=F", "CC=F")
LEGACY_POSITIONS = [p for p in DEFAULT_INVENTORY_POSITIONS if p.ticker in LEGACY_TICKERS]
LEGACY_RECEIVABLES = [
    ReceivableItem(buyer_name="Barry Callebaut AG", commodity="Cocoa", gross_amount=25_000_000.0,
                   days_past_due=18, is_investment_grade=True),
    ReceivableItem(buyer_name="Cargill Cocoa & Chocolate", commodity="Cocoa", gross_amount=21_500_000.0,
                   days_past_due=24, is_investment_grade=True),
    ReceivableItem(buyer_name="Olam Food Ingredients", commodity="Cocoa", gross_amount=8_500_000.0,
                   days_past_due=32, is_investment_grade=True),
    ReceivableItem(buyer_name="West Coast Lumber Supply", commodity="Lumber", gross_amount=5_200_000.0,
                   days_past_due=41, is_investment_grade=False),
    ReceivableItem(buyer_name="Midwest Copper Fabricators", commodity="Copper", gross_amount=4_100_000.0,
                   days_past_due=14, is_investment_grade=False),
    ReceivableItem(buyer_name="Artisanal Roasters Co.", commodity="Cocoa", gross_amount=2_400_000.0,
                   days_past_due=104, is_investment_grade=False),
]
LEGACY_TERMS = FacilityTerms(facility_limit=115_000_000.0, drawn_revolver=109_000_000.0,
                             unrestricted_cash=4_500_000.0, baseline_initial_margin=4_800_000.0)


@pytest.fixture()
def positions() -> list[InventoryPosition]:
    return list(LEGACY_POSITIONS)


@pytest.fixture()
def receivables() -> list[ReceivableItem]:
    return list(LEGACY_RECEIVABLES)


@pytest.fixture()
def terms() -> FacilityTerms:
    return LEGACY_TERMS


@pytest.fixture()
def default_positions() -> list[InventoryPosition]:
    return list(DEFAULT_INVENTORY_POSITIONS)


@pytest.fixture()
def default_receivables() -> list[ReceivableItem]:
    return list(DEFAULT_RECEIVABLES)


@pytest.fixture()
def default_terms() -> FacilityTerms:
    return DEFAULT_FACILITY_TERMS


@pytest.fixture()
def baseline_items(positions, snapshot) -> list[InventoryValuation]:
    return compute_portfolio_inventory(positions, snapshot)


@pytest.fixture()
def baseline_bb(baseline_items, receivables, terms) -> BorrowingBaseResult:
    return calculate_borrowing_base(baseline_items, receivables, terms)


@pytest.fixture(scope="session")
def sweep(snapshot) -> StressSweepResult:
    return run_stress_sweep(list(LEGACY_POSITIONS), list(LEGACY_RECEIVABLES), snapshot, LEGACY_TERMS)
