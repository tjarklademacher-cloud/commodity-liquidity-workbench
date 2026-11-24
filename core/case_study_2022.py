"""Flagship case study: a hypothetical hedged gas storage merchant through the 2022 European gas crisis.

Observed data: daily closes of Yahoo Finance's `TTF=F` series (Dutch TTF natural gas futures, EUR per MWh)
for 2022, fetched at run time. It is a continuous front-month series assembled by the data vendor, so it
contains roll jumps between contract months and is a proxy for the price of any single contract. Dates missing
from the vendor series (exchange holidays) are forward filled onto the engine's Monday-to-Friday calendar,
which produces zero price change on those days. Nothing is bundled or synthesized in place of the observed
series: if it cannot be fetched, the case study is not run.

Hypothetical company: "Hypothetical EU gas merchant". Every balance-sheet figure, facility term, customer,
haircut, margin and basis parameter below is an illustrative assumption chosen for the demonstration. It does
not represent any real company, and no real company's financial statements were used.

Mechanism demonstrated: the merchant holds gas in storage and is short TTF futures against it (an economically
effective hedge). As prices rise, variation margin is paid daily in cash while the stored gas gains value that
is only partly financeable (haircuts, facility commitment cap, weekly borrowing base redetermination). Higher
volatility raises volatility-scaled initial margin and dynamic haircuts at the same time. Customer delays and a
default reduce collections when liquidity is scarcest. The variants switch these effects on one at a time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from pydantic import BaseModel

from core.credit_risk import CreditScenario
from core.futures_curve import build_market_paths, business_dates, ou_basis_paths
from core.liquidity_metrics import deterministic_summary
from core.treasury_engine import TreasuryRun, default_credit_outcomes, run_treasury
from core.treasury_schemas import (
    Counterparty,
    FacilityConfig,
    FuturesPosition,
    HaircutPolicy,
    Instrument,
    Invoice,
    LiquiditySource,
    MarginPolicy,
    PhysicalPosition,
    PhysicalTrade,
    TreasuryBook,
    TreasuryConfig,
)

TICKER = "TTF=F"
START, END = "2022-01-03", "2022-12-30"
CURRENCY = "EUR"
LOT_MWH = 720.0
DATA_NOTE = ("Observed: Yahoo Finance TTF=F daily close (continuous front-month Dutch TTF gas futures, EUR/MWh), "
             "2022. Vendor-assembled continuous series with roll jumps; holidays forward filled.")
COMPANY_NOTE = ("Hypothetical company. Inventory, hedges, cash, facility, customers, haircuts, margin and basis "
                "parameters are illustrative assumptions, not data about any real firm.")
FETCH_TIMEOUT_SECONDS = 15


class CaseStudyAssumptions(BaseModel):
    storage_mwh: float = 1_000_000.0
    hedge_ratio: float = 1.0
    opening_cash: float = 25_000_000.0
    commitment: float = 400_000_000.0
    opening_drawn: float = 60_000_000.0
    interest_rate: float = 0.03
    commitment_fee: float = 0.004
    base_haircut: float = 0.20
    reference_vol: float = 0.60
    vol_coefficient: float = 0.50
    max_haircut: float = 0.60
    opening_im_pct: float = 0.15
    mpor_days: int = 2
    im_review_days: int = 5
    redetermination_days: int = 5
    drawdown_delay_days: int = 1
    min_operating_cash: float = 5_000_000.0
    alternative_line: float = 150_000_000.0
    alternative_line_delay: int = 2
    basis_vol: float = 0.25
    basis_kappa: float = 6.0
    location_shock: float = -0.08
    monthly_sales_mwh: float = 60_000.0
    sales_terms_days: int = 30
    injection_mwh: float = 50_000.0
    purchase_terms_days: int = 20


def fetch_ttf_2022(download: Callable[..., Any] = yf.download) -> pd.Series:
    """Observed TTF=F closes (EUR/MWh) for 2022. Raises ValueError when no usable data is returned."""
    raw = download(tickers=[TICKER], start=START, end=(pd.Timestamp(END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                   interval="1d", auto_adjust=True, progress=False, threads=False, timeout=FETCH_TIMEOUT_SECONDS)
    if raw is None or len(raw) == 0:
        raise ValueError("Yahoo Finance returned no TTF data.")
    close = raw["Close"] if "Close" in raw else raw
    if isinstance(close, pd.DataFrame):
        close = close[TICKER] if TICKER in close.columns else close.iloc[:, 0]
    s = pd.to_numeric(close, errors="coerce")
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).tz_localize(None) if getattr(s.index, "tz", None) else \
        pd.DatetimeIndex(pd.to_datetime(s.index))
    s = s.where(s > 0).dropna()
    if len(s) < 200:
        raise ValueError(f"Only {len(s)} TTF observations for 2022; at least 200 are needed.")
    return s.rename(TICKER)


def align_to_engine_calendar(observed: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray, int]:
    """Observed closes on the engine's business calendar from the first observation. Returns (dates, prices,
    number of forward-filled days)."""
    start = observed.index[0]
    end = observed.index[-1]
    n_days = len(pd.bdate_range(start, end)) - 1
    dates = business_dates(start, n_days)
    aligned = observed.reindex(dates)
    filled = int(aligned.isna().sum())
    aligned = aligned.ffill()
    return dates, aligned.to_numpy(dtype=float), filled


def _book(a: CaseStudyAssumptions, horizon: int, operations: bool, p0: float) -> TreasuryBook:
    lots = -round(a.hedge_ratio * a.storage_mwh / LOT_MWH)
    cps = [
        Counterparty(id="U", name="Customer A (utility, hypothetical)", industry="utility", region="DE", rating="BBB",
                     commodity_link=TICKER, commodity_loading=0.3),
        Counterparty(id="I", name="Customer B (industrial, hypothetical)", industry="chemicals", region="DE",
                     rating="BB", insurance_coverage=0.9, insurance_deductible=500_000.0, claim_delay_days=180,
                     commodity_link=TICKER, commodity_loading=-0.5),
        Counterparty(id="M", name="Customer C (municipal, hypothetical)", industry="municipal", region="AT", rating="A"),
        Counterparty(id="T", name="Customer D (trader, hypothetical)", industry="trading", region="CH", rating="BBB"),
    ]
    invs = [Invoice(counterparty_id="U", amount=14_000_000.0, due_in_days=20),
            Invoice(counterparty_id="I", amount=9_000_000.0, due_in_days=25),
            Invoice(counterparty_id="M", amount=6_000_000.0, due_in_days=15),
            Invoice(counterparty_id="T", amount=5_000_000.0, due_in_days=10)]
    trades: list[PhysicalTrade] = []
    if operations:
        # injection season purchases (April to September), withdrawal season sales (all year, monthly)
        for d in range(21, horizon + 1, 21):
            month = (d // 21) + 1
            if 4 <= month <= 9:
                trades.append(PhysicalTrade(day=d, instrument_id=TICKER, side="buy", quantity=a.injection_mwh,
                                            payment_days=a.purchase_terms_days))
            buyer = ["U", "I", "M", "T"][month % 4]
            trades.append(PhysicalTrade(day=d, instrument_id=TICKER, side="sell", quantity=a.monthly_sales_mwh,
                                        payment_days=a.sales_terms_days, counterparty_id=buyer))
    return TreasuryBook(
        instruments=[Instrument(id=TICKER, name="TTF natural gas", unit="MWh", contract_size=LOT_MWH, currency=CURRENCY,
                                exchange="ICE Endex (proxy)")],
        physical=[PhysicalPosition(instrument_id=TICKER, quantity=a.storage_mwh, purchase_price=p0,
                                   location="Hypothetical storage site", hedge_ratio_target=a.hedge_ratio)],
        futures=[FuturesPosition(instrument_id=TICKER, contract_id="TTF front (continuous proxy)",
                                 signed_contracts=lots, entry_price=p0)],
        counterparties=cps, invoices=invs, trades=trades)


def _config(a: CaseStudyAssumptions, horizon: int, dynamic: bool, vol_im: bool, extra_line: bool, p0: float,
            vol0: float, start: pd.Timestamp) -> TreasuryConfig:
    im_pc = a.opening_im_pct * p0 * LOT_MWH
    z = a.opening_im_pct / (vol0 * np.sqrt(a.mpor_days / 252.0))
    margin = MarginPolicy(mode="vol_scaled", z_score=float(min(6.0, z)), mpor_days=a.mpor_days,
                          review_days=a.im_review_days, floor_fraction=1.0) if vol_im else \
        MarginPolicy(mode="fixed_per_contract", im_per_contract=im_pc)
    hc = HaircutPolicy(base=a.base_haircut, vol_coefficient=a.vol_coefficient, reference_vol=a.reference_vol,
                       min_haircut=0.10, max_haircut=a.max_haircut, dynamic=dynamic)
    lines = [LiquiditySource(name="Alternative committed line", limit=a.alternative_line,
                             delay_days=a.alternative_line_delay, interest_rate=a.interest_rate + 0.02)] \
        if extra_line else []
    fac = FacilityConfig(commitment=a.commitment, opening_drawn=a.opening_drawn, interest_rate=a.interest_rate,
                         commitment_fee=a.commitment_fee, ar_ig_advance_rate=0.85, ar_non_ig_advance_rate=0.75,
                         concentration_cap_pct=0.25, drawdown_delay_days=a.drawdown_delay_days,
                         redetermination_days=a.redetermination_days, cure_grace_days=2)
    return TreasuryConfig(start_date=start.date(), horizon_days=horizon, opening_cash=a.opening_cash,
                          min_operating_cash=a.min_operating_cash, facility=fac, extra_lines=lines,
                          haircuts={TICKER: hc}, margins={TICKER: margin}, hedge_follows_inventory=True)


CASE_VARIANTS = (
    "Static financing",
    "Dynamic treasury (operations and payment timing)",
    "Plus dynamic haircuts and volatility-scaled margin",
    "Plus basis risk (synthetic location basis)",
    "Plus counterparty credit stress",
    "Combined adverse",
    "Combined adverse with alternative committed line",
)


def credit_stress(dates: pd.DatetimeIndex) -> CreditScenario:
    """Hypothetical: from mid-June customer A pays 45 days late; customer B defaults in September."""
    june = int(dates.searchsorted(pd.Timestamp("2022-06-15")))
    sept = int(dates.searchsorted(pd.Timestamp("2022-09-15")))
    return CreditScenario(name="Hypothetical delay and default", extra_delay_days={"U": 45},
                          delays_apply_from_day=june, default_day={"I": sept}, downgrade_day={"I": june})


def run_case_study(observed: pd.Series, a: CaseStudyAssumptions | None = None, seed: int = 2022,
                   ewma_lambda: float = 0.94) -> dict[str, object]:
    a = a or CaseStudyAssumptions()
    dates, px, filled = align_to_engine_calendar(observed)
    h = len(dates) - 1
    p0 = float(px[0])
    hist = np.log(observed.iloc[:21].to_numpy())
    vol0 = float(np.std(np.diff(hist), ddof=1) * np.sqrt(252.0)) if len(hist) > 5 else a.reference_vol
    front = px[None, :, None]
    basis = ou_basis_paths(1, h, np.zeros(1), np.zeros(1), np.array([a.basis_kappa]), np.array([a.basis_vol]), seed,
                           shock_path=np.where(np.arange(h + 1)[:, None] >= int(dates.searchsorted(pd.Timestamp("2022-06-01"))),
                                               a.location_shock, 0.0))
    stress = credit_stress(dates)
    runs: dict[str, TreasuryRun] = {}
    rows = []
    for name in CASE_VARIANTS:
        ops = name != "Static financing"
        dyn = name not in CASE_VARIANTS[:2]
        use_basis = name in (CASE_VARIANTS[3], CASE_VARIANTS[5], CASE_VARIANTS[6])
        use_credit = name in (CASE_VARIANTS[4], CASE_VARIANTS[5], CASE_VARIANTS[6])
        line = name == CASE_VARIANTS[6]
        book = _book(a, h, ops, p0)
        cfg = _config(a, h, dyn, dyn, line, p0, vol0, dates[0])
        mk = build_market_paths(dates, front, np.array([vol0]), basis=basis if use_basis else None,
                                ewma_lambda=ewma_lambda,
                                labels={"futures": "observed (Yahoo TTF=F, continuous front month)",
                                        "basis": "synthetic OU" if use_basis else "zero"})
        if not dyn:
            mk.vol = np.full_like(mk.vol, vol0)
        credit = default_credit_outcomes(book, cfg, mk, stress if use_credit else None)
        run = run_treasury(book, cfg, mk, credit)
        runs[name] = run
        d = deterministic_summary(run)
        rows.append({"variant": name, **d, "peak_vol": float(mk.vol[0, :, 0].max()),
                     "reconciliation_residual": run.reconciliation_residual})
    table = pd.DataFrame(rows)
    return {"table": table, "runs": runs, "dates": dates, "prices": px, "forward_filled_days": filled,
            "opening_vol": vol0, "assumptions": a, "data_note": DATA_NOTE, "company_note": COMPANY_NOTE,
            "basis_path": basis[0, :, 0]}
