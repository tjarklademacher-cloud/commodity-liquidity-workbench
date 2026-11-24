"""Treasury analytics on the workbench's six-commodity book: adapter, deterministic scenarios, haircut
sensitivity, Monte Carlo through the daily engine.

The adapter maps the existing inputs (InventoryPosition, ReceivableItem, FacilityTerms, MarginSettings) into
the instrument-agnostic engine. Mapping choices (all disclosed in the UI):
- One counterparty per receivable row. Rating bucket is illustrative: investment grade -> BBB, otherwise BB,
  with the illustrative PD table in `treasury_schemas`. This is not an assessment of any named company.
- Collections follow the existing ledger's rule: current invoices fall due after the invoice terms, overdue
  invoices are paid after the overdue delay, invoices past the aging limit are not collected in the horizon.
- Haircut policy per commodity from `CommoditySpec` (base, volatility slope, reference vol, unhedged penalty,
  floor and cap) under the haircut-only convention, so day-0 eligible inventory equals the legacy figure.
- Hedges are short futures in whole contracts per the margin settings' contract policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from core.advanced_schemas import MarginSettings
from core.credit_risk import CreditScenario, normal_scores, simulate_credit_outcomes
from core.futures_curve import MarketPaths, build_market_paths, business_dates, ou_basis_paths
from core.liquidity_metrics import LiquidityMetrics, compute_metrics, deterministic_summary, path_summary
from core.margin_engine import signed_short_contracts
from core.return_models import ReturnModel, price_paths, simulate_returns
from core.schemas import COMMODITY_SPECS, REQUIRED_TICKERS, FacilityTerms, InventoryPosition, MarketQuote, \
    ReceivableItem
from core.treasury_engine import TreasuryRun, default_credit_outcomes, invoice_schedule, run_treasury
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

HORIZON_CHOICES = (1, 5, 10, 30, 60, 90)
POOL_BUYER = "Trade buyers (pool)"


class TreasurySettings(BaseModel):
    """Interface-level settings. Defaults are illustrative; every rate, delay and size can be edited."""

    horizon_days: int = Field(default=30, ge=1, le=90)
    ramp_days: int = Field(default=5, ge=1, le=90)
    min_operating_cash: float = Field(default=0.0, ge=0.0)
    opex_per_day: float = Field(default=0.0, ge=0.0)
    revolver_rate: float = Field(default=0.0, ge=0.0, le=0.5)
    commitment_fee: float = Field(default=0.0, ge=0.0, le=0.05)
    drawdown_delay_days: int = Field(default=1, ge=0, le=10)
    redetermination_days: int = Field(default=1, ge=1, le=20)
    cure_grace_days: int = Field(default=2, ge=0, le=20)
    sweep_surplus: bool = True
    dynamic_haircuts: bool = True
    lender_stress_pts: float = Field(default=0.0, ge=0.0, le=0.5)
    max_haircut_increase: float | None = Field(default=None, ge=0.0, le=0.5)
    margin_mode: Literal["fixed_per_contract", "pct_notional", "vol_scaled"] = "fixed_per_contract"
    im_review_days: int = Field(default=1, ge=1, le=20)
    im_release_lag: int = Field(default=0, ge=0, le=10)
    im_scenario_multiplier: float = Field(default=1.0, ge=0.0, le=10.0)
    apply_trigger_rule: bool = True
    vol_multiplier: float = Field(default=1.0, ge=0.5, le=5.0)
    ewma_lambda: float = Field(default=0.94, gt=0.5, lt=1.0)
    invoice_terms_days: int = Field(default=60, ge=0, le=180)
    overdue_delay_days: int = Field(default=15, ge=0, le=180)
    extra_line_limit: float = Field(default=0.0, ge=0.0)
    extra_line_delay: int = Field(default=2, ge=0, le=10)
    emergency_limit: float = Field(default=0.0, ge=0.0)
    emergency_rate: float = Field(default=0.15, ge=0.0, le=1.0)
    turnover_pct: float = Field(default=0.0, ge=0.0, le=0.5)
    turnover_every_days: int = Field(default=5, ge=1, le=60)
    sales_terms_days: int = Field(default=30, ge=0, le=180)
    purchase_terms_days: int = Field(default=10, ge=0, le=180)
    basis_shock: float = Field(default=0.0, ge=-0.5, le=0.5)
    basis_vol: float = Field(default=0.0, ge=0.0, le=1.0)
    basis_kappa: float = Field(default=4.0, ge=0.0, le=50.0)
    basis_rho: float = Field(default=0.0, ge=-1.0, le=1.0)
    lgd: float = Field(default=0.6, ge=0.0, le=1.0)
    insurance_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    insurance_claim_days: int = Field(default=180, ge=0, le=720)


def build_book(positions: Sequence[InventoryPosition], receivables: Sequence[ReceivableItem],
               terms: FacilityTerms, margin: MarginSettings, ts: TreasurySettings,
               market_snapshot: Mapping[str, MarketQuote] | None = None) -> TreasuryBook:
    specs = COMMODITY_SPECS
    tick = [p.ticker for p in positions]
    instruments = [Instrument(id=t, name=specs[t].name, unit=specs[t].unit, contract_size=specs[t].contract_multiplier,
                              exchange=specs[t].exchange) for t in tick]
    physical = [PhysicalPosition(instrument_id=p.ticker, quantity=p.physical_qty) for p in positions]
    futures = []
    for p in positions:
        s, _, _ = signed_short_contracts(p.hedged_qty, p.ticker, margin, specs)
        if s != 0:
            futures.append(FuturesPosition(instrument_id=p.ticker, contract_id=f"{p.ticker} front", signed_contracts=s))
    name_to_ticker = {specs[t].name.lower(): t for t in REQUIRED_TICKERS}
    cps, invs = [], []
    for i, r in enumerate(receivables):
        overdue = r.days_past_due > 0
        beyond = r.days_past_due > terms.ar_aging_limit_days
        cps.append(Counterparty(
            id=f"cp{i}", name=r.buyer_name, rating="BBB" if r.is_investment_grade else "BB",
            expected_delay_days=365 if beyond else (ts.overdue_delay_days if overdue else 0),
            commodity_link=name_to_ticker.get(r.commodity.lower().split(" ")[0]),
            commodity_loading=0.0, industry=r.commodity, lgd=ts.lgd, insurance_coverage=ts.insurance_coverage,
            claim_delay_days=ts.insurance_claim_days))
        invs.append(Invoice(counterparty_id=f"cp{i}", amount=r.gross_amount, days_past_due=r.days_past_due,
                            due_in_days=0 if overdue else ts.invoice_terms_days, commodity=r.commodity))
    trades: list[PhysicalTrade] = []
    if ts.turnover_pct > 0:
        cps.append(Counterparty(id="pool", name=POOL_BUYER, rating="BBB", industry="diversified", lgd=ts.lgd))
        for d in range(ts.turnover_every_days, ts.horizon_days + 1, ts.turnover_every_days):
            for p in positions:
                q = p.physical_qty * ts.turnover_pct
                if q <= 0:
                    continue
                trades.append(PhysicalTrade(day=d, instrument_id=p.ticker, side="sell", quantity=q,
                                            payment_days=ts.sales_terms_days, counterparty_id="pool"))
                trades.append(PhysicalTrade(day=d, instrument_id=p.ticker, side="buy", quantity=q,
                                            payment_days=ts.purchase_terms_days))
    return TreasuryBook(instruments=instruments, physical=physical, futures=futures, counterparties=cps,
                        invoices=invs, trades=trades)


def build_config(positions: Sequence[InventoryPosition], terms: FacilityTerms, margin: MarginSettings,
                 ts: TreasurySettings, opening_vols: Mapping[str, float]) -> TreasuryConfig:
    specs = COMMODITY_SPECS
    haircuts, margins = {}, {}
    for p in positions:
        s = specs[p.ticker]
        haircuts[p.ticker] = HaircutPolicy(
            base=s.base_haircut, vol_coefficient=s.vol_slope, reference_vol=s.ref_vol,
            unhedged_penalty=s.unhedged_penalty_rate, min_haircut=s.min_haircut, max_haircut=s.max_haircut,
            dynamic=ts.dynamic_haircuts, max_increase_per_redetermination=ts.max_haircut_increase)
        prof = margin.profiles[p.ticker]
        margins[p.ticker] = MarginPolicy(
            mode=ts.margin_mode, im_per_contract=prof.im_per_contract, pct_notional=0.0,
            review_days=ts.im_review_days, release_lag_days=ts.im_release_lag,
            stepup_trigger=terms.im_stepup_trigger if ts.apply_trigger_rule else None,
            stepup_fraction=terms.im_stepup_fraction, provenance=prof.assumption_type)
    lines = []
    if terms.margin_line_limit > 0:
        lines.append(LiquiditySource(name="Margin financing line", limit=terms.margin_line_limit, delay_days=0,
                                     interest_rate=0.08, margin_only=True))
    if ts.extra_line_limit > 0:
        lines.append(LiquiditySource(name="Additional committed line", limit=ts.extra_line_limit,
                                     delay_days=ts.extra_line_delay, interest_rate=ts.revolver_rate + 0.01))
    emergency = LiquiditySource(name="Emergency funding", limit=ts.emergency_limit, delay_days=1,
                                interest_rate=ts.emergency_rate, enabled=ts.emergency_limit > 0,
                                provenance="synthetic_stress")
    fac = FacilityConfig(
        commitment=terms.facility_limit, opening_drawn=terms.drawn_revolver, interest_rate=ts.revolver_rate,
        commitment_fee=ts.commitment_fee, ar_ig_advance_rate=terms.ar_ig_advance_rate,
        ar_non_ig_advance_rate=terms.ar_non_ig_advance_rate, ar_aging_limit_days=terms.ar_aging_limit_days,
        concentration_cap_pct=terms.concentration_cap_pct, dilution_reserve_pct=terms.ar_dilution_reserve_pct,
        drawdown_delay_days=ts.drawdown_delay_days, redetermination_days=ts.redetermination_days,
        cure_grace_days=ts.cure_grace_days, sweep_surplus=ts.sweep_surplus)
    return TreasuryConfig(horizon_days=ts.horizon_days, opening_cash=terms.unrestricted_cash,
                          min_operating_cash=ts.min_operating_cash, opex_per_calendar_day=ts.opex_per_day,
                          facility=fac, extra_lines=lines, emergency=emergency, haircuts=haircuts, margins=margins,
                          hedge_follows_inventory=ts.turnover_pct > 0)


def _with_pct_margin(cfg: TreasuryConfig, f0: Mapping[str, float], ts: TreasurySettings,
                     vols: Mapping[str, float]) -> TreasuryConfig:
    """For pct_notional and vol_scaled modes, calibrate so the opening IM per contract equals the margin
    settings' per-contract figure (so modes differ only in how IM responds, not in its opening level)."""
    if ts.margin_mode == "fixed_per_contract":
        return cfg
    new = {}
    for t, pol in cfg.margins.items():
        size = COMMODITY_SPECS[t].contract_multiplier
        notional = max(f0[t] * size, 1e-9)
        if ts.margin_mode == "pct_notional":
            new[t] = pol.model_copy(update={"pct_notional": min(1.0, pol.im_per_contract / notional)})
        else:
            sig = max(vols[t], 1e-6)
            z = pol.im_per_contract / (sig * np.sqrt(pol.mpor_days / 252.0) * notional)
            new[t] = pol.model_copy(update={"z_score": float(min(6.0, z))})
    return cfg.model_copy(update={"margins": new})


def scenario_market(positions: Sequence[InventoryPosition], snapshot: Mapping[str, MarketQuote],
                    ts: TreasurySettings, shocks: Mapping[str, float], start: str = "2026-11-20") -> MarketPaths:
    """Deterministic path: futures move linearly to the shock over `ramp_days`, then stay flat; volatility
    follows max(EWMA of the path, opening vol scaled up linearly to `vol_multiplier`); basis moves to
    `basis_shock` over the same ramp. Scenario values, not forecasts."""
    tick = [p.ticker for p in positions]
    h = ts.horizon_days
    dates = business_dates(start, h)
    f0 = np.array([snapshot[t].spot_price for t in tick])
    sh = np.array([shocks.get(t, 0.0) for t in tick])
    frac = np.minimum(1.0, np.arange(h + 1) / ts.ramp_days)[:, None]
    front = (f0 * (1.0 + sh * frac))[None]
    vol0 = np.array([snapshot[t].realized_vol_30d for t in tick])
    basis = (ts.basis_shock * frac)[None] * np.ones((1, 1, len(tick)))
    mk = build_market_paths(dates, front, vol0, basis=basis, ewma_lambda=ts.ewma_lambda,
                            labels={"futures": "deterministic scenario", "basis": "scenario assumption"})
    scaled = vol0 * (1.0 + (ts.vol_multiplier - 1.0) * frac)
    mk.vol = np.maximum(mk.vol, scaled[None])
    return mk


def run_book_scenario(positions, receivables, snapshot, terms, margin, ts: TreasurySettings,
                      shocks: Mapping[str, float], credit_scenario: CreditScenario | None = None) -> TreasuryRun:
    book = build_book(positions, receivables, terms, margin, ts, snapshot)
    vols = {p.ticker: snapshot[p.ticker].realized_vol_30d for p in positions}
    cfg = _with_pct_margin(build_config(positions, terms, margin, ts, vols),
                           {t: snapshot[t].spot_price for t in vols}, ts, vols)
    mk = scenario_market(positions, snapshot, ts, shocks)
    frac = np.minimum(1.0, np.arange(ts.horizon_days + 1) / ts.ramp_days)[:, None]
    stress = ts.lender_stress_pts * frac * np.ones((1, len(positions)))
    im_mult = 1.0 + (ts.im_scenario_multiplier - 1.0) * frac * np.ones((1, len(positions)))
    credit = default_credit_outcomes(book, cfg, mk, credit_scenario)
    return run_treasury(book, cfg, mk, credit, stress, im_mult)


HAIRCUT_VARIANTS: dict[str, dict[str, object]] = {
    "Static haircuts": {"dynamic_haircuts": False, "lender_stress_pts": 0.0},
    "Dynamic (path volatility)": {"dynamic_haircuts": True, "lender_stress_pts": 0.0},
    "Lender tightening +5 points": {"dynamic_haircuts": True, "lender_stress_pts": 0.05},
    "Lender tightening +15 points": {"dynamic_haircuts": True, "lender_stress_pts": 0.15},
    "+15 points, capped at +2 per redetermination": {
        "dynamic_haircuts": True, "lender_stress_pts": 0.15, "max_haircut_increase": 0.02},
    "Dynamic, weekly redetermination": {"dynamic_haircuts": True, "lender_stress_pts": 0.0,
                                        "redetermination_days": 5},
}
"""Illustrative haircut assumptions for the sensitivity table (synthetic stress, not lender data)."""


def haircut_sensitivity(positions, receivables, snapshot, terms, margin, ts: TreasurySettings,
                        shocks: Mapping[str, float]) -> pd.DataFrame:
    rows = []
    base = None
    for name, upd in HAIRCUT_VARIANTS.items():
        s = TreasurySettings.model_validate({**ts.model_dump(), **upd})
        run = run_book_scenario(positions, receivables, snapshot, terms, margin, s, shocks)
        d = deterministic_summary(run)
        if base is None:
            base = d
        rows.append({"assumption": name, "min_liquidity": d["min_liquidity"],
                     "change_vs_static": d["min_liquidity"] - base["min_liquidity"],
                     "peak_funding": d["peak_funding"], "first_breach": d["first_breach_date"] or "none",
                     "breach_days": d["breach_days"],
                     "peak_avg_haircut": float(run.ledger["avg_haircut"][0].max())})
    return pd.DataFrame(rows)


class MonteCarloOutput(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    metrics: LiquidityMetrics
    method: str
    n_paths: int
    seed: int
    horizon_days: int
    credit_mode: str
    liquidity_fan: dict[str, list[float]]
    funding_samples: list[float]
    deterioration_samples: list[float]
    economic_pnl_samples: list[float]
    breach_samples: list[bool]
    peak_utilization_samples: list[float]
    opening_im: float
    notes: list[str]


def run_book_monte_carlo(positions, receivables, snapshot, terms, margin, ts: TreasurySettings,
                         model: ReturnModel, n_paths: int = 500, seed: int = 31, confidence: float = 0.95,
                         credit_mode: Literal["scheduled", "simulated"] = "scheduled",
                         asset_correlation: float = 0.2, wrong_way_loading: float = 0.0) -> MonteCarloOutput:
    """Simulated daily price paths (and synthetic basis if basis_vol > 0) through the daily engine. Each path
    carries its own state; maximum funding needs are taken over the whole path."""
    tick = [p.ticker for p in positions]
    if list(model.tickers) != tick:
        model_idx = [model.tickers.index(t) for t in tick]
    else:
        model_idx = list(range(len(tick)))
    h = ts.horizon_days
    rets = simulate_returns(model, h, n_paths, seed)[:, :, model_idx]
    f0 = np.array([snapshot[t].spot_price for t in tick])
    front = price_paths(f0, rets, model.return_type)
    front = np.maximum(front, 0.0)
    dates = business_dates("2026-11-20", h)
    vol0 = np.array([snapshot[t].realized_vol_30d for t in tick])
    basis = None
    if ts.basis_vol > 0 or ts.basis_shock != 0:
        sd = np.sqrt(np.diag(model.cov_daily))[model_idx]
        z_b = rets / np.where(sd > 0, sd, 1.0)
        frac = np.minimum(1.0, np.arange(h + 1) / ts.ramp_days)[:, None]
        basis = ou_basis_paths(n_paths, h, np.zeros(len(tick)), np.zeros(len(tick)), np.full(len(tick), ts.basis_kappa),
                               np.full(len(tick), ts.basis_vol), seed + 7, z_b, np.full(len(tick), ts.basis_rho),
                               shock_path=ts.basis_shock * frac * np.ones((1, len(tick))))
    mk = build_market_paths(dates, front, vol0 * ts.vol_multiplier, basis=basis, ewma_lambda=ts.ewma_lambda,
                            labels={"futures": f"simulated ({model.method})"})
    book = build_book(positions, receivables, terms, margin, ts, snapshot)
    vols = {t: snapshot[t].realized_vol_30d for t in tick}
    cfg = _with_pct_margin(build_config(positions, terms, margin, ts, vols), dict(zip(tick, f0)), ts, vols)
    stress = ts.lender_stress_pts * np.minimum(1.0, np.arange(h + 1) / ts.ramp_days)[:, None] * np.ones((1, len(tick)))
    im_mult = 1.0 + (ts.im_scenario_multiplier - 1.0) * np.minimum(1.0, np.arange(h + 1) / ts.ramp_days)[:, None]
    notes = list(model.notes)
    if credit_mode == "simulated":
        if wrong_way_loading != 0:
            book = book.model_copy(update={"counterparties": [
                c.model_copy(update={"commodity_loading": wrong_way_loading}) if c.commodity_link else c
                for c in book.counterparties]})
        cum = np.log(np.maximum(front[:, -1, :], 1e-12) / f0) if model.return_type == "log" else front[:, -1, :] - f0
        factors = {t: normal_scores(cum[:, i]) for i, t in enumerate(tick)}
        credit = simulate_credit_outcomes(book.counterparties, invoice_schedule(book, mk), dates, n_paths, seed + 13,
                                          asset_correlation, factors, min_eligible_rating=cfg.facility.min_eligible_rating)
        notes.append("Credit outcomes simulated with an illustrative one-factor model; PDs from the illustrative "
                     "rating table.")
    else:
        credit = default_credit_outcomes(book, cfg, mk)
    run = run_treasury(book, cfg, mk, credit, stress, im_mult, record="summary")
    met = compute_metrics(run, confidence)
    liq = run.ledger["available_liquidity"]
    s = path_summary(run)
    return MonteCarloOutput(
        metrics=met, method=model.method, n_paths=n_paths, seed=seed, horizon_days=h, credit_mode=credit_mode,
        liquidity_fan={f"p{q}": np.percentile(liq, q, axis=0).tolist() for q in (1, 5, 25, 50, 75, 95)},
        funding_samples=s["peak_funding"].tolist(), deterioration_samples=s["deterioration"].tolist(),
        economic_pnl_samples=run.ledger["economic_pnl"].sum(axis=1).tolist(), breach_samples=s["breach"].tolist(),
        peak_utilization_samples=s["peak_utilization"].tolist(), opening_im=float(run.ledger["im_posted"][0, 0]),
        notes=notes)


def credit_stress_scenarios(receivables: Sequence[ReceivableItem]) -> dict[str, CreditScenario]:
    """Deterministic, hypothetical credit stresses on the largest counterparty and on all buyers. Counterparty
    ids follow `build_book` (cp0, cp1, ... in receivable order)."""
    if not receivables:
        return {"No credit stress": CreditScenario()}
    big = f"cp{int(np.argmax([r.gross_amount for r in receivables]))}"
    return {
        "No credit stress": CreditScenario(),
        "All buyers pay 15 days late": CreditScenario(name="all late", all_extra_delay_days=15),
        "Largest buyer pays 30 days late": CreditScenario(name="largest late", extra_delay_days={big: 30}),
        "Largest buyer downgraded below eligibility on day 1": CreditScenario(name="downgrade", downgrade_day={big: 1}),
        "Largest buyer defaults on day 5": CreditScenario(name="default", default_day={big: 5}),
        "Largest buyer defaults and others pay 15 days late": CreditScenario(
            name="combined", default_day={big: 5}, all_extra_delay_days=15),
    }


def credit_stress_table(positions, receivables, snapshot, terms, margin, ts: TreasurySettings,
                        shocks: Mapping[str, float]) -> pd.DataFrame:
    rows, base = [], None
    for name, sc in credit_stress_scenarios(receivables).items():
        run = run_book_scenario(positions, receivables, snapshot, terms, margin, ts, shocks, sc)
        d = deterministic_summary(run)
        base = d if base is None else base
        rows.append({"scenario": name, "min_liquidity": d["min_liquidity"],
                     "change_vs_no_stress": d["min_liquidity"] - base["min_liquidity"],
                     "peak_funding": d["peak_funding"], "credit_loss": d["credit_loss"],
                     "first_breach": d["first_breach_date"] or "none", "breach_days": d["breach_days"],
                     "min_receivable_contribution": float(run.ledger["receivable_contribution"][0].min())})
    return pd.DataFrame(rows)
