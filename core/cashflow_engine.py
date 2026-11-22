"""Path-dependent daily treasury ledger: margin settlements, financing draws with settlement delays, receivable
collections, deficiency cures and interest. Vectorized across price paths; state carries from day to day.

Daily order of events (applied identically on every path):
 1. Opening balances: cash, revolver drawn, margin line drawn, posted initial margin, funding in transit.
 2. Settlement prices for the day (futures) and physical prices (futures x (1 + basis)).
 3. Variation margin on the day's futures change (net across commodities); initial margin re-set
    (post the increase, receive any release). IM multipliers: user multiplier per commodity, times
    (1 + step-up fraction) once a commodity's cumulative move reaches the trigger (persistent), if the rule is on.
 4. Receivables due today are collected and leave the borrowing base.
 5. Borrowing base recomputed from physical prices and outstanding receivables.
 6. Financing requested on earlier days arrives (debt rises on arrival).
 7. Interest on the opening revolver and margin line balances (simple interest, actual/day-count).
 8. Deficiency: if revolver debt exceeds the borrowing base for longer than the grace period, the excess is
    repaid from cash (cure).
 9. Repayments: margin line from the day's margin receipts (if enabled); surplus cash above the opening cash
    level sweeps to the revolver (if enabled).
10. Funding: any cash below the minimum is requested from the margin line (only up to the day's eligible margin
    payments) and then the revolver (only up to unused availability). Requests settle after the configured
    delay; with a delay, today's gap stays unfunded.
11. Closing: shortfall = cash below the minimum after same-day funding (an unmet payment obligation).

Definitions
- Liquidity pressure: the day's payments exceed receipts (a funding need arises).
- Financing draw: the need is met from a facility.
- Deficiency: revolver debt above the borrowing base (cured after the grace period).
- Funding failure (shortfall): an obligation that cannot be paid today after every same-day source.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from core.advanced_schemas import AdvancedSettings, CashflowSettings, ScenarioSpec, SimulationMethod
from core.borrowing_base import _value_receivables
from core.haircuts import compute_inventory_valuation
from core.margin_engine import signed_short_contracts
from core.market_calibration import CalibrationResult, simulate_daily_log_returns
from core.scenario_analytics import _vectorized_cap_solve
from core.schemas import (
    COMMODITY_SPECS,
    REQUIRED_TICKERS,
    CommoditySpec,
    FacilityTerms,
    InventoryPosition,
    MarketQuote,
    ReceivableItem,
)

LEDGER_FIELDS = (
    "futures_value_change", "vm_paid", "vm_received", "im_required", "im_posted", "im_released", "collections",
    "borrowing_base", "permitted_borrowing", "deficiency", "cure_paid", "interest", "revolver_arrival",
    "margin_line_arrival", "margin_line_repaid", "revolver_repaid", "funding_need", "margin_line_requested",
    "revolver_requested", "cash", "revolver_drawn", "margin_line_drawn", "revolver_unused", "margin_line_unused",
    "in_transit", "available_liquidity", "shortfall", "cumulative_interest",
)


class CashflowRun(BaseModel):
    """Ledger arrays of shape (n_paths, horizon + 1); index 0 is the opening position."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    horizon_days: int
    n_paths: int
    ledger: dict[str, np.ndarray]
    first_shortfall_day: np.ndarray
    max_shortfall: np.ndarray
    max_funding_need: np.ndarray
    min_available_liquidity: np.ndarray
    total_interest: np.ndarray
    max_utilization_day: np.ndarray


def collection_schedule(receivables: Sequence[ReceivableItem], terms: FacilityTerms,
                        settings: CashflowSettings) -> list[int | None]:
    """Illustrative collection day (1-based) for each receivable, or None if not collected in the horizon.

    Rule: buyers listed as non-collecting and receivables past the aging limit are not collected; other
    overdue invoices are collected `overdue_collection_delay_days` from today; current invoices (0 days past
    due) at the end of `invoice_terms_days`. No company-specific dates are assumed.
    """
    days: list[int | None] = []
    for r in receivables:
        if r.buyer_name in settings.noncollected_buyers or r.days_past_due > terms.ar_aging_limit_days:
            days.append(None)
            continue
        day = settings.overdue_collection_delay_days if r.days_past_due > 0 else settings.invoice_terms_days
        day = max(1, day)
        days.append(day if day <= settings.horizon_days else None)
    return days


def deterministic_paths(
    market_snapshot: Mapping[str, MarketQuote],
    settings: AdvancedSettings,
    scenario: ScenarioSpec,
    cf: CashflowSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """One path: futures and basis move linearly to the scenario values over `ramp_days`, then stay flat."""
    h = cf.horizon_days
    f0 = np.array([market_snapshot[t].spot_price for t in REQUIRED_TICKERS])
    shock = np.array([scenario.futures_shocks.get(t, 0.0) for t in REQUIRED_TICKERS])
    b0 = np.array([settings.basis[t].opening_basis_pct for t in REQUIRED_TICKERS])
    db = np.array([scenario.basis_shocks.get(t, 0.0) for t in REQUIRED_TICKERS])
    frac = np.minimum(1.0, np.arange(h + 1) / cf.ramp_days)[:, None]
    futures = f0 * (1.0 + shock * frac)
    basis = b0 + db * frac
    return futures[None, :, :], basis[None, :, :]


def run_cashflow_ledger(
    positions: Sequence[InventoryPosition],
    receivables: Sequence[ReceivableItem],
    market_snapshot: Mapping[str, MarketQuote],
    terms: FacilityTerms,
    settings: AdvancedSettings,
    cf: CashflowSettings,
    futures_paths: np.ndarray,
    basis_paths: np.ndarray | None = None,
    base_im_multipliers: Mapping[str, float] | None = None,
    apply_trigger_rule: bool = True,
    specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS,
) -> CashflowRun:
    n, h1, k = futures_paths.shape
    h = h1 - 1
    if h != cf.horizon_days:
        raise ValueError("futures_paths must have horizon_days + 1 columns")
    tick = list(REQUIRED_TICKERS)
    margin = settings.margin
    b0 = np.array([settings.basis[t].opening_basis_pct for t in tick])
    if basis_paths is None:
        basis_paths = np.broadcast_to(b0, (n, h1, k))
    signed = np.zeros(k)
    mult = np.array([specs[t].contract_multiplier for t in tick])
    im_pc = np.array([margin.profiles[t].im_per_contract for t in tick])
    addon = np.array([margin.profiles[t].fixed_addon_usd for t in tick])
    fin = np.array([margin.profiles[t].financeable for t in tick], dtype=float)
    elig0 = np.zeros(k)
    for p in positions:
        i = tick.index(p.ticker)
        s, _, _ = signed_short_contracts(p.hedged_qty, p.ticker, margin, specs)
        signed[i] += s
        p0 = max(market_snapshot[p.ticker].spot_price * (1.0 + b0[i]), 1e-12)
        quote = MarketQuote.model_validate({**market_snapshot[p.ticker].model_dump(), "spot_price": p0})
        elig0[i] += compute_inventory_valuation(p, quote, specs[p.ticker], 0.0).eligible_collateral
    base_m = np.array([float((base_im_multipliers or {}).get(t, 1.0)) for t in tick])
    has_pos = signed != 0.0

    rec_rows = _value_receivables(receivables, terms)
    pre_caps = np.array([r["pre_cap_eligible"] for r in rec_rows])
    gross = np.array([r.gross_amount for r in receivables])
    schedule = collection_schedule(receivables, terms, cf)

    def im_requirement(mult_now: np.ndarray, trig: np.ndarray) -> np.ndarray:
        if margin.im_mode == "legacy_stepup":
            step = np.where(trig.any(axis=1), terms.im_stepup_fraction * terms.baseline_initial_margin, 0.0)
            return terms.baseline_initial_margin + step
        per = np.abs(signed) * im_pc * mult_now + np.where(has_pos, addon, 0.0)
        return per.sum(axis=1)

    def fin_share(mult_now: np.ndarray) -> np.ndarray:
        per = np.abs(signed) * im_pc * mult_now + np.where(has_pos, addon, 0.0)
        tot = per.sum(axis=1)
        return np.where(tot > 0, (per * fin).sum(axis=1) / np.where(tot > 0, tot, 1.0), 1.0)

    L = {f: np.zeros((n, h1)) for f in LEDGER_FIELDS}
    trig = np.zeros((n, k), dtype=bool)
    mult_now = np.broadcast_to(base_m, (n, k)).copy()
    cash = np.full(n, terms.unrestricted_cash)
    debt = np.full(n, terms.drawn_revolver)
    mline = np.zeros(n)
    posted = im_requirement(mult_now, trig)
    max_delay = max(cf.revolver_delay_days, cf.margin_line_delay_days)
    pend_r = np.zeros((n, h1 + max_delay + 1))
    pend_m = np.zeros((n, h1 + max_delay + 1))
    def_since = np.full(n, -1)
    outstanding = np.ones(len(receivables), dtype=bool)
    cum_int = np.zeros(n)

    def bb_for(day: int, outstanding_mask: np.ndarray) -> np.ndarray:
        phys = futures_paths[:, day, :] * (1.0 + basis_paths[:, day, :])
        phys0 = futures_paths[:, 0, :] * (1.0 + basis_paths[:, 0, :])
        ratio = np.where(phys0 > 0, phys / np.where(phys0 > 0, phys0, 1.0), 0.0)
        bb_inv = (elig0 * ratio).sum(axis=1)
        return _vectorized_cap_solve(bb_inv, pre_caps[outstanding_mask].tolist(), terms.concentration_cap_pct)

    def record(day: int, bb: np.ndarray, permitted: np.ndarray, cap_m: float) -> None:
        in_transit = pend_r[:, day + 1:].sum(axis=1) + pend_m[:, day + 1:].sum(axis=1)
        rev_unused = np.maximum(0.0, permitted - debt - pend_r[:, day + 1:].sum(axis=1))
        ml_unused = np.maximum(0.0, cap_m - mline - pend_m[:, day + 1:].sum(axis=1))
        L["borrowing_base"][:, day] = bb
        L["permitted_borrowing"][:, day] = permitted
        L["deficiency"][:, day] = np.maximum(0.0, debt - bb)
        L["cash"][:, day] = cash
        L["revolver_drawn"][:, day] = debt
        L["margin_line_drawn"][:, day] = mline
        L["revolver_unused"][:, day] = rev_unused
        L["margin_line_unused"][:, day] = ml_unused
        L["in_transit"][:, day] = in_transit
        L["available_liquidity"][:, day] = cash + in_transit + rev_unused
        L["shortfall"][:, day] = np.maximum(0.0, cf.minimum_cash - cash)
        L["im_required"][:, day] = posted
        L["cumulative_interest"][:, day] = cum_int

    bb0 = bb_for(0, outstanding)
    record(0, bb0, np.minimum(terms.facility_limit, bb0), terms.margin_line_limit)

    for t in range(1, h + 1):
        debt_open, ml_open = debt.copy(), mline.copy()
        f_now, f_prev = futures_paths[:, t, :], futures_paths[:, t - 1, :]
        vm_i = signed * mult * (f_now - f_prev)
        vm_net = vm_i.sum(axis=1)
        vm_paid, vm_recv = np.maximum(0.0, -vm_net), np.maximum(0.0, vm_net)
        vm_fin = np.minimum(vm_paid, (np.maximum(0.0, -vm_i) * fin).sum(axis=1))
        if apply_trigger_rule:
            cum = np.where(futures_paths[:, 0, :] > 0, f_now / np.where(futures_paths[:, 0, :] > 0,
                                                                       futures_paths[:, 0, :], 1.0) - 1.0, 0.0)
            trig |= np.abs(cum) >= terms.im_stepup_trigger - 1e-9
        mult_now = base_m * np.where(trig, 1.0 + terms.im_stepup_fraction, 1.0)
        required = im_requirement(mult_now, trig)
        im_post = np.maximum(0.0, required - posted)
        im_rel = np.maximum(0.0, posted - required) if margin.release_excess_im else np.zeros(n)
        posted = required if margin.release_excess_im else np.maximum(posted, required)
        im_fin = im_post * (fin_share(mult_now) if margin.im_mode == "per_contract" else 1.0)

        due = np.array([d == t for d in schedule], dtype=bool) if len(receivables) else np.zeros(0, dtype=bool)
        coll = float(gross[due & outstanding].sum()) if len(receivables) else 0.0
        outstanding = outstanding & ~due
        bb = bb_for(t, outstanding)

        arr_r, arr_m = pend_r[:, t].copy(), pend_m[:, t].copy()
        debt += arr_r
        mline += arr_m
        interest = (debt_open * cf.revolver_rate + ml_open * cf.margin_line_rate) / cf.day_count
        cum_int += interest

        deficient = debt > bb + 1e-6
        def_since = np.where(deficient & (def_since < 0), t, np.where(deficient, def_since, -1))
        cure = np.where(deficient & (t - def_since >= cf.cure_grace_days), debt - bb, 0.0)
        cure = np.maximum(0.0, cure)

        cash = cash + vm_recv + im_rel + coll + arr_r + arr_m - vm_paid - im_post - interest - cure
        debt = debt - cure
        def_since = np.where(cure > 0, -1, def_since)

        rep_m = np.zeros(n)
        if cf.repay_margin_line_from_receipts:
            rep_m = np.minimum(mline, np.minimum(vm_recv + im_rel, np.maximum(0.0, cash - cf.minimum_cash)))
            mline -= rep_m
            cash -= rep_m
        rep_r = np.zeros(n)
        if cf.sweep_surplus_to_revolver:
            rep_r = np.minimum(debt, np.maximum(0.0, cash - terms.unrestricted_cash))
            debt -= rep_r
            cash -= rep_r

        need = np.maximum(0.0, cf.minimum_cash - cash)
        cap_m = terms.margin_line_limit * ((1.0 - cf.margin_line_cut_pct) if t >= cf.margin_line_cut_day else 1.0)
        ml_avail = np.maximum(0.0, cap_m - mline - pend_m[:, t + 1:].sum(axis=1))
        ml_req = np.minimum(need, np.minimum(vm_fin + im_fin, ml_avail))
        if cf.margin_line_delay_days == 0:
            cash += ml_req
            mline += ml_req
        else:
            pend_m[:, t + cf.margin_line_delay_days] += ml_req
        permitted = np.minimum(terms.facility_limit, bb)
        rev_avail = np.maximum(0.0, permitted - debt - pend_r[:, t + 1:].sum(axis=1))
        rev_req = np.minimum(need - ml_req, rev_avail)
        if cf.revolver_delay_days == 0:
            cash += rev_req
            debt += rev_req
        else:
            pend_r[:, t + cf.revolver_delay_days] += rev_req

        L["futures_value_change"][:, t] = vm_net
        L["vm_paid"][:, t] = vm_paid
        L["vm_received"][:, t] = vm_recv
        L["im_posted"][:, t] = im_post
        L["im_released"][:, t] = im_rel
        L["collections"][:, t] = coll
        L["cure_paid"][:, t] = cure
        L["interest"][:, t] = interest
        L["revolver_arrival"][:, t] = arr_r
        L["margin_line_arrival"][:, t] = arr_m
        L["margin_line_repaid"][:, t] = rep_m
        L["revolver_repaid"][:, t] = rep_r
        L["funding_need"][:, t] = need
        L["margin_line_requested"][:, t] = ml_req
        L["revolver_requested"][:, t] = rev_req
        record(t, bb, permitted, cap_m)

    short = L["shortfall"][:, 1:] > 1e-6
    first = np.where(short.any(axis=1), short.argmax(axis=1) + 1, 0)
    util = L["revolver_drawn"] / terms.facility_limit
    return CashflowRun(
        horizon_days=h, n_paths=n, ledger=L, first_shortfall_day=first,
        max_shortfall=L["shortfall"].max(axis=1), max_funding_need=L["funding_need"].max(axis=1),
        min_available_liquidity=L["available_liquidity"][:, 1:].min(axis=1) if h > 0 else L["available_liquidity"][:, 0],
        total_interest=L["cumulative_interest"][:, -1], max_utilization_day=util.argmax(axis=1),
    )


def ledger_frame(run: CashflowRun, path: int = 0) -> pd.DataFrame:
    """One path of the ledger as a table (day 0 = opening)."""
    data = {"day": np.arange(run.horizon_days + 1)}
    data.update({f: run.ledger[f][path] for f in LEDGER_FIELDS})
    return pd.DataFrame(data)


class CashflowMonteCarloResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    method: SimulationMethod
    n_paths: int
    seed: int
    horizon_days: int
    prob_shortfall_by_day: list[float]
    prob_any_shortfall: float
    first_shortfall_counts: list[int]
    expected_max_shortfall_given_shortfall: float | None
    min_liquidity_percentiles: dict[str, float]
    interest_percentiles: dict[str, float]
    reserve_95: float
    reserve_99: float
    liquidity_percentiles_by_day: dict[str, list[float]]


def run_cashflow_monte_carlo(
    positions: Sequence[InventoryPosition],
    receivables: Sequence[ReceivableItem],
    market_snapshot: Mapping[str, MarketQuote],
    terms: FacilityTerms,
    settings: AdvancedSettings,
    cf: CashflowSettings,
    calibration: CalibrationResult,
    prices_df: pd.DataFrame,
    method: SimulationMethod = "gaussian",
    n_paths: int = 2000,
    seed: int = 21,
    basis_vol_annual: float = 0.0,
    base_im_multipliers: Mapping[str, float] | None = None,
    apply_trigger_rule: bool = True,
    winsorize_sigma: float | None = 5.0,
    specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS,
) -> CashflowMonteCarloResult:
    """Simulated daily futures paths (and basis random walks if basis_vol_annual > 0) through the ledger.

    reserve_95 / reserve_99: extra opening cash that would have avoided any shortfall on 95% / 99% of paths,
    approximated by the quantile of each path's largest shortfall (sweep and interest effects ignored)."""
    h = cf.horizon_days
    rets = simulate_daily_log_returns(calibration, prices_df, method, h, n_paths, seed, winsorize_sigma)
    f0 = np.array([market_snapshot[t].spot_price for t in REQUIRED_TICKERS])
    cum = np.concatenate([np.zeros((n_paths, 1, len(f0))), np.cumsum(rets, axis=1)], axis=1)
    futures = f0 * np.exp(cum)
    b0 = np.array([settings.basis[t].opening_basis_pct for t in REQUIRED_TICKERS])
    if basis_vol_annual > 0:
        rng = np.random.default_rng(seed + 1)
        steps = rng.standard_normal((n_paths, h, len(f0))) * basis_vol_annual / np.sqrt(252.0)
        basis = b0 + np.concatenate([np.zeros((n_paths, 1, len(f0))), np.cumsum(steps, axis=1)], axis=1)
    else:
        basis = None
    run = run_cashflow_ledger(positions, receivables, market_snapshot, terms, settings, cf, futures, basis,
                              base_im_multipliers, apply_trigger_rule, specs)
    short = run.ledger["shortfall"][:, 1:] > 1e-6
    by_day = np.cumsum([(run.first_shortfall_day == d).sum() for d in range(1, h + 1)]) / n_paths
    hit = run.max_shortfall > 1e-6
    liq = run.ledger["available_liquidity"]
    return CashflowMonteCarloResult(
        method=method, n_paths=n_paths, seed=seed, horizon_days=h,
        prob_shortfall_by_day=by_day.tolist(), prob_any_shortfall=float(short.any(axis=1).mean()),
        first_shortfall_counts=[int((run.first_shortfall_day == d).sum()) for d in range(1, h + 1)],
        expected_max_shortfall_given_shortfall=float(run.max_shortfall[hit].mean()) if hit.any() else None,
        min_liquidity_percentiles={f"p{q}": float(np.percentile(run.min_available_liquidity, q)) for q in (1, 5, 50)},
        interest_percentiles={f"p{q}": float(np.percentile(run.total_interest, q)) for q in (50, 95)},
        reserve_95=float(np.percentile(run.max_shortfall, 95)), reserve_99=float(np.percentile(run.max_shortfall, 99)),
        liquidity_percentiles_by_day={f"p{q}": np.percentile(liq, q, axis=0).tolist() for q in (5, 25, 50, 75, 95)},
    )
