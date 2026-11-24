"""Daily treasury engine: one authoritative path-dependent state for cash, collateral, debt and margin.

Every simulation day inherits the previous day's closing state. Order of events on business day t (fixed and
identical on every path):

  1. Market: settlement prices F(t), physical prices P(t), path volatility vol(t) (from `MarketPaths`).
  2. Derivatives: variation margin on positions held over the day, VM = sum signed contracts x size x
     (F(t) - F(t-1)), netted across instruments in one clearing account. Rolls and hedge adjustments happen at
     the close at settlement prices (no VM on the trade day); transaction costs are paid in cash.
  3. Physical operations: storage cost on stock held, purchases (title and payable on the trade day, cash on
     the payment date, stock usable after transit), sales (stock leaves, an invoice is issued).
  4. Receivables: defaults are settled first (insurance and estate recovery become claims receivable, paid
     after their delays; the rest is a realized credit loss), then scheduled payments, partial payments and
     disputed remainders are collected.
  5. Payables due today are paid. Operating expenses, interest (on opening balances, actual/day-count over the
     calendar days since the previous business day) and commitment fees are paid.
  6. Initial margin requirement re-set from close positions (and volatility on review days); calls are paid,
     excess is released after the release lag. Posted IM is restricted cash, not an expense.
  7. Funding requested on earlier days arrives (debt rises on arrival).
  8. Collateral: dynamic haircuts, eligible inventory and receivables, concentration cap, sublimits, reserves.
     The reported borrowing base is updated on redetermination days only.
  9. Deficiency: drawn above permitted utilization; after the grace period the excess is a mandatory repayment.
 10. Repayments: margin-only lines from margin receipts, then surplus cash above the sweep target to the
     revolver and other lines.
 11. Funding waterfall for cash below the minimum operating level: revolver (within permitted availability),
     committed lines in order (margin-only lines only for today's margin payments), emergency funding if
     configured. Requests with a delay arrive later; today's gap remains.
 12. Close: an unfunded shortfall is cash below the minimum operating level after same-day funding.

Cash identity checked every day (residual stored in `reconciliation_residual`):
  closing cash = opening cash + collections + insurance receipts + recoveries + VM received + IM released
                 + financing received - supplier payments - storage - opex - interest and fees - VM paid
                 - IM posted - transaction costs - mandatory repayments - voluntary repayments
Net-worth identity (residual in `networth_residual`):
  NW = cash + restricted cash (posted IM, pending IM releases, other) + inventory at physical price
       + receivables + claims receivable - payables - debt
  NW(t) - NW(t-1) = inventory revaluation + futures P&L (= VM) + trade margin - storage - opex - interest
                    - fees - transaction costs - realized credit losses
Negative cash is allowed only as an unpaid obligation and is always reported as an unfunded shortfall.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.collateral import (
    HAIRCUT_COMPONENTS,
    availability,
    borrowing_base,
    dynamic_haircut,
    haircut_components,
    limit_haircut_increase,
)
from core.credit_risk import NEVER, CreditOutcomes, CreditScenario, InvoiceSchedule, default_settlement, \
    deterministic_outcomes
from core.futures_curve import MarketPaths, calendar_to_business_index, reference_contract
from core.treasury_schemas import RATING_SCALE, TreasuryBook, TreasuryConfig

TOL = 1e-6
SUMMARY_FIELDS = (
    "cash", "available_liquidity", "liquidity_headroom", "borrowing_base", "reported_borrowing_base",
    "permitted", "revolver_drawn", "lines_drawn", "emergency_drawn", "deficiency", "shortfall", "im_posted",
    "vm_net", "cum_vm", "inventory_value", "receivables", "credit_loss", "networth", "economic_pnl",
    "avg_haircut",
)


@dataclass
class TreasuryRun:
    dates: pd.DatetimeIndex
    ledger: dict[str, np.ndarray]           # (n, h+1)
    haircuts: np.ndarray | None             # (n, h+1, k) effective haircut per instrument
    instrument_ids: list[str]
    reconciliation_residual: float
    networth_residual: float
    breach_threshold: float
    notes: list[str]

    @property
    def n_paths(self) -> int:
        return next(iter(self.ledger.values())).shape[0]

    @property
    def horizon(self) -> int:
        return len(self.dates) - 1


# --------------------------------------------------------------------------- deterministic schedules

@dataclass
class _Schedules:
    owned: np.ndarray          # (h+1, k) owned quantity at close (stock + in transit)
    stock_elig: np.ndarray     # (h+1, k) arrived, financing-eligible stock at close
    transit: np.ndarray        # (h+1, k) in transit at close
    hold: np.ndarray           # (h+1, k, J) signed contracts at close
    traded: np.ndarray         # (h+1, k) contracts traded (rolls and hedge adjustments) on day t
    storage_rate: np.ndarray   # (k,) cost per unit per calendar day
    age0: np.ndarray           # (k,) quantity-weighted opening age (calendar days)
    hedge_ratio: np.ndarray    # (h+1, k) hedging futures quantity / owned quantity
    buys: list[tuple[int, int, float, object, int]]   # (day, k, qty, trade, pay_day)
    sells: list[tuple[int, int, float, object, int]]  # (day, k, qty, trade, invoice index)


def _schedules(book: TreasuryBook, cfg: TreasuryConfig, market: MarketPaths, n_open_invoices: int) -> _Schedules:
    h, k = market.horizon, market.n_instruments
    n_c = market.futures.shape[3]
    dates = market.dates
    size = np.array([i.contract_size for i in book.instruments])
    stock_e, stock_i = np.zeros(k), np.zeros(k)
    transit_arrivals: list[tuple[int, int, float, bool]] = []
    storage_cost = np.zeros(k)
    age_w, qty_w = np.zeros(k), np.zeros(k)
    ratio_target = np.full(k, np.nan)
    for p in book.physical:
        i = book.index(p.instrument_id)
        (stock_e if p.financing_eligible else stock_i)[i] += p.quantity
        if p.in_transit_qty > 0:
            transit_arrivals.append((max(0, p.transit_days_remaining), i, p.in_transit_qty, p.financing_eligible))
        tot = p.quantity + p.in_transit_qty
        storage_cost[i] += p.storage_cost_per_unit_day * p.quantity
        age_w[i] += p.age_days * tot
        qty_w[i] += tot
        if p.hedge_ratio_target is not None:
            ratio_target[i] = p.hedge_ratio_target
    storage_rate = np.where(qty_w > 0, storage_cost / np.where(stock_e + stock_i > 0, stock_e + stock_i, 1.0), 0.0)
    age0 = np.where(qty_w > 0, age_w / np.where(qty_w > 0, qty_w, 1.0), 0.0)

    hedge0 = np.zeros((k, n_c))
    other0 = np.zeros((k, n_c))
    for f in book.futures:
        i = book.index(f.instrument_id)
        if f.contract_index >= n_c:
            raise ValueError(f"contract index {f.contract_index} not in the curve ({n_c} contracts)")
        (hedge0 if f.hedges_physical else other0)[i, f.contract_index] += f.signed_contracts

    owned = np.zeros((h + 1, k))
    stock = np.zeros((h + 1, k))
    transit = np.zeros((h + 1, k))
    hedge = np.zeros((h + 1, k, n_c))
    other = np.zeros((h + 1, k, n_c))
    traded = np.zeros((h + 1, k))
    in_tr = np.zeros(k)
    for d, i, q, _ in transit_arrivals:
        if d > 0:
            in_tr[i] += q
        else:
            stock_e[i] += q
    pending_arrivals = [(d, i, q, e) for d, i, q, e in transit_arrivals if d > 0]
    owned_open = stock_e + stock_i + in_tr
    if np.all(np.isnan(ratio_target)) and cfg.hedge_follows_inventory:
        ratio_target = np.where(owned_open > 0, -hedge0.sum(axis=1) * size / np.where(owned_open > 0, owned_open, 1.0),
                                np.nan)
    owned[0], stock[0], transit[0] = owned_open, stock_e, in_tr
    hedge[0], other[0] = hedge0, other0
    ref = reference_contract(market.expiry_day, h, cfg.roll_days_before_expiry)

    trades = sorted(book.trades, key=lambda tr: tr.day)
    buys, sells = [], []
    inv_idx = n_open_invoices
    for t in range(1, h + 1):
        hd, ot = hedge[t - 1].copy(), other[t - 1].copy()
        for d, i, q, e in pending_arrivals:
            if d == t:
                in_tr[i] -= q
                if e:
                    stock_e[i] += q
                else:
                    stock_i[i] += q
        for tr in [x for x in trades if x.day == t]:
            i = book.index(tr.instrument_id)
            if tr.side == "buy":
                pay = calendar_to_business_index(dates, t, tr.payment_days)
                buys.append((t, i, tr.quantity, tr, pay))
                if tr.transit_days > 0:
                    in_tr[i] += tr.quantity
                    pending_arrivals.append((t + tr.transit_days, i, tr.quantity, True))
                else:
                    stock_e[i] += tr.quantity
            else:
                avail = stock_e[i] + stock_i[i]
                if tr.quantity > avail + 1e-9:
                    raise ValueError(f"sale of {tr.quantity} {tr.instrument_id} on day {t} exceeds stock {avail}")
                from_e = min(stock_e[i], tr.quantity)
                stock_e[i] -= from_e
                stock_i[i] -= tr.quantity - from_e
                sells.append((t, i, tr.quantity, tr, inv_idx))
                inv_idx += 1
        # rolls at the close of the roll day
        for i in range(k):
            for j in range(n_c - 1):
                roll_day = market.expiry_day[i, j] - cfg.roll_days_before_expiry
                if roll_day == t:
                    for arr in (hd, ot):
                        moved = arr[i, j]
                        if moved != 0:
                            traded[t, i] += 2 * abs(moved)
                            arr[i, j + 1] += moved
                            arr[i, j] = 0.0
        owned_now = stock_e + stock_i + in_tr
        if cfg.hedge_follows_inventory:
            for i in range(k):
                if np.isnan(ratio_target[i]):
                    continue
                target = -np.round(ratio_target[i] * owned_now[i] / size[i])
                delta = target - hd[i].sum()
                if abs(delta) > 1e-9:
                    hd[i, ref[t, i]] += delta
                    traded[t, i] += abs(delta)
        hedge[t], other[t] = hd, ot
        owned[t], stock[t], transit[t] = owned_now, stock_e.copy(), in_tr.copy()
    hold = hedge + other
    hq = -hedge.sum(axis=2) * size
    hedge_ratio = np.where(owned > 0, hq / np.where(owned > 0, owned, 1.0), 0.0)
    return _Schedules(owned, stock, transit, hold, traded, storage_rate, age0, hedge_ratio, buys, sells)


# --------------------------------------------------------------------------- engine

def default_credit_outcomes(book: TreasuryBook, cfg: TreasuryConfig, market: MarketPaths,
                            scenario: CreditScenario | None = None) -> CreditOutcomes:
    """Deterministic outcomes: every invoice paid on its due date plus the counterparty's expected delay,
    adjusted by an optional deterministic credit scenario."""
    sched = invoice_schedule(book, market)
    return deterministic_outcomes(book.counterparties, sched, market.dates, scenario or CreditScenario(),
                                  cfg.facility.min_eligible_rating, market.n_paths)


def invoice_schedule(book: TreasuryBook, market: MarketPaths) -> InvoiceSchedule:
    """Opening invoices first, then one invoice per scheduled sale in trade-day order."""
    cps = [c.id for c in book.counterparties]
    ci, issue, due = [], [], []
    for inv in book.invoices:
        ci.append(cps.index(inv.counterparty_id))
        issue.append(0)
        due.append(calendar_to_business_index(market.dates, 0, inv.due_in_days))
    for tr in sorted([t for t in book.trades if t.side == "sell"], key=lambda x: x.day):
        if tr.day > market.horizon:
            continue
        ci.append(cps.index(tr.counterparty_id))
        issue.append(tr.day)
        due.append(calendar_to_business_index(market.dates, tr.day, tr.payment_days))
    return InvoiceSchedule(np.array(ci, dtype=int), np.array(issue, dtype=int),
                           np.where(np.array(due) >= len(market.dates), NEVER, np.array(due)).astype(int))


def run_treasury(
    book: TreasuryBook,
    cfg: TreasuryConfig,
    market: MarketPaths,
    credit: CreditOutcomes | None = None,
    lender_stress: np.ndarray | None = None,
    im_multiplier: np.ndarray | None = None,
    record: str = "full",
) -> TreasuryRun:
    """Run the daily engine over every path of `market`. `lender_stress` (h+1, k) adds haircut points (synthetic
    lender tightening); `im_multiplier` (h+1, k) scales initial margin rates (explicit scenario step-ups)."""
    n, h, k = market.n_paths, market.horizon, market.n_instruments
    if len(book.instruments) != k:
        raise ValueError("market paths and book instruments differ in count")
    if cfg.horizon_days != h:
        raise ValueError("config horizon and market horizon differ")
    fac = cfg.facility
    dates = market.dates
    cal = (dates.normalize() - dates[0].normalize()).days.to_numpy()
    ids = [i.id for i in book.instruments]
    size = np.array([i.contract_size for i in book.instruments])
    n_c = market.futures.shape[3]
    sched = _schedules(book, cfg, market, len(book.invoices))
    inv_sched = invoice_schedule(book, market)
    if credit is None:
        credit = default_credit_outcomes(book, cfg, market)
    stress = np.zeros((h + 1, k)) if lender_stress is None else np.broadcast_to(lender_stress, (h + 1, k))
    im_mult = np.ones((h + 1, k)) if im_multiplier is None else np.broadcast_to(im_multiplier, (h + 1, k))
    hpol = [cfg.haircuts[i] for i in ids]
    mpol = [cfg.margins[i] for i in ids]
    notes: list[str] = []
    if np.any(np.abs(sched.hold - np.round(sched.hold)) > 1e-9):
        notes.append("Some futures positions are fractional contracts; exchanges trade whole contracts.")

    # ----- receivables state
    cps = book.counterparties
    m = len(inv_sched.issue_day)
    ci = inv_sched.counterparty_index
    incidence = np.zeros((m, len(cps)))
    if m:
        incidence[np.arange(m), ci] = 1.0
    face = np.zeros((n, m))
    for j, inv in enumerate(book.invoices):
        face[:, j] = inv.amount
    outstanding = face.copy()
    disputed = np.zeros((n, m), dtype=bool)
    dpd0 = np.array([inv.days_past_due for inv in book.invoices] + [0] * (m - len(book.invoices)), dtype=float)
    # contractual due date in calendar days from the start (opening invoices: due_in_days; sales: trade date
    # plus payment terms); days past due on day t = opening days past due + calendar days beyond the due date
    due_cal = np.array([float(inv.due_in_days) for inv in book.invoices]
                       + [float(cal[t_s] + tr.payment_days) for (t_s, _, _, tr, _) in sched.sells])
    rank = {r: i for i, r in enumerate(RATING_SCALE)}
    ig = np.array([c.is_investment_grade for c in cps], dtype=bool)
    ins_cash = np.zeros((n, h + 2))         # insurance claim receipts scheduled within the horizon
    rec_cash = np.zeros((n, h + 2))         # estate recovery receipts scheduled within the horizon
    claims_receivable = np.zeros(n)

    # ----- payables
    n_buy = len(sched.buys)
    ap = np.zeros((n, n_buy))
    ap_open = np.array([p.amount for p in book.payables], dtype=float)
    ap_open_day = np.array([calendar_to_business_index(dates, 0, p.due_in_days) for p in book.payables], dtype=int)

    # ----- margin state
    def im_rates(t: int, vol: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
        f = market.futures[:, t]                         # (n, k, J)
        out = np.empty((n, k, n_c))
        for i, pol in enumerate(mpol):
            if pol.mode == "fixed_per_contract":
                r = np.full((n, n_c), pol.im_per_contract)
            elif pol.mode == "pct_notional":
                r = pol.pct_notional * f[:, i, :] * size[i]
            else:
                r = pol.z_score * vol[:, i, None] * np.sqrt(pol.mpor_days / 252.0) * f[:, i, :] * size[i]
            if prev is not None:
                review = t % pol.review_days == 0
                r = r if review else prev[:, i, :]
            out[:, i, :] = r
        return out

    vol0 = market.vol[:, 0]
    rate = im_rates(0, vol0, None)
    rate0 = rate.copy()
    floor = np.array([p.floor_fraction for p in mpol])[None, :, None] * rate0
    trig = np.zeros((n, k), dtype=bool)
    ref = reference_contract(market.expiry_day, h, cfg.roll_days_before_expiry)
    ref_px = market.futures[:, np.arange(h + 1)[:, None], np.arange(k)[None, :], ref]   # (n, h+1, k)

    def im_requirement(t: int, rate_now: np.ndarray) -> np.ndarray:
        step = np.ones((n, k))
        for i, pol in enumerate(mpol):
            if pol.stepup_trigger is not None:
                step[:, i] = np.where(trig[:, i], 1.0 + pol.stepup_fraction, 1.0)
        eff = np.maximum(rate_now, floor) * (im_mult[t] * step)[:, :, None]
        return (np.abs(sched.hold[t])[None] * eff).sum(axis=(1, 2))

    posted = im_requirement(0, rate)
    lag_max = max([p.release_lag_days for p in mpol] + [0])
    release_q = np.zeros((n, h + lag_max + 2))

    # ----- financing state
    lines = [ln for ln in cfg.extra_lines if ln.enabled]
    emerg = cfg.emergency if cfg.emergency is not None and cfg.emergency.enabled else None
    sources = lines + ([emerg] if emerg else [])
    max_delay = max([fac.drawdown_delay_days] + [s.delay_days for s in sources])
    pend_r = np.zeros((n, h + max_delay + 2))
    pend_l = [np.zeros((n, h + max_delay + 2)) for _ in sources]
    debt = np.full(n, fac.opening_drawn)
    ldebt = [np.zeros(n) for _ in sources]
    cash = np.full(n, cfg.opening_cash)
    sweep_target = cfg.opening_cash if cfg.sweep_target_cash is None else cfg.sweep_target_cash
    def_since = np.full(n, -1)

    # ----- ledger
    fields = ("cash", "restricted_cash", "im_posted", "im_release_pending", "available_liquidity",
              "liquidity_headroom", "in_transit_funding", "borrowing_base", "reported_borrowing_base", "permitted",
              "revolver_drawn", "revolver_undrawn", "lines_drawn", "lines_undrawn", "emergency_drawn", "deficiency",
              "mandatory_repayment", "shortfall", "vm_net", "vm_paid", "vm_received", "cum_vm", "im_call",
              "im_released", "transaction_costs", "storage_costs", "opex", "interest", "commitment_fees",
              "supplier_payments", "collections", "insurance_receipts", "recoveries", "revolver_received",
              "lines_received", "emergency_received", "revolver_repaid", "lines_repaid", "funding_need",
              "revolver_requested", "lines_requested", "emergency_requested", "inventory_value",
              "inventory_eligible_value", "inventory_contribution", "receivable_contribution",
              "concentration_cutback", "receivables", "eligible_receivables", "overdue_receivables",
              "defaulted_exposure", "claims_receivable", "payables", "credit_loss", "networth", "economic_pnl",
              "inventory_revaluation", "trade_margin", "avg_haircut", "reconciliation", "debt_total")
    L = {f: np.zeros((n, h + 1)) for f in fields}
    hc_store = np.zeros((n, h + 1, k)) if record == "full" else None
    rep_hc = None
    rep_bb = None

    def phys(t: int) -> np.ndarray:
        return market.physical[:, t]

    def collateral(t: int, vol_now: np.ndarray, prev_hc: np.ndarray | None) -> tuple[dict, np.ndarray, np.ndarray]:
        p = phys(t)
        elig_qty = sched.stock_elig[t] + (sched.transit[t] if fac.in_transit_eligible else 0.0)
        mv_owned = sched.owned[t][None] * p
        tot = mv_owned.sum(axis=1, keepdims=True)
        share = np.where(tot > 0, mv_owned / np.where(tot > 0, tot, 1.0), 0.0)
        hc = np.empty((n, k))
        for i, pol in enumerate(hpol):
            v = vol_now[:, i] if pol.dynamic else vol0[:, i]
            st = stress[t, i] if pol.dynamic else stress[0, i]
            comps = haircut_components(pol, v, share[:, i], float(sched.hedge_ratio[t, i]),
                                       float(sched.age0[i] + cal[t]), st)
            raw = dynamic_haircut(pol, comps)
            if prev_hc is not None:
                raw = limit_haircut_increase(raw, prev_hc[:, i], pol.max_increase_per_redetermination)
            hc[:, i] = raw
        adj = (elig_qty[None] * p * (1.0 - hc)).sum(axis=1)
        # receivables
        cal_t = cal[t]
        issued = inv_sched.issue_day[None, :] <= t
        dpd = dpd0[None, :] + np.maximum(0.0, cal_t - due_cal[None, :])
        cp_ok = credit.ineligible_day[:, ci] > t if m else np.zeros((n, 0), dtype=bool)
        rating_ok = np.array([rank[cps[c].rating] <= rank[fac.min_eligible_rating] for c in ci], dtype=bool)
        elig = issued & (outstanding > TOL) & ~disputed & (dpd <= fac.ar_aging_limit_days) & cp_ok & rating_ok[None, :]
        elig_face = np.where(elig, outstanding, 0.0)
        net = elig_face * (1.0 - fac.dilution_reserve_pct)
        base_rate = np.where(ig[ci], fac.ar_ig_advance_rate, fac.ar_non_ig_advance_rate) if m else np.zeros(0)
        if fac.insured_ar_advance_rate is not None and m:
            ins_share = np.array([cps[c].insurance_coverage if not cps[c].insurance_excluded else 0.0 for c in ci])
            adv = ins_share * np.maximum(base_rate, fac.insured_ar_advance_rate) + (1 - ins_share) * base_rate
        else:
            adv = base_rate
        pre_cap = (net * adv) @ incidence if m else np.zeros((n, 0))
        bbd = borrowing_base(fac, adj, pre_cap)
        L["eligible_receivables"][:, t] = elig_face.sum(axis=1)
        L["overdue_receivables"][:, t] = np.where(issued & (dpd > 0), outstanding, 0.0).sum(axis=1)
        L["inventory_eligible_value"][:, t] = (elig_qty[None] * p).sum(axis=1)
        return bbd, hc, share

    def networth(t: int) -> np.ndarray:
        inv_val = (sched.owned[t][None] * phys(t)).sum(axis=1)
        L["inventory_value"][:, t] = inv_val
        L["receivables"][:, t] = outstanding.sum(axis=1)
        L["claims_receivable"][:, t] = claims_receivable
        L["payables"][:, t] = ap.sum(axis=1) + ap_open[ap_open_day > t].sum()
        restricted = posted + release_q[:, t + 1:].sum(axis=1) + cfg.other_restricted_cash
        debt_total = debt + sum(ldebt)
        L["debt_total"][:, t] = debt_total
        return (cash + restricted + inv_val + outstanding.sum(axis=1) + claims_receivable
                - L["payables"][:, t] - debt_total)

    def record_close(t: int, bbd: dict, hc: np.ndarray) -> None:
        nonlocal rep_bb, rep_hc
        av = availability(fac, rep_bb, debt, pend_r[:, t + 1:].sum(axis=1))
        in_transit = pend_r[:, t + 1:].sum(axis=1) + sum(pl[:, t + 1:].sum(axis=1) for pl in pend_l)
        l_und = sum(np.maximum(0.0, s.limit - ld - pl[:, t + 1:].sum(axis=1))
                    for s, ld, pl in zip(sources, ldebt, pend_l) if s is not emerg)
        L["cash"][:, t] = cash
        L["im_posted"][:, t] = posted
        L["im_release_pending"][:, t] = release_q[:, t + 1:].sum(axis=1)
        L["restricted_cash"][:, t] = posted + L["im_release_pending"][:, t] + cfg.other_restricted_cash
        L["borrowing_base"][:, t] = bbd["borrowing_base"]
        L["reported_borrowing_base"][:, t] = rep_bb
        L["inventory_contribution"][:, t] = bbd["inventory_contribution"]
        L["receivable_contribution"][:, t] = bbd["receivable_contribution"]
        L["concentration_cutback"][:, t] = bbd["concentration_cutback"]
        L["permitted"][:, t] = av["permitted"]
        L["revolver_drawn"][:, t] = debt
        L["revolver_undrawn"][:, t] = av["undrawn"]
        L["deficiency"][:, t] = av["deficiency"]
        L["lines_drawn"][:, t] = sum((ld for s, ld in zip(sources, ldebt) if s is not emerg), np.zeros(n))
        L["lines_undrawn"][:, t] = l_und if isinstance(l_und, np.ndarray) else np.zeros(n)
        L["emergency_drawn"][:, t] = ldebt[-1] if emerg else 0.0
        L["in_transit_funding"][:, t] = in_transit
        L["available_liquidity"][:, t] = cash + in_transit + av["undrawn"] + L["lines_undrawn"][:, t]
        L["liquidity_headroom"][:, t] = L["available_liquidity"][:, t] - cfg.min_operating_cash
        L["shortfall"][:, t] = np.maximum(0.0, cfg.min_operating_cash - cash)
        L["avg_haircut"][:, t] = np.where(L["inventory_eligible_value"][:, t] > 0,
                                          (sched.stock_elig[t][None] * phys(t) * hc).sum(axis=1)
                                          / np.maximum(L["inventory_eligible_value"][:, t], 1e-12), 0.0)
        if hc_store is not None:
            hc_store[:, t, :] = hc

    bbd, hc, _ = collateral(0, vol0, None)
    rep_hc, rep_bb = hc, bbd["borrowing_base"]
    record_close(0, bbd, hc)
    nw_prev = networth(0)
    L["networth"][:, 0] = nw_prev
    recon_max = 0.0
    nw_max = 0.0
    cum_vm = np.zeros(n)
    for t in range(1, h + 1):
        cash_open = cash.copy()
        debt_open, ldebt_open = debt.copy(), [ld.copy() for ld in ldebt]
        days = float(cal[t] - cal[t - 1])
        p_prev, p_now = phys(t - 1), phys(t)
        # 2. derivatives
        f_now, f_prev = market.futures[:, t], market.futures[:, t - 1]
        vm = (sched.hold[t - 1][None] * size[None, :, None] * (f_now - f_prev)).sum(axis=(1, 2))
        vm_paid, vm_recv = np.maximum(0.0, -vm), np.maximum(0.0, vm)
        cum_vm += vm
        tx = float((sched.traded[t] * cfg.transaction_cost_per_contract).sum())
        # 3. physical operations
        reval = (sched.owned[t - 1][None] * (p_now - p_prev)).sum(axis=1)
        storage = float(((sched.owned[t - 1] - sched.transit[t - 1]) * sched.storage_rate).sum()) * days
        margin = np.zeros(n)
        for (d, i, q, tr, pay) in sched.buys:
            if d != t:
                continue
            unit_px = p_now[:, i] * (1.0 + tr.price_premium_pct) if tr.price_mode == "market" else \
                np.full(n, float(tr.fixed_price))
            ap[:, sched.buys.index((d, i, q, tr, pay))] = q * unit_px
            margin += q * (p_now[:, i] - unit_px)
        for (d, i, q, tr, j) in sched.sells:
            if d != t:
                continue
            unit_px = p_now[:, i] * (1.0 + tr.price_premium_pct) if tr.price_mode == "market" else \
                np.full(n, float(tr.fixed_price))
            face[:, j] = q * unit_px
            outstanding[:, j] = q * unit_px
            margin += q * (unit_px - p_now[:, i])
        # 4. receivables: defaults, then payments
        loss = np.zeros(n)
        if m:
            dflt = (credit.default_day == t) & (outstanding > TOL)
            if dflt.any():
                for j in np.nonzero(dflt.any(axis=0))[0]:
                    rows = dflt[:, j]
                    cp = cps[ci[j]]
                    st = default_settlement(outstanding[rows, j], cp)
                    ins_day = calendar_to_business_index(dates, t, cp.claim_delay_days)
                    rec_day = calendar_to_business_index(dates, t, cp.recovery_delay_days)
                    claims_receivable[rows] += st["insured"] + st["recovery"]
                    if ins_day <= h:
                        ins_cash[rows, ins_day] += st["insured"]
                    if rec_day <= h:
                        rec_cash[rows, rec_day] += st["recovery"]
                    loss[rows] += st["loss"]
                    L["defaulted_exposure"][rows, t] += outstanding[rows, j]
                outstanding = np.where(dflt, 0.0, outstanding)
                disputed = np.where(dflt, False, disputed)
            pay_now = (credit.pay_day == t) & (outstanding > TOL) & (credit.default_day > t)
            paid = np.where(pay_now, outstanding * credit.pay_fraction, 0.0)
            outstanding = outstanding - paid
            disputed = disputed | (pay_now & (credit.pay_fraction < 1.0))
            rest = (credit.dispute_day == t) & disputed & (outstanding > TOL)
            paid = paid + np.where(rest, outstanding, 0.0)
            outstanding = np.where(rest, 0.0, outstanding)
            disputed = disputed & ~rest
            coll = paid.sum(axis=1)
        else:
            coll = np.zeros(n)
        ins_in, rec_in = ins_cash[:, t], rec_cash[:, t]
        claim_in = ins_in + rec_in
        claims_receivable = claims_receivable - claim_in
        # 5. payables, opex, interest, fees
        sup = ap[:, [b for b, x in enumerate(sched.buys) if x[4] == t]].sum(axis=1) if n_buy else np.zeros(n)
        if n_buy:
            ap[:, [b for b, x in enumerate(sched.buys) if x[4] == t]] = 0.0
        sup = sup + ap_open[ap_open_day == t].sum()
        opex = cfg.opex_per_calendar_day * days
        interest = (debt_open * fac.interest_rate + sum(ld * s.interest_rate for ld, s in zip(ldebt_open, sources))) \
            * days / cfg.day_count
        fees = np.maximum(0.0, fac.commitment - debt_open) * fac.commitment_fee * days / cfg.day_count
        # 6. initial margin
        vol_now = market.vol[:, t]
        for i, pol in enumerate(mpol):
            if pol.stepup_trigger is not None:
                trig[:, i] |= np.abs(ref_px[:, t, i] / np.maximum(ref_px[:, 0, i], 1e-12) - 1.0) >= pol.stepup_trigger - 1e-9
        rate = im_rates(t, vol_now, rate)
        req = im_requirement(t, rate)
        pend_total = release_q[:, t + 1:].sum(axis=1)
        need_more = np.maximum(0.0, req - posted)
        reuse = np.minimum(need_more, pend_total)
        if np.any(reuse > 0):
            scale = np.where(pend_total > 0, 1.0 - reuse / np.where(pend_total > 0, pend_total, 1.0), 1.0)
            release_q[:, t + 1:] *= scale[:, None]
        call = need_more - reuse
        posted = posted + need_more
        excess = np.maximum(0.0, posted - req)
        lag = max(p.release_lag_days for p in mpol) if mpol else 0
        posted = posted - excess
        release_q[:, t + lag] += excess
        im_rel = release_q[:, t].copy()
        release_q[:, t] = 0.0
        # 7. funding arrivals
        arr_r = pend_r[:, t].copy()
        debt = debt + arr_r
        arr_l = []
        for s_i, pl in enumerate(pend_l):
            a = pl[:, t].copy()
            ldebt[s_i] = ldebt[s_i] + a
            arr_l.append(a)
        cash = (cash + coll + claim_in + vm_recv + im_rel + arr_r + sum(arr_l, np.zeros(n))
                - sup - storage - opex - interest - fees - vm_paid - call - tx)
        # 8. collateral and reported borrowing base
        bbd, hc, _ = collateral(t, vol_now, rep_hc)
        if t % fac.redetermination_days == 0:
            rep_hc, rep_bb = hc, bbd["borrowing_base"]
        # 9. deficiency and mandatory repayment
        permitted = np.minimum(rep_bb, fac.commitment)
        deficient = debt > permitted + TOL
        def_since = np.where(deficient & (def_since < 0), t, np.where(deficient, def_since, -1))
        cure = np.where(deficient & (t - def_since >= fac.cure_grace_days), debt - permitted, 0.0)
        cash, debt = cash - cure, debt - cure
        def_since = np.where(cure > 0, -1, def_since)
        # 10. repayments
        margin_in = vm_recv + im_rel
        rep_l = np.zeros(n)
        for s_i, s in enumerate(sources):
            if s.margin_only and s.repay_from_margin_receipts:
                r_ = np.minimum(ldebt[s_i], np.minimum(margin_in, np.maximum(0.0, cash - cfg.min_operating_cash)))
                ldebt[s_i] -= r_
                cash = cash - r_
                rep_l += r_
        rep_r = np.zeros(n)
        if fac.sweep_surplus:
            surplus = np.maximum(0.0, cash - max(sweep_target, cfg.min_operating_cash))
            rep_r = np.minimum(debt, surplus)
            debt, cash = debt - rep_r, cash - rep_r
            for s_i, s in enumerate(sources):
                surplus = np.maximum(0.0, cash - max(sweep_target, cfg.min_operating_cash))
                r_ = np.minimum(ldebt[s_i], surplus)
                ldebt[s_i] -= r_
                cash = cash - r_
                rep_l += r_
        # 11. funding waterfall
        need = np.maximum(0.0, cfg.min_operating_cash - cash)
        L["funding_need"][:, t] = need
        rev_avail = np.maximum(0.0, permitted - debt - pend_r[:, t + 1:].sum(axis=1))
        req_r = np.minimum(need, rev_avail)
        if fac.drawdown_delay_days == 0:
            cash, debt = cash + req_r, debt + req_r
            recv_same = req_r
        else:
            pend_r[:, t + fac.drawdown_delay_days] += req_r
            recv_same = np.zeros(n)
        remaining = need - req_r
        margin_paid_today = vm_paid + call
        same_l = np.zeros(n)
        same_by = [np.zeros(n) for _ in sources]
        req_lines = np.zeros(n)
        req_em = np.zeros(n)
        for s_i, s in enumerate(sources):
            cap = np.maximum(0.0, s.limit - ldebt[s_i] - pend_l[s_i][:, t + 1:].sum(axis=1))
            if s.margin_only:
                cap = np.minimum(cap, np.maximum(0.0, margin_paid_today))
            take = np.minimum(remaining, cap)
            if s.margin_only:
                margin_paid_today = margin_paid_today - take
            if s.delay_days == 0:
                cash = cash + take
                ldebt[s_i] = ldebt[s_i] + take
                same_l += take
                same_by[s_i] = take
            else:
                pend_l[s_i][:, t + s.delay_days] += take
            remaining = remaining - take
            if s is emerg:
                req_em += take
            else:
                req_lines += take
        # 12. close
        flows_in = coll + claim_in + vm_recv + im_rel + arr_r + sum(arr_l, np.zeros(n)) + recv_same + same_l
        flows_out = sup + storage + opex + interest + fees + vm_paid + call + tx + cure + rep_r + rep_l
        resid = np.abs(cash_open + flows_in - flows_out - cash)
        recon_max = max(recon_max, float(resid.max()))
        L["reconciliation"][:, t] = resid
        for key, val in (("vm_net", vm), ("vm_paid", vm_paid), ("vm_received", vm_recv), ("cum_vm", cum_vm),
                         ("im_call", call), ("im_released", im_rel), ("transaction_costs", tx),
                         ("storage_costs", storage), ("opex", opex), ("interest", interest),
                         ("commitment_fees", fees), ("supplier_payments", sup), ("collections", coll),
                         ("insurance_receipts", ins_in), ("recoveries", rec_in),
                         ("revolver_received", arr_r + recv_same),
                         ("lines_received", sum((a + sb for a, sb, s in zip(arr_l, same_by, sources) if s is not emerg),
                                                np.zeros(n))),
                         ("emergency_received", arr_l[-1] + same_by[-1] if emerg else 0.0),
                         ("revolver_repaid", rep_r), ("lines_repaid", rep_l), ("mandatory_repayment", cure),
                         ("revolver_requested", req_r), ("lines_requested", req_lines),
                         ("emergency_requested", req_em), ("credit_loss", loss),
                         ("inventory_revaluation", reval), ("trade_margin", margin)):
            L[key][:, t] = val
        record_close(t, bbd, hc)
        nw = networth(t)
        L["networth"][:, t] = nw
        pnl = reval + vm + margin - storage - opex - interest - fees - tx - loss
        L["economic_pnl"][:, t] = pnl
        nw_max = max(nw_max, float(np.abs(nw - nw_prev - pnl).max()))
        nw_prev = nw
    if record == "summary":
        L = {f: L[f] for f in SUMMARY_FIELDS}
    return TreasuryRun(dates=dates, ledger=L, haircuts=hc_store, instrument_ids=ids,
                       reconciliation_residual=recon_max, networth_residual=nw_max,
                       breach_threshold=cfg.min_operating_cash, notes=notes)


def ledger_frame(run: TreasuryRun, path: int = 0) -> pd.DataFrame:
    """One path of the daily ledger as a table (day 0 = opening position). Amounts in functional currency."""
    data: dict[str, object] = {"day": np.arange(run.horizon + 1), "date": run.dates.strftime("%Y-%m-%d")}
    data.update({f: v[path] for f, v in run.ledger.items()})
    return pd.DataFrame(data)


HAIRCUT_COMPONENT_NAMES = HAIRCUT_COMPONENTS
