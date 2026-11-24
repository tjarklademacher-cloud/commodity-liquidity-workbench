"""Counterparty credit: expected credit loss, payment behaviour, defaults, insurance and rating migration.

This is an illustrative credit model for liquidity analysis. It is not an IFRS 9 implementation (no staging,
no lifetime ECL, no forward-looking macro scenarios weighting, no discounting) and not a regulatory model.

Horizon PD under a constant-hazard assumption: PD_h = 1 - (1 - PD_1y)^(days / 365). The constant hazard is a
convenience assumption; real default intensities vary with the cycle and the counterparty's condition.

Loss on default for exposure E (no double counting of recoveries):
    insured   = 0 if excluded, else min(limit, coverage x max(0, E - deductible))
    uninsured = E - insured
    recovery  = (1 - LGD) x uninsured           (estate recovery on the uninsured part only)
    loss      = LGD x uninsured                  so insured + recovery + loss = E
    ECL       = PD_h x LGD x (EAD - insured EAD)
Insurer credit risk is ignored. Insurance proceeds arrive `claim_delay_days` after default and estate recoveries
`recovery_delay_days` after default; neither is cash on the default date.

Correlated defaults (one-factor Gaussian copula with a commodity factor for wrong-way risk):
    X_i = sqrt(rho) x S_i + sqrt(1 - rho) x e_i,   S_i = beta_i x F_c + sqrt(1 - beta_i^2) x M
F_c is the counterparty's linked commodity factor (normal scores of that commodity's simulated horizon return,
so F_c is standard normal across paths by construction), M a common macro factor and e_i idiosyncratic. X_i is
standard normal, so the unconditional default probability equals PD_h for any rho and beta. Default when
X_i < Phi^-1(PD_h). beta_i > 0 means the counterparty weakens when its linked commodity falls.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from core.treasury_schemas import ILLUSTRATIVE_ANNUAL_PD, RATING_SCALE, Counterparty

NEVER = 10**6
_N = NormalDist()


def horizon_pd(annual_pd: float, days: float, year_days: float = 365.0) -> float:
    if not 0.0 <= annual_pd <= 1.0:
        raise ValueError("annual PD must be in [0, 1]")
    if days < 0:
        raise ValueError("days must be non-negative")
    return 1.0 - (1.0 - annual_pd) ** (days / year_days)


def insured_amount(exposure: np.ndarray | float, cp: Counterparty) -> np.ndarray:
    e = np.maximum(0.0, np.asarray(exposure, dtype=float))
    if cp.insurance_excluded or cp.insurance_coverage <= 0:
        return np.zeros_like(e)
    covered = cp.insurance_coverage * np.maximum(0.0, e - cp.insurance_deductible)
    return np.minimum(covered, cp.insurance_limit) if cp.insurance_limit is not None else covered


def default_settlement(exposure: np.ndarray | float, cp: Counterparty) -> dict[str, np.ndarray]:
    e = np.maximum(0.0, np.asarray(exposure, dtype=float))
    ins = insured_amount(e, cp)
    unins = e - ins
    return {"insured": ins, "recovery": (1.0 - cp.lgd) * unins, "loss": cp.lgd * unins}


def expected_credit_loss(exposure: float, cp: Counterparty, horizon_days: float) -> float:
    return horizon_pd(cp.pd_annual, horizon_days) * float(default_settlement(exposure, cp)["loss"])


def illustrative_transition_matrix() -> pd.DataFrame:
    """Illustrative one-year rating transition matrix built from simple rules, not agency data:
    default probability from ILLUSTRATIVE_ANNUAL_PD, one-notch upgrade 3%, one-notch downgrade 6%, two-notch
    downgrade 1% (where those ratings exist), remainder stays. Rows sum to one; D is absorbing."""
    states = list(RATING_SCALE)
    m = np.zeros((len(states), len(states)))
    for i, r in enumerate(states):
        if r == "D":
            m[i, i] = 1.0
            continue
        m[i, -1] = ILLUSTRATIVE_ANNUAL_PD[r]
        if i > 0:
            m[i, i - 1] = 0.03
        if i + 1 < len(states) - 1:
            m[i, i + 1] = 0.06
        if i + 2 < len(states) - 1:
            m[i, i + 2] = 0.01
        m[i, i] = 1.0 - m[i].sum()
    return pd.DataFrame(m, index=states, columns=states)


def horizon_transition(matrix: pd.DataFrame, days: float) -> pd.DataFrame:
    """Linear scaling P_h = I + (days / 365) (P - I). An approximation valid for short horizons (it keeps rows
    summing to one and entries non-negative for days <= 365); it is not a generator-matrix estimate."""
    f = days / 365.0
    if not 0.0 <= f <= 1.0:
        raise ValueError("horizon must be between 0 and 365 days")
    eye = np.eye(len(matrix))
    return pd.DataFrame(eye + f * (matrix.to_numpy() - eye), index=matrix.index, columns=matrix.columns)


@dataclass
class CreditOutcomes:
    """Per-path payment outcome of every invoice (opening and future sales) and eligibility events.

    pay_day[n, m]: business day the (first) payment arrives, NEVER if not in horizon or defaulted
    pay_fraction[n, m]: share of face paid on pay_day; the rest is disputed and paid on dispute_day
    dispute_day[n, m]: business day the disputed remainder is paid (NEVER if not in horizon)
    default_day[n, m]: business day of default (NEVER if none); a defaulted invoice pays nothing on pay_day
    ineligible_day[n, c]: business day a counterparty drops below the facility's minimum eligible rating
    """

    pay_day: np.ndarray
    pay_fraction: np.ndarray
    dispute_day: np.ndarray
    default_day: np.ndarray
    ineligible_day: np.ndarray
    label: str = ""


class CreditScenario(BaseModel):
    """Deterministic credit stress. Delays are extra calendar days on top of each counterparty's expected delay."""

    name: str = "No credit stress"
    extra_delay_days: dict[str, int] = Field(default_factory=dict)
    default_day: dict[str, int] = Field(default_factory=dict)
    partial_payment: dict[str, float] = Field(default_factory=dict)
    dispute_days: int = Field(default=30, ge=0, le=365)
    downgrade_day: dict[str, int] = Field(default_factory=dict)
    all_extra_delay_days: int = Field(default=0, ge=0, le=365)
    delays_apply_from_day: int = Field(default=0, ge=0, description="extra delays apply to invoices issued on or "
                                       "after this business day")


@dataclass
class InvoiceSchedule:
    """Static facts about each invoice needed to build outcomes: counterparty index, issue day and contractual
    due day (business-day indices from the engine's calendar)."""

    counterparty_index: np.ndarray
    issue_day: np.ndarray
    due_day: np.ndarray


def _shift_days(dates: pd.DatetimeIndex, base_days: np.ndarray, extra_calendar_days: np.ndarray) -> np.ndarray:
    """Business-day index of base day + extra calendar days (NEVER when beyond the horizon)."""
    day_num = (dates.normalize() - dates[0].normalize()).days.to_numpy()
    base = np.asarray(base_days, dtype=np.int64)
    valid = base < len(dates)
    safe = np.where(valid, base, 0)
    target = day_num[safe] + np.asarray(extra_calendar_days, dtype=np.int64)
    pos = np.searchsorted(day_num, target, side="left")
    pos = np.maximum(pos, safe)
    return np.where(valid & (pos < len(dates)), pos, NEVER).astype(int)


def deterministic_outcomes(counterparties: Sequence[Counterparty], schedule: InvoiceSchedule,
                           dates: pd.DatetimeIndex, scenario: CreditScenario, min_eligible_rating: str,
                           n_paths: int = 1) -> CreditOutcomes:
    ids = [c.id for c in counterparties]
    m = len(schedule.issue_day)
    applies = schedule.issue_day >= scenario.delays_apply_from_day
    delays = np.array([counterparties[ci].expected_delay_days
                       + (scenario.all_extra_delay_days + scenario.extra_delay_days.get(ids[ci], 0) if ok else 0)
                       for ci, ok in zip(schedule.counterparty_index, applies)], dtype=int)
    base_due = np.maximum(schedule.due_day, schedule.issue_day + 0)
    pay_day = _shift_days(dates, base_due, delays) if m else np.zeros(0, dtype=int)
    frac = np.array([scenario.partial_payment.get(ids[ci], 1.0) for ci in schedule.counterparty_index]) \
        if m else np.zeros(0)
    dispute = np.where(frac < 1.0, _shift_days(dates, np.minimum(pay_day, len(dates)), np.full(m, scenario.dispute_days)),
                       NEVER) if m else np.zeros(0, dtype=int)
    dflt = np.array([scenario.default_day.get(ids[ci], NEVER) for ci in schedule.counterparty_index], dtype=int) \
        if m else np.zeros(0, dtype=int)
    dflt = np.where(dflt < schedule.issue_day, np.maximum(dflt, schedule.issue_day), dflt) if m else dflt
    rank = {r: i for i, r in enumerate(RATING_SCALE)}
    inel = np.array([0 if rank[c.rating] > rank[min_eligible_rating] else scenario.downgrade_day.get(c.id, NEVER)
                     for c in counterparties], dtype=int)
    for i, c in enumerate(counterparties):
        if c.id in scenario.default_day:
            inel[i] = min(inel[i], scenario.default_day[c.id])
    rep = lambda a: np.repeat(a[None, ...], n_paths, axis=0)  # noqa: E731
    return CreditOutcomes(rep(pay_day), rep(frac), rep(dispute), rep(dflt), rep(inel), scenario.name)


def normal_scores(x: np.ndarray) -> np.ndarray:
    """Rank-based normal scores across paths (column-wise): exactly standard normal marginals in the sample."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    ranks = np.argsort(np.argsort(x, axis=0, kind="stable"), axis=0) + 1
    return np.vectorize(_N.inv_cdf)(ranks / (n + 1.0))


def simulate_credit_outcomes(
    counterparties: Sequence[Counterparty],
    schedule: InvoiceSchedule,
    dates: pd.DatetimeIndex,
    n_paths: int,
    seed: int,
    asset_correlation: float = 0.2,
    commodity_factors: dict[str, np.ndarray] | None = None,
    extended_delay_prob: float = 0.10,
    extended_delay_days: int = 30,
    dispute_prob: float = 0.0,
    dispute_fraction: float = 0.2,
    dispute_days: int = 45,
    min_eligible_rating: str = "CCC",
    migration: bool = True,
    pd_multiplier: float = 1.0,
) -> CreditOutcomes:
    """Probabilistic payment, dispute, default and downgrade outcomes per path. All parameters are illustrative.

    - Default within the horizon with probability min(1, pd_multiplier x PD_h); default day uniform over the
      horizon's business days.
    - Extended delay of `extended_delay_days` with probability `extended_delay_prob`, driven by the same latent
      variable (weaker counterparties pay later).
    - Disputes: with probability `dispute_prob`, `dispute_fraction` of the face is withheld for `dispute_days`.
    - Migration: the horizon-scaled illustrative transition matrix; a counterparty whose new rating is below
      the facility minimum becomes ineligible from a uniformly drawn day."""
    if not 0.0 <= asset_correlation < 1.0:
        raise ValueError("asset correlation must be in [0, 1)")
    rng = np.random.default_rng(seed)
    h = len(dates) - 1
    cal_days = max(1, (dates[-1] - dates[0]).days)
    c = len(counterparties)
    macro = rng.standard_normal(n_paths)
    x = np.empty((n_paths, c))
    for i, cp in enumerate(counterparties):
        f = None
        if commodity_factors is not None and cp.commodity_link in commodity_factors and cp.commodity_loading != 0:
            f = commodity_factors[cp.commodity_link]
        b = cp.commodity_loading if f is not None else 0.0
        s = b * (f if f is not None else 0.0) + np.sqrt(1.0 - b * b) * macro
        x[:, i] = np.sqrt(asset_correlation) * s + np.sqrt(1.0 - asset_correlation) * rng.standard_normal(n_paths)
    pd_h = np.array([min(1.0, pd_multiplier * horizon_pd(cp.pd_annual, cal_days)) for cp in counterparties])
    thr_d = np.array([_N.inv_cdf(min(max(p, 1e-12), 1 - 1e-12)) if 0 < p < 1 else (-np.inf if p == 0 else np.inf)
                      for p in pd_h])
    defaulted = x < thr_d[None, :]
    d_day = np.where(defaulted, rng.integers(1, h + 1, size=(n_paths, c)), NEVER)
    thr_delay = _N.inv_cdf(min(max(extended_delay_prob, 1e-12), 1 - 1e-12)) if extended_delay_prob > 0 else -np.inf
    delayed = (x < thr_delay) & ~defaulted
    disputed = rng.random((n_paths, c)) < dispute_prob

    rank = {r: i for i, r in enumerate(RATING_SCALE)}
    inel = np.full((n_paths, c), NEVER, dtype=int)
    if migration:
        trans = horizon_transition(illustrative_transition_matrix(), min(365, cal_days))
        min_rank = rank[min_eligible_rating]
        for i, cp in enumerate(counterparties):
            if rank[cp.rating] > min_rank:
                inel[:, i] = 0
                continue
            row = trans.loc[cp.rating].to_numpy()
            p_below = row[min_rank + 1:-1].sum()
            lo = pd_h[i]
            thr_hi = _N.inv_cdf(min(1 - 1e-12, lo + p_below)) if p_below > 0 else -np.inf
            down = (x[:, i] < thr_hi) & ~defaulted[:, i]
            inel[:, i] = np.where(down, rng.integers(1, h + 1, size=n_paths), NEVER)
    inel = np.minimum(inel, d_day)

    ci = schedule.counterparty_index
    m = len(ci)
    exp_delay = np.array([counterparties[j].expected_delay_days for j in ci], dtype=int)
    base = np.broadcast_to(schedule.due_day, (n_paths, m))
    extra = exp_delay[None, :] + np.where(delayed[:, ci], extended_delay_days, 0)
    pay_day = _shift_days(dates, np.asarray(base), extra) if m else np.zeros((n_paths, 0), dtype=int)
    frac = np.where(disputed[:, ci], 1.0 - dispute_fraction, 1.0) if m else np.zeros((n_paths, 0))
    disp = np.where(frac < 1.0, _shift_days(dates, np.minimum(pay_day, len(dates)), np.full(pay_day.shape, dispute_days)),
                    NEVER) if m else np.zeros((n_paths, 0), dtype=int)
    inv_default = np.where(d_day[:, ci] < NEVER, np.maximum(d_day[:, ci], schedule.issue_day[None, :]), NEVER) \
        if m else np.zeros((n_paths, 0), dtype=int)
    return CreditOutcomes(pay_day, frac, disp, inv_default, inel, "Simulated credit outcomes")


def counterparty_exposure_table(counterparties: Sequence[Counterparty], exposures: dict[str, float],
                                overdue: dict[str, float], horizon_days: float) -> pd.DataFrame:
    """Exposure, concentration, insured share and ECL by counterparty for the analytics view."""
    total = sum(exposures.values()) or 1.0
    rows = []
    for cp in counterparties:
        e = exposures.get(cp.id, 0.0)
        ins = float(insured_amount(e, cp))
        rows.append({
            "counterparty": cp.name, "rating": cp.rating, "industry": cp.industry, "region": cp.region,
            "exposure": e, "share": e / total, "overdue": overdue.get(cp.id, 0.0), "insured": ins,
            "uninsured": e - ins, "pd_annual": cp.pd_annual, "pd_horizon": horizon_pd(cp.pd_annual, horizon_days),
            "lgd": cp.lgd, "ecl": expected_credit_loss(e, cp, horizon_days),
            "over_limit": cp.credit_limit is not None and e > cp.credit_limit,
            "pd_source": "illustrative rating table" if cp.annual_pd is None else cp.provenance,
        })
    return pd.DataFrame(rows).sort_values("exposure", ascending=False, ignore_index=True)
