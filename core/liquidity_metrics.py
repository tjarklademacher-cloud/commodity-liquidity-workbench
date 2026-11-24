"""Path-based liquidity risk metrics computed from a `TreasuryRun`.

Definitions (amounts in the run's functional currency; days are business days after the start):
- Available liquidity L(t) = unrestricted cash + funding in transit + undrawn permitted revolver availability
  + undrawn committed lines (emergency funding excluded).
- Minimum liquidity on a path = min over t = 1..H of L(t).
- Liquidity deterioration D = L(0) - minimum liquidity (positive = deterioration). L(0) is the baseline.
- Liquidity-at-Risk at confidence c, LaR_c = the c-quantile of D across paths: with probability c the
  deterioration over the horizon does not exceed LaR_c. This is a deterioration, not a remaining level; the
  legacy workbench's "LaR" (a 5th percentile of the buffer level) is a different convention.
- Expected liquidity shortfall ELS_c = mean of D over paths with D >= LaR_c (severity in the tail). This is
  not the market-risk Expected Shortfall of a P&L distribution.
- Breach: closing unrestricted cash below the minimum operating cash after same-day funding on some day
  (threshold reported as `breach_threshold`). P(breach) = share of paths with at least one breach day.
- Peak funding requirement on a path = max over t of the unfunded shortfall (minimum cash less cash).
- Expected unfunded requirement = mean peak funding requirement over all paths.
Monte Carlo sampling uncertainty: Wilson score intervals for probabilities and distribution-free order
statistic intervals for quantiles. A share of deterministic scenarios that breach is not a probability and is
labelled `scenario_share` where used.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel

from core.treasury_engine import TreasuryRun

Z95 = 1.959963984540054


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def quantile_interval(x: np.ndarray, q: float, z: float = Z95) -> tuple[float, float]:
    s = np.sort(np.asarray(x, dtype=float))
    n = s.size
    half = z * math.sqrt(n * q * (1 - q))
    lo = int(max(0, math.floor(n * q - half)))
    hi = int(min(n - 1, math.ceil(n * q + half)))
    return float(s[lo]), float(s[hi])


class LiquidityMetrics(BaseModel):
    n_paths: int
    horizon_days: int
    confidence: float
    breach_threshold: float
    baseline_liquidity: float
    min_liquidity_median: float
    min_liquidity_p05: float
    lar: float
    lar_interval: tuple[float, float]
    els: float
    prob_breach: float
    prob_breach_interval: tuple[float, float]
    expected_unfunded: float
    funding_p95: float
    funding_p99: float
    funding_given_breach: float | None
    first_breach_day_median: float | None
    survival_curve: list[float]
    breach_days_mean: float
    peak_utilization_p95: float
    peak_cum_vm_paid_p95: float
    peak_im_p95: float
    credit_loss_mean: float
    tail_decomposition: dict[str, float]


def path_summary(run: TreasuryRun) -> dict[str, np.ndarray]:
    L = run.ledger
    liq = L["available_liquidity"]
    h = run.horizon
    window = liq[:, 1:] if h > 0 else liq
    short = L["shortfall"][:, 1:] > 1e-6 if h > 0 else L["shortfall"] > 1e-6
    first = np.where(short.any(axis=1), short.argmax(axis=1) + 1, 0)
    return {
        "min_liquidity": window.min(axis=1),
        "argmin_day": window.argmin(axis=1) + 1,
        "deterioration": liq[:, 0] - window.min(axis=1),
        "peak_funding": L["shortfall"].max(axis=1),
        "breach": short.any(axis=1),
        "first_breach_day": first,
        "breach_days": short.sum(axis=1),
        "peak_utilization": (L["revolver_drawn"] / np.maximum(L["permitted"], 1e-9)).max(axis=1),
        "peak_cum_vm_paid": np.maximum(0.0, -L["cum_vm"]).max(axis=1),
        "peak_im": L["im_posted"].max(axis=1),
        "credit_loss": L["credit_loss"].sum(axis=1),
    }


def compute_metrics(run: TreasuryRun, confidence: float = 0.95) -> LiquidityMetrics:
    if not 0.5 <= confidence < 1.0:
        raise ValueError("confidence must be in [0.5, 1)")
    s = path_summary(run)
    n, h = run.n_paths, run.horizon
    d = s["deterioration"]
    lar = float(np.quantile(d, confidence))
    tail = d >= lar - 1e-9
    k = int(s["breach"].sum())
    surv = [1.0 - float(((s["first_breach_day"] > 0) & (s["first_breach_day"] <= t)).mean()) for t in range(1, h + 1)]
    L = run.ledger
    decomp: dict[str, float] = {}
    if tail.any():
        idx = np.nonzero(tail)[0]
        day = s["argmin_day"][idx]

        def at(field: str) -> np.ndarray:
            return L[field][idx, day] - L[field][idx, 0]

        decomp = {
            "unrestricted cash": float(at("cash").mean()),
            "undrawn availability": float((at("available_liquidity") - at("cash")).mean()),
            "cumulative variation margin": float(L["cum_vm"][idx, day].mean()),
            "initial margin posted": float(-at("im_posted").mean()) if "im_posted" in L else 0.0,
            "borrowing base": float(at("reported_borrowing_base").mean()),
            "receivables outstanding": float(-at("receivables").mean()) if "receivables" in L else 0.0,
        }
    fb = s["first_breach_day"][s["breach"]]
    return LiquidityMetrics(
        n_paths=n, horizon_days=h, confidence=confidence, breach_threshold=run.breach_threshold,
        baseline_liquidity=float(L["available_liquidity"][:, 0].mean()),
        min_liquidity_median=float(np.median(s["min_liquidity"])),
        min_liquidity_p05=float(np.quantile(s["min_liquidity"], 0.05)),
        lar=lar, lar_interval=quantile_interval(d, confidence), els=float(d[tail].mean()),
        prob_breach=k / n, prob_breach_interval=wilson_interval(k, n),
        expected_unfunded=float(s["peak_funding"].mean()),
        funding_p95=float(np.quantile(s["peak_funding"], 0.95)),
        funding_p99=float(np.quantile(s["peak_funding"], 0.99)),
        funding_given_breach=float(s["peak_funding"][s["breach"]].mean()) if k else None,
        first_breach_day_median=float(np.median(fb)) if k else None,
        survival_curve=surv, breach_days_mean=float(s["breach_days"].mean()),
        peak_utilization_p95=float(np.quantile(s["peak_utilization"], 0.95)),
        peak_cum_vm_paid_p95=float(np.quantile(s["peak_cum_vm_paid"], 0.95)),
        peak_im_p95=float(np.quantile(s["peak_im"], 0.95)),
        credit_loss_mean=float(s["credit_loss"].mean()), tail_decomposition=decomp,
    )


def deterministic_summary(run: TreasuryRun, path: int = 0) -> dict[str, float | int | str | None]:
    """Headline figures of one deterministic path (no probabilities)."""
    s = path_summary(run)
    L = run.ledger
    fb = int(s["first_breach_day"][path])
    return {
        "min_liquidity": float(s["min_liquidity"][path]),
        "min_liquidity_date": run.dates[int(s["argmin_day"][path])].strftime("%Y-%m-%d"),
        "deterioration": float(s["deterioration"][path]),
        "peak_funding": float(s["peak_funding"][path]),
        "first_breach_date": run.dates[fb].strftime("%Y-%m-%d") if fb else None,
        "breach_days": int(s["breach_days"][path]),
        "peak_utilization": float(s["peak_utilization"][path]),
        "peak_cum_vm_paid": float(s["peak_cum_vm_paid"][path]),
        "peak_im": float(s["peak_im"][path]),
        "credit_loss": float(s["credit_loss"][path]),
        "economic_pnl": float(L["economic_pnl"][path].sum()),
        "closing_cash": float(L["cash"][path, -1]),
        "peak_emergency": float(L["emergency_drawn"][path].max()),
    }
