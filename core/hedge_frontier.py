"""Multi-objective hedge frontier: economic risk against funding liquidity, by transparent grid search.

For each candidate hedge ratio (share of physical quantity sold forward in whole futures contracts) the same
simulated market paths (common random numbers: same seed, same return model) run through the daily treasury
engine. Reported per ratio:
- economic P&L over the horizon per path (inventory revaluation + futures P&L + trade margin - costs - credit
  losses): mean, standard deviation, 5% quantile loss and the mean of the worst 5% (economic downside)
- funding: 95th percentile peak unfunded requirement, probability of a liquidity breach, mean peak revolver
  utilization
- opening initial margin and the transaction cost of moving from the current hedge to the candidate
Constraints mark candidates infeasible; Pareto efficiency is assessed on (economic downside, funding p95)
among feasible candidates. Three labelled choices are reported, each conditional on these assumptions:
- economically conservative: feasible ratio with the smallest economic downside
- liquidity conservative: feasible ratio with the smallest funding p95
- balanced: feasible ratio minimizing the larger of the two objectives after scaling each to [0, 1] across
  feasible candidates (a minimax rule; other weightings give other answers)
There is no universally optimal hedge ratio.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.schemas import COMMODITY_SPECS, InventoryPosition

ALL = "ALL"


@dataclass
class HedgeConstraints:
    min_ratio: float = 0.0
    max_ratio: float = 1.0
    max_economic_downside: float | None = None
    max_breach_probability: float | None = None
    max_funding_p95: float | None = None
    max_peak_utilization: float | None = None


def _with_ratio(positions: Sequence[InventoryPosition], scope: str, ratio: float) -> list[InventoryPosition]:
    return [InventoryPosition.model_validate({**p.model_dump(), "hedged_qty": p.physical_qty * ratio})
            if scope in (ALL, p.ticker) else p for p in positions]


def pareto_mask(x: np.ndarray, y: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    """True where no other feasible point is at least as good on both (lower is better) and better on one."""
    n = len(x)
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        if not feasible[i]:
            continue
        dom = feasible & (x <= x[i] + 1e-9) & (y <= y[i] + 1e-9) & ((x < x[i] - 1e-9) | (y < y[i] - 1e-9))
        out[i] = not dom.any()
    return out


def run_hedge_frontier(positions, receivables, snapshot, terms, margin, ts, model, scope: str = ALL,
                       ratios: Sequence[float] | None = None, constraints: HedgeConstraints | None = None,
                       n_paths: int = 300, seed: int = 31, transaction_cost_per_contract: float = 0.0) -> pd.DataFrame:
    from core.treasury_analytics import run_book_monte_carlo
    c = constraints or HedgeConstraints()
    grid = np.round(np.linspace(c.min_ratio, c.max_ratio, 11) if ratios is None else np.asarray(ratios), 6)
    rows = []
    current = {p.ticker: p.hedged_qty for p in positions}
    for r in grid:
        trial = _with_ratio(positions, scope, float(r))
        out = run_book_monte_carlo(trial, receivables, snapshot, terms, margin, ts, model, n_paths, seed)
        pnl = np.asarray(out.economic_pnl_samples)
        q05 = float(np.quantile(pnl, 0.05))
        worst = pnl[pnl <= q05 + 1e-9]
        traded = sum(abs(p.hedged_qty - current[p.ticker]) / COMMODITY_SPECS[p.ticker].contract_multiplier
                     for p in trial)
        rows.append({
            "hedge_ratio": float(r), "pnl_mean": float(pnl.mean()), "pnl_sd": float(pnl.std(ddof=1)),
            "economic_downside": max(0.0, -q05), "economic_tail_mean": float(-worst.mean()),
            "funding_p95": out.metrics.funding_p95, "prob_breach": out.metrics.prob_breach,
            "els": out.metrics.els, "lar": out.metrics.lar,
            "peak_utilization_mean": float(np.mean(out.peak_utilization_samples)),
            "opening_im": out.opening_im, "transaction_cost": traded * transaction_cost_per_contract,
        })
    df = pd.DataFrame(rows)
    feas = np.ones(len(df), dtype=bool)
    if c.max_economic_downside is not None:
        feas &= df["economic_downside"].to_numpy() <= c.max_economic_downside
    if c.max_breach_probability is not None:
        feas &= df["prob_breach"].to_numpy() <= c.max_breach_probability
    if c.max_funding_p95 is not None:
        feas &= df["funding_p95"].to_numpy() <= c.max_funding_p95
    if c.max_peak_utilization is not None:
        feas &= df["peak_utilization_mean"].to_numpy() <= c.max_peak_utilization
    df["feasible"] = feas
    x, y = df["economic_downside"].to_numpy(), df["funding_p95"].to_numpy()
    df["pareto"] = pareto_mask(x, y, feas)
    df["label"] = ""
    if feas.any():
        idx = np.nonzero(feas)[0]
        econ = idx[np.lexsort((y[idx], x[idx]))[0]]   # smallest downside, ties to smaller funding need
        liq = idx[np.lexsort((x[idx], y[idx]))[0]]    # smallest funding need, ties to smaller downside
        span_x = max(np.ptp(x[idx]), 1e-9)
        span_y = max(np.ptp(y[idx]), 1e-9)
        score = np.maximum((x[idx] - x[idx].min()) / span_x, (y[idx] - y[idx].min()) / span_y)
        bal = idx[np.lexsort((x[idx], score))[0]]
        df.loc[liq, "label"] = "liquidity conservative"
        df.loc[econ, "label"] = "economically conservative" if econ != liq else "economic and liquidity"
        if bal not in (econ, liq):
            df.loc[bal, "label"] = "balanced"
    return df
