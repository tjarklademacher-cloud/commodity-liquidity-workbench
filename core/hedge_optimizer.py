"""Hedge-ratio frontier: liquidity protection vs. price protection as the futures hedge ratio changes.

For each candidate hedge ratio the book is re-valued (the unhedged haircut penalty changes with the ratio)
and run through the same 41-point parallel stress sweep as the main stress view, using the vectorized
engine that reproduces `run_stress_scenario` exactly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from core.borrowing_base import calculate_borrowing_base
from core.haircuts import compute_portfolio_inventory
from core.scenario_analytics import evaluate_shock_matrix
from core.schemas import (
    COMMODITY_SPECS,
    REQUIRED_TICKERS,
    CommoditySpec,
    FacilityTerms,
    HedgeFrontierResult,
    InventoryPosition,
    MarketQuote,
    ReceivableItem,
)

ALL_COMMODITIES = "ALL"
CRASH_SHOCK = -0.20


def _with_ratio(positions: Sequence[InventoryPosition], scope: str, ratio: float) -> list[InventoryPosition]:
    out = []
    for p in positions:
        if scope == ALL_COMMODITIES or p.ticker == scope:
            out.append(InventoryPosition(ticker=p.ticker, physical_qty=p.physical_qty,
                                         hedged_qty=p.physical_qty * ratio))
        else:
            out.append(p)
    return out


def _current_ratio(positions: Sequence[InventoryPosition], snapshot: Mapping[str, MarketQuote], scope: str) -> float:
    chosen = [p for p in positions if scope == ALL_COMMODITIES or p.ticker == scope]
    physical_value = sum(p.physical_qty * snapshot[p.ticker].spot_price for p in chosen)
    hedged_value = sum(p.hedged_qty * snapshot[p.ticker].spot_price for p in chosen)
    return hedged_value / physical_value if physical_value > 0 else 0.0


def run_hedge_frontier(
    positions: Sequence[InventoryPosition],
    receivables: Sequence[ReceivableItem],
    market_snapshot: Mapping[str, MarketQuote],
    terms: FacilityTerms,
    scope: str = ALL_COMMODITIES,
    specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS,
    steps: int = 21,
    shock_min: float = -0.20,
    shock_max: float = 0.20,
    shock_steps: int = 41,
    basis_shock_pct: float = 0.05,
) -> HedgeFrontierResult:
    """Sweep the hedge ratio of `scope` (a ticker, or ALL) from 0% to 100% of physical quantity.

    The recommended ratio maximizes the share of parallel shocks with a non-negative net liquidity buffer;
    ties go to the higher worst-case buffer, then to the higher ratio (more price protection). It is a
    liquidity objective only. The multi-objective outputs are reported without weights:
      max_margin_need  = VM payable + IM step-up at the largest rally in the grid
      basis_exposure   = hedged notional x basis_shock_pct (P&L hit if the basis moved that far against
                         the hedged inventory; illustrative size)
      pareto_efficient = no other ratio has both a higher worst-case buffer and a lower unhedged crash loss
    """
    if scope != ALL_COMMODITIES and scope not in {p.ticker for p in positions}:
        raise ValueError(f"No position for {scope}")
    ratios = np.round(np.linspace(0.0, 1.0, steps), 6)
    shocks = np.round(np.linspace(shock_min, shock_max, shock_steps), 6)
    matrix = np.repeat(shocks[:, None], len(REQUIRED_TICKERS), axis=1)
    i_up = int(np.argmin(np.abs(shocks - shock_max)))
    i_down = int(np.argmin(np.abs(shocks - shock_min)))

    min_buffer, positive_share, up, down, crash_loss, base_bb = [], [], [], [], [], []
    margin_need, basis_exp = [], []
    for ratio in ratios:
        trial = _with_ratio(positions, scope, float(ratio))
        res = evaluate_shock_matrix(trial, receivables, market_snapshot, terms, matrix, specs)
        nlb = res["net_liquidity_buffer"]
        margin_need.append(float(res["vm_outflow"][i_up] + res["im_addon_outflow"][i_up]))
        basis_exp.append(float(sum(p.hedged_qty * market_snapshot[p.ticker].spot_price for p in trial
                                   if scope == ALL_COMMODITIES or p.ticker == scope) * basis_shock_pct))
        min_buffer.append(float(nlb.min()))
        positive_share.append(float((nlb >= 0.0).mean()))
        up.append(float(nlb[i_up]))
        down.append(float(nlb[i_down]))
        crash_loss.append(float(sum(
            max(0.0, (p.physical_qty - p.hedged_qty)) * market_snapshot[p.ticker].spot_price * -CRASH_SHOCK
            for p in trial if scope == ALL_COMMODITIES or p.ticker == scope
        )))
        items = compute_portfolio_inventory(trial, market_snapshot, specs)
        base_bb.append(calculate_borrowing_base(items, receivables, terms).final_borrowing_base)

    order = sorted(range(len(ratios)), key=lambda i: (positive_share[i], min_buffer[i], ratios[i]), reverse=True)
    best = order[0]
    pareto = [
        not any((min_buffer[j] >= min_buffer[i] and crash_loss[j] <= crash_loss[i])
                and (min_buffer[j] > min_buffer[i] + 1e-6 or crash_loss[j] < crash_loss[i] - 1e-6)
                for j in range(len(ratios)))
        for i in range(len(ratios))
    ]
    return HedgeFrontierResult(
        scope=scope,
        ratios=ratios.tolist(),
        min_buffer=min_buffer,
        positive_share=positive_share,
        buffer_up_20=up,
        buffer_down_20=down,
        crash_value_loss=crash_loss,
        baseline_bb=base_bb,
        current_ratio=_current_ratio(positions, market_snapshot, scope),
        recommended_ratio=float(ratios[best]),
        recommended_index=best,
        max_margin_need=margin_need,
        basis_exposure=basis_exp,
        basis_shock_pct=basis_shock_pct,
        pareto_efficient=pareto,
    )
