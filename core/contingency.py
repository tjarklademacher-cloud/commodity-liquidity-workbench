"""Contingency funding plan: reverse stress test, early-warning trigger ladder and sized remedial actions.

Every number comes from the same vectorized engine as the stress views (`evaluate_shock_matrix`), so an
action sized here restores a non-negative net liquidity buffer when re-run through the stress engine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from core.scenario_analytics import evaluate_shock_matrix
from core.schemas import (
    COMMODITY_SPECS,
    REQUIRED_TICKERS,
    CommodityBreakingMove,
    CommoditySpec,
    ContingencyPlan,
    FacilityTerms,
    FundingAction,
    InventoryPosition,
    LiquidityRegime,
    MarketQuote,
    ReceivableItem,
    StressSweepResult,
    TriggerLevel,
)

MAX_SINGLE_MOVE = 1.0
SINGLE_MOVE_STEP = 0.005


def _row(shocks: Mapping[str, float]) -> np.ndarray:
    return np.array([[float(shocks.get(t, 0.0)) for t in REQUIRED_TICKERS]])


def _crossing(grid: np.ndarray, values: np.ndarray) -> float | None:
    """First grid point where values turns negative, linearly interpolated from the previous point."""
    neg = np.flatnonzero(values < 0.0)
    if neg.size == 0:
        return None
    i = int(neg[0])
    if i == 0:
        return float(grid[0])
    x0, x1, y0, y1 = grid[i - 1], grid[i], values[i - 1], values[i]
    return float(x0 + (0.0 - y0) / (y1 - y0) * (x1 - x0))


def breaking_moves(
    positions: Sequence[InventoryPosition],
    receivables: Sequence[ReceivableItem],
    market_snapshot: Mapping[str, MarketQuote],
    terms: FacilityTerms,
    specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS,
) -> list[CommodityBreakingMove]:
    """Smallest single-commodity rally that drives the buffer below zero, and smallest fall that opens a
    borrowing base deficiency, searched up to +/-100% with all other prices unchanged."""
    grid = np.round(np.arange(0.0, MAX_SINGLE_MOVE + 1e-12, SINGLE_MOVE_STEP), 6)
    out = []
    for ticker in dict.fromkeys(p.ticker for p in positions):
        col = REQUIRED_TICKERS.index(ticker)
        up = np.zeros((grid.size, len(REQUIRED_TICKERS)))
        up[:, col] = grid
        down = np.zeros_like(up)
        down[:, col] = -grid
        r_up = evaluate_shock_matrix(positions, receivables, market_snapshot, terms, up, specs)
        r_down = evaluate_shock_matrix(positions, receivables, market_snapshot, terms, down, specs)
        squeeze = _crossing(grid, r_up["net_liquidity_buffer"])
        deficiency = _crossing(grid, r_down["shocked_bb"] - terms.drawn_revolver)
        out.append(CommodityBreakingMove(
            ticker=ticker, name=specs[ticker].name,
            squeeze_rally_pct=squeeze, deficiency_fall_pct=-deficiency if deficiency is not None else None,
        ))
    return out


def trigger_ladder(sweep: StressSweepResult) -> list[TriggerLevel]:
    """Early-warning levels read off the parallel sweep: where each regime first appears moving away from 0%."""
    pts = sweep.points
    levels: list[TriggerLevel] = []
    up = [p for p in pts if p.shock_pct > 0]
    down = sorted([p for p in pts if p.shock_pct < 0], key=lambda p: -p.shock_pct)
    first_trap = next((p for p in up if p.regime != LiquidityRegime.HEALTHY), None)
    if first_trap is not None:
        levels.append(TriggerLevel(shock_pct=first_trap.shock_pct, regime=str(first_trap.regime),
                                   description="Rally at which margin calls first outrun new revolver headroom: "
                                               "pre-arrange margin funding."))
    if sweep.upside_squeeze_breakeven_pct is not None:
        levels.append(TriggerLevel(shock_pct=sweep.upside_squeeze_breakeven_pct,
                                   regime=str(LiquidityRegime.CRITICAL_SQUEEZE),
                                   description="Rally at which the net buffer reaches zero: funding must be in place."))
    first_def = next((p for p in down if p.regime == LiquidityRegime.BB_DEFICIENCY_WARNING), None)
    if sweep.downside_deficiency_breakeven_pct is not None:
        levels.append(TriggerLevel(shock_pct=sweep.downside_deficiency_breakeven_pct,
                                   regime=str(LiquidityRegime.BB_DEFICIENCY_WARNING),
                                   description="Fall at which the borrowing base drops below the drawn balance: "
                                               "a deficiency cure becomes due."))
    elif first_def is not None:
        levels.append(TriggerLevel(shock_pct=first_def.shock_pct, regime=str(first_def.regime),
                                   description="Fall at which a borrowing base deficiency first appears."))
    first_crit_down = next((p for p in down if p.regime == LiquidityRegime.CRITICAL_SQUEEZE), None)
    if first_crit_down is not None:
        levels.append(TriggerLevel(shock_pct=first_crit_down.shock_pct, regime=str(first_crit_down.regime),
                                   description="Fall at which the deficiency cure exhausts liquidity."))
    return sorted(levels, key=lambda lv: lv.shock_pct)


def _nlb(positions, receivables, snapshot, terms, shocks, specs) -> float:
    return float(evaluate_shock_matrix(positions, receivables, snapshot, terms, _row(shocks),
                                       specs)["net_liquidity_buffer"][0])


def _terms_with(terms: FacilityTerms, **changes: float) -> FacilityTerms:
    return FacilityTerms.model_validate({**terms.model_dump(), **changes})


def build_contingency_plan(
    positions: Sequence[InventoryPosition],
    receivables: Sequence[ReceivableItem],
    market_snapshot: Mapping[str, MarketQuote],
    terms: FacilityTerms,
    sweep: StressSweepResult,
    target_shocks: Mapping[str, float],
    target_label: str,
    specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS,
) -> ContingencyPlan:
    """Size each remedial action needed to bring the target scenario's net liquidity buffer back to zero."""
    shocks = {t: float(target_shocks.get(t, 0.0)) for t in REQUIRED_TICKERS}
    res = evaluate_shock_matrix(positions, receivables, market_snapshot, terms, _row(shocks), specs)
    nlb = float(res["net_liquidity_buffer"][0])
    shortfall = max(0.0, -nlb)
    margin_calls = float(res["vm_outflow"][0] + res["im_addon_outflow"][0])
    current_draw = float(res["margin_line_draw"][0])
    shocked_bb = float(res["shocked_bb"][0])
    actions: list[FundingAction] = []

    actions.append(FundingAction(
        key="cash", label="Raise cash (equity, shareholder loan or asset sale)", required_usd=shortfall, feasible=True,
        detail="Adds one for one to the buffer.", side_effect="Costs capital or requires a sale at stressed prices.",
    ))

    spare_line = max(0.0, margin_calls - current_draw)
    if shortfall <= spare_line + 1e-6:
        new_limit = terms.margin_line_limit + shortfall
        check = _nlb(positions, receivables, market_snapshot, _terms_with(terms, margin_line_limit=new_limit),
                     shocks, specs)
        actions.append(FundingAction(
            key="margin_line", label="Arrange or enlarge a margin financing line", required_usd=shortfall,
            capacity_usd=spare_line, feasible=check >= -1e-6,
            detail="Adds this amount to the existing line, which then funds this scenario's margin calls.",
            side_effect="Bank commitment fee; usable only for exchange margin, not for a deficiency cure.",
        ))
    else:
        actions.append(FundingAction(
            key="margin_line", label="Arrange or enlarge a margin financing line", required_usd=None,
            capacity_usd=spare_line, feasible=False,
            detail="Margin calls not yet covered by a line are smaller than the shortfall; a line alone cannot close it.",
            side_effect="Would need to be combined with another action.",
        ))

    limit_gain = max(0.0, min(shocked_bb, terms.facility_limit + shortfall) - terms.facility_limit)
    if shortfall > 0 and shocked_bb - terms.facility_limit >= shortfall - 1e-6:
        check = _nlb(positions, receivables, market_snapshot,
                     _terms_with(terms, facility_limit=terms.facility_limit + shortfall), shocks, specs)
        actions.append(FundingAction(
            key="facility", label="Increase the revolver limit (accordion)", required_usd=shortfall,
            capacity_usd=max(0.0, shocked_bb - terms.facility_limit), feasible=check >= -1e-6,
            detail="The stressed borrowing base already exceeds the facility limit, so a higher limit is drawable "
                   "at once.",
            side_effect="Needs lender approval; higher commitment fees.",
        ))
    else:
        actions.append(FundingAction(
            key="facility", label="Increase the revolver limit (accordion)",
            required_usd=None if shortfall > 0 else 0.0, capacity_usd=limit_gain, feasible=shortfall == 0,
            detail="The stressed borrowing base caps what a higher limit can add.",
            side_effect="Collateral, not the limit, is the binding constraint.",
        ))

    hedged_value = sum(p.hedged_qty * market_snapshot[p.ticker].spot_price for p in positions)
    if shortfall <= 0:
        scale = 1.0
        feasible = True
    else:
        def nlb_at(k: float) -> float:
            trial = [InventoryPosition(ticker=p.ticker, physical_qty=p.physical_qty, hedged_qty=p.hedged_qty * k)
                     for p in positions]
            return _nlb(trial, receivables, market_snapshot, terms, shocks, specs)
        feasible = nlb_at(0.0) >= 0.0
        scale = 0.0
        if feasible:
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = (lo + hi) / 2.0
                if nlb_at(mid) >= 0.0:
                    lo = mid
                else:
                    hi = mid
            scale = lo
    cut_value = (1.0 - scale) * hedged_value
    actions.append(FundingAction(
        key="hedges", label="Reduce futures hedges pro rata",
        required_usd=cut_value if feasible else None, feasible=feasible,
        detail=(f"Cut every futures hedge by {1.0 - scale:.0%}; the size shown is the hedged notional removed."
                if feasible else "Even with no hedges the buffer stays negative in this scenario."),
        side_effect="Leaves that inventory value exposed to the next price move.",
    ))

    actions.sort(key=lambda a: (not a.feasible, a.required_usd if a.required_usd is not None else float("inf")))
    return ContingencyPlan(
        target_label=target_label,
        target_shocks=shocks,
        target_net_buffer=nlb,
        shortfall=shortfall,
        actions=actions,
        breaking_moves=breaking_moves(positions, receivables, market_snapshot, terms, specs),
        triggers=trigger_ladder(sweep),
    )
