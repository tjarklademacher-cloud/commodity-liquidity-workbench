"""Margin-trap liquidity stress engine, sweep runner, breakevens and certificate exporters."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

import numpy as np

from core.borrowing_base import calculate_borrowing_base
from core.haircuts import compute_portfolio_inventory
from core.schemas import (
    COMMODITY_SPECS,
    REQUIRED_TICKERS,
    BorrowingBaseResult,
    CommoditySpec,
    FacilityTerms,
    InventoryPosition,
    LiquidityRegime,
    MarketLoadResult,
    MarketQuote,
    ReceivableItem,
    StressScenarioResult,
    StressSweepResult,
)


def classify_liquidity_regime(
    net_liquidity_buffer: float,
    vm_outflow: float,
    headroom_change: float,
    bb_deficiency_cure: float,
    margin_line_vm_cover: float = 0.0,
) -> tuple[LiquidityRegime, str]:
    """Strict-priority treasury regime classification with a plain-English diagnosis.

    `margin_line_vm_cover` is the part of the variation margin call funded by a dedicated margin line; only the
    uncovered remainder can create a margin trap.
    """
    if net_liquidity_buffer < 0.0:
        return (
            LiquidityRegime.CRITICAL_SQUEEZE,
            f"Cash calls exceed available liquidity by ${-net_liquidity_buffer / 1e6:,.2f}M: "
            "futures margin and cure obligations cannot be funded from cash, revolver headroom and any margin line.",
        )
    uncovered_vm = vm_outflow - min(max(0.0, margin_line_vm_cover), vm_outflow)
    if uncovered_vm > 0.0 and uncovered_vm > max(0.0, headroom_change):
        covered = (f" (after ${(vm_outflow - uncovered_vm) / 1e6:,.2f}M from the margin line)"
                   if uncovered_vm < vm_outflow else "")
        return (
            LiquidityRegime.MARGIN_TRAP_WARNING,
            f"Futures variation margin of ${uncovered_vm / 1e6:,.2f}M{covered} outpaces the "
            f"${max(0.0, headroom_change) / 1e6:,.2f}M of new revolver headroom: the inventory rally is "
            "trapped behind haircuts, caps and the facility ceiling while hedges drain cash.",
        )
    if bb_deficiency_cure > 0.0:
        return (
            LiquidityRegime.BB_DEFICIENCY_WARNING,
            f"Borrowing base has fallen below the drawn revolver: a ${bb_deficiency_cure / 1e6:,.2f}M "
            "mandatory deficiency cure payment is due, partly offset by futures VM inflows.",
        )
    return (
        LiquidityRegime.HEALTHY,
        "Liquid resources comfortably cover all margin and cure obligations; no deficiency and no margin trap.",
    )


def _resolve_shocks_map(
    positions: Sequence[InventoryPosition],
    shock_pct: float,
    per_commodity_shocks: Mapping[str, float] | None,
) -> dict[str, float]:
    tickers = list(dict.fromkeys([*REQUIRED_TICKERS, *(p.ticker for p in positions)]))
    if per_commodity_shocks is None:
        return {t: float(shock_pct) for t in tickers}
    return {t: float(per_commodity_shocks.get(t, shock_pct)) for t in tickers}


def run_stress_scenario(
    positions: Sequence[InventoryPosition],
    receivables: Sequence[ReceivableItem],
    market_snapshot: Mapping[str, MarketQuote],
    terms: FacilityTerms,
    specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS,
    shock_pct: float = 0.0,
    per_commodity_shocks: Mapping[str, float] | None = None,
    baseline_bb_result: BorrowingBaseResult | None = None,
) -> StressScenarioResult:
    """Revalue collateral under price shocks and net futures margin calls against liquidity."""
    if baseline_bb_result is None:
        baseline_items = compute_portfolio_inventory(positions, market_snapshot, specs)
        baseline_bb_result = calculate_borrowing_base(baseline_items, receivables, terms)
    shocks_map = _resolve_shocks_map(positions, shock_pct, per_commodity_shocks)

    shocked_items = compute_portfolio_inventory(positions, market_snapshot, specs, shocks_map)
    shocked_bb_result = calculate_borrowing_base(shocked_items, receivables, terms)

    baseline_bb = baseline_bb_result.final_borrowing_base
    shocked_bb = shocked_bb_result.final_borrowing_base
    baseline_headroom = max(0.0, min(terms.facility_limit, baseline_bb) - terms.drawn_revolver)
    shocked_headroom = max(0.0, min(terms.facility_limit, shocked_bb) - terms.drawn_revolver)
    headroom_change = shocked_headroom - baseline_headroom

    futures_mtm_pnl = -sum(
        pos.hedged_qty * market_snapshot[pos.ticker].spot_price * shocks_map[pos.ticker] for pos in positions
    )
    vm_outflow = max(0.0, -futures_mtm_pnl)
    vm_inflow = max(0.0, futures_mtm_pnl)
    max_abs_shock = max((abs(v) for v in shocks_map.values()), default=0.0)
    im_addon_outflow = (
        terms.im_stepup_fraction * terms.baseline_initial_margin
        if max_abs_shock >= (terms.im_stepup_trigger - 1e-9)
        else 0.0
    )
    bb_deficiency_cure = max(0.0, terms.drawn_revolver - shocked_bb)
    total_cash_outflows = vm_outflow + im_addon_outflow + bb_deficiency_cure
    margin_line_draw = min(terms.margin_line_limit, vm_outflow + im_addon_outflow)
    total_liquid_resources = terms.unrestricted_cash + shocked_headroom + vm_inflow + margin_line_draw
    net_liquidity_buffer = total_liquid_resources - total_cash_outflows
    regime, explanation = classify_liquidity_regime(
        net_liquidity_buffer, vm_outflow, headroom_change, bb_deficiency_cure,
        margin_line_vm_cover=min(terms.margin_line_limit, vm_outflow),
    )
    return StressScenarioResult(
        shock_pct=float(shock_pct),
        shocks_map=shocks_map,
        baseline_bb=baseline_bb,
        shocked_bb=shocked_bb,
        bb_change=shocked_bb - baseline_bb,
        baseline_headroom=baseline_headroom,
        shocked_headroom=shocked_headroom,
        headroom_change=headroom_change,
        futures_mtm_pnl=futures_mtm_pnl,
        vm_outflow=vm_outflow,
        vm_inflow=vm_inflow,
        im_addon_outflow=im_addon_outflow,
        bb_deficiency_cure=bb_deficiency_cure,
        total_cash_outflows=total_cash_outflows,
        total_liquid_resources=total_liquid_resources,
        net_liquidity_buffer=net_liquidity_buffer,
        margin_line_draw=margin_line_draw,
        regime=regime,
        regime_explanation=explanation,
        shocked_bb_result=shocked_bb_result,
    )


def _interpolate(x1: float, y1: float, x2: float, y2: float, target: float) -> float:
    return x1 + (target - y1) / (y2 - y1) * (x2 - x1)


def run_stress_sweep(
    positions: Sequence[InventoryPosition],
    receivables: Sequence[ReceivableItem],
    market_snapshot: Mapping[str, MarketQuote],
    terms: FacilityTerms,
    specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS,
    shock_min: float = -0.20,
    shock_max: float = 0.20,
    steps: int = 41,
) -> StressSweepResult:
    """Run parallel shocks across a rounded linear grid and locate exact breakevens."""
    grid = np.round(np.linspace(shock_min, shock_max, steps), 6)
    baseline_items = compute_portfolio_inventory(positions, market_snapshot, specs)
    baseline_bb_result = calculate_borrowing_base(baseline_items, receivables, terms)
    points = [
        run_stress_scenario(
            positions, receivables, market_snapshot, terms, specs,
            shock_pct=float(s), baseline_bb_result=baseline_bb_result,
        )
        for s in grid
    ]

    upside: float | None = None
    for p1, p2 in zip(points[:-1], points[1:]):
        if p1.shock_pct >= 0.0 and p1.net_liquidity_buffer >= 0.0 and p2.net_liquidity_buffer < 0.0:
            upside = _interpolate(
                p1.shock_pct, p1.net_liquidity_buffer, p2.shock_pct, p2.net_liquidity_buffer, 0.0
            )
            break

    downside: float | None = None
    start_candidates = [i for i, p in enumerate(points) if p.shock_pct >= 0.0]
    start = start_candidates[0] if start_candidates else len(points) - 1
    for i in range(start, 0, -1):
        p1, p2 = points[i], points[i - 1]
        if p1.shocked_bb >= terms.drawn_revolver and p2.shocked_bb < terms.drawn_revolver:
            downside = _interpolate(p1.shock_pct, p1.shocked_bb, p2.shock_pct, p2.shocked_bb, terms.drawn_revolver)
            break

    counts = {str(r): 0 for r in LiquidityRegime}
    for p in points:
        counts[str(p.regime)] += 1
    worst = min(points, key=lambda p: p.net_liquidity_buffer)
    return StressSweepResult(
        points=points,
        upside_squeeze_breakeven_pct=upside,
        downside_deficiency_breakeven_pct=downside,
        worst_net_buffer_point=worst,
        regime_counts=counts,
    )


def _certificate_payload(
    bb_result: BorrowingBaseResult,
    stress_result: StressScenarioResult,
    market_meta: MarketLoadResult,
    borrower: str = "",
) -> dict:
    return {
        "certificate": {
            "title": "Borrowing Base Certificate - Commodity ABL Facility",
            "borrower": borrower,
            "currency": "USD",
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "valuation_date": market_meta.as_of_date,
            "market_data_source": market_meta.source_label,
            "used_fallback": market_meta.used_fallback,
        },
        "market_snapshot": {
            t: {
                "spot_price": q.spot_price,
                "realized_vol_30d": q.realized_vol_30d,
                "vol_fallback_used": q.vol_fallback_used,
            }
            for t, q in market_meta.snapshot.items()
        },
        "inventory": [
            {
                "ticker": i.ticker,
                "commodity": i.name,
                "physical_qty": i.physical_qty,
                "hedged_qty": i.hedged_qty,
                "spot_price": i.spot_price,
                "hedge_ratio": i.hedge_ratio,
                "total_haircut": i.total_haircut,
                "advance_rate": i.advance_rate,
                "gross_market_value": i.gross_market_value,
                "eligible_collateral": i.eligible_collateral,
            }
            for i in bb_result.inventory_items
        ],
        "receivables": [
            {
                "buyer_name": r.buyer_name,
                "commodity": r.commodity,
                "gross_amount": r.gross_amount,
                "days_past_due": r.days_past_due,
                "is_investment_grade": r.is_investment_grade,
                "pre_cap_eligible": r.pre_cap_eligible,
                "final_eligible": r.final_eligible,
                "concentration_cutback": r.concentration_cutback,
                "is_capped": r.is_capped,
            }
            for r in bb_result.receivable_items
        ],
        "borrowing_base": {
            "gross_inventory": bb_result.gross_inventory,
            "inventory_haircut_dollars": bb_result.inventory_haircut_dollars,
            "eligible_inventory": bb_result.eligible_inventory,
            "gross_receivables": bb_result.gross_receivables,
            "aging_ineligible_dollars": bb_result.aging_ineligible_dollars,
            "ar_advance_haircut_dollars": bb_result.ar_advance_haircut_dollars,
            "pre_cap_eligible_receivables": bb_result.pre_cap_eligible_receivables,
            "concentration_cap_pct": bb_result.concentration_cap_pct,
            "cap_dollar_limit": bb_result.cap_dollar_limit,
            "concentration_cutback_dollars": bb_result.concentration_cutback_dollars,
            "final_eligible_receivables": bb_result.final_eligible_receivables,
            "final_borrowing_base": bb_result.final_borrowing_base,
            "facility_limit": bb_result.facility_limit,
            "drawn_revolver": bb_result.drawn_revolver,
            "available_headroom": bb_result.available_headroom,
            "facility_utilization_pct": bb_result.facility_utilization_pct,
            "collateral_coverage_ratio": bb_result.collateral_coverage_ratio,
            "capped_buyer_names": bb_result.capped_buyer_names,
            "solver_iterations": bb_result.solver_iterations,
            "solver_used_closed_form": bb_result.solver_used_closed_form,
            "solver_residual": bb_result.solver_residual,
        },
        "active_stress": {
            "shock_pct": stress_result.shock_pct,
            "shocks_map": stress_result.shocks_map,
            "shocked_bb": stress_result.shocked_bb,
            "shocked_headroom": stress_result.shocked_headroom,
            "headroom_change": stress_result.headroom_change,
            "futures_mtm_pnl": stress_result.futures_mtm_pnl,
            "vm_outflow": stress_result.vm_outflow,
            "vm_inflow": stress_result.vm_inflow,
            "im_addon_outflow": stress_result.im_addon_outflow,
            "margin_line_draw": stress_result.margin_line_draw,
            "bb_deficiency_cure": stress_result.bb_deficiency_cure,
            "total_cash_outflows": stress_result.total_cash_outflows,
            "total_liquid_resources": stress_result.total_liquid_resources,
            "net_liquidity_buffer": stress_result.net_liquidity_buffer,
            "regime": str(stress_result.regime),
            "regime_explanation": stress_result.regime_explanation,
        },
    }


def build_bb_certificate_csv(
    bb_result: BorrowingBaseResult,
    stress_result: StressScenarioResult,
    market_meta: MarketLoadResult,
    borrower: str = "",
) -> str:
    """Flatten the certificate to a three-column (section, item, value) audit CSV."""
    payload = _certificate_payload(bb_result, stress_result, market_meta, borrower)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["section", "item", "value"])
    for section in ("certificate", "borrowing_base", "active_stress"):
        for key, value in payload[section].items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            writer.writerow([section, key, value])
    for ticker, quote in payload["market_snapshot"].items():
        for key, value in quote.items():
            writer.writerow(["market_snapshot", f"{ticker}.{key}", value])
    for row in payload["inventory"]:
        for key, value in row.items():
            if key != "ticker":
                writer.writerow(["inventory", f"{row['ticker']}.{key}", value])
    for row in payload["receivables"]:
        for key, value in row.items():
            if key != "buyer_name":
                writer.writerow(["receivables", f"{row['buyer_name']}.{key}", value])
    return buf.getvalue()


def build_bb_certificate_json(
    bb_result: BorrowingBaseResult,
    stress_result: StressScenarioResult,
    market_meta: MarketLoadResult,
    borrower: str = "",
) -> str:
    """Serialize the certificate as indented JSON."""
    return json.dumps(_certificate_payload(bb_result, stress_result, market_meta, borrower), indent=2)
