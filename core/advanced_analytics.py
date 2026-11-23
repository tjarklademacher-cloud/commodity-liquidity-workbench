"""Stress sweep, metals x softs grid and hedge frontier run through the advanced engine.

Each point is a full `run_advanced_scenario` call: whole contracts, per-contract initial margin, physical
basis and the margin line, with the active scenario's IM multipliers, basis shocks, liquidation discounts and
margin-line cut held fixed while futures prices move. When the per-contract IM model is on, the step-up
trigger rule is applied per commodity at each point, exactly as for the active scenario.

Results reuse the legacy result types (StressSweepResult, ShockGridResult, HedgeFrontierResult) so the same
charts and tables render them. The mapping keeps the liquidity identity: IM released by a fall counts with
variation margin receipts as a margin inflow.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from core.advanced_scenario import auto_im_multipliers, run_advanced_scenario
from core.advanced_schemas import AdvancedScenarioResult, AdvancedSettings, ScenarioSpec
from core.hedge_optimizer import ALL_COMMODITIES, run_hedge_frontier
from core.schemas import (
    METALS_TICKERS,
    REQUIRED_TICKERS,
    SOFTS_TICKERS,
    FacilityTerms,
    HedgeFrontierResult,
    InventoryPosition,
    LiquidityRegime,
    MarketQuote,
    ReceivableItem,
    ShockGridResult,
    StressScenarioResult,
    StressSweepResult,
)
from core.stress_engine import summarize_sweep


def as_stress_point(adv: AdvancedScenarioResult, shock_pct: float) -> StressScenarioResult:
    """Express an advanced result in the legacy point schema (same buffer, regime and borrowing base)."""
    return StressScenarioResult(
        shock_pct=shock_pct, shocks_map=dict(adv.scenario.futures_shocks),
        baseline_bb=adv.baseline_bb, shocked_bb=adv.shocked_bb, bb_change=adv.shocked_bb - adv.baseline_bb,
        baseline_headroom=adv.baseline_headroom, shocked_headroom=adv.shocked_headroom,
        headroom_change=adv.headroom_change, futures_mtm_pnl=adv.vm_net,
        vm_outflow=adv.vm_payable, vm_inflow=adv.vm_receivable + adv.im_released,
        im_addon_outflow=adv.im_additional, bb_deficiency_cure=adv.bb_deficiency_cure,
        total_cash_outflows=adv.total_cash_outflows, total_liquid_resources=adv.total_liquid_resources,
        net_liquidity_buffer=adv.net_liquidity_buffer, margin_line_draw=adv.margin_line_draw,
        regime=adv.regime, regime_explanation=adv.regime_explanation, shocked_bb_result=adv.shocked_bb_result,
    )


def _spec_at(base: ScenarioSpec, shocks: Mapping[str, float], terms: FacilityTerms, settings: AdvancedSettings,
             user_multipliers: Mapping[str, float], apply_trigger_rule: bool) -> ScenarioSpec:
    mults = (auto_im_multipliers(shocks, terms, user_multipliers, apply_trigger_rule)
             if settings.margin.im_mode == "per_contract" else dict(user_multipliers))
    return base.model_copy(update={"futures_shocks": {t: float(shocks.get(t, 0.0)) for t in REQUIRED_TICKERS},
                                   "im_multipliers": mults})


def _evaluate(positions, receivables, snapshot, terms, settings, base, user_multipliers, apply_trigger_rule,
              shocks: Mapping[str, float]) -> AdvancedScenarioResult:
    spec = _spec_at(base, shocks, terms, settings, user_multipliers, apply_trigger_rule)
    return run_advanced_scenario(positions, receivables, snapshot, terms, settings, spec)


def run_advanced_sweep(
    positions: Sequence[InventoryPosition],
    receivables: Sequence[ReceivableItem],
    market_snapshot: Mapping[str, MarketQuote],
    terms: FacilityTerms,
    settings: AdvancedSettings,
    base_spec: ScenarioSpec,
    user_multipliers: Mapping[str, float] | None = None,
    apply_trigger_rule: bool = True,
    shock_min: float = -0.20,
    shock_max: float = 0.20,
    steps: int = 41,
) -> StressSweepResult:
    """41-point parallel sweep through the advanced engine; breakevens use the legacy interpolation."""
    grid = np.round(np.linspace(shock_min, shock_max, steps), 6)
    mults = dict(user_multipliers or {})
    points = [
        as_stress_point(_evaluate(positions, receivables, market_snapshot, terms, settings, base_spec, mults,
                                  apply_trigger_rule, {t: float(s) for t in REQUIRED_TICKERS}), float(s))
        for s in grid
    ]
    return summarize_sweep(points, terms)


def run_advanced_shock_grid(
    positions: Sequence[InventoryPosition],
    receivables: Sequence[ReceivableItem],
    market_snapshot: Mapping[str, MarketQuote],
    terms: FacilityTerms,
    settings: AdvancedSettings,
    base_spec: ScenarioSpec,
    user_multipliers: Mapping[str, float] | None = None,
    apply_trigger_rule: bool = True,
    shock_min: float = -0.20,
    shock_max: float = 0.20,
    steps: int = 17,
) -> ShockGridResult:
    """Metals (gold, copper) against lumber and softs (cocoa, coffee, sugar) through the advanced engine, on the
    same axes as the legacy grid."""
    axis = np.round(np.linspace(shock_min, shock_max, steps), 6)
    mults = dict(user_multipliers or {})
    nlb = np.zeros((steps, steps))
    bb = np.zeros((steps, steps))
    regimes: list[list[str]] = []
    for i, m in enumerate(axis):
        row = []
        for j, s in enumerate(axis):
            shocks = {t: 0.0 for t in REQUIRED_TICKERS}
            shocks.update({t: float(m) for t in METALS_TICKERS})
            shocks.update({t: float(s) for t in SOFTS_TICKERS})
            res = _evaluate(positions, receivables, market_snapshot, terms, settings, base_spec, mults,
                            apply_trigger_rule, shocks)
            nlb[i, j], bb[i, j] = res.net_liquidity_buffer, res.shocked_bb
            row.append(str(res.regime))
        regimes.append(row)
    wi, wj = np.unravel_index(int(np.argmin(nlb)), nlb.shape)
    squeeze = str(LiquidityRegime.CRITICAL_SQUEEZE)
    return ShockGridResult(
        metals_axis=axis.tolist(), softs_axis=axis.tolist(), net_buffer=nlb.tolist(), shocked_bb=bb.tolist(),
        regime=regimes, squeeze_share=float(np.mean([[r == squeeze for r in row] for row in regimes])),
        worst_net_buffer=float(nlb[wi, wj]), worst_metals_shock=float(axis[wi]), worst_softs_shock=float(axis[wj]),
    )


def run_advanced_hedge_frontier(
    positions: Sequence[InventoryPosition],
    receivables: Sequence[ReceivableItem],
    market_snapshot: Mapping[str, MarketQuote],
    terms: FacilityTerms,
    settings: AdvancedSettings,
    base_spec: ScenarioSpec,
    user_multipliers: Mapping[str, float] | None = None,
    apply_trigger_rule: bool = True,
    scope: str = ALL_COMMODITIES,
) -> HedgeFrontierResult:
    """Hedge ratio frontier with every ratio run through the advanced 41-point sweep (whole contracts, so a
    ratio rounds to the nearest contract count)."""
    mults = dict(user_multipliers or {})

    def evaluate(trial: list[InventoryPosition], matrix: np.ndarray) -> dict[str, np.ndarray | float]:
        results = [
            _evaluate(trial, receivables, market_snapshot, terms, settings, base_spec, mults, apply_trigger_rule,
                      dict(zip(REQUIRED_TICKERS, row.tolist())))
            for row in matrix
        ]
        return {
            "net_liquidity_buffer": np.array([r.net_liquidity_buffer for r in results]),
            "vm_outflow": np.array([r.vm_payable for r in results]),
            "im_addon_outflow": np.array([r.im_additional for r in results]),
            "baseline_bb": results[0].baseline_bb,
        }

    return run_hedge_frontier(positions, receivables, market_snapshot, terms, scope=scope, evaluate=evaluate)
