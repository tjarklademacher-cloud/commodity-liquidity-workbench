"""Advanced instantaneous scenario: margin engine + basis risk + borrowing base + liquidity, with a
step-by-step reconciliation to the legacy stress result.

Neutral settings (continuous contracts, legacy IM step-up, zero opening basis, no basis shock, no
liquidation discount, no margin-line cut) reproduce `stress_engine.run_stress_scenario` exactly.

Liquidity identity (same structure as the legacy engine):
  resources = unrestricted cash + shocked revolver headroom + VM receivable + IM released + margin line draw
  outflows  = VM payable + additional IM + borrowing base deficiency cure
  net liquidity buffer = resources - outflows
Variation margin nets across commodities (one clearing account); initial margin nets at portfolio level.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from core.advanced_schemas import (
    AdvancedScenarioResult,
    AdvancedSettings,
    CommodityMarginLine,
    MarginSettings,
    ReconciliationStep,
    ScenarioSpec,
)
from core.basis_risk import default_basis_assumptions, economic_pnl, scenario_prices
from core.borrowing_base import calculate_borrowing_base
from core.haircuts import compute_inventory_valuation
from core.margin_engine import (
    default_margin_settings,
    im_change,
    initial_margin_required,
    signed_short_contracts,
)
from core.schemas import (
    COMMODITY_SPECS,
    REQUIRED_TICKERS,
    CommoditySpec,
    FacilityTerms,
    InventoryPosition,
    MarketQuote,
    ReceivableItem,
)
from core.stress_engine import classify_liquidity_regime


def default_advanced_settings() -> AdvancedSettings:
    return AdvancedSettings(margin=default_margin_settings(), basis=default_basis_assumptions())


def neutral_settings(settings: AdvancedSettings) -> AdvancedSettings:
    """Same margin profiles, but legacy conventions: continuous contracts, legacy IM rule, zero basis."""
    margin = MarginSettings.model_validate({**settings.margin.model_dump(), "contract_policy": "continuous",
                                            "im_mode": "legacy_stepup"})
    return AdvancedSettings(margin=margin, basis=default_basis_assumptions())


def _quote_at(quote: MarketQuote, price: float) -> MarketQuote:
    data = quote.model_dump()
    data["spot_price"] = max(price, 1e-12)
    return MarketQuote.model_validate(data)


def run_advanced_scenario(
    positions: Sequence[InventoryPosition],
    receivables: Sequence[ReceivableItem],
    market_snapshot: Mapping[str, MarketQuote],
    terms: FacilityTerms,
    settings: AdvancedSettings,
    scenario: ScenarioSpec,
    specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS,
) -> AdvancedScenarioResult:
    margin = settings.margin
    baseline_items, shocked_items, lines = [], [], []
    vm_net = 0.0
    im_req_0 = 0.0
    im_req_1 = 0.0
    fin_vm_payable = 0.0
    fin_im_increment = 0.0
    totals = {k: 0.0 for k in ("physical_pnl", "futures_pnl", "basis_pnl", "residual_price_pnl",
                               "combined_pnl", "liquidation_discount_cost")}
    phys_value_0 = phys_value_1 = 0.0
    for pos in positions:
        t = pos.ticker
        spec = specs[t]
        quote = market_snapshot[t]
        b0 = settings.basis[t].opening_basis_pct
        f0, f1, p0, p1, collateral = scenario_prices(
            quote.spot_price, scenario.futures_shocks.get(t, 0.0), b0,
            scenario.basis_shocks.get(t, 0.0), scenario.liquidation_discounts.get(t, 0.0),
        )
        base_quote = _quote_at(quote, p0)
        base_item = compute_inventory_valuation(pos, base_quote, spec, 0.0)
        shock = (collateral / p0 - 1.0) if p0 > 0 else 0.0
        shocked_item = compute_inventory_valuation(pos, base_quote, spec, shock)
        baseline_items.append(base_item)
        shocked_items.append(shocked_item)

        signed, exact, residual = signed_short_contracts(pos.hedged_qty, t, margin, specs)
        vm = signed * spec.contract_multiplier * (f1 - f0)
        vm_net += vm
        profile = margin.profiles[t]
        mult = scenario.im_multipliers.get(t, 1.0)
        r0 = initial_margin_required(signed, profile, 1.0)
        r1 = initial_margin_required(signed, profile, mult)
        im_req_0 += r0
        im_req_1 += r1
        if profile.financeable:
            fin_vm_payable += max(0.0, -vm)
            fin_im_increment += max(0.0, r1 - r0)
        pnl = economic_pnl(pos.physical_qty, signed, spec.contract_multiplier, f0, f1, p0, p1, b0,
                           scenario.liquidation_discounts.get(t, 0.0))
        for k in totals:
            totals[k] += pnl[k]
        phys_value_0 += pos.physical_qty * p0
        phys_value_1 += pos.physical_qty * p1
        lines.append(CommodityMarginLine(
            ticker=t, name=spec.name, physical_qty=pos.physical_qty, hedged_qty=pos.hedged_qty,
            contracts_exact=exact, contracts=signed, residual_hedge_qty=residual,
            futures_price_0=f0, futures_price_1=f1, physical_price_0=p0, physical_price_1=p1,
            collateral_price_1=collateral, futures_notional_1=abs(signed) * spec.contract_multiplier * f1,
            vm_cash=vm, im_per_contract=profile.im_per_contract, im_multiplier=mult, im_required_0=r0,
            im_required_1=r1, financeable=profile.financeable, **pnl,
        ))

    baseline_bb = calculate_borrowing_base(baseline_items, receivables, terms)
    shocked_bb = calculate_borrowing_base(shocked_items, receivables, terms)
    baseline_headroom = max(0.0, min(terms.facility_limit, baseline_bb.final_borrowing_base) - terms.drawn_revolver)
    shocked_headroom = max(0.0, min(terms.facility_limit, shocked_bb.final_borrowing_base) - terms.drawn_revolver)
    headroom_change = shocked_headroom - baseline_headroom

    vm_payable = max(0.0, -vm_net)
    vm_receivable = max(0.0, vm_net)
    if margin.im_mode == "legacy_stepup":
        max_abs = max((abs(scenario.futures_shocks.get(t, 0.0)) for t in REQUIRED_TICKERS), default=0.0)
        addon = (terms.im_stepup_fraction * terms.baseline_initial_margin
                 if max_abs >= terms.im_stepup_trigger - 1e-9 else 0.0)
        posted, required = terms.baseline_initial_margin, terms.baseline_initial_margin + addon
        im_additional, im_released = addon, 0.0
        eligible_im = addon
    else:
        posted, required = im_req_0, im_req_1
        im_additional, im_released = im_change(posted, required, margin.release_excess_im)
        eligible_im = min(im_additional, fin_im_increment)
    eligible_vm = min(vm_payable, fin_vm_payable)

    capacity = terms.margin_line_limit * (1.0 - scenario.margin_line_cut_pct)
    eligible_need = eligible_vm + eligible_im
    draw = min(capacity, eligible_need)
    margin_need = vm_payable + im_additional
    cure = max(0.0, terms.drawn_revolver - shocked_bb.final_borrowing_base)
    carry = terms.funding_carry_cost(phys_value_0)
    resources = terms.unrestricted_cash + shocked_headroom + vm_receivable + im_released + draw
    outflows = vm_payable + im_additional + cure + carry
    nlb = resources - outflows
    regime, explanation = classify_liquidity_regime(nlb, vm_payable, headroom_change, cure,
                                                    margin_line_vm_cover=min(draw, vm_payable))
    return AdvancedScenarioResult(
        scenario=scenario, lines=lines, baseline_bb=baseline_bb.final_borrowing_base,
        shocked_bb_result=shocked_bb, shocked_bb=shocked_bb.final_borrowing_base,
        baseline_headroom=baseline_headroom, shocked_headroom=shocked_headroom, headroom_change=headroom_change,
        vm_net=vm_net, vm_payable=vm_payable, vm_receivable=vm_receivable, im_mode=margin.im_mode,
        im_posted_opening=posted, im_required=required, im_additional=im_additional, im_released=im_released,
        margin_cash_need=margin_need, margin_line_capacity=capacity, margin_line_eligible_need=eligible_need,
        margin_line_draw=draw, margin_funded_from_cash=max(0.0, margin_need - draw), bb_deficiency_cure=cure,
        funding_carry_outflow=carry,
        total_liquid_resources=resources, total_cash_outflows=outflows, net_liquidity_buffer=nlb,
        regime=regime, regime_explanation=explanation,
        physical_inventory_value_0=phys_value_0, physical_inventory_value_1=phys_value_1, **totals,
    )


def reconcile_to_legacy(
    positions: Sequence[InventoryPosition],
    receivables: Sequence[ReceivableItem],
    market_snapshot: Mapping[str, MarketQuote],
    terms: FacilityTerms,
    settings: AdvancedSettings,
    scenario: ScenarioSpec,
    specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS,
) -> list[ReconciliationStep]:
    """Sequential attribution from the legacy buffer to the advanced buffer, one assumption at a time.

    Steps: legacy (neutral settings) -> whole-contract rounding -> per-contract IM model -> opening basis
    -> basis shock -> liquidation discount -> margin line cut. Changes sum exactly to the total difference.
    """
    neutral = neutral_settings(settings)
    base_spec = ScenarioSpec(name=scenario.name, futures_shocks=scenario.futures_shocks,
                             im_multipliers=scenario.im_multipliers)
    steps: list[tuple[str, AdvancedSettings, ScenarioSpec]] = [("Legacy model", neutral, base_spec)]
    s1 = AdvancedSettings(margin=MarginSettings.model_validate(
        {**neutral.margin.model_dump(), "contract_policy": settings.margin.contract_policy}), basis=neutral.basis)
    steps.append(("Whole-contract hedge rounding", s1, base_spec))
    s2 = AdvancedSettings(margin=settings.margin, basis=neutral.basis)
    steps.append(("Per-contract initial margin model", s2, base_spec))
    s3 = AdvancedSettings(margin=settings.margin, basis=dict(settings.basis))
    steps.append(("Opening physical basis", s3, base_spec))
    spec4 = ScenarioSpec.model_validate({**base_spec.model_dump(), "basis_shocks": scenario.basis_shocks})
    steps.append(("Basis shock", s3, spec4))
    spec5 = ScenarioSpec.model_validate({**spec4.model_dump(), "liquidation_discounts": scenario.liquidation_discounts})
    steps.append(("Liquidation discount on collateral", s3, spec5))
    steps.append(("Margin line reduction", s3, scenario))
    out: list[ReconciliationStep] = []
    prev = None
    for label, st, spec in steps:
        nlb = run_advanced_scenario(positions, receivables, market_snapshot, terms, st, spec, specs).net_liquidity_buffer
        out.append(ReconciliationStep(label=label, net_liquidity_buffer=nlb, change=0.0 if prev is None else nlb - prev))
        prev = nlb
    return out


def legacy_equivalent_scenario(shocks: Mapping[str, float]) -> ScenarioSpec:
    return ScenarioSpec(name="Legacy", futures_shocks={t: float(shocks.get(t, 0.0)) for t in REQUIRED_TICKERS})



MARGIN_PRESETS: dict[str, dict] = {
    "Price shock only": {"im_multipliers": {}, "margin_line_cut_pct": 0.0},
    "IM increase without price move (x1.5 all)": {
        "futures_shocks": {t: 0.0 for t in REQUIRED_TICKERS},
        "im_multipliers": {t: 1.5 for t in REQUIRED_TICKERS}, "margin_line_cut_pct": 0.0},
    "Rally with margin hike (+15% prices, IM x1.5)": {
        "futures_shocks": {t: 0.15 for t in REQUIRED_TICKERS},
        "im_multipliers": {t: 1.5 for t in REQUIRED_TICKERS}, "margin_line_cut_pct": 0.0},
    "Softs volatility shock (IM x1.75 cocoa, coffee, sugar)": {
        "im_multipliers": {"CC=F": 1.75, "KC=F": 1.75, "SB=F": 1.75}, "margin_line_cut_pct": 0.0},
    "Cocoa squeeze (+25% cocoa, IM x1.6)": {
        "futures_shocks": {t: (0.25 if t == "CC=F" else 0.0) for t in REQUIRED_TICKERS},
        "im_multipliers": {"CC=F": 1.6}, "margin_line_cut_pct": 0.0},
    "Funding disruption (margin line cut 50%)": {"im_multipliers": {}, "margin_line_cut_pct": 0.5},
}
"""Illustrative margin stress presets; not historical exchange decisions."""


def auto_im_multipliers(
    futures_shocks: Mapping[str, float],
    terms: FacilityTerms,
    base_multipliers: Mapping[str, float] | None = None,
    apply_trigger_rule: bool = True,
) -> dict[str, float]:
    """Per-commodity IM multipliers: user multipliers, times (1 + step-up fraction) for any commodity whose
    own move reaches the step-up trigger when the rule is on (the per-commodity analogue of the legacy rule)."""
    base = dict(base_multipliers or {})
    out = {}
    for t in REQUIRED_TICKERS:
        m = float(base.get(t, 1.0))
        if apply_trigger_rule and abs(float(futures_shocks.get(t, 0.0))) >= terms.im_stepup_trigger - 1e-9:
            m *= 1.0 + terms.im_stepup_fraction
        out[t] = m
    return out


def buffer_sensitivities(
    positions: Sequence[InventoryPosition],
    receivables: Sequence[ReceivableItem],
    market_snapshot: Mapping[str, MarketQuote],
    terms: FacilityTerms,
    settings: AdvancedSettings,
    scenario: ScenarioSpec,
    specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS,
) -> list[ReconciliationStep]:
    """One-at-a-time changes to the active scenario's net liquidity buffer (each relative to the active case)."""
    base = run_advanced_scenario(positions, receivables, market_snapshot, terms, settings, scenario, specs)
    b = base.net_liquidity_buffer

    def with_terms(**kw: float) -> FacilityTerms:
        return FacilityTerms.model_validate({**terms.model_dump(), **kw})

    def with_spec(**kw: object) -> ScenarioSpec:
        return ScenarioSpec.model_validate({**scenario.model_dump(), **kw})

    cases: list[tuple[str, object]] = [
        ("Initial margin requirement x1.5 (all commodities)",
         ("spec", with_spec(im_multipliers={t: 1.5 * scenario.im_multipliers.get(t, 1.0) for t in REQUIRED_TICKERS}))),
        ("Basis 5% weaker on all commodities",
         ("spec", with_spec(basis_shocks={t: scenario.basis_shocks.get(t, 0.0) - 0.05 for t in REQUIRED_TICKERS}))),
        ("Hedge ratio 10 points lower (all commodities)",
         ("pos", [InventoryPosition(ticker=p.ticker, physical_qty=p.physical_qty,
                                    hedged_qty=max(0.0, p.hedged_qty - 0.10 * p.physical_qty)) for p in positions])),
        ("Receivable advance rates 10 points lower",
         ("terms", with_terms(ar_ig_advance_rate=max(0.0, terms.ar_ig_advance_rate - 0.10),
                              ar_non_ig_advance_rate=max(0.0, terms.ar_non_ig_advance_rate - 0.10)))),
        ("Margin financing line +10M", ("terms", with_terms(margin_line_limit=terms.margin_line_limit + 10e6))),
        ("Unrestricted cash +5M", ("terms", with_terms(unrestricted_cash=terms.unrestricted_cash + 5e6))),
    ]
    out = [ReconciliationStep(label="Active scenario", net_liquidity_buffer=b, change=0.0)]
    for label, (kind, value) in cases:
        pos, tm, sp = positions, terms, scenario
        if kind == "spec":
            sp = value
        elif kind == "pos":
            pos = value
        else:
            tm = value
        nlb = run_advanced_scenario(pos, receivables, market_snapshot, tm, settings, sp, specs).net_liquidity_buffer
        out.append(ReconciliationStep(label=label, net_liquidity_buffer=nlb, change=nlb - b))
    return out
