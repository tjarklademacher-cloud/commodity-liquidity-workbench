"""Borrowing base engine: AR eligibility, fixed-point concentration caps and waterfall bridge."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from core.schemas import (
    CAP_TIER_OPTIONS,
    BorrowingBaseResult,
    CapTierComparison,
    FacilityTerms,
    InventoryValuation,
    ReceivableItem,
    ReceivableValuation,
    WaterfallLine,
)


def _closed_form_cap_solution(eligible_inventory: float, pre_cap_amounts: Sequence[float], cap_pct: float) -> float:
    """Exact O(n log n) sorted-suffix solution of BB = BB_inv + sum_j min(a_j, cap * BB)."""
    a_sorted = sorted([max(0.0, float(x)) for x in pre_cap_amounts], reverse=True)
    n = len(a_sorted)
    suffix = [0.0] * (n + 1)
    for idx in range(n - 1, -1, -1):
        suffix[idx] = suffix[idx + 1] + a_sorted[idx]
    k = 0
    while k <= n and 1.0 - k * cap_pct > 1e-12:
        cand_bb = (eligible_inventory + suffix[k]) / (1.0 - k * cap_pct)
        lim = cap_pct * cand_bb
        upper_ok = k == 0 or a_sorted[k - 1] >= lim - 1e-9
        lower_ok = k == n or a_sorted[k] <= lim + 1e-9
        if upper_ok and lower_ok:
            return max(0.0, cand_bb)
        k += 1
    return 0.0


def solve_concentration_cap_fixed_point(
    eligible_inventory: float,
    pre_cap_amounts: Sequence[float],
    cap_pct: float,
    max_iterations: int = 25,
) -> tuple[float, int, bool]:
    """Solve the self-referential single-buyer cap. Returns (solved_bb, iterations, used_closed_form)."""
    pre_caps = [max(0.0, float(x)) for x in pre_cap_amounts]
    n = len(pre_caps)
    bb_inv = max(0.0, float(eligible_inventory))
    bb = bb_inv + sum(pre_caps)
    capped_mask = [False] * n
    for it in range(1, max_iterations + 1):
        new_mask = [val > (cap_pct * bb + 1e-9) for val in pre_caps]
        if new_mask == capped_mask and it > 1:
            return bb, it, False
        denom = 1.0 - sum(new_mask) * cap_pct
        if denom <= 1e-9:
            return _closed_form_cap_solution(bb_inv, pre_caps, cap_pct), it, True
        uncapped_sum = sum(val for val, capped in zip(pre_caps, new_mask) if not capped)
        bb = (bb_inv + uncapped_sum) / denom
        capped_mask = new_mask
    return _closed_form_cap_solution(bb_inv, pre_caps, cap_pct), max_iterations, True


def _value_receivables(receivables: Sequence[ReceivableItem], terms: FacilityTerms) -> list[dict]:
    rows = []
    for rec in receivables:
        is_aging_eligible = rec.days_past_due <= terms.ar_aging_limit_days
        eligible_gross_amount = rec.gross_amount if is_aging_eligible else 0.0
        ar_advance_rate = (
            (terms.ar_ig_advance_rate if rec.is_investment_grade else terms.ar_non_ig_advance_rate)
            if is_aging_eligible
            else 0.0
        )
        pre_cap_eligible = eligible_gross_amount * ar_advance_rate
        rows.append(
            {
                "rec": rec,
                "is_aging_eligible": is_aging_eligible,
                "aging_ineligible_amount": rec.gross_amount - eligible_gross_amount,
                "eligible_gross_amount": eligible_gross_amount,
                "ar_advance_rate": ar_advance_rate,
                "ar_haircut_amount": eligible_gross_amount - pre_cap_eligible,
                "pre_cap_eligible": pre_cap_eligible,
            }
        )
    return rows


def calculate_borrowing_base(
    inventory_items: Sequence[InventoryValuation],
    receivables: Sequence[ReceivableItem],
    terms: FacilityTerms,
) -> BorrowingBaseResult:
    """Compute the full borrowing base certificate for the given collateral pool and terms."""
    gross_inventory = sum(i.gross_market_value for i in inventory_items)
    eligible_inventory = sum(i.eligible_collateral for i in inventory_items)
    inventory_haircut_dollars = gross_inventory - eligible_inventory
    total_hedged_notional = sum(i.hedged_notional for i in inventory_items)
    weighted_inv_advance_rate = eligible_inventory / gross_inventory if gross_inventory > 0.0 else 0.0
    weighted_hedge_ratio = (
        sum(i.hedge_ratio * i.gross_market_value for i in inventory_items) / gross_inventory
        if gross_inventory > 0.0
        else 0.0
    )

    rec_rows = _value_receivables(receivables, terms)
    pre_caps = [r["pre_cap_eligible"] for r in rec_rows]
    cap_pct = terms.concentration_cap_pct
    solved_bb, iterations, used_closed_form = solve_concentration_cap_fixed_point(
        eligible_inventory, pre_caps, cap_pct
    )
    cap_dollar_limit = cap_pct * solved_bb
    finals = [min(a, cap_dollar_limit) for a in pre_caps]
    final_eligible_receivables = sum(finals)
    final_borrowing_base = eligible_inventory + final_eligible_receivables
    solver_residual = abs(final_borrowing_base - solved_bb)

    receivable_items: list[ReceivableValuation] = []
    capped_buyer_names: list[str] = []
    for row, final in zip(rec_rows, finals):
        rec: ReceivableItem = row["rec"]
        cutback = row["pre_cap_eligible"] - final
        is_capped = cutback > 1e-6
        if is_capped:
            capped_buyer_names.append(rec.buyer_name)
        receivable_items.append(
            ReceivableValuation(
                buyer_name=rec.buyer_name,
                commodity=rec.commodity,
                gross_amount=rec.gross_amount,
                days_past_due=rec.days_past_due,
                is_investment_grade=rec.is_investment_grade,
                is_aging_eligible=row["is_aging_eligible"],
                aging_ineligible_amount=row["aging_ineligible_amount"],
                eligible_gross_amount=row["eligible_gross_amount"],
                ar_advance_rate=row["ar_advance_rate"],
                ar_haircut_amount=row["ar_haircut_amount"],
                pre_cap_eligible=row["pre_cap_eligible"],
                cap_dollar_limit=cap_dollar_limit,
                final_eligible=final,
                concentration_cutback=cutback,
                is_capped=is_capped,
                pct_of_bb=final / final_borrowing_base if final_borrowing_base > 0.0 else 0.0,
            )
        )

    gross_receivables = sum(r.gross_amount for r in receivables)
    aging_ineligible_dollars = sum(r["aging_ineligible_amount"] for r in rec_rows)
    ar_advance_haircut_dollars = sum(r["ar_haircut_amount"] for r in rec_rows)
    pre_cap_eligible_receivables = sum(pre_caps)
    concentration_cutback_dollars = pre_cap_eligible_receivables - final_eligible_receivables

    steps: list[tuple[str, float, str]] = [
        ("Gross Inventory", gross_inventory, "absolute"),
        ("Inv. Haircuts", -inventory_haircut_dollars, "relative"),
        ("Eligible Inventory", eligible_inventory, "total"),
        ("Gross AR", gross_receivables, "relative"),
        (f"Aging >{terms.ar_aging_limit_days}d", -aging_ineligible_dollars, "relative"),
        ("AR Haircuts", -ar_advance_haircut_dollars, "relative"),
        ("Conc. Cutback", -concentration_cutback_dollars, "relative"),
        ("Final Borrowing Base", final_borrowing_base, "total"),
    ]
    waterfall: list[WaterfallLine] = []
    running = 0.0
    for label, amount, measure in steps:
        running = amount if measure in ("absolute", "total") else running + amount
        waterfall.append(WaterfallLine(label=label, amount=amount, measure=measure, running_total=running))

    available_headroom = max(0.0, min(terms.facility_limit, final_borrowing_base) - terms.drawn_revolver)
    return BorrowingBaseResult(
        inventory_items=list(inventory_items),
        receivable_items=receivable_items,
        waterfall=waterfall,
        gross_inventory=gross_inventory,
        inventory_haircut_dollars=inventory_haircut_dollars,
        eligible_inventory=eligible_inventory,
        weighted_inv_advance_rate=weighted_inv_advance_rate,
        total_hedged_notional=total_hedged_notional,
        weighted_hedge_ratio=weighted_hedge_ratio,
        gross_receivables=gross_receivables,
        aging_ineligible_dollars=aging_ineligible_dollars,
        ar_advance_haircut_dollars=ar_advance_haircut_dollars,
        pre_cap_eligible_receivables=pre_cap_eligible_receivables,
        concentration_cutback_dollars=concentration_cutback_dollars,
        final_eligible_receivables=final_eligible_receivables,
        concentration_cap_pct=cap_pct,
        cap_dollar_limit=cap_dollar_limit,
        capped_buyer_names=capped_buyer_names,
        solver_iterations=iterations,
        solver_used_closed_form=used_closed_form,
        solver_residual=solver_residual,
        final_borrowing_base=final_borrowing_base,
        facility_limit=terms.facility_limit,
        drawn_revolver=terms.drawn_revolver,
        available_headroom=available_headroom,
        facility_utilization_pct=terms.drawn_revolver / terms.facility_limit,
        collateral_coverage_ratio=(
            final_borrowing_base / terms.drawn_revolver if terms.drawn_revolver > 0.0 else 0.0
        ),
    )


def compare_concentration_cap_tiers(
    inventory_items: Sequence[InventoryValuation],
    receivables: Sequence[ReceivableItem],
    terms: FacilityTerms,
    tiers: Sequence[float] = CAP_TIER_OPTIONS,
) -> list[CapTierComparison]:
    """Re-solve the borrowing base under each single-buyer cap tier."""
    comparisons: list[CapTierComparison] = []
    base_terms = terms.model_dump()
    for tier in tiers:
        tier_terms = FacilityTerms.model_validate({**base_terms, "concentration_cap_pct": float(tier)})
        res = calculate_borrowing_base(inventory_items, receivables, tier_terms)
        comparisons.append(
            CapTierComparison(
                cap_pct=float(tier),
                final_borrowing_base=res.final_borrowing_base,
                cap_dollar_limit=res.cap_dollar_limit,
                total_cutback_dollars=res.concentration_cutback_dollars,
                capped_buyer_count=len(res.capped_buyer_names),
                capped_buyer_names=res.capped_buyer_names,
                available_headroom=res.available_headroom,
                solver_iterations=res.solver_iterations,
            )
        )
    return comparisons


def receivable_valuations_to_dataframe(items: Sequence[ReceivableValuation]) -> pd.DataFrame:
    """Flatten receivable valuations into a DataFrame (one row per buyer)."""
    return pd.DataFrame([item.model_dump() for item in items])
