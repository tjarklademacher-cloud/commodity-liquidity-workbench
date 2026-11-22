"""Borrowing base: AR eligibility, concentration cap solver, waterfall and cap tiers."""

from __future__ import annotations

import numpy as np
import pytest

from core.borrowing_base import (
    _closed_form_cap_solution,
    calculate_borrowing_base,
    compare_concentration_cap_tiers,
    receivable_valuations_to_dataframe,
    solve_concentration_cap_fixed_point,
)
from core.schemas import FacilityTerms, ReceivableItem

M = 0.01e6


def _fixed_point_residual(bb_inv: float, amounts, cap: float, bb: float) -> float:
    return abs(bb - (bb_inv + sum(min(a, cap * bb) for a in amounts)))


def test_aging_exclusion(baseline_bb):
    art = next(r for r in baseline_bb.receivable_items if r.buyer_name == "Artisanal Roasters Co.")
    assert not art.is_aging_eligible
    assert art.final_eligible == 0.0
    assert art.pre_cap_eligible == 0.0
    assert art.aging_ineligible_amount == pytest.approx(2.4e6)


def test_ig_and_non_ig_advance_rates(baseline_bb):
    rates = {r.buyer_name: r.ar_advance_rate for r in baseline_bb.receivable_items}
    assert rates["Barry Callebaut AG"] == 0.85
    assert rates["Olam Food Ingredients"] == 0.85
    assert rates["West Coast Lumber Supply"] == 0.75
    assert rates["Midwest Copper Fabricators"] == 0.75


def test_ar_aggregates(baseline_bb):
    assert baseline_bb.gross_receivables == pytest.approx(66.70e6, abs=M)
    assert baseline_bb.aging_ineligible_dollars == pytest.approx(2.40e6, abs=M)
    assert baseline_bb.ar_advance_haircut_dollars == pytest.approx(10.575e6, abs=M)
    assert baseline_bb.pre_cap_eligible_receivables == pytest.approx(53.725e6, abs=M)


def test_default_cap_checkpoint(baseline_bb):
    assert baseline_bb.final_borrowing_base == pytest.approx(114.12e6, abs=M)
    assert baseline_bb.cap_dollar_limit == pytest.approx(17.12e6, abs=M)
    assert baseline_bb.concentration_cutback_dollars == pytest.approx(5.29e6, abs=M)
    assert baseline_bb.capped_buyer_names == ["Barry Callebaut AG", "Cargill Cocoa & Chocolate"]
    assert baseline_bb.solver_iterations == 2
    assert not baseline_bb.solver_used_closed_form
    assert baseline_bb.solver_residual < 0.01
    assert baseline_bb.available_headroom == pytest.approx(5.12e6, abs=M)


@pytest.mark.parametrize(
    ("cap", "bb", "limit", "cutback", "capped"),
    [
        (0.15, 114.12e6, 17.12e6, 5.29e6, ["Barry Callebaut AG", "Cargill Cocoa & Chocolate"]),
        (0.175, 118.98e6, 20.82e6, 0.43e6, ["Barry Callebaut AG"]),
        (0.20, 119.41e6, 23.88e6, 0.0, []),
    ],
)
def test_cap_tiers(baseline_items, receivables, terms, cap, bb, limit, cutback, capped):
    tiers = {round(c.cap_pct, 4): c for c in compare_concentration_cap_tiers(baseline_items, receivables, terms)}
    tier = tiers[round(cap, 4)]
    assert tier.final_borrowing_base == pytest.approx(bb, abs=M)
    assert tier.cap_dollar_limit == pytest.approx(limit, abs=M)
    assert tier.total_cutback_dollars == pytest.approx(cutback, abs=M)
    assert tier.capped_buyer_names == capped
    assert tier.capped_buyer_count == len(capped)


def test_waterfall_identity(baseline_bb):
    wf = baseline_bb.waterfall
    assert [w.label for w in wf] == [
        "Gross Inventory", "Inv. Haircuts", "Eligible Inventory", "Gross AR",
        "Aging >90d", "AR Haircuts", "Conc. Cutback", "Final Borrowing Base",
    ]
    assert [w.measure for w in wf] == [
        "absolute", "relative", "total", "relative", "relative", "relative", "relative", "total",
    ]
    assert wf[0].amount + wf[1].amount == pytest.approx(wf[2].amount, abs=1e-6)
    bridged = wf[2].amount + sum(w.amount for w in wf[3:7])
    assert bridged == pytest.approx(baseline_bb.final_borrowing_base, abs=1e-6)
    assert wf[6].running_total == pytest.approx(wf[7].amount, abs=1e-6)


def test_receivable_dataframe(baseline_bb):
    df = receivable_valuations_to_dataframe(baseline_bb.receivable_items)
    assert len(df) == 6
    assert df["final_eligible"].max() <= baseline_bb.cap_dollar_limit + 1e-6


def test_zero_inventory_single_buyer_collapses_to_zero():
    bb, iters, _ = solve_concentration_cap_fixed_point(0.0, [10_000_000.0], 0.15)
    assert bb == pytest.approx(0.0, abs=1e-9)
    assert iters >= 1
    assert _closed_form_cap_solution(0.0, [10_000_000.0], 0.15) == 0.0


def test_k_times_cap_equals_one_closed_form():
    amounts = [100.0] * 6
    bb_cf = _closed_form_cap_solution(1.0, amounts, 0.20)
    assert bb_cf == pytest.approx(601.0)
    assert _fixed_point_residual(1.0, amounts, 0.20, bb_cf) < 1e-9
    assert _closed_form_cap_solution(0.0, [50.0] * 4, 0.25) == pytest.approx(200.0)
    bb, _, used_closed_form = solve_concentration_cap_fixed_point(1.0, amounts, 0.20)
    assert not used_closed_form
    assert bb == pytest.approx(bb_cf)


def test_iteration_exhaustion_falls_back_to_closed_form():
    pre = [21.25e6, 18.275e6, 7.225e6, 3.9e6, 3.075e6]
    bb, iters, used_closed_form = solve_concentration_cap_fixed_point(65.68e6, pre, 0.15, max_iterations=1)
    assert used_closed_form
    assert iters == 1
    assert bb == pytest.approx((65.68e6 + 7.225e6 + 3.9e6 + 3.075e6) / 0.70, rel=1e-12)


def test_exactly_at_cap_boundary():
    amounts = [20.0] * 5
    bb, _, _ = solve_concentration_cap_fixed_point(0.0, amounts, 0.20)
    assert bb == pytest.approx(100.0)
    assert _fixed_point_residual(0.0, amounts, 0.20, bb) < 1e-9


def test_empty_receivables(baseline_items, terms):
    res = calculate_borrowing_base(baseline_items, [], terms)
    assert res.final_borrowing_base == pytest.approx(res.eligible_inventory)
    assert res.capped_buyer_names == []
    assert res.gross_receivables == 0.0


def test_huge_single_buyer(baseline_items, terms):
    whale = [ReceivableItem(buyer_name="Whale", commodity="Cocoa", gross_amount=1e10,
                            days_past_due=5, is_investment_grade=True)]
    res = calculate_borrowing_base(baseline_items, whale, terms)
    expected = res.eligible_inventory / (1.0 - terms.concentration_cap_pct)
    assert res.final_borrowing_base == pytest.approx(expected, rel=1e-12)
    assert res.capped_buyer_names == ["Whale"]
    assert res.receivable_items[0].final_eligible == pytest.approx(0.15 * expected)


def test_all_aged_out(baseline_items, terms):
    aged = [ReceivableItem(buyer_name="Old", commodity="Copper", gross_amount=5e6,
                           days_past_due=91, is_investment_grade=True)]
    res = calculate_borrowing_base(baseline_items, aged, terms)
    assert res.final_eligible_receivables == 0.0
    assert res.aging_ineligible_dollars == pytest.approx(5e6)


def test_closed_form_matches_iteration_on_default():
    pre = [21.25e6, 18.275e6, 7.225e6, 3.9e6, 3.075e6, 0.0]
    bb_iter, _, _ = solve_concentration_cap_fixed_point(65.68e6, pre, 0.15)
    bb_cf = _closed_form_cap_solution(65.68e6, pre, 0.15)
    assert bb_iter == pytest.approx(bb_cf, rel=1e-12)


def test_randomized_fuzz_1000_portfolios():
    rng = np.random.default_rng(2026)
    for _ in range(1000):
        n = int(rng.integers(0, 15))
        bb_inv = float(rng.choice([0.0, rng.uniform(0.0, 2e8)]))
        amounts = (rng.lognormal(mean=15.0, sigma=1.5, size=n) * rng.choice([0.0, 1.0], size=n, p=[0.1, 0.9])).tolist()
        cap = float(rng.choice([0.05, 0.10, 0.15, 0.175, 0.20, 0.25, 0.5, 1.0]))
        bb, iters, used_cf = solve_concentration_cap_fixed_point(bb_inv, amounts, cap)
        assert bb >= 0.0
        assert 1 <= iters <= 25
        limit = cap * bb
        finals = [min(a, limit) for a in amounts]
        assert all(f <= limit + 1e-6 for f in finals)
        assert _fixed_point_residual(bb_inv, amounts, cap, bb) < 1e-4
        if not used_cf and bb > 0.0:
            assert bb == pytest.approx(_closed_form_cap_solution(bb_inv, amounts, cap), rel=1e-9, abs=1e-6)


def test_terms_validation_through_model_validate(baseline_items, receivables):
    terms = FacilityTerms.model_validate({"concentration_cap_pct": 0.20})
    res = calculate_borrowing_base(baseline_items, receivables, terms)
    assert res.concentration_cutback_dollars == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError):
        FacilityTerms.model_validate({"concentration_cap_pct": 0.0})
