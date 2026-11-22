"""Margin-trap stress engine: regimes, IM boundary, breakevens and certificate export."""

from __future__ import annotations

import csv
import io
import json

import numpy as np
import pytest

from core.schemas import FacilityTerms, LiquidityRegime
from core.stress_engine import (
    build_bb_certificate_csv,
    build_bb_certificate_json,
    classify_liquidity_regime,
    run_stress_scenario,
)

M = 0.01e6


def _run(positions, receivables, snapshot, terms, shock=0.0, per=None):
    return run_stress_scenario(positions, receivables, snapshot, terms, shock_pct=shock, per_commodity_shocks=per)


def test_sweep_grid_is_exact(sweep):
    shocks = [p.shock_pct for p in sweep.points]
    assert len(shocks) == 41
    assert shocks[0] == -0.20 and shocks[-1] == 0.20
    assert 0.10 in shocks and -0.10 in shocks and 0.0 in shocks


def test_sweep_regime_counts(sweep):
    assert sweep.regime_counts == {
        "HEALTHY": 7,
        "MARGIN_TRAP_WARNING": 11,
        "BB_DEFICIENCY_WARNING": 15,
        "CRITICAL_SQUEEZE": 8,
    }
    assert {p.regime for p in sweep.points} == set(LiquidityRegime)


def test_sweep_dataframe(sweep):
    df = sweep.to_dataframe()
    assert len(df) == 41
    assert df["net_liquidity_buffer"].dtype.kind == "f"
    assert df.loc[df["shock_pct"] == 0.14, "regime"].item() == "CRITICAL_SQUEEZE"


def test_baseline_healthy(positions, receivables, snapshot, terms):
    r = _run(positions, receivables, snapshot, terms, 0.0)
    assert r.shocked_headroom == pytest.approx(5.12e6, abs=M)
    assert r.net_liquidity_buffer == pytest.approx(9.62e6, abs=M)
    assert r.regime == LiquidityRegime.HEALTHY
    assert r.im_addon_outflow == 0.0


def test_plus_one_pct_cap_multiplier(positions, receivables, snapshot, terms):
    r = _run(positions, receivables, snapshot, terms, 0.01)
    assert r.shocked_bb == pytest.approx(115.06e6, abs=M)
    assert r.shocked_headroom == pytest.approx(6.00e6, abs=M)
    assert r.headroom_change == pytest.approx(0.88e6, abs=M)
    assert r.vm_outflow == pytest.approx(0.77e6, abs=M)
    assert r.net_liquidity_buffer == pytest.approx(9.73e6, abs=M)
    assert r.regime == LiquidityRegime.HEALTHY
    assert r.bb_change > 0.0


def test_im_stepup_boundary_symmetry(positions, receivables, snapshot, terms):
    up = _run(positions, receivables, snapshot, terms, float(np.round(np.linspace(-0.2, 0.2, 41), 6)[30]))
    down = _run(positions, receivables, snapshot, terms, -0.10)
    assert up.shock_pct == 0.10
    assert up.im_addon_outflow == pytest.approx(0.96e6, abs=M)
    assert down.im_addon_outflow == pytest.approx(0.96e6, abs=M)
    assert up.shocked_bb == pytest.approx(123.21e6, abs=M)
    assert up.vm_outflow == pytest.approx(7.68e6, abs=M)
    assert up.net_liquidity_buffer == pytest.approx(1.86e6, abs=M)
    assert up.regime == LiquidityRegime.MARGIN_TRAP_WARNING
    assert down.vm_inflow == pytest.approx(up.vm_outflow, rel=1e-12)
    raw_float = float(np.linspace(-0.20, 0.20, 41)[30])
    just_below = _run(positions, receivables, snapshot, terms, raw_float)
    assert just_below.im_addon_outflow == pytest.approx(0.96e6, abs=M)
    nine = _run(positions, receivables, snapshot, terms, 0.09)
    assert nine.im_addon_outflow == 0.0


def test_plus_14_critical_squeeze(positions, receivables, snapshot, terms):
    r = _run(positions, receivables, snapshot, terms, 0.14)
    assert r.shocked_bb == pytest.approx(126.30e6, abs=M)
    assert r.shocked_headroom == pytest.approx(6.00e6, abs=M)
    assert r.vm_outflow == pytest.approx(10.76e6, abs=M)
    assert r.im_addon_outflow == pytest.approx(0.96e6, abs=M)
    assert r.net_liquidity_buffer == pytest.approx(-1.22e6, abs=M)
    assert r.regime == LiquidityRegime.CRITICAL_SQUEEZE


def test_plus_20_critical_squeeze(positions, receivables, snapshot, terms):
    r = _run(positions, receivables, snapshot, terms, 0.20)
    assert r.shocked_bb == pytest.approx(130.93e6, abs=M)
    assert r.vm_outflow == pytest.approx(15.37e6, abs=M)
    assert r.net_liquidity_buffer == pytest.approx(-5.83e6, abs=M)
    assert r.regime == LiquidityRegime.CRITICAL_SQUEEZE


def test_minus_20_bb_deficiency(positions, receivables, snapshot, terms):
    r = _run(positions, receivables, snapshot, terms, -0.20)
    assert r.shocked_bb == pytest.approx(95.35e6, abs=M)
    assert r.shocked_headroom == 0.0
    assert r.vm_inflow == pytest.approx(15.37e6, abs=M)
    assert r.vm_outflow == 0.0
    assert r.im_addon_outflow == pytest.approx(0.96e6, abs=M)
    assert r.bb_deficiency_cure == pytest.approx(13.65e6, abs=M)
    assert r.net_liquidity_buffer == pytest.approx(5.26e6, abs=M)
    assert r.regime == LiquidityRegime.BB_DEFICIENCY_WARNING


def test_minus_20_cure_at_full_draw(positions, receivables, snapshot):
    terms = FacilityTerms.model_validate({"drawn_revolver": 115_000_000.0})
    r = _run(positions, receivables, snapshot, terms, -0.20)
    assert r.bb_deficiency_cure == pytest.approx(19.65e6, abs=M)


def test_breakevens(sweep):
    assert sweep.upside_squeeze_breakeven_pct == pytest.approx(0.1241, abs=0.0005)
    assert sweep.downside_deficiency_breakeven_pct == pytest.approx(-0.0545, abs=0.0005)


def test_breakeven_points_are_zero_crossings(positions, receivables, snapshot, terms, sweep):
    up = _run(positions, receivables, snapshot, terms, sweep.upside_squeeze_breakeven_pct)
    down = _run(positions, receivables, snapshot, terms, sweep.downside_deficiency_breakeven_pct)
    assert abs(up.net_liquidity_buffer) < 0.15e6
    assert down.shocked_bb == pytest.approx(terms.drawn_revolver, abs=0.05e6)


def test_worst_point(sweep):
    assert sweep.worst_net_buffer_point.shock_pct == 0.20
    assert sweep.worst_net_buffer_point.net_liquidity_buffer == min(p.net_liquidity_buffer for p in sweep.points)


def test_mixed_per_commodity_shocks(positions, receivables, snapshot, terms):
    r = _run(positions, receivables, snapshot, terms, 0.0, {"GC=F": 0.20, "HG=F": 0.0, "LBR=F": 0.0, "CC=F": -0.20})
    gold_hedged = 9_000.0 * 2710.0
    cocoa_hedged = 2_610.0 * 8150.0
    expected_mtm = -(gold_hedged * 0.20 + cocoa_hedged * -0.20)
    assert r.futures_mtm_pnl == pytest.approx(expected_mtm, rel=1e-12)
    assert r.vm_outflow == pytest.approx(max(0.0, -expected_mtm))
    assert r.vm_inflow == pytest.approx(max(0.0, expected_mtm))
    assert r.im_addon_outflow == pytest.approx(0.96e6, abs=M)
    small = _run(positions, receivables, snapshot, terms, 0.0, {"GC=F": 0.05, "HG=F": -0.05, "LBR=F": 0.0, "CC=F": 0.0})
    assert small.im_addon_outflow == 0.0


def test_maxed_out_facility_floors_headroom(positions, receivables, snapshot):
    terms = FacilityTerms.model_validate({"facility_limit": 100_000_000.0, "drawn_revolver": 112_000_000.0})
    for shock in (-0.20, 0.0, 0.14):
        r = _run(positions, receivables, snapshot, terms, shock)
        assert r.shocked_headroom == 0.0
        assert r.baseline_headroom == 0.0


def test_classifier_priority():
    assert classify_liquidity_regime(-1.0, 5.0, 0.0, 3.0)[0] == LiquidityRegime.CRITICAL_SQUEEZE
    assert classify_liquidity_regime(1.0, 5.0, 2.0, 3.0)[0] == LiquidityRegime.MARGIN_TRAP_WARNING
    assert classify_liquidity_regime(1.0, 1.0, 2.0, 3.0)[0] == LiquidityRegime.BB_DEFICIENCY_WARNING
    assert classify_liquidity_regime(1.0, 0.0, 0.0, 0.0)[0] == LiquidityRegime.HEALTHY


def test_certificate_exports(market, positions, receivables, terms, baseline_bb):
    stress = _run(positions, receivables, market.snapshot, terms, 0.14)
    csv_text = build_bb_certificate_csv(baseline_bb, stress, market)
    rows = list(csv.reader(io.StringIO(csv_text)))
    assert rows[0] == ["section", "item", "value"]
    lookup = {(r[0], r[1]): r[2] for r in rows[1:]}
    assert float(lookup[("borrowing_base", "final_borrowing_base")]) == pytest.approx(114.12e6, abs=M)
    assert lookup[("active_stress", "regime")] == "CRITICAL_SQUEEZE"
    payload = json.loads(build_bb_certificate_json(baseline_bb, stress, market))
    assert payload["certificate"]["valuation_date"] == "2026-11-20"
    assert payload["active_stress"]["net_liquidity_buffer"] == pytest.approx(-1.22e6, abs=M)
    assert len(payload["receivables"]) == 6
