"""Architecture boundary guards, deprecation scans, chart serialization and Streamlit AppTest E2E."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.schemas import COMMODITY_SPECS, LiquidityRegime
from core.stress_engine import run_stress_scenario
from core.scenario_analytics import run_historical_replay, run_liquidity_monte_carlo, run_shock_grid
from ui.charts import (
    build_active_shock_sources_uses_chart,
    build_borrowing_base_waterfall,
    build_collateral_mix_donut,
    build_correlation_heatmap,
    build_facility_gauge,
    build_haircut_decomposition_chart,
    build_mc_distribution_chart,
    build_receivables_concentration_chart,
    build_regime_ribbon,
    build_replay_chart,
    build_shock_heatmap,
    build_stress_sweep_chart,
    build_volatility_history_chart,
)

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_CORE_IMPORTS = ("streamlit", "plotly")


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", sorted((ROOT / "core").glob("*.py")), ids=lambda p: p.name)
def test_core_has_no_ui_imports(path):
    assert not (_imported_roots(path) & set(FORBIDDEN_CORE_IMPORTS)), f"{path.name} imports a UI library"


def test_core_import_is_transitively_ui_free():
    code = (
        "import sys, core, core.schemas, core.data_feed, core.haircuts, core.borrowing_base, core.stress_engine; "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('streamlit', 'plotly')); "
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def test_no_use_container_width_anywhere():
    files = [ROOT / "app.py", *sorted((ROOT / "ui").glob("*.py"))]
    offenders = [f.name for f in files if "use_container_width" in f.read_text(encoding="utf-8")]
    assert offenders == []


def test_user_facing_copy_has_no_em_dashes():
    files = [ROOT / "app.py", ROOT / "README.md", *sorted((ROOT / "ui").glob("*.py"))]
    offenders = [f.name for f in files if "\u2014" in f.read_text(encoding="utf-8")]
    assert offenders == []


def test_repo_has_no_local_paths_or_secrets():
    files = [p for p in ROOT.rglob("*") if p.is_file() and ".venv" not in p.parts and ".git" not in p.parts
             and "__pycache__" not in p.parts and p.suffix in {".py", ".md", ".toml", ".txt", ".csv"}]
    for f in files:
        text = f.read_text(encoding="utf-8")
        for needle in ("/" + "Users/", "scratch" + "-workspaces", "Application" + " Support"):
            assert needle not in text, f"{needle} found in {f.name}"


def test_no_deprecated_pandas_or_plotly_patterns():
    files = [ROOT / "app.py", *sorted((ROOT / "ui").glob("*.py")), *sorted((ROOT / "core").glob("*.py"))]
    banned = ("applymap(", "inplace=True", "titlefont", "titleside", "persist_state=")
    for f in files:
        text = f.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{token} found in {f.name}"


def test_all_six_charts_serialize(market, baseline_bb, sweep, positions, receivables, terms):
    stress = run_stress_scenario(positions, receivables, market.snapshot, terms, shock_pct=0.14)
    figures = {
        "waterfall": build_borrowing_base_waterfall(baseline_bb),
        "haircuts": build_haircut_decomposition_chart(baseline_bb.inventory_items),
        "volatility": build_volatility_history_chart(market.vol_df, COMMODITY_SPECS),
        "receivables": build_receivables_concentration_chart(baseline_bb),
        "sweep": build_stress_sweep_chart(sweep, 0.14),
        "sweep_custom": build_stress_sweep_chart(sweep, 0.14, is_custom_per_commodity=True),
        "sources_uses": build_active_shock_sources_uses_chart(stress, terms),
    }
    for name, fig in figures.items():
        payload = json.loads(fig.to_json())
        assert payload["data"], f"{name} has no traces"
    wf = figures["waterfall"].data[0]
    assert list(wf.measure) == ["absolute", "relative", "total", "relative", "relative", "relative", "relative", "total"]
    assert wf.y[2] is None and wf.y[7] is None
    assert wf.text[-1] == "$114.12M"
    for trace in figures["haircuts"].data:
        assert trace.orientation == "h"
    stacked = [sum(t.x[i] for t in figures["haircuts"].data) for i in range(4)]
    assert stacked == pytest.approx([100.0] * 4, abs=1e-9)
    sweep_fig = figures["sweep"]
    assert "yaxis2" in sweep_fig.layout and sweep_fig.layout.yaxis2.overlaying is None
    regime_bands = [sh for sh in sweep_fig.layout.shapes if sh.type == "rect"]
    assert len(regime_bands) == 4


def test_extended_charts_serialize(market, baseline_bb, sweep, positions, receivables, terms):
    grid = run_shock_grid(positions, receivables, market.snapshot, terms)
    mc = run_liquidity_monte_carlo(positions, receivables, market.snapshot, terms, market.prices_df, n_sims=1000)
    replay = run_historical_replay(positions, receivables, market.prices_df, market.vol_df, terms)
    figures = [
        build_facility_gauge(baseline_bb),
        build_collateral_mix_donut(baseline_bb),
        build_regime_ribbon(sweep, 0.14),
        build_regime_ribbon(sweep, None),
        build_shock_heatmap(grid, (0.14, 0.14)),
        build_shock_heatmap(grid, None),
        build_mc_distribution_chart(mc),
        build_correlation_heatmap(mc, COMMODITY_SPECS),
        build_replay_chart(replay, terms),
    ]
    for fig in figures:
        assert json.loads(fig.to_json())["data"]
    assert any(sh.type == "rect" for sh in figures[-1].layout.shapes)
    donut = figures[1].data[0]
    assert sum(donut.values) == pytest.approx(baseline_bb.final_borrowing_base / 1e6, rel=1e-12)


@pytest.fixture()
def app() -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    return at


def _banner_text(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown if "regime-banner" in m.value)


def test_app_initial_load_clean(app):
    assert len(app.exception) == 0
    assert len(app.error) == 0
    assert len(app.warning) == 0
    assert any(e.label == "Methodology & Commercial Use Case" for e in app.expander)
    assert "Bundled sample data" in " ".join(m.value for m in app.markdown)
    assert len(app.tabs) == 8
    assert "kpi-grid" in " ".join(m.value for m in app.markdown)


def test_app_default_is_critical_squeeze(app):
    assert app.session_state["active_regime"] == LiquidityRegime.CRITICAL_SQUEEZE
    assert "CRITICAL_SQUEEZE" in _banner_text(app)
    assert "-$1.22M" in _banner_text(app)


def test_app_zero_shock_is_healthy(app):
    app.slider(key="parallel_shock").set_value(0).run()
    assert len(app.exception) == 0
    assert app.session_state["active_regime"] == LiquidityRegime.HEALTHY
    assert "HEALTHY" in _banner_text(app)


def test_app_minus_20_is_bb_deficiency(app):
    app.slider(key="parallel_shock").set_value(-20).run()
    assert len(app.exception) == 0
    assert app.session_state["active_regime"] == LiquidityRegime.BB_DEFICIENCY_WARNING
    assert "BB_DEFICIENCY_WARNING" in _banner_text(app)


def test_app_cap_tier_and_preset(app):
    app.radio(key="cap_tier").set_value(20.0).run()
    assert len(app.exception) == 0
    assert "Cap tier: 20.0%" in " ".join(m.value for m in app.markdown)
    app.selectbox(key="preset_name").set_value("West Africa Cocoa Spike (+30% CC, +5% Others)").run()
    app.button(key="apply_preset_btn").click().run()
    assert len(app.exception) == 0
    assert app.session_state["override_enabled"] is True
    assert app.session_state["shock_CC=F"] == 30
    app.button(key="reset_btn").click().run()
    assert len(app.exception) == 0
    assert app.session_state["active_regime"] == LiquidityRegime.CRITICAL_SQUEEZE


def test_app_inventory_edit_flows_downstream(app):
    nonce = app.session_state["editor_nonce"]
    app.session_state[f"inv_editor_{nonce}"] = {
        "edited_rows": {3: {"Physical Qty": 0.0, "Hedged Qty": 0.0}},
        "added_rows": [],
        "deleted_rows": [],
    }
    app.slider(key="parallel_shock").set_value(0).run()
    assert len(app.exception) == 0
    edited = app.session_state["inventory_edited_df"]
    assert float(edited.loc[3, "Physical Qty"]) == 0.0


def test_app_monte_carlo_controls(app):
    app.session_state["mc_horizon"] = 10
    app.session_state["mc_sims"] = 1000
    app.run()
    assert len(app.exception) == 0
    assert "10-Day LaR (95%)" in " ".join(m.value for m in app.markdown)


def test_app_per_commodity_override_runs(app):
    app.checkbox(key="override_enabled").check().run()
    app.slider(key="shock_CC=F").set_value(30).run()
    assert len(app.exception) == 0
    assert len(app.error) == 0
