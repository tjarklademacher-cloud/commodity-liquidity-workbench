"""AppTest coverage for the advanced model UI, scenario comparison, saved configurations and reset."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import ui.advanced_views as av
from core.advanced_scenario import default_advanced_settings
from core.advanced_schemas import CashflowSettings, ScenarioSpec
from core.scenario_comparison import ScenarioState, WorkbenchConfig, dump_config
from core.schemas import (
    DEFAULT_FACILITY_TERMS,
    DEFAULT_INVENTORY_POSITIONS,
    DEFAULT_RECEIVABLES,
    REQUIRED_TICKERS,
    LiquidityRegime,
)

ROOT = Path(__file__).resolve().parent.parent
TABS = ["Overview", "Borrowing base", "Inventory", "Receivables", "Stress sweep", "Scenario heatmap",
        "Monte Carlo LaR", "Hedging", "Funding plan", "Replays"]


@pytest.fixture()
def app() -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    return at


def _text(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


def test_legacy_mode_is_default_and_unchanged(app):
    assert not app.exception
    assert app.session_state["model_mode"] == "legacy"
    assert [t.label for t in app.tabs] == TABS
    text = _text(app)
    assert "-$1.79M" in text and "LaR 95% · 5-day" in text
    assert app.session_state["active_regime"] == LiquidityRegime.CRITICAL_SQUEEZE


def test_switch_to_advanced_mode(app):
    app.session_state["model_mode"] = "advanced"
    app.run()
    assert not app.exception
    assert [t.label for t in app.tabs] == TABS
    text = _text(app)
    # Default advanced settings at +14%: per-commodity IM with the trigger rule (see PROJECT_CONTEXT.md).
    assert "-$2.61M" in text
    assert "First unfunded day" in text and "Day 3" in text
    assert "Margin dashboard, active scenario (advanced model)" in text
    assert app.session_state["active_regime"] == LiquidityRegime.CRITICAL_SQUEEZE


def test_neutral_advanced_settings_reproduce_legacy(app):
    app.session_state["model_mode"] = "advanced"
    app.session_state["contract_policy"] = "continuous"
    app.session_state["im_mode"] = "legacy_stepup"
    app.run()
    assert not app.exception
    assert "-$1.79M" in _text(app)


def test_advanced_settings_change_the_result(app):
    app.session_state["model_mode"] = "advanced"
    app.session_state["margin_line_m"] = 20.0
    app.run()
    assert not app.exception
    assert app.session_state["active_regime"] != LiquidityRegime.CRITICAL_SQUEEZE
    app.session_state["ml_cut_pct"] = 100
    app.run()
    assert "-$2.61M" in _text(app)


def test_margin_preset_sets_shocks_and_multipliers(app):
    app.session_state["model_mode"] = "advanced"
    app.run()
    app.selectbox(key="margin_preset").set_value("Cocoa squeeze (+25% cocoa, IM x1.6)").run()
    app.button(key="apply_margin_preset_btn").click().run()
    assert not app.exception
    assert app.session_state["override_enabled"] is True
    assert app.session_state["shock_CC=F"] == 25
    df = app.session_state["margin_df"]
    assert float(df.loc[df["Ticker"] == "CC=F", "Stress IM multiplier"].iloc[0]) == 1.6


def test_basis_preset_and_factor_feed(app):
    app.session_state["model_mode"] = "advanced"
    app.session_state["feed_mode"] = "Factor Sample (Synthetic)"
    app.run()
    app.selectbox(key="basis_preset").set_value("D. Adverse liquidation (8% discount to futures)").run()
    app.button(key="apply_basis_preset_btn").click().run()
    assert not app.exception
    df = app.session_state["basis_df"]
    assert df["Liquidation discount (%)"].tolist() == [8.0] * 6
    assert "Synthetic factor sample" in _text(app)


def test_save_active_as_alternative(app):
    assert "compare_alt" not in app.session_state
    app.button(key="save_alt_btn").click().run()
    assert not app.exception
    alt = ScenarioSpec.model_validate_json(app.session_state["compare_alt"])
    assert alt.futures_shocks == {t: 0.14 for t in REQUIRED_TICKERS}


def test_reset_restores_all_advanced_state(app):
    app.session_state["model_mode"] = "advanced"
    app.session_state["ml_cut_pct"] = 50
    app.session_state["cf_horizon"] = 30
    app.run()
    app.button(key="save_alt_btn").click().run()
    app.button(key="reset_btn").click().run()
    assert not app.exception
    ss = app.session_state
    assert ss["model_mode"] == "legacy" and ss["ml_cut_pct"] == 0 and ss["cf_horizon"] == 20
    assert "compare_alt" not in ss
    assert ss["margin_df"].equals(av.default_margin_df())


# ---- Configuration restore (callback exercised against a plain session-state stand-in) -----------------------

class _FakeState(dict):
    def __getattr__(self, k):
        return self[k]

    def __setattr__(self, k, v):
        self[k] = v


def _config(**overrides) -> str:
    s = default_advanced_settings()
    cfg = WorkbenchConfig(
        model_mode="advanced", positions=list(DEFAULT_INVENTORY_POSITIONS), receivables=list(DEFAULT_RECEIVABLES),
        terms=DEFAULT_FACILITY_TERMS.model_copy(update={"unrestricted_cash": 9.5e6}), margin=s.margin, basis=s.basis,
        scenario_extras=ScenarioSpec(im_multipliers={"CC=F": 1.5}, margin_line_cut_pct=0.25),
        cashflow=CashflowSettings(horizon_days=15),
        scenario=ScenarioState(parallel_shock_pct=10),
    )
    data = json.loads(dump_config(cfg))
    data.update(overrides)
    return json.dumps(data)


def _run_callback(monkeypatch, raw: bytes) -> _FakeState:
    ss = _FakeState(editor_nonce=0, config_upload=SimpleNamespace(getvalue=lambda: raw))
    monkeypatch.setattr(av, "st", SimpleNamespace(session_state=ss))
    av.apply_uploaded_config()
    return ss


def test_config_restore_valid(monkeypatch):
    ss = _run_callback(monkeypatch, _config().encode())
    assert "config_error" not in ss, ss.get("config_error")
    assert ss["model_mode"] == "advanced" and ss["unrestricted_cash_m"] == 9.5 and ss["parallel_shock"] == 10
    assert ss["ml_cut_pct"] == 25 and ss["cf_horizon"] == 15 and ss.editor_nonce == 1
    df = ss["margin_df"]
    assert float(df.loc[df["Ticker"] == "CC=F", "Stress IM multiplier"].iloc[0]) == 1.5


@pytest.mark.parametrize("raw, fragment", [
    (b"not json", "not valid JSON"),
    (json.dumps({"kind": "other"}).encode(), "not a Commodity Liquidity Workbench"),
    (_config(version=99).encode(), "Unsupported configuration version"),
    (_config(positions=[]).encode(), "Invalid value at positions"),
    (b"\xff\xfe", "not UTF-8"),
])
def test_config_restore_rejects_bad_files(monkeypatch, raw, fragment):
    ss = _run_callback(monkeypatch, raw)
    assert fragment in ss["config_error"]
    assert "model_mode" not in ss and ss.editor_nonce == 0


def test_config_restore_rejects_unsupported_cap_tier(monkeypatch):
    data = json.loads(_config())
    data["terms"]["concentration_cap_pct"] = 0.33
    ss = _run_callback(monkeypatch, json.dumps(data).encode())
    assert "not one of the supported tiers" in ss["config_error"] and "model_mode" not in ss


def test_intro_card_shows_once_and_dismisses(app):
    assert "Start here" in _text(app)
    app.button(key="intro_dismiss_btn").click().run()
    assert not app.exception
    assert app.session_state["intro_dismissed"] is True
    assert "Start here" not in _text(app)
    assert "-$1.79M" in _text(app)


def test_shared_link_round_trip_applies_scenario():
    source = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    source.run()
    source.session_state["parallel_shock"] = -7
    source.session_state["unrestricted_cash_m"] = 9.5
    source.run()
    share_box = next(c.value for c in source.code if "?s=" in c.value)
    token = share_box.split("?s=", 1)[1]

    target = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    target.query_params["s"] = token
    target.run()
    assert not target.exception
    assert target.session_state["parallel_shock"] == -7
    assert target.session_state["unrestricted_cash_m"] == 9.5
    assert "shared link" in target.session_state["config_message"]
    target.run()
    assert target.session_state["parallel_shock"] == -7


def test_damaged_shared_link_changes_nothing():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.query_params["s"] = "not-a-real-token"
    at.run()
    assert not at.exception
    assert at.session_state["parallel_shock"] == 14
    assert at.session_state["config_error"].startswith("Shared link not applied")
