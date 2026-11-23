"""Commodity Liquidity Workbench: Streamlit entry point (presentation only; models live in core/)."""

from __future__ import annotations

import json
import os
import time

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from core.advanced_scenario import (
    auto_im_multipliers,
    buffer_sensitivities,
    neutral_settings,
    reconcile_to_legacy,
    run_advanced_scenario,
)
from core.advanced_schemas import AdvancedSettings, ScenarioSpec
from core.borrowing_base import calculate_borrowing_base, compare_concentration_cap_tiers
from core.cashflow_engine import deterministic_paths, ledger_frame, run_cashflow_ledger
from core.market_calibration import percentile_interval
from core.scenario_comparison import compare_scenarios
from core.data_feed import DEFAULT_SAMPLE_CSV_PATH
from core.haircuts import compute_portfolio_inventory
from core.schemas import (
    CAP_TIER_OPTIONS,
    COMMODITY_SPECS,
    DEFAULT_FACILITY_TERMS,
    REQUIRED_TICKERS,
    STRESS_SCENARIO_PRESETS,
    FacilityTerms,
    MarketLoadResult,
)
from core.contingency import build_contingency_plan
from core.crisis_replay import CRISIS_EPISODES
from core.hedge_optimizer import ALL_COMMODITIES
from core.stress_engine import run_stress_scenario
from ui.advanced_views import (
    MODEL_ADVANCED,
    MODEL_LABELS,
    RISK_MEASURE_NOTES,
    SIM_METHOD_LABELS,
    apply_uploaded_config,
    build_config,
    cashflow_mc_kpis,
    clear_alternative,
    commit_pending_edits,
    init_advanced_state,
    ledger_kpis,
    margin_kpis,
    render_basis_settings,
    render_calibration_table,
    render_cashflow_settings,
    render_comparison,
    render_ledger_table,
    render_margin_audit,
    render_margin_settings,
    render_provenance,
    render_simulation_settings,
    render_steps_table,
    reset_advanced_state,
    save_active_as_alternative,
)
from ui.cache import (
    LIVE_TTL_SECONDS,
    cached_backtest,
    cached_calibration,
    cached_cashflow_monte_carlo,
    cached_factor_market,
    cached_fx_rate,
    cached_crisis_replay,
    cached_custom_market,
    cached_hedge_frontier,
    cached_live_market,
    cached_monte_carlo,
    cached_runway,
    cached_sample_market,
    cached_scenario_suite,
    market_data_status,
    model_inputs_json,
)
from ui.charts import (
    build_active_shock_sources_uses_chart,
    build_backtest_chart,
    build_economic_pnl_chart,
    build_ledger_chart,
    build_liquidity_fan_chart,
    build_margin_contribution_chart,
    build_pareto_chart,
    build_probability_chart,
    build_borrowing_base_waterfall,
    build_buffer_profile_chart,
    build_correlation_heatmap,
    build_crisis_replay_chart,
    build_haircut_decomposition_chart,
    build_hedge_frontier_chart,
    build_mc_distribution_chart,
    build_receivables_concentration_chart,
    build_replay_chart,
    build_runway_breach_chart,
    build_runway_fan_chart,
    build_shock_heatmap,
    build_stress_sweep_chart,
    build_volatility_history_chart,
)
from ui.components import (
    certificate_download_buttons,
    covenant_audit_rows,
    default_inventory_df,
    default_receivables_df,
    haircut_methodology_expander,
    inventory_editor,
    methodology_expander,
    provenance_note,
    receivables_editor,
    regime_explanation_text,
    render_action_list,
    render_audit_rows,
    render_trigger_list,
    render_author_panel,
    render_briefing,
    render_header,
    render_intro,
    render_kpi_row,
    render_verdict,
    section_head,
    solver_audit_rows,
    stress_methodology_expander,
)
from ui.report import build_report_html
from ui.theme import (
    CURRENCY_SYMBOLS,
    PLOTLY_CONFIG,
    DisplayCurrency,
    bridge_label,
    fmt_millions,
    fmt_pct,
    fmt_signed_millions,
    inject_custom_css,
    localize_figure,
    money_divisor,
    money_unit,
    set_display_currency,
    regime_label,
    regime_status,
)

FEED_SAMPLE = "Calibrated Sample (Offline)"
FEED_LIVE = "Live Yahoo Finance (yfinance)"
FEED_UPLOAD = "Upload Custom CSV"
FEED_FACTOR = "Factor Sample (Synthetic)"
PRESET_NAMES = list(STRESS_SCENARIO_PRESETS)

WIDGET_DEFAULTS: dict[str, object] = {
    # A hosted copy sets WORKBENCH_DEFAULT_FEED=live so visitors land on live prices;
    # tests and local runs keep the sample.
    "feed_mode": FEED_LIVE if os.environ.get("WORKBENCH_DEFAULT_FEED", "").strip().lower() == "live" else FEED_SAMPLE,
    "winsorize": True,
    "facility_limit_m": DEFAULT_FACILITY_TERMS.facility_limit / 1e6,
    "drawn_revolver_m": DEFAULT_FACILITY_TERMS.drawn_revolver / 1e6,
    "unrestricted_cash_m": DEFAULT_FACILITY_TERMS.unrestricted_cash / 1e6,
    "baseline_im_m": DEFAULT_FACILITY_TERMS.baseline_initial_margin / 1e6,
    "cap_tier": round(DEFAULT_FACILITY_TERMS.concentration_cap_pct * 100, 1),
    "im_trigger_pct": int(round(DEFAULT_FACILITY_TERMS.im_stepup_trigger * 100)),
    "im_addon_pct": int(round(DEFAULT_FACILITY_TERMS.im_stepup_fraction * 100)),
    "parallel_shock": int(round(DEFAULT_FACILITY_TERMS.default_active_shock_pct * 100)),
    "preset_name": PRESET_NAMES[0],
    "override_enabled": False,
    **{f"shock_{t}": int(round(DEFAULT_FACILITY_TERMS.default_active_shock_pct * 100)) for t in REQUIRED_TICKERS},
    "mc_horizon": 5,
    "mc_sims": 5000,
    "hedge_scope": ALL_COMMODITIES,
    "replay_source": "loaded",
    "margin_line_m": DEFAULT_FACILITY_TERMS.margin_line_limit / 1e6,
    "report_ccy": "USD",
    "cfp_target": "active",
}
MC_HORIZONS = [1, 5, 10, 20]
MC_SIM_OPTIONS = [1000, 2500, 5000, 10000]


def init_session_state() -> None:
    ss = st.session_state
    if "editor_nonce" not in ss:
        ss.editor_nonce = 0
    if "inventory_df" not in ss:
        ss["inventory_df"] = default_inventory_df()
    if "receivables_df" not in ss:
        ss["receivables_df"] = default_receivables_df()
    for key, value in WIDGET_DEFAULTS.items():
        if key not in ss:
            ss[key] = value
    init_advanced_state(ss)


def reset_to_defaults() -> None:
    ss = st.session_state
    for key, value in WIDGET_DEFAULTS.items():
        ss[key] = value
    ss["inventory_df"] = default_inventory_df()
    ss["receivables_df"] = default_receivables_df()
    reset_advanced_state(ss)
    ss.editor_nonce += 1


def apply_preset() -> None:
    ss = st.session_state
    commit_pending_edits(ss)
    shocks = STRESS_SCENARIO_PRESETS[ss["preset_name"]]
    values = [int(round(shocks[t] * 100)) for t in REQUIRED_TICKERS]
    if len(set(values)) == 1:
        ss["override_enabled"] = False
        ss["parallel_shock"] = values[0]
    else:
        ss["override_enabled"] = True
    for t, v in zip(REQUIRED_TICKERS, values):
        ss[f"shock_{t}"] = v
    ss.editor_nonce += 1


FEED_LABELS = {FEED_SAMPLE: "Bundled sample", FEED_FACTOR: "Factor sample (synthetic, correlated)",
               FEED_LIVE: "Live (Yahoo Finance)", FEED_UPLOAD: "Upload CSV"}


def side_section(title: str, first: bool = False) -> None:
    css = "side-section first" if first else "side-section"
    st.sidebar.markdown(f'<div class="{css}">{title}</div>', unsafe_allow_html=True)


def sidebar_model_mode() -> str:
    side_section("Model", first=True)
    st.sidebar.radio("Model mode", list(MODEL_LABELS), key="model_mode", horizontal=True,
                     format_func=lambda k: MODEL_LABELS[k],
                     help="Legacy: the audited instantaneous model. Advanced: whole contracts, per-contract "
                          "initial margin, physical basis, a daily cash-flow ledger and calibrated simulation. "
                          "With neutral advanced settings both give the same buffer.")
    return st.session_state["model_mode"]


def sidebar_facility_terms() -> FacilityTerms:
    side_section("Portfolio and facility")
    st.sidebar.number_input("Facility limit (USD M)", min_value=1.0, step=1.0, format="%.1f", key="facility_limit_m")
    st.sidebar.number_input("Drawn balance (USD M)", min_value=0.0, step=1.0, format="%.1f", key="drawn_revolver_m")
    st.sidebar.number_input("Unrestricted cash (USD M)", min_value=0.0, step=0.5, format="%.1f",
                            key="unrestricted_cash_m")
    st.sidebar.number_input("Baseline initial margin (USD M)", min_value=0.0, step=0.1, format="%.1f",
                            key="baseline_im_m")
    st.sidebar.radio(
        "Single-buyer cap (% of borrowing base)", [round(c * 100, 1) for c in CAP_TIER_OPTIONS], key="cap_tier",
        format_func=lambda v: f"{v:.1f}%", horizontal=True,
    )
    st.sidebar.number_input("Margin financing line (USD M)", min_value=0.0, step=1.0, format="%.1f",
                            key="margin_line_m",
                            help="Dedicated bank line that funds exchange margin calls (variation and initial "
                                 "margin). It cannot be used for a borrowing base deficiency cure. 0 = none.")
    st.sidebar.slider("IM step-up trigger (largest move, %)", 5, 20, step=1, key="im_trigger_pct")
    st.sidebar.slider("IM step-up size (% of baseline IM)", 0, 50, step=1, key="im_addon_pct")
    ss = st.session_state
    payload = {
        **ss.get("terms_base", {}),
        "facility_limit": float(ss["facility_limit_m"]) * 1e6,
        "drawn_revolver": float(ss["drawn_revolver_m"]) * 1e6,
        "unrestricted_cash": float(ss["unrestricted_cash_m"]) * 1e6,
        "baseline_initial_margin": float(ss["baseline_im_m"]) * 1e6,
        "concentration_cap_pct": float(ss["cap_tier"]) / 100.0,
        "im_stepup_trigger": float(ss["im_trigger_pct"]) / 100.0,
        "im_stepup_fraction": float(ss["im_addon_pct"]) / 100.0,
        "margin_line_limit": float(ss["margin_line_m"]) * 1e6,
    }
    try:
        return FacilityTerms.model_validate(payload)
    except ValidationError as exc:
        st.sidebar.warning(f"Invalid facility terms ({exc.errors()[0]['msg']}); using calibrated defaults.")
        return DEFAULT_FACILITY_TERMS


@st.fragment(run_every=60)
def _live_refresh_timer() -> None:
    """Reruns the whole app once the live price cache has expired, so an open page picks up fresh prices.

    The check runs every minute but only triggers a full rerun when the last full run is older than the cache TTL;
    the shared cache means Yahoo is queried at most once per TTL however many visitors are connected.
    """
    last = st.session_state.get("_live_full_run_at")
    if last is not None and time.monotonic() - last >= LIVE_TTL_SECONDS:
        st.rerun(scope="app")


def sidebar_market_feed() -> MarketLoadResult:
    side_section("Market data")
    mode = st.sidebar.radio("Price source", [FEED_SAMPLE, FEED_FACTOR, FEED_LIVE, FEED_UPLOAD], key="feed_mode",
                            format_func=lambda v: FEED_LABELS[v])
    winsorize = st.sidebar.checkbox("Winsorize daily returns at 5 sigma", key="winsorize",
                                    help="Caps futures roll gaps before computing 30-day volatility.")
    market: MarketLoadResult
    if mode == FEED_LIVE:
        with st.spinner("Loading front-month futures from Yahoo Finance"):
            market = cached_live_market("1y", winsorize)
        st.session_state["_live_full_run_at"] = time.monotonic()
        _live_refresh_timer()
        st.sidebar.caption(f"Live prices refresh every {LIVE_TTL_SECONDS // 60} minutes while this page is open. "
                           "Yahoo quotes are delayed.")
    elif mode == FEED_UPLOAD:
        uploaded = st.sidebar.file_uploader("Price CSV (Date, GC=F, HG=F, LBR=F, CC=F, KC=F, SB=F)", type=["csv"])
        template = pd.read_csv(DEFAULT_SAMPLE_CSV_PATH).tail(60).to_csv(index=False)
        st.sidebar.download_button(
            "Download CSV template", data=template, file_name="price_template.csv",
            mime="text/csv", width="stretch", key="dl_template",
        )
        sample = cached_sample_market(winsorize)
        if uploaded is None:
            market = sample.model_copy(update={"status_message": "No file uploaded yet; showing the bundled sample."})
        else:
            try:
                market = cached_custom_market(uploaded.getvalue(), winsorize)
            except (ValueError, KeyError, pd.errors.ParserError) as exc:
                market = sample.model_copy(
                    update={
                        "used_fallback": True,
                        "source_label": "Calibrated Sample (Upload Fallback)",
                        "status_message": f"Uploaded CSV rejected ({exc}). Using the bundled sample.",
                    }
                )
    elif mode == FEED_FACTOR:
        market = cached_factor_market(winsorize)
    else:
        market = cached_sample_market(winsorize)
    st.sidebar.caption(market.status_message)
    st.sidebar.selectbox("Reporting currency", ["USD", "EUR", "CHF", "GBP"], key="report_ccy",
                         help="Converts displayed outputs only. Inputs, the model and certificate exports stay in USD.")
    return market


def sidebar_shock_controls() -> tuple[float, dict[str, float] | None]:
    side_section("Stress scenario")
    st.sidebar.slider("Parallel price shock (%)", -20, 20, step=1, key="parallel_shock")
    st.sidebar.selectbox("Scenario preset", PRESET_NAMES, key="preset_name")
    st.sidebar.button("Apply preset", width="stretch", on_click=apply_preset, key="apply_preset_btn")
    override = st.sidebar.checkbox("Set shocks per commodity", key="override_enabled")
    parallel = st.session_state["parallel_shock"] / 100.0
    if not override:
        return parallel, None
    per: dict[str, float] = {}
    for t in REQUIRED_TICKERS:
        st.sidebar.slider(f"{COMMODITY_SPECS[t].name} shock (%)", -30, 30, step=1, key=f"shock_{t}")
        per[t] = st.session_state[f"shock_{t}"] / 100.0
    return parallel, per


def sidebar_simulation() -> tuple[int, int]:
    side_section("Simulation")
    st.sidebar.segmented_control("Monte Carlo horizon (trading days)", MC_HORIZONS, key="mc_horizon",
                                 format_func=lambda d: f"{d}d")
    st.sidebar.select_slider("Monte Carlo paths", MC_SIM_OPTIONS, key="mc_sims", format_func=lambda n: f"{n:,}")
    ss = st.session_state
    return int(ss.get("mc_horizon") or 5), int(ss.get("mc_sims") or 5000)


def sidebar_configuration(config_json: str, file_stub: str) -> None:
    side_section("Configuration")
    st.sidebar.download_button("Download configuration (JSON)", data=config_json,
                               file_name=f"workbench_config_{file_stub}.json", mime="application/json",
                               width="stretch", key="dl_config")
    st.sidebar.file_uploader("Restore a saved configuration", type=["json"], key="config_upload",
                             help="Data only: every field is validated and nothing in the file is executed. "
                                  "Market data is not part of a configuration.")
    st.sidebar.button("Apply configuration", width="stretch", on_click=apply_uploaded_config, key="apply_config_btn")
    ss = st.session_state
    if ss.get("config_error"):
        st.sidebar.error(f"Configuration not applied. {ss['config_error']}")
    elif ss.get("config_message"):
        st.sidebar.success(ss["config_message"])


def sidebar_reset() -> None:
    side_section("Reset")
    st.sidebar.button("Reset all inputs to defaults", width="stretch", on_click=reset_to_defaults, key="reset_btn")


def panel(key: str):
    """White analytical surface (styled via the stable st-key-panel_* class)."""
    return st.container(key=f"panel_{key}")


def chart(fig, key: str, title: str = "", subtitle: str = "") -> None:
    with panel(key):
        if title:
            section_head(title, subtitle)
        st.plotly_chart(localize_figure(fig), width="stretch", theme=None, config=PLOTLY_CONFIG, key=key)


def _money_column(label: str) -> st.column_config.NumberColumn:
    """USD millions with two decimals; the unit sits in the header so negatives read as -1.22."""
    return st.column_config.NumberColumn(f"{label} ({money_unit()})", format="%.2f")


def main() -> None:
    st.set_page_config(
        page_title="Commodity Liquidity Workbench",
        layout="wide",
        initial_sidebar_state="auto",
    )
    inject_custom_css()
    init_session_state()

    model_mode = sidebar_model_mode()
    advanced = model_mode == MODEL_ADVANCED
    terms = sidebar_facility_terms()
    market = sidebar_market_feed()
    ccy_code = st.session_state["report_ccy"]
    fx_rate, fx_source, fx_date = cached_fx_rate(ccy_code)
    currency = DisplayCurrency(code=ccy_code, symbol=CURRENCY_SYMBOLS[ccy_code], usd_per_unit=fx_rate,
                               source=fx_source, as_of=fx_date)
    set_display_currency(currency)
    parallel_shock, per_commodity = sidebar_shock_controls()
    horizon, n_sims = sidebar_simulation()
    is_custom = per_commodity is not None
    ss = st.session_state

    header = st.container()
    (tab_overview, tab_cert, tab_inv, tab_ar, tab_stress, tab_heat, tab_mc, tab_hedge, tab_plan,
     tab_replay) = st.tabs(
        ["Overview", "Borrowing base", "Inventory", "Receivables", "Stress sweep", "Scenario heatmap",
         "Monte Carlo LaR", "Hedging", "Funding plan", "Replays"]
    )
    nonce = ss.editor_nonce

    with tab_inv:
        section_head("Physical inventory and hedge book",
                     "Edit physical and hedged quantities. Calculated columns and every other view update on edit.")
        positions, inv_errors, inv_edited = inventory_editor(ss["inventory_df"], market.snapshot, nonce)
        ss["inventory_edited_df"] = inv_edited
        for msg in inv_errors:
            st.warning(msg)

    with tab_ar:
        section_head("Trade receivables ledger",
                     "Edit, add or delete buyer exposures. Rows that fail validation are excluded and listed below.")
        receivables, ar_errors, ar_edited = receivables_editor(ss["receivables_df"], nonce)
        ss["receivables_edited_df"] = ar_edited
        for msg in ar_errors:
            st.warning(msg)

    in_use = "" if advanced else " Used when the model mode is Advanced."
    with tab_stress:
        with st.expander("Exchange margin assumptions" + ("" if advanced else " (advanced model)")):
            st.caption("Per-commodity initial margin, contract rounding, margin stress and the margin line."
                       + in_use)
            margin_settings, user_mults, margin_errors = render_margin_settings(nonce)
        for msg in margin_errors:
            st.warning(msg)
    with tab_hedge:
        with st.expander("Physical basis assumptions" + ("" if advanced else " (advanced model)")):
            basis_map, basis_shocks, liq_discounts, basis_errors = render_basis_settings(nonce)
            if not advanced:
                st.caption(in_use.strip())
        for msg in basis_errors:
            st.warning(msg)
    with tab_plan:
        with st.expander("Daily cash-flow ledger settings" + ("" if advanced else " (advanced model)")):
            cf_settings, cf_errors = render_cashflow_settings(receivables)
            if not advanced:
                st.caption(in_use.strip())
        for msg in cf_errors:
            st.warning(msg)
    with tab_mc:
        with st.expander("Calibration and simulation settings" + ("" if advanced else " (advanced model)")):
            render_simulation_settings()
            st.caption("Calibration window and method drive the cash-flow simulation and the provenance panel. The "
                       "legacy liquidity-at-risk below always uses the full loaded history." + in_use)

    items = compute_portfolio_inventory(positions, market.snapshot)
    bb = calculate_borrowing_base(items, receivables, terms)
    stress = run_stress_scenario(
        positions, receivables, market.snapshot, terms,
        shock_pct=parallel_shock, per_commodity_shocks=per_commodity, baseline_bb_result=bb,
    )
    tiers = compare_concentration_cap_tiers(items, receivables, terms)
    inputs = model_inputs_json(positions, receivables, terms, market.snapshot)
    with st.spinner("Running scenario analytics"):
        sweep, grid, replay = cached_scenario_suite(*inputs, market.prices_df, market.vol_df)
        mc = cached_monte_carlo(*inputs, market.prices_df, horizon, n_sims, bool(ss["winsorize"]))
        runway = cached_runway(*inputs, market.prices_df, n_sims, bool(ss["winsorize"]))

    settings = AdvancedSettings(margin=margin_settings, basis=basis_map)
    ml_cut = float(ss["ml_cut_pct"]) / 100.0
    active_spec = ScenarioSpec(name="Active scenario", futures_shocks=dict(stress.shocks_map),
                               im_multipliers=user_mults, basis_shocks=basis_shocks,
                               liquidation_discounts=liq_discounts, margin_line_cut_pct=ml_cut)
    ss["active_spec_json"] = active_spec.model_dump_json()
    trigger_rule = bool(ss["apply_trigger_rule"])
    instant_spec = active_spec
    if settings.margin.im_mode == "per_contract":
        instant_spec = active_spec.model_copy(update={"im_multipliers": auto_im_multipliers(
            active_spec.futures_shocks, terms, user_mults, trigger_rule)})
    calibration = cached_calibration(market.prices_df, ss["calib_window"], bool(ss["winsorize"]))
    adv = adv_recon = run = ledger_df = cfmc = None
    if advanced:
        with st.spinner("Running the advanced model"):
            adv = run_advanced_scenario(positions, receivables, market.snapshot, terms, settings, instant_spec)
            adv_recon = reconcile_to_legacy(positions, receivables, market.snapshot, terms, settings, instant_spec)
            fp, bp = deterministic_paths(market.snapshot, settings, active_spec, cf_settings)
            run = run_cashflow_ledger(positions, receivables, market.snapshot, terms, settings, cf_settings, fp, bp,
                                      user_mults, trigger_rule)
            ledger_df = ledger_frame(run)
            cfmc = cached_cashflow_monte_carlo(
                *inputs, settings.model_dump_json(), cf_settings.model_dump_json(), market.prices_df,
                ss["calib_window"], ss["sim_method"], int(ss["cf_mc_paths"]), float(ss["basis_vol"]) / 100,
                json.dumps(user_mults, sort_keys=True), trigger_rule, bool(ss["winsorize"]),
            )
        sens_settings, sens_spec = settings, instant_spec
    else:
        sens_settings = neutral_settings(settings)
        sens_spec = ScenarioSpec(name="Active scenario", futures_shocks=dict(stress.shocks_map))
    sensitivities = buffer_sensitivities(positions, receivables, market.snapshot, terms, sens_settings, sens_spec)
    if not advanced:
        sensitivities = [s for s in sensitivities if not s.label.startswith(("Basis", "Initial margin requirement"))]
    active_nlb = adv.net_liquidity_buffer if advanced else stress.net_liquidity_buffer
    active_regime = adv.regime if advanced else stress.regime
    active_explanation = adv.regime_explanation if advanced else regime_explanation_text(stress)
    ss["active_regime"] = str(active_regime)

    if is_custom:
        scenario = "Per commodity " + ", ".join(
            f"{COMMODITY_SPECS[t].name} {v:+.0%}" for t, v in per_commodity.items()
        )
        same_metals = per_commodity["GC=F"] == per_commodity["HG=F"]
        same_softs = per_commodity["LBR=F"] == per_commodity["CC=F"]
        active_point = (per_commodity["GC=F"], per_commodity["LBR=F"]) if same_metals and same_softs else None
    else:
        scenario = f"Parallel {parallel_shock:+.0%}"
        active_point = (parallel_shock, parallel_shock)

    baseline_point = next((p for p in sweep.points if p.shock_pct == 0.0), None)
    worst_point = sweep.worst_net_buffer_point
    if ss["cfp_target"] == "worst":
        plan_shocks, plan_label = worst_point.shocks_map, f"Worst parallel shock ({worst_point.shock_pct:+.0%})"
    else:
        plan_shocks, plan_label = stress.shocks_map, f"Active scenario ({scenario})"
    plan = build_contingency_plan(positions, receivables, market.snapshot, terms, sweep, plan_shocks, plan_label)
    availability = min(terms.facility_limit, bb.final_borrowing_base)
    usage = terms.drawn_revolver / availability if availability > 0 else float("inf")
    be_up = sweep.upside_squeeze_breakeven_pct
    be_down = sweep.downside_deficiency_breakeven_pct
    status = regime_status(active_regime)
    worst_spec = ScenarioSpec(name=f"Worst parallel shock ({sweep.worst_net_buffer_point.shock_pct:+.0%})",
                              futures_shocks=dict(sweep.worst_net_buffer_point.shocks_map))
    alt_spec = ScenarioSpec.model_validate_json(ss["compare_alt"]) if "compare_alt" in ss else worst_spec
    comparison = compare_scenarios(
        positions, receivables, market.snapshot, terms, settings, model_mode,
        [ScenarioSpec(name="Baseline (no shock)", futures_shocks={t: 0.0 for t in REQUIRED_TICKERS}),
         active_spec, alt_spec],
        cf_settings if advanced else None, trigger_rule,
    )

    with header:
        render_header(
            [
                ("Model", MODEL_LABELS[model_mode]),
                ("Scenario", scenario),
                ("Cap tier", f"{terms.concentration_cap_pct:.1%}"),
                ("Monte Carlo", f"{n_sims:,} paths, {horizon}-day"),
                ("Commodities", str(len(positions))),
                ("Prices as of", market.as_of_date),
                ("Currency", ccy_code if ccy_code == "USD"
                 else f"{ccy_code} at {fx_rate:.4f} USD ({fx_source}, {fx_date})"),
            ],
            market_data_status(market),
        )
        render_intro()
        baseline_text = (f"No-shock baseline {fmt_signed_millions(comparison[0].metrics['net_liquidity_buffer'])}."
                         if comparison else "")
        if advanced:
            margin_need, vm_part, im_part = adv.margin_cash_need, adv.vm_payable, adv.im_additional
            line_cap, line_draw = adv.margin_line_capacity, adv.margin_line_draw
        else:
            margin_need = stress.vm_outflow + stress.im_addon_outflow
            vm_part, im_part = stress.vm_outflow, stress.im_addon_outflow
            line_cap, line_draw = terms.margin_line_limit, stress.margin_line_draw
        if advanced:
            first_day = int(run.first_shortfall_day[0])
            last_card = {
                "label": "First unfunded day",
                "value": f"Day {first_day}" if first_day > 0 else "None",
                "context": f"Daily ledger, {cf_settings.horizon_days} business days. "
                           f"P(any unfunded day) {fmt_pct(cfmc.prob_any_shortfall, 1)} over {cfmc.n_paths:,} paths",
                "status": "breach" if first_day > 0 else None, "status_text": "Shortfall",
            }
        else:
            last_card = {
                "label": f"LaR 95% · {mc.horizon_days}-day",
                "value": fmt_signed_millions(mc.p05_net_buffer),
                "context": f"5th percentile buffer over {mc.n_sims:,} paths. "
                           f"P(critical squeeze) {fmt_pct(mc.prob_critical, 1)}",
                "status": "breach" if mc.p05_net_buffer < 0 else None, "status_text": "Below zero",
            }
        render_verdict(
            {
                "label": "Net liquidity buffer under active stress",
                "value": fmt_signed_millions(active_nlb),
                "status": status, "status_text": regime_label(active_regime),
                "context": f"{active_explanation} {baseline_text}",
            },
            [
                {
                    "label": "Revolver headroom",
                    "value": fmt_millions(bb.available_headroom),
                    "context": f"Base {fmt_millions(bb.final_borrowing_base)}, limit "
                               f"{fmt_millions(terms.facility_limit)}, drawn {fmt_millions(terms.drawn_revolver)}. "
                               f"Availability used {fmt_pct(usage, 1)}",
                    "status": "breach" if bb.final_borrowing_base < terms.drawn_revolver else None,
                    "status_text": "Deficiency",
                },
                {
                    "label": "Margin cash need",
                    "value": fmt_millions(margin_need),
                    "context": f"Variation margin {fmt_millions(vm_part)}, additional initial margin "
                               f"{fmt_millions(im_part)} at the active shock",
                },
                {
                    "label": "Margin financing",
                    "value": fmt_millions(line_draw),
                    "context": f"Drawn of {fmt_millions(line_cap)} capacity. Funds margin calls only, "
                               "never a deficiency cure",
                },
                last_card,
            ],
        )
        methodology_expander()

    with tab_overview:
        explanation_now = regime_explanation_text(stress)
        report_args = (currency, market, terms, bb, stress, sweep, mc, runway, plan, scenario, explanation_now)
        report_kwargs = {
            "comparison": comparison, "model_label": MODEL_LABELS[model_mode],
            "advanced": ({"adv": adv, "recon": adv_recon, "sens": sensitivities, "run": run, "ledger_df": ledger_df,
                          "cf": cf_settings, "cfmc": cfmc, "calibration": calibration, "margin": settings.margin}
                         if advanced else None),
        }
        col_note, col_btn = st.columns([3, 1], gap="medium")
        with col_note:
            provenance_note("Download a self-contained HTML report of this view for a credit or risk committee: "
                            "verdict, scenario comparison, borrowing base, stress sweep, liquidity-at-risk, runway and "
                            "funding plan" + (", plus margin, ledger, simulation and provenance sections."
                                              if advanced else "."))
        with col_btn:
            st.download_button(
                "Credit committee report (HTML)", data=lambda: build_report_html(*report_args, **report_kwargs),
                file_name=f"liquidity_report_{market.as_of_date.replace('-', '')}_{ccy_code}.html",
                mime="text/html", width="stretch", key="dl_report", type="primary",
            )
        col_main, col_side = st.columns([2.1, 1], gap="medium")
        with col_main:
            chart(build_buffer_profile_chart(sweep, None if is_custom else parallel_shock), "chart_profile",
                  "Net liquidity buffer across parallel price shocks",
                  "Same percentage shock to every commodity. Shading: green healthy, amber margin trap or BB "
                  "deficiency, red critical squeeze.")
        with col_side:
            with panel("briefing"):
                section_head("Treasury briefing", "Derived from the current inputs.")
                findings = [(status, f"<b>{regime_label(active_regime)}</b> at {scenario.lower()}"
                                     f"{' (advanced model)' if advanced else ''}: {active_explanation}")]
                findings.append(("neutral", "Parallel squeeze breakeven <b>"
                                 + (fmt_pct(be_up) if be_up is not None else "none to +20%")
                                 + "</b>, deficiency breakeven <b>"
                                 + (fmt_pct(be_down) if be_down is not None else "none to -20%") + "</b>."))
                cap_point = next(
                    (p for p in sweep.points if p.shock_pct > 0 and p.shocked_bb >= terms.facility_limit), None
                )
                if cap_point is not None:
                    findings.append(("neutral", f"Headroom <b>caps out at {cap_point.shock_pct:+.0%}</b>. Above it "
                                                "the base exceeds the facility limit, so rallies add margin calls "
                                                "but no new availability."))
                if bb.capped_buyer_names:
                    tier_gain = tiers[-1].final_borrowing_base - bb.final_borrowing_base
                    findings.append(("neutral", f"<b>{len(bb.capped_buyer_names)} buyer(s) capped</b>, "
                                                f"{fmt_millions(bb.concentration_cutback_dollars)} cut back. The "
                                                f"{tiers[-1].cap_pct:.1%} tier would add {fmt_millions(tier_gain)}."))
                findings.append(("breach" if grid.squeeze_share > 0 else "neutral",
                                 f"<b>{fmt_pct(grid.squeeze_share, 0)}</b> of mixed metals/softs grid scenarios "
                                 f"end in a critical squeeze. Worst: metals {grid.worst_metals_shock:+.0%}, softs "
                                 f"{grid.worst_softs_shock:+.0%} ({fmt_signed_millions(grid.worst_net_buffer)})."))
                fix = next((a for a in plan.actions if a.feasible), None)
                if plan.shortfall > 0 and fix is not None:
                    findings.append(("caution", f"Smallest single fix for {plan.target_label.lower()}: "
                                                f"<b>{fix.label.lower()}</b>, {fmt_millions(fix.required_usd)} "
                                                "(see Funding plan)."))
                findings.append(("breach" if runway.prob_breach_within_horizon > 0 else "neutral",
                                 f"<b>{fmt_pct(runway.prob_breach_within_horizon, 1)}</b> of {runway.n_paths:,} "
                                 f"simulated paths run out of liquidity within {runway.horizon_days} trading days."))
                if replay is not None:
                    findings.append(("caution" if replay.deficiency_days else "neutral",
                                     f"Replay: base below drawn on <b>{replay.deficiency_days} of {replay.n_days} "
                                     f"days</b>. Worst 5-day margin call {fmt_millions(replay.max_5d_vm_outflow)}."))
                render_briefing(findings)
            with panel("mix"):
                section_head("Eligible collateral", f"Total {fmt_millions(bb.final_borrowing_base)}")
                mix = pd.DataFrame(
                    [{"Component": i.name, "Eligible": i.eligible_collateral / money_divisor(),
                      "Share": i.eligible_collateral / bb.final_borrowing_base * 100 if bb.final_borrowing_base else 0.0}
                     for i in bb.inventory_items]
                    + [{"Component": "Receivables (after cap)", "Eligible": bb.final_eligible_receivables / money_divisor(),
                        "Share": bb.final_eligible_receivables / bb.final_borrowing_base * 100
                        if bb.final_borrowing_base else 0.0}]
                )
                st.dataframe(mix, width="stretch", hide_index=True,
                             column_config={"Eligible": _money_column("Eligible"),
                                            "Share": st.column_config.NumberColumn("Share", format="%.1f%%")})
        with panel("comparison"):
            section_head("Scenario comparison",
                         "Baseline, active scenario and a saved alternative, computed with the same engine and "
                         f"conventions ({MODEL_LABELS[model_mode].lower()} model).")
            render_comparison(comparison)
            b1, b2, _ = st.columns([1, 1, 2])
            with b1:
                st.button("Save active as alternative", on_click=save_active_as_alternative, width="stretch",
                          key="save_alt_btn")
            with b2:
                st.button("Reset alternative to worst parallel shock", on_click=clear_alternative, width="stretch",
                          key="clear_alt_btn", disabled="compare_alt" not in ss)
        with panel("drivers"):
            section_head("Risk drivers and responses",
                         "One change at a time against the active scenario's net liquidity buffer"
                         + (" (advanced model)." if advanced else " (legacy model)."))
            render_steps_table(sensitivities)

    with tab_cert:
        col_main, col_side = st.columns([1.8, 1], gap="medium")
        with col_main:
            chart(build_borrowing_base_waterfall(bb), "chart_waterfall", "Borrowing base bridge",
                  "From gross collateral to the certified base. Dashed: facility limit. Dotted: drawn balance.")
        with col_side:
            with panel("covenant"):
                section_head("Covenant position")
                covenant_audit_rows(bb)
                st.write("")
                certificate_download_buttons(bb, stress, market)
        with panel("recon"):
            section_head("Reconciliation ledger", "Each step reconciles to the final base within $0.000001.")
            kinds = {"absolute": "Gross", "relative": "Adjustment", "total": "Subtotal"}
            recon = pd.DataFrame(
                [{"Line": bridge_label(w.label), "Type": "Total" if i == len(bb.waterfall) - 1 else kinds[w.measure],
                  "Amount": w.amount / money_divisor(), "Running total": w.running_total / money_divisor()}
                 for i, w in enumerate(bb.waterfall)]
            )
            styled = recon.style.apply(
                lambda r: ["font-weight: 600" if r["Type"] in ("Subtotal", "Total") else "" for _ in r], axis=1
            ).format({"Amount": "{:,.2f}", "Running total": "{:,.2f}"})
            st.dataframe(styled, width="stretch", hide_index=True,
                         column_config={"Amount": f"Amount ({money_unit()})",
                                        "Running total": f"Running total ({money_unit()})"})

    with tab_inv:
        col_hc, col_vol = st.columns([1.15, 1], gap="medium")
        with col_hc:
            chart(build_haircut_decomposition_chart(bb.inventory_items), "chart_haircuts",
                  "Advance rate and haircut components", "Share of gross market value per commodity.")
        with col_vol:
            chart(build_volatility_history_chart(market.vol_df, COMMODITY_SPECS), "chart_vol",
                  "30-day realized volatility",
                  "Annualized. Dotted: haircut reference volatility ("
                  + ", ".join(f"{s.name} {s.ref_vol:.0%}" for s in COMMODITY_SPECS.values()) + ").")
        haircut_methodology_expander()

    with tab_ar:
        col_chart, col_side = st.columns([2.1, 1], gap="medium")
        with col_chart:
            chart(build_receivables_concentration_chart(bb), "chart_ar", "Concentration by buyer",
                  "Eligible receivables before and after the single-buyer cap. Amber bars are capped.")
        with col_side:
            with panel("tiers"):
                section_head("Cap tier sensitivity", "Borrowing base re-solved at each cap tier.")
                tier_df = pd.DataFrame(
                    [{"Tier": f"{t.cap_pct:.1%}", "Base": t.final_borrowing_base / money_divisor(),
                      "Cutback": t.total_cutback_dollars / money_divisor(), "Capped": t.capped_buyer_count} for t in tiers]
                )
                st.dataframe(tier_df, width="stretch", hide_index=True,
                             column_config={"Base": _money_column("Base"), "Cutback": _money_column("Cutback")})
            with panel("solver"):
                section_head("Fixed-point solver check")
                solver_audit_rows(bb)

    with tab_stress:
        chart(build_stress_sweep_chart(sweep, parallel_shock, is_custom), "chart_sweep",
              "Margin-trap stress sweep",
              f"41 parallel shocks from -20% to +20%. Squeeze breakeven "
              f"{fmt_pct(be_up) if be_up is not None else 'none in range'}, deficiency breakeven "
              f"{fmt_pct(be_down) if be_down is not None else 'none in range'}. Regime counts: "
              + ", ".join(f"{regime_label(k)} {v}" for k, v in sweep.regime_counts.items()) + ".")
        if advanced:
            with panel("margin_dashboard"):
                section_head("Margin dashboard, active scenario (advanced model)",
                             f"{scenario}. Whole contracts, per-contract initial margin, basis and margin line "
                             "applied. The 41-point sweep above uses the legacy engine.")
                render_kpi_row(margin_kpis(adv))
            col_main, col_side = st.columns([2.1, 1], gap="medium")
            with col_main:
                chart(build_margin_contribution_chart(adv), "chart_margin_contrib", "Margin cash by commodity",
                      "Variation margin paid and additional initial margin (right), variation margin received (left).")
            with col_side:
                with panel("adv_ledger"):
                    section_head("Active shock ledger", "Advanced model")
                    render_audit_rows([
                        ("Shocked borrowing base", fmt_millions(adv.shocked_bb), False),
                        ("Shocked headroom", fmt_millions(adv.shocked_headroom), False),
                        ("Variation margin received", fmt_millions(adv.vm_receivable), False),
                        ("Initial margin released", fmt_millions(adv.im_released), False),
                        ("Margin line draw", fmt_millions(adv.margin_line_draw), False),
                        ("Variation margin paid", fmt_millions(adv.vm_payable), False),
                        ("Additional initial margin", fmt_millions(adv.im_additional), False),
                        ("BB deficiency cure", fmt_millions(adv.bb_deficiency_cure), False),
                        ("Liquid resources", fmt_millions(adv.total_liquid_resources), False),
                        ("Cash outflows", fmt_millions(adv.total_cash_outflows), False),
                        ("Net liquidity buffer", f"<b>{fmt_signed_millions(adv.net_liquidity_buffer)}</b>", False),
                    ])
            col_rec, col_sens = st.columns(2, gap="medium")
            with col_rec:
                with panel("reconciliation"):
                    section_head("Reconciliation from the legacy model",
                                 "Each assumption added in turn; the changes sum exactly to the difference.")
                    render_steps_table(adv_recon, "Advanced model")
            with col_sens:
                with panel("sensitivity"):
                    section_head("Sensitivity of the buffer", "One change at a time against the active scenario.")
                    render_steps_table(sensitivities)
            with st.expander("Margin calculation audit by commodity"):
                render_margin_audit(adv, settings.margin)
        else:
            col_main, col_side = st.columns([2.1, 1], gap="medium")
            with col_main:
                chart(build_active_shock_sources_uses_chart(stress, terms), "chart_sources_uses",
                      "Cash sources and uses at the active shock", scenario)
            with col_side:
                with panel("active_ledger"):
                    section_head("Active shock ledger", scenario)
                    render_audit_rows(
                        [
                            ("Shocked borrowing base", fmt_millions(stress.shocked_bb), False),
                            ("Change vs. baseline", fmt_signed_millions(stress.bb_change), False),
                            ("Shocked headroom", fmt_millions(stress.shocked_headroom), False),
                            ("Futures MtM", fmt_signed_millions(stress.futures_mtm_pnl), False),
                            ("Variation margin call", fmt_millions(stress.vm_outflow), False),
                            ("Variation margin inflow", fmt_millions(stress.vm_inflow), False),
                            ("Initial margin step-up", fmt_millions(stress.im_addon_outflow), False),
                            ("BB deficiency cure", fmt_millions(stress.bb_deficiency_cure), False),
                            ("Liquid resources", fmt_millions(stress.total_liquid_resources), False),
                            ("Cash outflows", fmt_millions(stress.total_cash_outflows), False),
                            ("Net liquidity buffer", f"<b>{fmt_signed_millions(stress.net_liquidity_buffer)}</b>", False),
                        ]
                    )
        with st.expander("Full 41-point sweep table"):
            sweep_df = sweep.to_dataframe()
            view = pd.DataFrame({
                "Shock": sweep_df["shock_pct"] * 100,
                "Borrowing base": sweep_df["shocked_bb"] / money_divisor(),
                "Headroom": sweep_df["shocked_headroom"] / money_divisor(),
                "VM call": sweep_df["vm_outflow"] / money_divisor(),
                "VM inflow": sweep_df["vm_inflow"] / money_divisor(),
                "IM step-up": sweep_df["im_addon_outflow"] / money_divisor(),
                "BB cure": sweep_df["bb_deficiency_cure"] / money_divisor(),
                "Net buffer": sweep_df["net_liquidity_buffer"] / money_divisor(),
                "Regime": [regime_label(r) for r in sweep_df["regime"]],
            })
            st.dataframe(
                view, width="stretch", hide_index=True,
                column_config={"Shock": st.column_config.NumberColumn("Shock", format="%+.0f%%"),
                               **{c: _money_column(c) for c in ["Borrowing base", "Headroom", "VM call", "VM inflow",
                                                                 "IM step-up", "BB cure", "Net buffer"]}},
            )
            st.download_button(
                "Download sweep (CSV, full precision)", data=sweep_df.to_csv(index=False),
                file_name=f"stress_sweep_{market.as_of_date.replace('-', '')}.csv",
                mime="text/csv", key="dl_sweep",
            )
        stress_methodology_expander()

    with tab_heat:
        active_nlb = None
        if active_point is not None and active_point[0] in grid.metals_axis and active_point[1] in grid.softs_axis:
            i, j = grid.metals_axis.index(active_point[0]), grid.softs_axis.index(active_point[1])
            active_nlb = grid.net_buffer[i][j]
        squeeze_count = sum(1 for row in grid.regime for r in row if r == "CRITICAL_SQUEEZE")
        render_kpi_row(
            [
                {"label": "Grid scenarios in critical squeeze", "value": fmt_pct(grid.squeeze_share, 1),
                 "context": f"{squeeze_count} of {len(grid.metals_axis) ** 2} combinations"},
                {"label": "Worst grid scenario", "value": fmt_signed_millions(grid.worst_net_buffer),
                 "context": f"Metals {grid.worst_metals_shock:+.1%}, softs {grid.worst_softs_shock:+.1%}"},
                {"label": "Active scenario", "value": fmt_signed_millions(stress.net_liquidity_buffer),
                 "status": status, "status_text": regime_label(stress.regime),
                 "context": "On the grid (circled)" if active_nlb is not None
                            else "Between grid points; exact engine value"},
            ]
        )
        chart(build_shock_heatmap(grid, active_point), "chart_heatmap",
              "Net liquidity buffer by mixed shock",
              "Gold and copper move together (vertical), lumber and cocoa together (horizontal), 2.5% steps. "
              "Dashed line: buffer of $0. Darker cells are worse.")

    with tab_mc:
        render_kpi_row(
            [
                {"label": "LaR 95%", "value": fmt_signed_millions(mc.p05_net_buffer),
                 "context": f"5th percentile buffer, {mc.horizon_days}-day",
                 "status": "breach" if mc.p05_net_buffer < 0 else None, "status_text": "Below zero"},
                {"label": "LaR 99%", "value": fmt_signed_millions(mc.p01_net_buffer),
                 "context": f"1st percentile buffer, {mc.horizon_days}-day",
                 "status": "breach" if mc.p01_net_buffer < 0 else None, "status_text": "Below zero"},
                {"label": "Expected shortfall 95%", "value": fmt_signed_millions(mc.es95_net_buffer),
                 "context": "Mean buffer of the worst 5% of paths"},
                {"label": "P(critical squeeze)", "value": fmt_pct(mc.prob_critical, 2),
                 "context": f"{mc.regime_counts['CRITICAL_SQUEEZE']:,} of {mc.n_sims:,} paths"},
                {"label": "P(margin trap)", "value": fmt_pct(mc.prob_margin_trap, 1),
                 "context": f"P(BB deficiency) {fmt_pct(mc.prob_deficiency, 1)}"},
            ]
        )
        chart(build_mc_distribution_chart(mc), "chart_mc", "Simulated net liquidity buffer",
              f"{mc.n_sims:,} correlated lognormal paths over {mc.horizon_days} trading days, seed {mc.seed}. "
              "Red: paths at or below the 5th percentile.")
        lar_lo, lar_hi = percentile_interval(mc.net_buffer_samples, 0.05)
        provenance_note(f"Sampling interval for LaR 95% (95% confidence, order statistics over {mc.n_sims:,} paths): "
                        f"{fmt_signed_millions(lar_lo)} to {fmt_signed_millions(lar_hi)}. More paths narrow the "
                        "interval; they do not make the model more accurate. " + RISK_MEASURE_NOTES)
        col_corr, col_vols = st.columns([1.25, 1], gap="medium")
        with col_corr:
            chart(build_correlation_heatmap(mc, COMMODITY_SPECS), "chart_corr", "Return correlation",
                  f"Daily log returns, last {mc.lookback_days} observations.")
        with col_vols:
            with panel("mc_vols"):
                section_head("Horizon volatility and worst path",
                             "Standard deviation of the simulated move, and the move on the worst path.")
                vol_df = pd.DataFrame(
                    [{"Commodity": COMMODITY_SPECS[t].name, "Horizon vol": mc.horizon_vols[t] * 100,
                      "Worst-path move": mc.worst_scenario_shocks[t] * 100} for t in REQUIRED_TICKERS]
                )
                st.dataframe(vol_df, width="stretch", hide_index=True,
                             column_config={"Horizon vol": st.column_config.NumberColumn(
                                                f"{mc.horizon_days}d vol", format="%.2f%%"),
                                            "Worst-path move": st.column_config.NumberColumn(format="%+.2f%%")})
        if market_data_status(market)[0].startswith("Bundled"):
            provenance_note("The bundled sample draws each commodity independently, so correlations are near "
                            "zero. Switch to the live feed for observed co-movement.")

        with panel("runway"):
            section_head("Liquidity runway, next 20 trading days",
                         f"{runway.n_paths:,} daily correlated paths (same covariance as above, seed {runway.seed}). "
                         "Each day applies the cumulative price move since today; a path runs out of liquidity "
                         "on the first day its net buffer is below zero.")
            day20 = {k: v[-1] for k, v in runway.percentiles.items()}
            render_kpi_row(
                [
                    {"label": "Shortfall within 20 days", "value": fmt_pct(runway.prob_breach_within_horizon, 2),
                     "context": f"{sum(runway.first_breach_counts):,} of {runway.n_paths:,} paths"},
                    {"label": "Median day of first shortfall",
                     "value": f"Day {runway.median_first_breach_day:.0f}"
                     if runway.median_first_breach_day is not None else "None",
                     "context": "Among paths that run short"},
                    {"label": "Day-20 buffer, 5th percentile", "value": fmt_signed_millions(day20["p5"]),
                     "context": "Adverse tail after four weeks"},
                    {"label": "Day-20 buffer, median", "value": fmt_signed_millions(day20["p50"]),
                     "context": f"Today without shock {fmt_signed_millions(baseline_point.net_liquidity_buffer)}"
                     if baseline_point is not None else ""},
                ]
            )
            col_fan, col_prob = st.columns([2.1, 1], gap="medium")
            with col_fan:
                section_head("Net liquidity buffer by day", "Median with 25th to 75th and 5th to 95th percentile bands.")
                st.plotly_chart(localize_figure(build_runway_fan_chart(runway)), width="stretch", theme=None, config=PLOTLY_CONFIG,
                                key="chart_runway_fan")
            with col_prob:
                section_head("Share of paths already short", "Cumulative, by trading day.")
                st.plotly_chart(build_runway_breach_chart(runway), width="stretch", theme=None,
                                config=PLOTLY_CONFIG, key="chart_runway_prob")

        if advanced:
            with panel("cf_mc"):
                section_head("Daily cash-flow simulation (advanced model)",
                             f"{cfmc.n_paths:,} paths of {cfmc.horizon_days} business days through the daily ledger: "
                             f"method: {SIM_METHOD_LABELS[cfmc.method]}, {calibration.window_label} calibration, "
                             f"seed {cfmc.seed}. Funding delays, interest, collections and cure grace included.")
                render_kpi_row(cashflow_mc_kpis(cfmc))
                days = list(range(cfmc.horizon_days + 1))
                col_fan, col_prob = st.columns([2.1, 1], gap="medium")
                with col_fan:
                    section_head("Available liquidity by day", "Median with 25th to 75th and 5th to 95th percentiles.")
                    st.plotly_chart(localize_figure(build_liquidity_fan_chart(
                        days, cfmc.liquidity_percentiles_by_day, "Available liquidity (USD M)")), width="stretch",
                        theme=None, config=PLOTLY_CONFIG, key="chart_cf_fan")
                with col_prob:
                    section_head("Share of paths with an unfunded day", "Cumulative, by business day.")
                    st.plotly_chart(build_probability_chart(days, [0.0, *cfmc.prob_shortfall_by_day], "Paths short (%)"),
                                    width="stretch", theme=None, config=PLOTLY_CONFIG, key="chart_cf_prob")
        col_prov, col_cal = st.columns([1.6, 1], gap="medium")
        with col_prov:
            with panel("provenance"):
                section_head("Data provenance", "What the simulation is calibrated on.")
                render_provenance(market, calibration, synthetic=not market.source_label.startswith(("Live", "Custom"))
                                  or market.used_fallback)
        with col_cal:
            with panel("calibrated_vols"):
                section_head("Calibrated volatility", f"{calibration.window_label} window, annualized.")
                render_calibration_table(calibration)
        with st.expander("Counterfactual backtest of the 5-day margin forecast"):
            st.select_slider("Rolling estimation window (days)", [60, 125, 250], key="bt_window")
            bt = cached_backtest(inputs[0], inputs[3], market.prices_df, int(ss["bt_window"]), bool(ss["winsorize"]))
            if bt.n_forecasts == 0:
                st.info("Not enough price history in the active feed for this window.")
            else:
                render_kpi_row([
                    {"label": "Forecasts", "value": f"{bt.n_forecasts:,}",
                     "context": f"{bt.horizon_days}-day windows, {bt.window_days}-day estimation"},
                    {"label": "Breaches", "value": f"{bt.n_breaches:,}",
                     "context": f"Realized outflow above the {bt.level:.0%} forecast"},
                    {"label": "Breach rate", "value": fmt_pct(bt.breach_rate, 1),
                     "context": f"Expected {fmt_pct(bt.expected_rate, 1)}"},
                ])
                st.plotly_chart(localize_figure(build_backtest_chart(bt.frame)), width="stretch", theme=None,
                                config=PLOTLY_CONFIG, key="chart_backtest")
            provenance_note("Counterfactual: today's hedge book applied to past price moves of the active feed. "
                            + bt.note + (" The bundled feeds are synthetic, so this checks internal consistency "
                                         "only, not real-world performance." if not market.source_label.startswith(
                                             ("Live", "Custom")) or market.used_fallback else ""))

    with tab_hedge:
        scope_names = {ALL_COMMODITIES: "All commodities together",
                       **{t: COMMODITY_SPECS[t].name for t in REQUIRED_TICKERS}}
        col_sel, col_note = st.columns([1, 2.4], gap="medium")
        with col_sel:
            scope = st.selectbox("Commodity to re-hedge", list(scope_names), key="hedge_scope",
                                 format_func=lambda k: scope_names[k])
        with col_note:
            provenance_note("Sweeps the futures hedge from 0% to 100% of physical quantity"
                            + (" for every commodity at once." if scope == ALL_COMMODITIES
                               else f" for {scope_names[scope].lower()}; other commodities keep today's hedges.")
                            + " Each ratio is run through the same 41 parallel shocks as the stress sweep.")
        frontier = cached_hedge_frontier(*inputs, scope)
        nlbs = [p.net_liquidity_buffer for p in sweep.points]
        cur_share = sum(v >= 0 for v in nlbs) / len(nlbs)
        best = frontier.recommended_index
        render_kpi_row(
            [
                {"label": "Current hedge ratio", "value": fmt_pct(frontier.current_ratio, 0),
                 "context": "Hedged value / physical value" if scope == ALL_COMMODITIES else "Hedged / physical qty"},
                {"label": "Liquidity-optimal ratio", "value": fmt_pct(frontier.recommended_ratio, 0),
                 "context": "Most shocks with a buffer at or above zero"},
                {"label": "Shocks with buffer >= 0", "value": f"{cur_share:.0%} to {frontier.positive_share[best]:.0%}",
                 "context": "Current vs. liquidity-optimal, of 41 shocks"},
                {"label": "Worst-case buffer", "value": fmt_signed_millions(frontier.min_buffer[best]),
                 "context": f"At the optimal ratio; current {fmt_signed_millions(min(nlbs))}"},
                {"label": "Unprotected in a 20% fall", "value": fmt_millions(frontier.crash_value_loss[best]),
                 "context": "Inventory value lost at the optimal ratio"},
            ]
        )
        chart(build_hedge_frontier_chart(frontier), "chart_hedge", "Hedge ratio trade-off",
              "More hedging protects inventory value and the borrowing base in a fall, but adds variation margin "
              "calls in a rally. The liquidity-optimal ratio ignores price risk; read it against the lower panel.")
        col_par, col_tab = st.columns([1.4, 1], gap="medium")
        with col_par:
            chart(build_pareto_chart(frontier), "chart_pareto", "Liquidity protection against price protection",
                  "Each point is a hedge ratio. Pareto-efficient ratios: no other ratio has both a better worst-case "
                  "buffer and a smaller unhedged loss. No weights are applied; the choice between them is a policy "
                  "decision.")
        with col_tab:
            with panel("hedge_objectives"):
                section_head("Objectives by hedge ratio",
                             f"Margin need at +20%; basis exposure = hedged notional x {frontier.basis_shock_pct:.0%} "
                             "(illustrative basis move).")
                obj = pd.DataFrame({
                    "Hedge ratio": [r * 100 for r in frontier.ratios],
                    "Worst buffer": [v / money_divisor() for v in frontier.min_buffer],
                    "Crash loss": [v / money_divisor() for v in frontier.crash_value_loss],
                    "Margin need": [v / money_divisor() for v in frontier.max_margin_need],
                    "Basis exposure": [v / money_divisor() for v in frontier.basis_exposure],
                    "Efficient": frontier.pareto_efficient,
                })
                obj = obj.rename(columns={"Hedge ratio": "Ratio", "Basis exposure": "Basis"})
                st.dataframe(obj.iloc[::2], width="stretch", hide_index=True, height=420, column_config={
                    "Ratio": st.column_config.NumberColumn(format="%.0f%%", width="small"),
                    "Efficient": st.column_config.CheckboxColumn(width="small"),
                    **{c: st.column_config.NumberColumn(c, format="%+.2f" if c == "Worst buffer" else "%.2f")
                       for c in ["Worst buffer", "Crash loss", "Margin need", "Basis"]}})
                st.caption(f"Amounts in {money_unit()}. Every second ratio shown.")
        if advanced:
            col_pnl, col_cash = st.columns([1.4, 1], gap="medium")
            with col_pnl:
                chart(build_economic_pnl_chart(adv), "chart_econ_pnl", "Economic P&L by commodity, active scenario",
                      "Physical inventory, futures hedge and combined. Economic value, not cash: the hedge P&L is "
                      "settled daily as variation margin, the physical P&L only when the inventory is sold.")
            with col_cash:
                with panel("econ_vs_cash"):
                    section_head("Economic result against cash", "Active scenario, advanced model.")
                    render_audit_rows([
                        ("Physical inventory P&L", fmt_signed_millions(adv.physical_pnl), False),
                        ("Futures hedge P&L", fmt_signed_millions(adv.futures_pnl), False),
                        ("Combined economic P&L", f"<b>{fmt_signed_millions(adv.combined_pnl)}</b>", False),
                        ("of which basis", fmt_signed_millions(adv.basis_pnl), False),
                        ("of which unhedged price exposure", fmt_signed_millions(adv.residual_price_pnl), False),
                        ("Liquidation discount (collateral only)", fmt_signed_millions(-adv.liquidation_discount_cost),
                         False),
                        ("Cash today: variation margin, net", fmt_signed_millions(adv.vm_net), False),
                        ("Cash today: initial margin change", fmt_signed_millions(adv.im_released - adv.im_additional),
                         False),
                    ])
                    st.caption("A fully hedged book can be flat economically and still need cash today: the hedge "
                               "loss is paid at once while the offsetting physical gain waits for the sale.")

    with tab_plan:
        st.radio("Plan against", ["active", "worst"], key="cfp_target", horizontal=True,
                 format_func=lambda k: {"active": "Active scenario",
                                        "worst": "Worst parallel shock in the sweep"}[k])
        provenance_note("Each action is sized with the stress engine to bring this scenario's net liquidity buffer "
                        "back to zero on its own. In practice desks combine smaller amounts of several actions.")
        fix = next((a for a in plan.actions if a.feasible), None)
        render_kpi_row(
            [
                {"label": "Net buffer in this scenario", "value": fmt_signed_millions(plan.target_net_buffer),
                 "context": plan.target_label,
                 "status": "breach" if plan.target_net_buffer < 0 else None, "status_text": "Shortfall"},
                {"label": "Shortfall to close", "value": fmt_millions(plan.shortfall),
                 "context": "Amount each action must cover"},
                {"label": "Smallest sufficient action",
                 "value": fmt_millions(fix.required_usd) if fix is not None and plan.shortfall > 0 else "None needed",
                 "context": fix.label if fix is not None and plan.shortfall > 0 else "Buffer already at or above zero"},
                {"label": "Margin financing line", "value": fmt_millions(terms.margin_line_limit),
                 "context": "Set in the sidebar; funds margin calls only"},
            ]
        )
        with panel("cfp_actions"):
            section_head("Remedial actions, ranked",
                         "Sufficient actions first, smallest size first. Size is in the reporting currency.")
            render_action_list(plan.actions)
        col_trig, col_rev = st.columns(2, gap="medium")
        with col_trig:
            with panel("cfp_triggers"):
                section_head("Early-warning triggers", "Parallel moves at which the regime changes, from the sweep.")
                render_trigger_list(plan.triggers, regime_label, regime_status)
        with col_rev:
            with panel("cfp_reverse"):
                section_head("Reverse stress: one commodity at a time",
                             "Smallest move of a single commodity, others unchanged, searched to +/-100%.")
                st.dataframe(
                    pd.DataFrame([{"Commodity": b.name,
                                   "Rally that exhausts liquidity": b.squeeze_rally_pct * 100
                                   if b.squeeze_rally_pct is not None else None,
                                   "Fall that opens a deficiency": b.deficiency_fall_pct * 100
                                   if b.deficiency_fall_pct is not None else None} for b in plan.breaking_moves]),
                    width="stretch", hide_index=True,
                    column_config={"Rally that exhausts liquidity": st.column_config.NumberColumn(format="%+.1f%%"),
                                   "Fall that opens a deficiency": st.column_config.NumberColumn(format="%+.1f%%")},
                )
                st.caption("Blank: no breaking move within +/-100% for that commodity alone.")
        if advanced:
            with panel("ledger"):
                section_head("Daily cash-flow ledger, active scenario (advanced model)",
                             f"The scenario move is reached over {cf_settings.ramp_days} business days and then held. "
                             f"Revolver funds arrive after {cf_settings.revolver_delay_days} day(s), margin line after "
                             f"{cf_settings.margin_line_delay_days}; deficiency cure due after "
                             f"{cf_settings.cure_grace_days} day(s) of grace.")
                render_kpi_row(ledger_kpis(run, cf_settings.horizon_days))
                st.plotly_chart(localize_figure(build_ledger_chart(ledger_df, terms)), width="stretch", theme=None,
                                config=PLOTLY_CONFIG, key="chart_ledger")
                with st.expander("Daily ledger table"):
                    render_ledger_table(ledger_df, market.as_of_date.replace("-", ""))
                    st.caption("Daily order: price move and margin settlement, collections, interest, borrowing base "
                               "and cure, funding arrivals, repayments, new requests, shortfall. Collections follow "
                               "an illustrative rule: current invoices at invoice terms, overdue ones after the "
                               "overdue delay, ineligible or listed buyers not at all.")
        else:
            provenance_note("Switch Model mode to Advanced in the sidebar for the daily cash-flow ledger: funding "
                            "delays, interest, collections and the first unfunded day.")

    extras = ScenarioSpec(name="Saved extras", im_multipliers=user_mults, basis_shocks=basis_shocks,
                          liquidation_discounts=liq_discounts, margin_line_cut_pct=ml_cut)
    sidebar_configuration(build_config(positions, receivables, terms, settings.margin, basis_map, extras, cf_settings),
                          market.as_of_date.replace("-", ""))
    sidebar_reset()
    render_author_panel()

    with tab_replay:
        replay_names = {"loaded": "Loaded price history (past year)",
                        **{k: e.name for k, e in CRISIS_EPISODES.items()}}
        replay_choice = st.selectbox("Replay", list(replay_names), key="replay_source",
                                     format_func=lambda k: replay_names[k])
        if replay_choice != "loaded":
            episode = CRISIS_EPISODES[replay_choice]
            try:
                with st.spinner("Loading observed futures prices for this episode"):
                    crisis = cached_crisis_replay(*inputs, replay_choice, bool(ss["winsorize"]))
            except Exception as exc:  # noqa: BLE001 - network or data failure must not break the page
                crisis = None
                st.info(f"Historical prices for this episode could not be loaded ({type(exc).__name__}). "
                        "This replay needs internet access to Yahoo Finance; the other views are unaffected.")
            if crisis is not None:
                missing = [COMMODITY_SPECS[t].name for t in crisis.missing_tickers]
                provenance_note(
                    f"{episode.note} Prices: Yahoo Finance front-month futures, observed {crisis.first_date} to "
                    f"{crisis.last_date}. Each day's cumulative move is applied to today's spot prices and haircuts "
                    "use the 30-day volatility observed that day."
                    + (f" Not covered (no Yahoo history in this window, held at today's price): {', '.join(missing)}."
                       if missing else "")
                )
                biggest = max(crisis.max_moves, key=lambda t: abs(crisis.max_moves[t]))
                render_kpi_row(
                    [
                        {"label": "Worst net buffer", "value": fmt_signed_millions(crisis.worst_net_buffer),
                         "context": f"on {crisis.worst_date}",
                         "status": regime_status(crisis.worst_regime), "status_text": regime_label(crisis.worst_regime)},
                        {"label": "Days in critical squeeze", "value": f"{crisis.critical_days} / {len(crisis.frame)}",
                         "context": "Trading days with a buffer below zero"},
                        {"label": "Peak variation margin call", "value": fmt_millions(crisis.peak_vm_outflow),
                         "context": f"cumulative, on {crisis.peak_vm_date}"},
                        {"label": "Largest move", "value": f"{crisis.max_moves[biggest]:+.1%}",
                         "context": f"{COMMODITY_SPECS[biggest].name}, from episode start"},
                    ]
                )
                chart(build_crisis_replay_chart(crisis, COMMODITY_SPECS, terms), "chart_crisis",
                      f"Today's book through the {episode.name.split(' (')[0].lower()}",
                      "Top: observed prices. Below: modelled with today's quantities, receivables, drawn balance "
                      "and facility terms. Dashed: facility limit. Dotted: drawn balance.")
        if replay_choice == "loaded":
            if market.source_label.startswith("Live"):
                source_note = "Prices: Yahoo Finance front-month futures (observed)."
            elif market.source_label.startswith("Custom"):
                source_note = "Prices: uploaded CSV."
            else:
                source_note = "Prices: bundled calibrated sample (synthetic series, not observed market history)."
            provenance_note(f"{source_note} Borrowing base, margin and buffer are modelled from those prices with "
                            "today's quantities, receivables and drawn balance held fixed.")
            if replay is None:
                st.info("Not enough price and volatility history in the active feed to run a replay.")
            else:
                render_kpi_row(
                    [
                        {"label": "BB deficiency days", "value": f"{replay.deficiency_days} / {replay.n_days}",
                         "context": f"Base below drawn balance, {replay.start_date} to {replay.end_date}"},
                        {"label": "Minimum headroom", "value": fmt_millions(replay.min_headroom),
                         "context": f"on {replay.min_headroom_date}"},
                        {"label": "Worst 5-day margin call", "value": fmt_millions(replay.max_5d_vm_outflow),
                         "context": f"window ending {replay.max_5d_vm_outflow_date}"},
                        {"label": "Lowest stressed buffer", "value": fmt_signed_millions(replay.min_stressed_buffer),
                         "context": f"on {replay.min_stressed_buffer_date}",
                         "status": "breach" if replay.min_stressed_buffer < 0 else None, "status_text": "Below zero"},
                    ]
                )
                chart(build_replay_chart(replay, terms), "chart_replay", "Replay of today's book",
                      "Stressed buffer = cash + headroom + 5-day net variation margin - deficiency cure.")
                with st.expander("Daily replay table"):
                    rf = replay.frame.reset_index()
                    view = pd.DataFrame({
                        "Date": rf["date"].dt.date,
                        "Borrowing base": rf["borrowing_base"] / money_divisor(),
                        "Headroom": rf["headroom"] / money_divisor(),
                        "BB cure": rf["bb_deficiency_cure"] / money_divisor(),
                        "Daily VM": rf["daily_vm"] / money_divisor(),
                        "5-day VM": rf["rolling_vm"] / money_divisor(),
                        "Stressed buffer": rf["stressed_buffer"] / money_divisor(),
                    })
                    st.dataframe(view, width="stretch", hide_index=True,
                                 column_config={c: _money_column(c) for c in view.columns if c != "Date"})
                    st.download_button(
                        "Download replay (CSV, full precision)", data=rf.to_csv(index=False),
                        file_name=f"historical_replay_{market.as_of_date.replace('-', '')}.csv",
                        mime="text/csv", key="dl_replay",
                    )


main()
