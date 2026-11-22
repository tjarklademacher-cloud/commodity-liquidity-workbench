"""Commodity Collateral, Borrowing Base & Liquidity Stress Workbench: Streamlit entry point."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from core.borrowing_base import calculate_borrowing_base, compare_concentration_cap_tiers
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
from core.stress_engine import run_stress_scenario
from ui.cache import (
    cached_custom_market,
    cached_live_market,
    cached_monte_carlo,
    cached_sample_market,
    cached_scenario_suite,
    market_data_status,
    model_inputs_json,
)
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
from ui.components import (
    certificate_download_buttons,
    covenant_audit_card,
    default_inventory_df,
    default_receivables_df,
    haircut_methodology_expander,
    methodology_expander,
    inventory_editor,
    receivables_editor,
    render_hero,
    render_insights,
    render_kpi_cards,
    render_regime_banner,
    section_head,
    solver_audit_card,
    stress_methodology_expander,
)
from ui.theme import (
    PLOTLY_CONFIG,
    REGIME_TONES,
    fmt_millions,
    fmt_pct,
    fmt_signed_millions,
    inject_custom_css,
)

FEED_SAMPLE = "Calibrated Sample (Offline)"
FEED_LIVE = "Live Yahoo Finance (yfinance)"
FEED_UPLOAD = "Upload Custom CSV"
PRESET_NAMES = list(STRESS_SCENARIO_PRESETS)

WIDGET_DEFAULTS: dict[str, object] = {
    "feed_mode": FEED_SAMPLE,
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


def reset_to_defaults() -> None:
    ss = st.session_state
    for key, value in WIDGET_DEFAULTS.items():
        ss[key] = value
    ss["inventory_df"] = default_inventory_df()
    ss["receivables_df"] = default_receivables_df()
    ss.editor_nonce += 1


def apply_preset() -> None:
    ss = st.session_state
    if "inventory_edited_df" in ss:
        ss["inventory_df"] = ss["inventory_edited_df"].copy()
    if "receivables_edited_df" in ss:
        ss["receivables_df"] = ss["receivables_edited_df"].copy()
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


def side_section(number: int, title: str) -> None:
    st.sidebar.markdown(
        f'<div class="side-section"><span class="num">{number}</span>{title}</div>', unsafe_allow_html=True
    )


def sidebar_brand() -> None:
    st.sidebar.markdown(
        '<div class="side-brand"><div class="logo">ABL</div><div><div class="name">Liquidity Desk</div>'
        '<div class="tag">Collateral · Margin · Treasury</div></div></div>',
        unsafe_allow_html=True,
    )


def sidebar_market_feed() -> MarketLoadResult:
    side_section(1, "Market Data Feed")
    mode = st.sidebar.radio("Price source", [FEED_SAMPLE, FEED_LIVE, FEED_UPLOAD], key="feed_mode")
    winsorize = st.sidebar.checkbox("5-Sigma Roll-Jump Winsorization", key="winsorize")
    market: MarketLoadResult
    if mode == FEED_LIVE:
        with st.spinner("Fetching front-month futures from Yahoo Finance"):
            market = cached_live_market("1y", winsorize)
    elif mode == FEED_UPLOAD:
        uploaded = st.sidebar.file_uploader("Price CSV (Date, GC=F, HG=F, LBR=F, CC=F)", type=["csv"])
        template = pd.read_csv(DEFAULT_SAMPLE_CSV_PATH).tail(60).to_csv(index=False)
        st.sidebar.download_button(
            "Download CSV Template", data=template, file_name="price_template.csv",
            mime="text/csv", width="stretch", key="dl_template",
        )
        sample = cached_sample_market(winsorize)
        if uploaded is None:
            market = sample.model_copy(update={"status_message": "No file uploaded yet; showing calibrated sample."})
        else:
            try:
                market = cached_custom_market(uploaded.getvalue(), winsorize)
            except (ValueError, KeyError, pd.errors.ParserError) as exc:
                market = sample.model_copy(
                    update={
                        "used_fallback": True,
                        "source_label": "Calibrated Sample (Upload Fallback)",
                        "status_message": f"Uploaded CSV rejected ({exc}). Using calibrated sample.",
                    }
                )
    else:
        market = cached_sample_market(winsorize)
    if market.used_fallback:
        st.sidebar.warning(market.status_message)
    else:
        st.sidebar.caption(f"**{market.source_label}** · as of {market.as_of_date}. {market.status_message}")
    return market


def sidebar_facility_terms() -> FacilityTerms:
    side_section(2, "Facility & Liquidity Terms")
    st.sidebar.number_input("Facility Limit ($M)", min_value=1.0, step=1.0, format="%.1f", key="facility_limit_m")
    st.sidebar.number_input("Drawn Revolver ($M)", min_value=0.0, step=1.0, format="%.1f", key="drawn_revolver_m")
    st.sidebar.number_input("Unrestricted Cash ($M)", min_value=0.0, step=0.5, format="%.1f", key="unrestricted_cash_m")
    st.sidebar.number_input("Baseline Initial Margin ($M)", min_value=0.0, step=0.1, format="%.1f", key="baseline_im_m")
    st.sidebar.radio(
        "Single-Buyer Cap Tier", [round(c * 100, 1) for c in CAP_TIER_OPTIONS], key="cap_tier",
        format_func=lambda v: f"{v:.1f}%", horizontal=True,
    )
    st.sidebar.slider("IM Step-Up Trigger (%)", 5, 20, step=1, key="im_trigger_pct")
    st.sidebar.slider("IM Step-Up Add-On (%)", 0, 50, step=1, key="im_addon_pct")
    ss = st.session_state
    payload = {
        "facility_limit": float(ss["facility_limit_m"]) * 1e6,
        "drawn_revolver": float(ss["drawn_revolver_m"]) * 1e6,
        "unrestricted_cash": float(ss["unrestricted_cash_m"]) * 1e6,
        "baseline_initial_margin": float(ss["baseline_im_m"]) * 1e6,
        "concentration_cap_pct": float(ss["cap_tier"]) / 100.0,
        "im_stepup_trigger": float(ss["im_trigger_pct"]) / 100.0,
        "im_stepup_fraction": float(ss["im_addon_pct"]) / 100.0,
    }
    try:
        return FacilityTerms.model_validate(payload)
    except ValidationError as exc:
        st.sidebar.warning(f"Invalid facility terms ({exc.errors()[0]['msg']}); using calibrated defaults.")
        return DEFAULT_FACILITY_TERMS


def sidebar_shock_controls() -> tuple[float, dict[str, float] | None]:
    side_section(3, "Stress Shock Controls")
    st.sidebar.slider("Parallel Price Shock (%)", -20, 20, step=1, key="parallel_shock")
    st.sidebar.selectbox("Scenario Preset", PRESET_NAMES, key="preset_name")
    st.sidebar.button("Apply Preset", width="stretch", on_click=apply_preset, key="apply_preset_btn")
    override = st.sidebar.checkbox("Enable Per-Commodity Shock Overrides", key="override_enabled")
    parallel = st.session_state["parallel_shock"] / 100.0
    if not override:
        return parallel, None
    per: dict[str, float] = {}
    for t in REQUIRED_TICKERS:
        st.sidebar.slider(f"{COMMODITY_SPECS[t].name} ({t}) Shock (%)", -30, 30, step=1, key=f"shock_{t}")
        per[t] = st.session_state[f"shock_{t}"] / 100.0
    return parallel, per


def sidebar_reset() -> None:
    side_section(4, "Reset")
    st.sidebar.button(
        "Reset All to Calibrated Defaults", width="stretch", on_click=reset_to_defaults, key="reset_btn"
    )



def chart(fig, key: str) -> None:
    with st.container(border=True):
        st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONFIG, key=key)


def _regime_label(regime) -> str:
    return str(regime).replace("_", " ")


def main() -> None:
    st.set_page_config(
        page_title="Commodity Collateral & Liquidity Stress Workbench",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()
    init_session_state()

    sidebar_brand()
    market = sidebar_market_feed()
    terms = sidebar_facility_terms()
    parallel_shock, per_commodity = sidebar_shock_controls()
    sidebar_reset()
    is_custom = per_commodity is not None
    ss = st.session_state

    header = st.container()
    (tab_overview, tab_cert, tab_inv, tab_ar, tab_stress, tab_heat, tab_mc, tab_replay) = st.tabs(
        [
            ":material/dashboard: Overview",
            ":material/receipt_long: BB Certificate",
            ":material/inventory_2: Inventory",
            ":material/account_balance: Receivables",
            ":material/bolt: Stress Sweep",
            ":material/grid_on: Heatmap",
            ":material/ssid_chart: Monte Carlo LaR",
            ":material/history: Replay",
        ]
    )
    nonce = ss.editor_nonce

    with tab_inv:
        section_head("Physical Inventory & Hedge Book",
                     "Edit Physical Qty and Hedged Qty; calculated columns and every downstream view update instantly.")
        positions, inv_errors, inv_edited = inventory_editor(ss["inventory_df"], market.snapshot, nonce)
        ss["inventory_edited_df"] = inv_edited
        for msg in inv_errors:
            st.warning(msg)

    with tab_ar:
        section_head("Trade Receivables Ledger",
                     "Edit, add or delete buyer exposures. Rows failing validation are excluded and reported.")
        receivables, ar_errors, ar_edited = receivables_editor(ss["receivables_df"], nonce)
        ss["receivables_edited_df"] = ar_edited
        for msg in ar_errors:
            st.warning(msg)

    with tab_mc:
        section_head("Monte Carlo Liquidity-at-Risk",
                     "Correlated lognormal price moves drawn from the historical covariance (Cholesky), "
                     "each revalued through the full borrowing base and margin engine.")
        c_h, c_n = st.columns([1, 2])
        with c_h:
            st.segmented_control("Horizon (trading days)", MC_HORIZONS, key="mc_horizon",
                                 format_func=lambda d: f"{d}d")
        with c_n:
            st.select_slider("Simulated paths", MC_SIM_OPTIONS, key="mc_sims", format_func=lambda n: f"{n:,}")
    horizon = int(ss.get("mc_horizon") or 5)
    n_sims = int(ss.get("mc_sims") or 5000)

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
    ss["active_regime"] = str(stress.regime)

    if is_custom:
        mode_label = "Per-commodity: " + ", ".join(f"{t} {v:+.0%}" for t, v in per_commodity.items())
        same_metals = per_commodity["GC=F"] == per_commodity["HG=F"]
        same_softs = per_commodity["LBR=F"] == per_commodity["CC=F"]
        active_point = (per_commodity["GC=F"], per_commodity["LBR=F"]) if same_metals and same_softs else None
    else:
        mode_label = f"Parallel shock {parallel_shock:+.0%}"
        active_point = (parallel_shock, parallel_shock)

    with header:
        render_hero(market, terms, mode_label, *market_data_status(market))
        render_regime_banner(stress, mode_label)
        methodology_expander()

    regime_tone = REGIME_TONES[stress.regime]
    availability = min(terms.facility_limit, bb.final_borrowing_base)
    usage = terms.drawn_revolver / availability if availability > 0 else 1.0
    usage_tone = "emerald" if usage < 0.85 else ("amber" if usage < 0.95 else "red")
    be_up = sweep.upside_squeeze_breakeven_pct
    be_down = sweep.downside_deficiency_breakeven_pct
    replay_bb = replay.frame["borrowing_base"].tail(60).tolist() if replay is not None else None
    replay_hr = replay.frame["headroom"].tail(60).tolist() if replay is not None else None

    with tab_overview:
        render_kpi_cards(
            [
                {"label": "Final Borrowing Base", "value": fmt_millions(bb.final_borrowing_base), "tone": "blue",
                 "sub": f"Inventory {fmt_millions(bb.eligible_inventory)} + AR {fmt_millions(bb.final_eligible_receivables)}",
                 "spark": replay_bb},
                {"label": "Available Headroom", "value": fmt_millions(bb.available_headroom),
                 "tone": "emerald" if bb.available_headroom > 0 else "red",
                 "sub": f"Limit {fmt_millions(terms.facility_limit)} · drawn {fmt_millions(terms.drawn_revolver)}",
                 "spark": replay_hr},
                {"label": "Availability Usage", "value": fmt_pct(usage, 1), "tone": usage_tone,
                 "sub": f"Coverage {bb.collateral_coverage_ratio:.2f}x drawn",
                 "chip": "TIGHT" if usage >= 0.95 else ("WATCH" if usage >= 0.85 else "COMFORTABLE")},
                {"label": "Active Stress Net Buffer", "value": fmt_signed_millions(stress.net_liquidity_buffer),
                 "tone": regime_tone, "chip": _regime_label(stress.regime), "sub": mode_label,
                 "spark": [p.net_liquidity_buffer for p in sweep.points]},
                {"label": f"{mc.horizon_days}-Day LaR (95%)", "value": fmt_signed_millions(mc.p05_net_buffer),
                 "tone": "violet" if mc.p05_net_buffer >= 0 else "red",
                 "sub": f"P(squeeze) {fmt_pct(mc.prob_critical, 1)} · P(trap) {fmt_pct(mc.prob_margin_trap, 1)}"},
                {"label": "Squeeze Breakeven", "value": fmt_pct(be_up) if be_up is not None else "None ≤ +20%",
                 "tone": "amber",
                 "sub": f"Deficiency at {fmt_pct(be_down) if be_down is not None else 'none ≥ −20%'}"},
            ]
        )
        col_g, col_d = st.columns(2)
        with col_g:
            chart(build_facility_gauge(bb), "chart_gauge")
        with col_d:
            chart(build_collateral_mix_donut(bb), "chart_mix")
        insights = [(regime_tone, f"<b>{_regime_label(stress.regime)}</b>: {stress.regime_explanation}")]
        cap_point = next((p for p in sweep.points if p.shock_pct > 0 and p.shocked_bb >= terms.facility_limit), None)
        if cap_point is not None:
            insights.append(("amber", f"Headroom <b>caps out at {cap_point.shock_pct:+.0%}</b>: above it the "
                                      "borrowing base exceeds the facility limit, so rallies add VM calls but no "
                                      "new availability."))
        if bb.capped_buyer_names:
            tier_gain = tiers[-1].final_borrowing_base - bb.final_borrowing_base
            insights.append(("blue", f"<b>{len(bb.capped_buyer_names)} buyer(s) capped</b> "
                                     f"({fmt_millions(bb.concentration_cutback_dollars)} cutback); the "
                                     f"{tiers[-1].cap_pct:.1%} tier would add {fmt_millions(tier_gain)} of BB."))
        insights.append(("violet", f"<b>{mc.horizon_days}-day LaR 95%</b> of {fmt_signed_millions(mc.p05_net_buffer)}; "
                                   f"{fmt_pct(mc.prob_critical, 1)} of {mc.n_sims:,} correlated paths end in a squeeze."))
        insights.append(("red", f"<b>{fmt_pct(grid.squeeze_share, 0)}</b> of mixed metals/softs scenarios are "
                                f"CRITICAL; worst at metals {grid.worst_metals_shock:+.0%} / softs "
                                f"{grid.worst_softs_shock:+.0%} ({fmt_signed_millions(grid.worst_net_buffer)})."))
        if replay is not None:
            insights.append(("orange", f"Replay since {replay.start_date}: BB below drawn on "
                                       f"<b>{replay.deficiency_days} of {replay.n_days} days</b>; worst 5-day VM call "
                                       f"{fmt_millions(replay.max_5d_vm_outflow)}."))
        render_insights("Treasury Briefing", insights)
        chart(build_regime_ribbon(sweep, None if is_custom else parallel_shock), "chart_ribbon")

    with tab_cert:
        render_kpi_cards(
            [
                {"label": "Final Borrowing Base", "value": fmt_millions(bb.final_borrowing_base), "tone": "blue",
                 "sub": "Eligible inventory + capped eligible AR"},
                {"label": "Facility Limit", "value": fmt_millions(terms.facility_limit), "tone": "slate",
                 "sub": f"Drawn {fmt_millions(terms.drawn_revolver)}"},
                {"label": "Available Headroom", "value": fmt_millions(bb.available_headroom),
                 "tone": "emerald" if bb.available_headroom > 0 else "red",
                 "sub": "max(0, min(limit, BB) − drawn)"},
                {"label": "Revolver Utilization", "value": fmt_pct(bb.facility_utilization_pct), "tone": usage_tone,
                 "sub": "Drawn ÷ facility limit"},
                {"label": "Active Stress Net Buffer", "value": fmt_signed_millions(stress.net_liquidity_buffer),
                 "tone": regime_tone, "chip": _regime_label(stress.regime)},
            ]
        )
        chart(build_borrowing_base_waterfall(bb), "chart_waterfall")
        col_table, col_audit = st.columns([3, 2])
        with col_table:
            section_head("Borrowing Base Reconciliation", "Every bridge step reconciles to the final base within $1e-6.")
            recon = pd.DataFrame(
                [{"Line": w.label, "Amount ($M)": w.amount / 1e6, "Running Total ($M)": w.running_total / 1e6}
                 for w in bb.waterfall]
            )
            st.dataframe(
                recon, width="stretch", hide_index=True,
                column_config={
                    "Amount ($M)": st.column_config.NumberColumn(format="%.2f"),
                    "Running Total ($M)": st.column_config.NumberColumn(format="%.2f"),
                },
            )
        with col_audit:
            covenant_audit_card(bb)
            st.write("")
            certificate_download_buttons(bb, stress, market)

    with tab_inv:
        col_hc, col_vol = st.columns(2)
        with col_hc:
            chart(build_haircut_decomposition_chart(bb.inventory_items), "chart_haircuts")
        with col_vol:
            chart(build_volatility_history_chart(market.vol_df, COMMODITY_SPECS), "chart_vol")
        haircut_methodology_expander()

    with tab_ar:
        col_chart, col_side = st.columns([3, 2])
        with col_chart:
            chart(build_receivables_concentration_chart(bb), "chart_ar")
        with col_side:
            section_head("Concentration Cap Tier Sensitivity", "Borrowing base re-solved under each cap tier.")
            tier_df = pd.DataFrame(
                [
                    {
                        "Tier": f"{t.cap_pct:.1%}",
                        "BB ($M)": t.final_borrowing_base / 1e6,
                        "Cutback ($M)": t.total_cutback_dollars / 1e6,
                        "Capped": t.capped_buyer_count,
                    }
                    for t in tiers
                ]
            )
            st.dataframe(
                tier_df, width="stretch", hide_index=True,
                column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ["BB ($M)", "Cutback ($M)"]},
            )
            solver_audit_card(bb)

    with tab_stress:
        render_kpi_cards(
            [
                {"label": "Shocked BB", "value": fmt_millions(stress.shocked_bb), "tone": "blue",
                 "sub": f"{fmt_signed_millions(stress.bb_change)} vs. baseline"},
                {"label": "Shocked Headroom", "value": fmt_millions(stress.shocked_headroom),
                 "tone": "emerald" if stress.headroom_change >= 0 else "orange",
                 "sub": f"{fmt_signed_millions(stress.headroom_change)} change"},
                {"label": "Net Futures VM", "value": fmt_signed_millions(stress.vm_inflow - stress.vm_outflow),
                 "tone": "red" if stress.vm_outflow > 0 else "emerald",
                 "sub": f"Call {fmt_millions(stress.vm_outflow)} · inflow {fmt_millions(stress.vm_inflow)}"},
                {"label": "IM Step-Up + BB Cure",
                 "value": fmt_millions(stress.im_addon_outflow + stress.bb_deficiency_cure), "tone": "amber",
                 "sub": f"IM {fmt_millions(stress.im_addon_outflow)} · cure {fmt_millions(stress.bb_deficiency_cure)}"},
                {"label": "Net Liquidity Buffer", "value": fmt_signed_millions(stress.net_liquidity_buffer),
                 "tone": regime_tone, "chip": _regime_label(stress.regime),
                 "spark": [p.net_liquidity_buffer for p in sweep.points]},
            ]
        )
        chart(build_stress_sweep_chart(sweep, parallel_shock, is_custom), "chart_sweep")
        st.caption(
            f"Upside squeeze breakeven: {fmt_pct(be_up) if be_up is not None else 'none in range'} · "
            f"Downside deficiency breakeven: {fmt_pct(be_down) if be_down is not None else 'none in range'} · "
            "Regime counts across 41 shocks: "
            + ", ".join(f"{_regime_label(k).title()} {v}" for k, v in sweep.regime_counts.items())
        )
        chart(build_active_shock_sources_uses_chart(stress, terms), "chart_sources_uses")
        with st.expander("Full 41-Point Stress Sweep Table"):
            sweep_df = sweep.to_dataframe()
            st.dataframe(sweep_df, width="stretch", hide_index=True)
            st.download_button(
                "Download Stress Sweep CSV", data=sweep_df.to_csv(index=False),
                file_name=f"stress_sweep_{market.as_of_date.replace('-', '')}.csv",
                mime="text/csv", width="stretch", key="dl_sweep",
            )
        stress_methodology_expander()

    with tab_heat:
        section_head("Mixed-Shock Scenario Heatmap",
                     "Gold & Copper move together on the vertical axis, Lumber & Cocoa on the horizontal; "
                     "the dashed line marks where the net liquidity buffer hits $0.")
        active_nlb = None
        if active_point is not None and active_point[0] in grid.metals_axis and active_point[1] in grid.softs_axis:
            i, j = grid.metals_axis.index(active_point[0]), grid.softs_axis.index(active_point[1])
            active_nlb = grid.net_buffer[i][j]
        render_kpi_cards(
            [
                {"label": "Scenarios in Critical Squeeze", "value": fmt_pct(grid.squeeze_share, 1), "tone": "red",
                 "sub": f"of {len(grid.metals_axis) ** 2} grid points"},
                {"label": "Worst Grid Scenario", "value": fmt_signed_millions(grid.worst_net_buffer), "tone": "orange",
                 "sub": f"Metals {grid.worst_metals_shock:+.1%} · softs {grid.worst_softs_shock:+.1%}"},
                {"label": "Active Scenario", "value": fmt_signed_millions(stress.net_liquidity_buffer),
                 "tone": regime_tone, "chip": _regime_label(stress.regime),
                 "sub": "On grid" if active_nlb is not None else "Off the 2.5% grid (exact engine value)"},
            ]
        )
        chart(build_shock_heatmap(grid, active_point), "chart_heatmap")

    with tab_mc:
        render_kpi_cards(
            [
                {"label": "LaR 95% (5th pct buffer)", "value": fmt_signed_millions(mc.p05_net_buffer),
                 "tone": "violet" if mc.p05_net_buffer >= 0 else "red",
                 "sub": f"{mc.horizon_days}-day horizon"},
                {"label": "Expected Shortfall 95%", "value": fmt_signed_millions(mc.es95_net_buffer),
                 "tone": "orange", "sub": "Mean of worst 5% paths"},
                {"label": "P(Critical Squeeze)", "value": fmt_pct(mc.prob_critical, 2), "tone": "red",
                 "sub": f"{mc.regime_counts['CRITICAL_SQUEEZE']:,} paths"},
                {"label": "P(Margin Trap)", "value": fmt_pct(mc.prob_margin_trap, 1), "tone": "amber",
                 "sub": f"{mc.regime_counts['MARGIN_TRAP_WARNING']:,} paths"},
                {"label": "P(BB Deficiency)", "value": fmt_pct(mc.prob_deficiency, 1), "tone": "orange",
                 "sub": f"{mc.regime_counts['BB_DEFICIENCY_WARNING']:,} paths"},
            ]
        )
        chart(build_mc_distribution_chart(mc), "chart_mc")
        col_corr, col_vols = st.columns([1.2, 1])
        with col_corr:
            chart(build_correlation_heatmap(mc, COMMODITY_SPECS), "chart_corr")
        with col_vols:
            section_head("Horizon Volatility & Worst Path",
                         f"Calibrated on the last {mc.lookback_days} daily returns of the active feed.")
            vol_df = pd.DataFrame(
                [
                    {"Commodity": COMMODITY_SPECS[t].name,
                     f"{mc.horizon_days}d σ": f"{mc.horizon_vols[t]:.2%}",
                     "Worst-path move": f"{mc.worst_scenario_shocks[t]:+.2%}"}
                    for t in REQUIRED_TICKERS
                ]
            )
            st.dataframe(vol_df, width="stretch", hide_index=True)
            st.caption("The calibrated sample draws each commodity independently, so correlations are near zero; "
                       "switch to the live feed for real cross-commodity co-movement.")

    with tab_replay:
        section_head("Historical Replay",
                     "Today's quantities, receivables and drawn balance replayed through each historical day's "
                     "prices and 30-day volatilities.")
        if replay is None:
            st.info("Not enough price and volatility history in the active feed to run a replay.")
        else:
            render_kpi_cards(
                [
                    {"label": "Days in BB Deficiency", "value": f"{replay.deficiency_days} / {replay.n_days}",
                     "tone": "orange" if replay.deficiency_days else "emerald",
                     "sub": f"{replay.start_date} → {replay.end_date}"},
                    {"label": "Minimum Headroom", "value": fmt_millions(replay.min_headroom), "tone": "blue",
                     "sub": f"on {replay.min_headroom_date}", "spark": replay.frame["headroom"].tolist()},
                    {"label": "Worst 5-Day VM Call", "value": fmt_millions(replay.max_5d_vm_outflow), "tone": "red",
                     "sub": f"window ending {replay.max_5d_vm_outflow_date}"},
                    {"label": "Lowest Stressed Buffer", "value": fmt_signed_millions(replay.min_stressed_buffer),
                     "tone": "red" if replay.min_stressed_buffer < 0 else "emerald",
                     "sub": f"on {replay.min_stressed_buffer_date}",
                     "spark": replay.frame["stressed_buffer"].tolist()},
                ]
            )
            chart(build_replay_chart(replay, terms), "chart_replay")
            with st.expander("Daily Replay Table"):
                replay_df = replay.frame.reset_index()
                st.dataframe(replay_df, width="stretch", hide_index=True)
                st.download_button(
                    "Download Replay CSV", data=replay_df.to_csv(index=False),
                    file_name=f"historical_replay_{market.as_of_date.replace('-', '')}.csv",
                    mime="text/csv", width="stretch", key="dl_replay",
                )


main()
