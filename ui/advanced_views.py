"""Advanced model presentation: settings widgets, validated margin and basis editors, dashboards, ledger,
cash-flow Monte Carlo, scenario comparison and saved configurations.

Everything here is presentation and session state. Calculations come from core/ (margin_engine,
basis_risk, advanced_scenario, cashflow_engine, market_calibration, scenario_comparison).
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from core.advanced_schemas import (
    AdvancedScenarioResult,
    BasisAssumption,
    CashflowSettings,
    MarginProfile,
    MarginSettings,
    ReconciliationStep,
    ScenarioSpec,
)
from core.advanced_scenario import MARGIN_PRESETS
from core.basis_risk import BASIS_PRESETS
from core.margin_engine import default_margin_settings
from core.scenario_comparison import (
    COMPARISON_METRICS,
    ComparisonColumn,
    ConfigError,
    ScenarioState,
    SimulationState,
    WorkbenchConfig,
    dump_config,
    load_config,
)
from core.schemas import (
    CAP_TIER_OPTIONS,
    COMMODITY_SPECS,
    REQUIRED_TICKERS,
    FacilityTerms,
    InventoryPosition,
    ReceivableItem,
)
from ui.components import render_audit_rows
from ui.theme import fmt_millions, fmt_pct, fmt_signed_millions, money_divisor, money_unit, regime_label

MODEL_LEGACY = "legacy"
MODEL_ADVANCED = "advanced"
MODEL_LABELS = {MODEL_LEGACY: "Legacy", MODEL_ADVANCED: "Advanced"}
CONTRACT_POLICY_LABELS = {
    "nearest": "Nearest whole contract", "floor": "Round down", "ceil": "Round up",
    "continuous": "Fractional (legacy)",
}
IM_MODE_LABELS = {"per_contract": "Per contract (exchange style)", "legacy_stepup": "Baseline plus step-up (legacy)"}
SIM_METHOD_LABELS = {"gaussian": "Gaussian (calibrated covariance)", "bootstrap": "Historical block bootstrap",
                     "stress": "Stressed covariance (vol x1.5, correlation 0.6)"}
CALIB_WINDOWS = ["60 days", "125 days", "250 days", "Maximum"]
CF_MC_PATHS = [500, 1000, 2000, 5000]

ADVANCED_WIDGET_DEFAULTS: dict[str, object] = {
    "model_mode": MODEL_LEGACY,
    "apply_trigger_rule": True,
    "contract_policy": "nearest",
    "im_mode": "per_contract",
    "release_excess_im": True,
    "margin_preset": next(iter(MARGIN_PRESETS)),
    "basis_preset": next(iter(BASIS_PRESETS)),
    "ml_cut_pct": 0,
    "calib_window": "250 days",
    "sim_method": "gaussian",
    "basis_vol": 0,
    "bt_window": 125,
    "cf_horizon": 20,
    "cf_ramp": 5,
    "cf_rev_rate": 7.0,
    "cf_ml_rate": 8.0,
    "cf_day_count": 360,
    "cf_rev_delay": 1,
    "cf_ml_delay": 0,
    "cf_cure_grace": 2,
    "cf_invoice_terms": 60,
    "cf_overdue_delay": 15,
    "cf_noncollected": [],
    "cf_repay_ml": True,
    "cf_sweep": True,
    "cf_min_cash_m": 0.0,
    "cf_ml_cut_day": 1,
    "cf_mc_paths": 2000,
}

MARGIN_COLUMNS = ["Ticker", "Commodity", "IM per contract", "Maintenance per contract", "Fixed add-on",
                  "Financeable", "Stress IM multiplier", "Source", "Source date"]
BASIS_COLUMNS = ["Ticker", "Commodity", "Opening basis (%)", "Basis shock (%)", "Liquidation discount (%)"]


def _py(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


# ---- Default frames ---------------------------------------------------------------------------------------

def margin_df_from_settings(margin: MarginSettings, multipliers: Mapping[str, float] | None = None) -> pd.DataFrame:
    mult = multipliers or {}
    return pd.DataFrame([{
        "Ticker": t, "Commodity": COMMODITY_SPECS[t].name,
        "IM per contract": margin.profiles[t].im_per_contract,
        "Maintenance per contract": margin.profiles[t].maintenance_per_contract,
        "Fixed add-on": margin.profiles[t].fixed_addon_usd,
        "Financeable": margin.profiles[t].financeable,
        "Stress IM multiplier": float(mult.get(t, 1.0)),
        "Source": margin.profiles[t].source, "Source date": margin.profiles[t].source_date,
    } for t in REQUIRED_TICKERS])


def default_margin_df() -> pd.DataFrame:
    return margin_df_from_settings(default_margin_settings())


def basis_df_from_values(basis: Mapping[str, BasisAssumption], shocks: Mapping[str, float] | None = None,
                         discounts: Mapping[str, float] | None = None) -> pd.DataFrame:
    shocks, discounts = shocks or {}, discounts or {}
    return pd.DataFrame([{
        "Ticker": t, "Commodity": COMMODITY_SPECS[t].name,
        "Opening basis (%)": basis[t].opening_basis_pct * 100,
        "Basis shock (%)": float(shocks.get(t, 0.0)) * 100,
        "Liquidation discount (%)": float(discounts.get(t, 0.0)) * 100,
    } for t in REQUIRED_TICKERS])


def default_basis_df() -> pd.DataFrame:
    return basis_df_from_values({t: BasisAssumption(ticker=t) for t in REQUIRED_TICKERS})


def init_advanced_state(ss: Any) -> None:
    if "margin_df" not in ss:
        ss["margin_df"] = default_margin_df()
    if "basis_df" not in ss:
        ss["basis_df"] = default_basis_df()
    for key, value in ADVANCED_WIDGET_DEFAULTS.items():
        if key not in ss:
            ss[key] = list(value) if isinstance(value, list) else value


def reset_advanced_state(ss: Any) -> None:
    for key, value in ADVANCED_WIDGET_DEFAULTS.items():
        ss[key] = list(value) if isinstance(value, list) else value
    ss["margin_df"] = default_margin_df()
    ss["basis_df"] = default_basis_df()
    for key in ("compare_alt", "terms_base", "config_message", "config_error"):
        ss.pop(key, None)


def commit_pending_edits(ss: Any) -> None:
    """Make the current editor contents canonical (used before any callback that bumps the editor nonce)."""
    for edited, canonical in (("inventory_edited_df", "inventory_df"), ("receivables_edited_df", "receivables_df"),
                              ("margin_edited_df", "margin_df"), ("basis_edited_df", "basis_df")):
        if edited in ss:
            ss[canonical] = ss[edited].copy()


# ---- Callbacks --------------------------------------------------------------------------------------------

def _set_shocks(ss: Any, shocks: Mapping[str, float]) -> None:
    values = [int(round(float(shocks.get(t, 0.0)) * 100)) for t in REQUIRED_TICKERS]
    if len(set(values)) == 1 and -20 <= values[0] <= 20:
        ss["override_enabled"] = False
        ss["parallel_shock"] = values[0]
    else:
        ss["override_enabled"] = True
    for t, v in zip(REQUIRED_TICKERS, values):
        ss[f"shock_{t}"] = max(-30, min(30, v))


def apply_margin_preset() -> None:
    ss = st.session_state
    commit_pending_edits(ss)
    preset = MARGIN_PRESETS[ss["margin_preset"]]
    if "futures_shocks" in preset:
        _set_shocks(ss, preset["futures_shocks"])
    df = ss["margin_df"].copy()
    df["Stress IM multiplier"] = [float(preset["im_multipliers"].get(t, 1.0)) for t in df["Ticker"]]
    ss["margin_df"] = df
    ss["ml_cut_pct"] = int(round(preset.get("margin_line_cut_pct", 0.0) * 100))
    ss.editor_nonce += 1


def apply_basis_preset() -> None:
    ss = st.session_state
    commit_pending_edits(ss)
    preset = BASIS_PRESETS[ss["basis_preset"]]
    df = ss["basis_df"].copy()
    df["Basis shock (%)"] = [float(preset["basis_shocks"].get(t, 0.0)) * 100 for t in df["Ticker"]]
    df["Liquidation discount (%)"] = [float(preset["liquidation_discounts"].get(t, 0.0)) * 100 for t in df["Ticker"]]
    ss["basis_df"] = df
    ss.editor_nonce += 1


def save_active_as_alternative() -> None:
    ss = st.session_state
    if "active_spec_json" in ss:
        spec = ScenarioSpec.model_validate_json(ss["active_spec_json"])
        ss["compare_alt"] = spec.model_copy(update={"name": "Alternative (saved)"}).model_dump_json()


def clear_alternative() -> None:
    st.session_state.pop("compare_alt", None)


# ---- Editors ----------------------------------------------------------------------------------------------

def margin_editor(baseline_df: pd.DataFrame, nonce: int) -> tuple[dict[str, MarginProfile], dict[str, float],
                                                                    list[str], pd.DataFrame]:
    """Validated fixed-row margin profile editor. Edited IM figures are marked as user assumptions."""
    edited = st.data_editor(
        baseline_df, key=f"margin_editor_{nonce}", width="stretch", num_rows="fixed", hide_index=True,
        disabled=["Ticker", "Commodity", "Source", "Source date"],
        column_config={
            "IM per contract": st.column_config.NumberColumn("IM per contract (USD)", min_value=0.0, format="localized"),
            "Maintenance per contract": st.column_config.NumberColumn("Maintenance (USD)", min_value=0.0,
                                                                      format="localized"),
            "Fixed add-on": st.column_config.NumberColumn("Fixed add-on (USD)", min_value=0.0, format="localized"),
            "Financeable": st.column_config.CheckboxColumn("Line can fund"),
            "Stress IM multiplier": st.column_config.NumberColumn("Stress IM multiplier", min_value=0.0,
                                                                  max_value=10.0, step=0.05, format="%.2f"),
        },
    )
    defaults = default_margin_settings().profiles
    profiles: dict[str, MarginProfile] = {}
    multipliers: dict[str, float] = {}
    errors: list[str] = []
    for idx, row in edited.iterrows():
        t = str(row["Ticker"])
        d = defaults[t]
        payload = {
            "ticker": t, "im_per_contract": _py(row["IM per contract"]),
            "maintenance_per_contract": _py(row["Maintenance per contract"]),
            "fixed_addon_usd": _py(row["Fixed add-on"]) or 0.0, "financeable": bool(_py(row["Financeable"])),
            "source": d.source, "source_date": d.source_date, "assumption_type": d.assumption_type,
        }
        changed = (payload["im_per_contract"] != d.im_per_contract
                   or payload["maintenance_per_contract"] != d.maintenance_per_contract
                   or payload["fixed_addon_usd"] != d.fixed_addon_usd)
        if changed:
            payload.update(source="User input", source_date="", assumption_type="user")
        try:
            profiles[t] = MarginProfile.model_validate(payload)
        except ValidationError as exc:
            errors.append(f"{COMMODITY_SPECS[t].name} margin: {exc.errors()[0]['msg']}. Using the illustrative default.")
            profiles[t] = d
        mult = _py(row["Stress IM multiplier"])
        if mult is None or not 0.0 <= float(mult) <= 10.0:
            errors.append(f"{COMMODITY_SPECS[t].name} stress IM multiplier must be between 0 and 10. Using 1.0.")
            mult = 1.0
        multipliers[t] = float(mult)
    return profiles, multipliers, errors, edited.loc[:, MARGIN_COLUMNS]


def basis_editor(baseline_df: pd.DataFrame, nonce: int) -> tuple[dict[str, BasisAssumption], dict[str, float],
                                                                   dict[str, float], list[str], pd.DataFrame]:
    """Validated fixed-row basis editor (percent of the futures price)."""
    edited = st.data_editor(
        baseline_df, key=f"basis_editor_{nonce}", width="stretch", num_rows="fixed", hide_index=True,
        disabled=["Ticker", "Commodity"],
        column_config={
            "Opening basis (%)": st.column_config.NumberColumn(min_value=-50.0, max_value=50.0, step=0.5,
                                                               format="%+.2f%%"),
            "Basis shock (%)": st.column_config.NumberColumn(min_value=-50.0, max_value=50.0, step=0.5,
                                                             format="%+.2f%%"),
            "Liquidation discount (%)": st.column_config.NumberColumn(min_value=0.0, max_value=50.0, step=0.5,
                                                                      format="%.2f%%"),
        },
    )
    basis: dict[str, BasisAssumption] = {}
    shocks: dict[str, float] = {}
    discounts: dict[str, float] = {}
    errors: list[str] = []
    for _, row in edited.iterrows():
        t = str(row["Ticker"])
        name = COMMODITY_SPECS[t].name
        try:
            basis[t] = BasisAssumption.model_validate(
                {"ticker": t, "opening_basis_pct": (_py(row["Opening basis (%)"]) or 0.0) / 100})
        except ValidationError as exc:
            errors.append(f"{name} opening basis: {exc.errors()[0]['msg']}. Using 0%.")
            basis[t] = BasisAssumption(ticker=t)
        shock = (_py(row["Basis shock (%)"]) or 0.0) / 100
        disc = (_py(row["Liquidation discount (%)"]) or 0.0) / 100
        if not -0.5 <= shock <= 0.5:
            errors.append(f"{name} basis shock must be between -50% and +50%. Using 0%.")
            shock = 0.0
        if not 0.0 <= disc <= 0.5:
            errors.append(f"{name} liquidation discount must be between 0% and 50%. Using 0%.")
            disc = 0.0
        shocks[t], discounts[t] = shock, disc
    return basis, shocks, discounts, errors, edited.loc[:, BASIS_COLUMNS]


# ---- Settings panels --------------------------------------------------------------------------------------

def render_margin_settings(nonce: int) -> tuple[MarginSettings, dict[str, float], list[str]]:
    ss = st.session_state
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.selectbox("Hedge contracts", list(CONTRACT_POLICY_LABELS), key="contract_policy",
                     format_func=lambda k: CONTRACT_POLICY_LABELS[k],
                     help="Exchange hedges trade in whole contracts. The unhedged residual stays in the inventory.")
    with c2:
        st.selectbox("Initial margin model", list(IM_MODE_LABELS), key="im_mode",
                     format_func=lambda k: IM_MODE_LABELS[k])
    with c3:
        st.slider("Margin line reduction (%)", 0, 100, step=1, key="ml_cut_pct",
                  help="Share of the margin financing line withdrawn by the lender in the scenario.")
    c4, c5 = st.columns(2, gap="medium")
    with c4:
        st.checkbox("Release initial margin above the new requirement", key="release_excess_im")
    with c5:
        st.checkbox("Apply the step-up trigger per commodity", key="apply_trigger_rule",
                    help="Multiplies a commodity's IM by (1 + step-up size) when its own move reaches the "
                         "sidebar trigger, on top of the stress multiplier below.")
    p1, p2 = st.columns([3, 1], gap="medium", vertical_alignment="bottom")
    with p1:
        st.selectbox("Margin stress preset (illustrative)", list(MARGIN_PRESETS), key="margin_preset")
    with p2:
        st.button("Apply margin preset", on_click=apply_margin_preset, width="stretch", key="apply_margin_preset_btn")
    profiles, multipliers, errors, edited = margin_editor(ss["margin_df"], nonce)
    ss["margin_edited_df"] = edited
    st.caption("Default IM figures are illustrative (a fixed share of contract notional at the bundled sample "
               "price, 2026-11-20). They are not current exchange requirements. Edited rows become user inputs.")
    settings = MarginSettings(profiles=profiles, contract_policy=ss["contract_policy"], im_mode=ss["im_mode"],
                              release_excess_im=bool(ss["release_excess_im"]))
    return settings, multipliers, errors


def render_basis_settings(nonce: int) -> tuple[dict[str, BasisAssumption], dict[str, float], dict[str, float],
                                               list[str]]:
    ss = st.session_state
    p1, p2 = st.columns([3, 1], gap="medium", vertical_alignment="bottom")
    with p1:
        st.selectbox("Basis scenario preset (illustrative)", list(BASIS_PRESETS), key="basis_preset")
    with p2:
        st.button("Apply basis preset", on_click=apply_basis_preset, width="stretch", key="apply_basis_preset_btn")
    basis, shocks, discounts, errors, edited = basis_editor(ss["basis_df"], nonce)
    ss["basis_edited_df"] = edited
    st.caption("Basis = physical price minus futures price, as a percent of futures. Opening basis prices the "
               "inventory; the basis shock moves physical independently of futures; the liquidation discount "
               "applies to collateral value only. All values are user assumptions, not observed basis.")
    return basis, shocks, discounts, errors


def render_cashflow_settings(receivables: Sequence[ReceivableItem]) -> tuple[CashflowSettings, list[str]]:
    ss = st.session_state
    names = sorted({r.buyer_name for r in receivables})
    ss["cf_noncollected"] = [n for n in ss.get("cf_noncollected", []) if n in names]
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.number_input("Horizon (business days)", 1, 60, step=1, key="cf_horizon")
        st.number_input("Price move spread over (days)", 1, 60, step=1, key="cf_ramp",
                        help="The scenario move is reached linearly over this many days, then held.")
    with c2:
        st.number_input("Revolver rate (% a year)", 0.0, 50.0, step=0.25, format="%.2f", key="cf_rev_rate")
        st.number_input("Margin line rate (% a year)", 0.0, 50.0, step=0.25, format="%.2f", key="cf_ml_rate")
        st.radio("Day count", [360, 365], key="cf_day_count", horizontal=True, format_func=lambda d: f"ACT/{d}")
    with c3:
        st.number_input("Revolver funding delay (days)", 0, 5, step=1, key="cf_rev_delay")
        st.number_input("Margin line funding delay (days)", 0, 5, step=1, key="cf_ml_delay")
        st.number_input("Deficiency cure grace (days)", 0, 10, step=1, key="cf_cure_grace")
    with c4:
        st.number_input("Invoice terms (days)", 1, 180, step=1, key="cf_invoice_terms")
        st.number_input("Overdue collection delay (days)", 0, 180, step=1, key="cf_overdue_delay")
        st.number_input("Minimum cash (USD M)", 0.0, 1000.0, step=0.5, format="%.1f", key="cf_min_cash_m")
    c5, c6, c7 = st.columns([2, 1, 1], gap="medium")
    with c5:
        st.multiselect("Buyers that do not pay within the horizon", names, key="cf_noncollected")
    with c6:
        st.checkbox("Repay margin line from margin receipts", key="cf_repay_ml")
        st.checkbox("Sweep surplus cash to the revolver", key="cf_sweep")
    with c7:
        st.number_input("Margin line reduction from day", 1, 60, step=1, key="cf_ml_cut_day")
    payload = {
        "horizon_days": int(ss["cf_horizon"]), "ramp_days": int(ss["cf_ramp"]),
        "revolver_rate": float(ss["cf_rev_rate"]) / 100, "margin_line_rate": float(ss["cf_ml_rate"]) / 100,
        "day_count": int(ss["cf_day_count"]), "revolver_delay_days": int(ss["cf_rev_delay"]),
        "margin_line_delay_days": int(ss["cf_ml_delay"]), "cure_grace_days": int(ss["cf_cure_grace"]),
        "invoice_terms_days": int(ss["cf_invoice_terms"]), "overdue_collection_delay_days": int(ss["cf_overdue_delay"]),
        "noncollected_buyers": list(ss["cf_noncollected"]), "repay_margin_line_from_receipts": bool(ss["cf_repay_ml"]),
        "sweep_surplus_to_revolver": bool(ss["cf_sweep"]), "minimum_cash": float(ss["cf_min_cash_m"]) * 1e6,
        "margin_line_cut_pct": float(ss["ml_cut_pct"]) / 100, "margin_line_cut_day": int(ss["cf_ml_cut_day"]),
    }
    try:
        return CashflowSettings.model_validate(payload), []
    except ValidationError as exc:
        return CashflowSettings(), [f"Cash-flow settings: {exc.errors()[0]['msg']}. Using defaults."]


def cashflow_settings_from_state(receivables: Sequence[ReceivableItem]) -> CashflowSettings:
    """Same as render_cashflow_settings without drawing widgets (for callers that only need the value)."""
    ss = st.session_state
    names = {r.buyer_name for r in receivables}
    try:
        return CashflowSettings.model_validate({
            "horizon_days": int(ss["cf_horizon"]), "ramp_days": int(ss["cf_ramp"]),
            "revolver_rate": float(ss["cf_rev_rate"]) / 100, "margin_line_rate": float(ss["cf_ml_rate"]) / 100,
            "day_count": int(ss["cf_day_count"]), "revolver_delay_days": int(ss["cf_rev_delay"]),
            "margin_line_delay_days": int(ss["cf_ml_delay"]), "cure_grace_days": int(ss["cf_cure_grace"]),
            "invoice_terms_days": int(ss["cf_invoice_terms"]),
            "overdue_collection_delay_days": int(ss["cf_overdue_delay"]),
            "noncollected_buyers": [n for n in ss["cf_noncollected"] if n in names],
            "repay_margin_line_from_receipts": bool(ss["cf_repay_ml"]),
            "sweep_surplus_to_revolver": bool(ss["cf_sweep"]), "minimum_cash": float(ss["cf_min_cash_m"]) * 1e6,
            "margin_line_cut_pct": float(ss["ml_cut_pct"]) / 100, "margin_line_cut_day": int(ss["cf_ml_cut_day"]),
        })
    except ValidationError:
        return CashflowSettings()


def render_simulation_settings() -> None:
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.selectbox("Calibration window", CALIB_WINDOWS, key="calib_window")
    with c2:
        st.selectbox("Simulation method", list(SIM_METHOD_LABELS), key="sim_method",
                     format_func=lambda k: SIM_METHOD_LABELS[k])
    with c3:
        st.slider("Basis volatility (% a year)", 0, 50, step=1, key="basis_vol",
                  help="Independent random walk of each commodity's basis in the cash-flow simulation. "
                       "0 = basis fixed. An assumption; no basis history is bundled.")
    with c4:
        st.select_slider("Cash-flow simulation paths", CF_MC_PATHS, key="cf_mc_paths", format_func=lambda n: f"{n:,}")


# ---- Result panels ----------------------------------------------------------------------------------------

def _m(v: float) -> float:
    return v / money_divisor()


def margin_kpis(adv: AdvancedScenarioResult) -> list[dict[str, Any]]:
    return [
        {"label": "Variation margin, net", "value": fmt_signed_millions(adv.vm_net),
         "context": f"Paid {fmt_millions(adv.vm_payable)}, received {fmt_millions(adv.vm_receivable)}"},
        {"label": "Initial margin required", "value": fmt_millions(adv.im_required),
         "context": f"Posted {fmt_millions(adv.im_posted_opening)}; additional {fmt_millions(adv.im_additional)}, "
                    f"released {fmt_millions(adv.im_released)}"},
        {"label": "Margin cash need", "value": fmt_millions(adv.margin_cash_need),
         "context": "Variation margin paid plus additional initial margin"},
        {"label": "Margin line draw", "value": fmt_millions(adv.margin_line_draw),
         "context": f"Capacity {fmt_millions(adv.margin_line_capacity)}, eligible need "
                    f"{fmt_millions(adv.margin_line_eligible_need)}"},
        {"label": "Funded from cash", "value": fmt_millions(adv.margin_funded_from_cash),
         "context": "Margin cash need not covered by the line; met from cash and revolver headroom"},
    ]


def margin_audit_frame(adv: AdvancedScenarioResult) -> pd.DataFrame:
    u = money_divisor()
    return pd.DataFrame([{
        "Commodity": ln.name, "Hedged qty": ln.hedged_qty, "Contracts (exact)": ln.contracts_exact,
        "Contracts (traded)": abs(ln.contracts), "Residual unhedged qty": ln.residual_hedge_qty,
        "Futures price 0": ln.futures_price_0, "Futures price 1": ln.futures_price_1,
        "Physical price 1": ln.physical_price_1, "Collateral price 1": ln.collateral_price_1,
        "VM": ln.vm_cash / u, "IM per contract (USD)": ln.im_per_contract, "IM multiplier": ln.im_multiplier,
        "IM before": ln.im_required_0 / u, "IM after": ln.im_required_1 / u,
        "Line can fund": ln.financeable,
    } for ln in adv.lines])


def render_margin_audit(adv: AdvancedScenarioResult, margin: MarginSettings) -> None:
    df = margin_audit_frame(adv)
    unit = money_unit()
    st.dataframe(df, width="stretch", hide_index=True, column_config={
        "Hedged qty": st.column_config.NumberColumn(format="localized"),
        "Contracts (exact)": st.column_config.NumberColumn(format="%.2f"),
        "Contracts (traded)": st.column_config.NumberColumn(format="%.0f"),
        "Residual unhedged qty": st.column_config.NumberColumn(format="localized"),
        "Futures price 0": st.column_config.NumberColumn(format="%.4f"),
        "Futures price 1": st.column_config.NumberColumn(format="%.4f"),
        "Physical price 1": st.column_config.NumberColumn(format="%.4f"),
        "Collateral price 1": st.column_config.NumberColumn(format="%.4f"),
        "VM": st.column_config.NumberColumn(f"VM, + = received ({unit})", format="%+.3f"),
        "IM per contract (USD)": st.column_config.NumberColumn(format="localized"),
        "IM multiplier": st.column_config.NumberColumn(format="%.3f"),
        "IM before": st.column_config.NumberColumn(f"IM before ({unit})", format="%.3f"),
        "IM after": st.column_config.NumberColumn(f"IM after ({unit})", format="%.3f"),
    })
    src = pd.DataFrame([{"Commodity": COMMODITY_SPECS[t].name, "Source": p.source,
                         "Date": p.source_date or "n/a", "Type": p.assumption_type}
                        for t, p in margin.profiles.items()])
    st.caption("Margin assumption sources")
    st.dataframe(src, width="stretch", hide_index=True)
    st.caption("VM = signed contracts x contract multiplier x futures price change (shorts gain when prices fall). "
               "IM = contracts x IM per contract x multiplier + fixed add-on. Posted IM is held at the exchange and "
               "is not counted as liquidity; only the change in requirement moves cash.")


def steps_frame(steps: Sequence[ReconciliationStep]) -> pd.DataFrame:
    return pd.DataFrame([{"Step": s.label, "Net liquidity buffer": _m(s.net_liquidity_buffer),
                          "Change": None if i == 0 else _m(s.change)} for i, s in enumerate(steps)])


def render_steps_table(steps: Sequence[ReconciliationStep], total_label: str | None = None) -> None:
    df = steps_frame(steps)
    if total_label is not None and len(steps) > 1:
        df = pd.concat([df, pd.DataFrame([{"Step": total_label, "Net liquidity buffer": _m(steps[-1].net_liquidity_buffer),
                                           "Change": _m(steps[-1].net_liquidity_buffer - steps[0].net_liquidity_buffer)}])],
                       ignore_index=True)
    unit = money_unit()
    st.dataframe(df, width="stretch", hide_index=True, column_config={
        "Step": st.column_config.TextColumn(width="medium"),
        "Net liquidity buffer": st.column_config.NumberColumn(f"Buffer ({unit})", format="%+.2f", width="small"),
        "Change": st.column_config.NumberColumn("Change", format="%+.2f", width="small"),
    })


def ledger_kpis(run: Any, horizon: int) -> list[dict[str, Any]]:
    first = int(run.first_shortfall_day[0])
    return [
        {"label": "First unfunded day", "value": f"Day {first}" if first > 0 else "None",
         "context": f"Within {horizon} business days", "status": "breach" if first > 0 else "healthy",
         "status_text": "Shortfall" if first > 0 else "Funded"},
        {"label": "Largest unpaid obligation", "value": fmt_millions(float(run.max_shortfall[0])),
         "context": "Cash below minimum after same-day funding"},
        {"label": "Largest daily funding need", "value": fmt_millions(float(run.max_funding_need[0])),
         "context": "Before facility draws"},
        {"label": "Lowest available liquidity", "value": fmt_signed_millions(float(run.min_available_liquidity[0])),
         "context": "Cash, funds in transit and unused revolver"},
        {"label": "Interest over horizon", "value": fmt_millions(float(run.total_interest[0])),
         "context": "Revolver and margin line"},
    ]


LEDGER_VIEW = [
    ("day", "Day"), ("futures_value_change", "Futures value change"), ("vm_paid", "VM paid"),
    ("vm_received", "VM received"), ("im_required", "IM required"), ("im_posted", "IM posted"),
    ("im_released", "IM released"), ("collections", "Collections"), ("borrowing_base", "Borrowing base"),
    ("permitted_borrowing", "Permitted borrowing"), ("deficiency", "Deficiency"), ("cure_paid", "Cure paid"),
    ("interest", "Interest"), ("revolver_requested", "Revolver requested"), ("revolver_arrival", "Revolver arrived"),
    ("revolver_repaid", "Revolver repaid"), ("margin_line_requested", "Margin line requested"),
    ("margin_line_arrival", "Margin line arrived"), ("margin_line_repaid", "Margin line repaid"),
    ("cash", "Cash"), ("revolver_drawn", "Revolver drawn"), ("margin_line_drawn", "Margin line drawn"),
    ("in_transit", "In transit"), ("available_liquidity", "Available liquidity"), ("shortfall", "Unpaid obligation"),
]


def render_ledger_table(df: pd.DataFrame, file_stub: str) -> None:
    u = money_divisor()
    view = pd.DataFrame({label: (df[col] if col == "day" else df[col] / u) for col, label in LEDGER_VIEW})
    st.dataframe(view, width="stretch", hide_index=True,
                 column_config={label: st.column_config.NumberColumn(f"{label} ({money_unit()})", format="%.3f")
                                for _, label in LEDGER_VIEW[1:]})
    st.download_button("Download ledger (CSV, USD, full precision)", data=df.to_csv(index=False),
                       file_name=f"cashflow_ledger_{file_stub}.csv", mime="text/csv", key="dl_ledger")


def cashflow_mc_kpis(res: Any) -> list[dict[str, Any]]:
    cond = res.expected_max_shortfall_given_shortfall
    return [
        {"label": "P(any unfunded day)", "value": fmt_pct(res.prob_any_shortfall, 2),
         "context": f"{res.n_paths:,} paths, {res.horizon_days} days",
         "status": "breach" if res.prob_any_shortfall > 0.05 else None, "status_text": "Above 5%"},
        {"label": "Mean largest gap, if short", "value": fmt_millions(cond) if cond is not None else "None",
         "context": "Average of each short path's largest unpaid amount"},
        {"label": "Lowest liquidity, 5th percentile", "value": fmt_signed_millions(res.min_liquidity_percentiles["p5"]),
         "context": f"Median {fmt_signed_millions(res.min_liquidity_percentiles['p50'])}"},
        {"label": "Liquidity reserve, 95%", "value": fmt_millions(res.reserve_95),
         "context": f"99%: {fmt_millions(res.reserve_99)}. Extra opening cash, approximate"},
    ]


def render_provenance(market: Any, calibration: Any, synthetic: bool) -> None:
    rows = [
        ("Price source", market.source_label),
        ("Data type", "Synthetic (illustrative)" if synthetic else "Observed or uploaded prices"),
        ("Prices as of", market.as_of_date),
        ("Calibration window", f"{calibration.window_label}: {calibration.n_obs} joint daily returns, "
                               f"{calibration.start_date} to {calibration.end_date}"),
        ("Coverage", ", ".join(f"{COMMODITY_SPECS[t].name} {v:.0%}" for t, v in calibration.coverage.items())),
        ("Covariance repair", "Repaired to nearest positive semidefinite matrix" if calibration.psd_repaired
         else f"Not needed (smallest eigenvalue {calibration.min_eigenvalue:.2e})"),
        ("Volatilities", "Reference volatilities (too few observations)" if calibration.used_reference_vols
         else "Estimated from the window"),
    ]
    render_audit_rows([(k, html.escape(str(v)), True) for k, v in rows])
    if calibration.note:
        st.caption(calibration.note)


def render_calibration_table(calibration: Any) -> None:
    vols = pd.DataFrame([{"Commodity": COMMODITY_SPECS[t].name, "Annual vol": v * 100}
                         for t, v in calibration.annual_vols.items()])
    st.dataframe(vols, width="stretch", hide_index=True,
                 column_config={"Annual vol": st.column_config.NumberColumn(format="%.1f%%")})


RISK_MEASURE_NOTES = (
    "Liquidity-at-risk (LaR) q%: the (100-q)th percentile of the simulated net liquidity buffer at the horizon. "
    "Expected shortfall 95%: the mean buffer of the worst 5% of paths. P(any unfunded day): share of paths whose "
    "daily ledger shows cash below the minimum after same-day funding on at least one day. Liquidity reserve 95%: "
    "the 95th percentile of each path's largest unpaid amount. Percentile intervals use order statistics; with "
    "few paths in the tail they are wide."
)


COMPARISON_HEADERS = ("Baseline", "Active", "Alternative")
LEVEL_METRICS = {"physical_inventory_value", "borrowing_base", "revolver_availability", "im_requirement",
                 "margin_line_draw", "bb_cure", "funding_shortfall"}


def comparison_frame(cols: Sequence[ComparisonColumn]) -> pd.DataFrame:
    """Metric rows, one column per scenario (short headers) plus the two variances."""
    heads = COMPARISON_HEADERS[:len(cols)]
    rows = []
    for key, label in COMPARISON_METRICS:
        row: dict[str, Any] = {"Metric": label}
        for h, c in zip(heads, cols):
            v = c.metrics.get(key)
            row[h] = None if v is None else _m(v)
        if len(cols) >= 2:
            a, b = cols[0].metrics.get(key), cols[1].metrics.get(key)
            row["Active less baseline"] = None if a is None or b is None else _m(b - a)
        if len(cols) >= 3:
            b, c3 = cols[1].metrics.get(key), cols[2].metrics.get(key)
            row["Alternative less active"] = None if b is None or c3 is None else _m(c3 - b)
        rows.append(row)
    return pd.DataFrame(rows)


def render_comparison(cols: Sequence[ComparisonColumn]) -> None:
    df = comparison_frame(cols)
    numeric = [c for c in df.columns if c != "Metric"]
    st.dataframe(df, width="stretch", hide_index=True,
                 column_config={"Metric": st.column_config.TextColumn(width="medium"),
                                **{c: st.column_config.NumberColumn(c, format="%+.2f" if "less" in c else "%.2f")
                                   for c in numeric}})
    names = "; ".join(f"{h}: {html.escape(c.label)}, {regime_label(c.regime).lower()}"
                      for h, c in zip(COMPARISON_HEADERS, cols))
    st.caption(f"{money_unit()}. {names}. Variation margin and combined P&L are signed (+ = received or gain). "
               "Blank cells are not applicable in this model mode.")


# ---- Saved configurations ---------------------------------------------------------------------------------

def build_config(
    positions: Sequence[InventoryPosition], receivables: Sequence[ReceivableItem], terms: FacilityTerms,
    margin: MarginSettings, basis: Mapping[str, BasisAssumption], extras: ScenarioSpec, cashflow: CashflowSettings,
) -> str:
    ss = st.session_state
    config = WorkbenchConfig(
        model_mode=ss["model_mode"], positions=list(positions), receivables=list(receivables), terms=terms,
        margin=margin, basis=dict(basis), scenario_extras=extras, apply_trigger_rule=bool(ss["apply_trigger_rule"]),
        cashflow=cashflow,
        scenario=ScenarioState(parallel_shock_pct=int(ss["parallel_shock"]),
                               override_enabled=bool(ss["override_enabled"]),
                               per_commodity_pct={t: int(ss[f"shock_{t}"]) for t in REQUIRED_TICKERS}),
        simulation=SimulationState(mc_horizon=int(ss["mc_horizon"] or 5), mc_sims=int(ss["mc_sims"] or 5000),
                                   calibration_window=ss["calib_window"], method=ss["sim_method"],
                                   basis_vol_annual=float(ss["basis_vol"]) / 100),
        report_ccy=ss["report_ccy"],
    )
    return dump_config(config)


def check_config_for_ui(cfg: WorkbenchConfig) -> None:
    """Checks beyond the schema that the widgets need (all done before any state changes)."""
    t = cfg.terms
    cap = round(t.concentration_cap_pct * 100, 1)
    if cap not in [round(c * 100, 1) for c in CAP_TIER_OPTIONS]:
        raise ConfigError(f"Concentration cap {cap}% is not one of the supported tiers.")
    if not 5 <= round(t.im_stepup_trigger * 100) <= 20 or not 0 <= round(t.im_stepup_fraction * 100) <= 50:
        raise ConfigError("IM step-up trigger must be 5% to 20% and step-up size 0% to 50%.")
    if not -20 <= cfg.scenario.parallel_shock_pct <= 20:
        raise ConfigError("Parallel shock must be between -20% and +20%.")
    if any(tk not in cfg.margin.profiles or tk not in cfg.basis for tk in REQUIRED_TICKERS):
        raise ConfigError("Margin and basis settings must cover all six commodities.")


def _apply_config(ss: Any, cfg: WorkbenchConfig) -> None:
    t = cfg.terms
    cap = round(t.concentration_cap_pct * 100, 1)
    ss["terms_base"] = t.model_dump()
    ss["facility_limit_m"] = t.facility_limit / 1e6
    ss["drawn_revolver_m"] = t.drawn_revolver / 1e6
    ss["unrestricted_cash_m"] = t.unrestricted_cash / 1e6
    ss["baseline_im_m"] = t.baseline_initial_margin / 1e6
    ss["cap_tier"] = cap
    ss["im_trigger_pct"] = int(round(t.im_stepup_trigger * 100))
    ss["im_addon_pct"] = int(round(t.im_stepup_fraction * 100))
    ss["margin_line_m"] = t.margin_line_limit / 1e6
    by_ticker = {p.ticker: p for p in cfg.positions}
    ss["inventory_df"] = pd.DataFrame([{
        "Ticker": tk, "Commodity": COMMODITY_SPECS[tk].name, "Unit": COMMODITY_SPECS[tk].unit,
        "Physical Qty": by_ticker[tk].physical_qty, "Hedged Qty": by_ticker[tk].hedged_qty,
    } for tk in REQUIRED_TICKERS])
    ss["receivables_df"] = pd.DataFrame([{
        "Buyer Name": r.buyer_name, "Commodity": r.commodity, "Gross Amount ($)": r.gross_amount,
        "Days Past Due": r.days_past_due, "Investment Grade": r.is_investment_grade,
    } for r in cfg.receivables], columns=["Buyer Name", "Commodity", "Gross Amount ($)", "Days Past Due",
                                          "Investment Grade"])
    ss["margin_df"] = margin_df_from_settings(cfg.margin, cfg.scenario_extras.im_multipliers)
    ss["basis_df"] = basis_df_from_values(cfg.basis, cfg.scenario_extras.basis_shocks,
                                          cfg.scenario_extras.liquidation_discounts)
    ss["contract_policy"] = cfg.margin.contract_policy
    ss["im_mode"] = cfg.margin.im_mode
    ss["release_excess_im"] = cfg.margin.release_excess_im
    ss["ml_cut_pct"] = int(round(cfg.scenario_extras.margin_line_cut_pct * 100))
    ss["apply_trigger_rule"] = cfg.apply_trigger_rule
    cf = cfg.cashflow
    ss.update({
        "cf_horizon": cf.horizon_days, "cf_ramp": cf.ramp_days, "cf_rev_rate": cf.revolver_rate * 100,
        "cf_ml_rate": cf.margin_line_rate * 100, "cf_day_count": cf.day_count, "cf_rev_delay": cf.revolver_delay_days,
        "cf_ml_delay": cf.margin_line_delay_days, "cf_cure_grace": cf.cure_grace_days,
        "cf_invoice_terms": cf.invoice_terms_days, "cf_overdue_delay": cf.overdue_collection_delay_days,
        "cf_noncollected": [n for n in cf.noncollected_buyers if n in {r.buyer_name for r in cfg.receivables}],
        "cf_repay_ml": cf.repay_margin_line_from_receipts, "cf_sweep": cf.sweep_surplus_to_revolver,
        "cf_min_cash_m": cf.minimum_cash / 1e6, "cf_ml_cut_day": cf.margin_line_cut_day,
    })
    ss["parallel_shock"] = cfg.scenario.parallel_shock_pct
    ss["override_enabled"] = cfg.scenario.override_enabled
    for tk in REQUIRED_TICKERS:
        ss[f"shock_{tk}"] = int(cfg.scenario.per_commodity_pct.get(tk, cfg.scenario.parallel_shock_pct))
    sim = cfg.simulation
    ss["mc_horizon"], ss["mc_sims"] = sim.mc_horizon, sim.mc_sims
    ss["calib_window"], ss["sim_method"] = sim.calibration_window, sim.method
    ss["basis_vol"] = int(round(sim.basis_vol_annual * 100))
    ss["report_ccy"] = cfg.report_ccy
    ss["model_mode"] = cfg.model_mode


def apply_uploaded_config() -> None:
    """Callback: validate the uploaded JSON and, only if every field passes, replace the current inputs."""
    ss = st.session_state
    ss.pop("config_error", None)
    ss.pop("config_message", None)
    uploaded = ss.get("config_upload")
    if uploaded is None:
        ss["config_error"] = "Choose a configuration file first."
        return
    raw = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded
    try:
        cfg = load_config(raw)
        check_config_for_ui(cfg)
    except ConfigError as exc:
        ss["config_error"] = str(exc)
        return
    _apply_config(ss, cfg)
    ss.editor_nonce += 1
    ss["config_message"] = (f"Configuration applied: {len(cfg.positions)} commodities, {len(cfg.receivables)} "
                            f"receivables, {MODEL_LABELS[cfg.model_mode].lower()} model.")
