"""Views for the daily treasury engine: Treasury, Credit and Case study 2022 tabs, the Overview projection
panel and the stochastic hedge frontier panel. All calculations live in `core/`; this module only gathers
widget state (keys prefixed `tr_`, `cr_`, `cs_`), calls cached engines and renders results."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from core.advanced_schemas import MarginSettings
from core.case_study_2022 import CASE_VARIANTS, COMPANY_NOTE, DATA_NOTE, LOT_MWH
from core.credit_risk import counterparty_exposure_table
from core.liquidity_metrics import deterministic_summary
from core.schemas import FacilityTerms, InventoryPosition, MarketQuote, ReceivableItem
from core.treasury_analytics import (
    HAIRCUT_VARIANTS,
    TreasurySettings,
    build_book,
)
from core.treasury_engine import ledger_frame
from ui.cache import (
    cached_book_scenario,
    cached_case_study,
    cached_credit_stress,
    cached_haircut_sensitivity,
    cached_mo_frontier,
    cached_treasury_mc,
    cached_ttf_2022,
    cached_validation,
)
from ui.charts import build_liquidity_fan_chart
from ui.components import provenance_note, render_kpi_row, section_head
from ui.theme import PLOTLY_CONFIG, fmt_millions, fmt_pct, fmt_signed_millions, localize_figure, money_divisor, \
    money_unit
from ui.treasury_charts import (
    build_case_liquidity_chart,
    build_case_price_chart,
    build_cash_flow_waterfall,
    build_distribution_chart,
    build_exposure_chart,
    build_pareto_frontier_chart,
    build_sensitivity_bars,
    build_survival_chart,
    build_treasury_path_chart,
)

METHOD_LABELS = {"gaussian": "Gaussian (baseline)", "student_t": "Student t (fat tails)",
                 "bootstrap": "Block bootstrap (historical)", "ewma": "EWMA volatility", "garch": "GARCH(1,1)"}
MARGIN_LABELS = {"fixed_per_contract": "Fixed per contract", "pct_notional": "Share of notional",
                 "vol_scaled": "Volatility-scaled proxy"}
CORR_LABELS = {"unconditional": "Unconditional", "rolling": "Rolling 60 days", "ewma": "EWMA"}
WINDOWS = {"60 days": 60, "125 days": 125, "250 days": 250, "Maximum": None}

TREASURY_DEFAULTS: dict[str, Any] = {
    "tr_horizon": 30, "tr_paths": 500, "tr_method": "gaussian", "tr_nu": 5.0, "tr_block": 5, "tr_lambda": 0.94,
    "tr_corr": "unconditional", "tr_window": "250 days", "tr_seed": 31, "tr_conf": 0.95, "tr_credit": "scheduled",
    "tr_wrongway": 0.0, "tr_dynamic": True, "tr_lender_pts": 0.0, "tr_margin_mode": "fixed_per_contract",
    "tr_vol_mult": 1.0, "tr_min_cash_m": 0.0, "tr_rev_rate": 0.0, "tr_delay": 1, "tr_redet": 1,
    "tr_turnover": 0.0, "tr_basis_vol": 0.0, "tr_extra_line_m": 0.0, "tr_emergency_m": 0.0,
    "tr_validate": False, "cr_lgd": 0.6, "cr_insurance": 0.0, "hf_max_breach": 1.0,
    "cs_alt_line_m": 150.0, "cs_cash_m": 25.0, "cs_hedge": 1.0, "cs_loaded": False,
}


def init_treasury_state() -> None:
    for k, v in TREASURY_DEFAULTS.items():
        st.session_state.setdefault(k, v)


def reset_treasury_state() -> None:
    for k, v in TREASURY_DEFAULTS.items():
        st.session_state[k] = v


def _ss() -> Mapping[str, Any]:
    return st.session_state


def treasury_settings() -> TreasurySettings:
    ss = _ss()
    return TreasurySettings.model_validate({
        "horizon_days": int(ss["tr_horizon"]), "ramp_days": min(5, int(ss["tr_horizon"])),
        "min_operating_cash": float(ss["tr_min_cash_m"]) * 1e6, "revolver_rate": float(ss["tr_rev_rate"]) / 100,
        "drawdown_delay_days": int(ss["tr_delay"]), "redetermination_days": int(ss["tr_redet"]),
        "dynamic_haircuts": bool(ss["tr_dynamic"]), "lender_stress_pts": float(ss["tr_lender_pts"]) / 100,
        "margin_mode": ss["tr_margin_mode"], "vol_multiplier": float(ss["tr_vol_mult"]),
        "turnover_pct": float(ss["tr_turnover"]) / 100, "basis_vol": float(ss["tr_basis_vol"]) / 100,
        "extra_line_limit": float(ss["tr_extra_line_m"]) * 1e6, "emergency_limit": float(ss["tr_emergency_m"]) * 1e6,
        "lgd": float(ss["cr_lgd"]), "insurance_coverage": float(ss["cr_insurance"]),
        "ewma_lambda": float(ss["tr_lambda"]),
    })


def model_key() -> tuple:
    ss = _ss()
    return (ss["tr_method"], WINDOWS[ss["tr_window"]], float(ss["tr_nu"]), int(ss["tr_block"]), float(ss["tr_lambda"]),
            ss["tr_corr"])


def render_treasury_settings() -> None:
    """Simulation settings, grouped by model layer. Widgets carry state by key only."""
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.caption("Horizon and paths")
        st.segmented_control("Horizon (business days)", [1, 5, 10, 30, 60, 90], key="tr_horizon")
        st.selectbox("Monte Carlo paths", [250, 500, 1000, 2000], key="tr_paths")
        st.number_input("Random seed", min_value=0, max_value=10_000, step=1, key="tr_seed")
        st.selectbox("Confidence level", [0.95, 0.99], key="tr_conf", format_func=lambda v: f"{v:.0%}")
    with c2:
        st.caption("Return model")
        st.selectbox("Distribution and volatility", list(METHOD_LABELS), key="tr_method",
                     format_func=METHOD_LABELS.get)
        st.selectbox("Calibration window", list(WINDOWS), key="tr_window")
        st.selectbox("Correlation", list(CORR_LABELS), key="tr_corr", format_func=CORR_LABELS.get)
        st.number_input("Student t degrees of freedom (> 2)", min_value=2.1, max_value=50.0, step=0.5, key="tr_nu")
        st.number_input("Bootstrap block length (days)", min_value=1, max_value=20, step=1, key="tr_block")
        st.selectbox("EWMA decay (lambda)", [0.90, 0.94, 0.97, 0.99], key="tr_lambda")
    with c3:
        st.caption("Collateral and margin")
        st.toggle("Dynamic haircuts (path volatility)", key="tr_dynamic")
        st.slider("Lender tightening (haircut points)", 0.0, 20.0, step=1.0, key="tr_lender_pts")
        st.selectbox("Initial margin model", list(MARGIN_LABELS), key="tr_margin_mode", format_func=MARGIN_LABELS.get)
        st.slider("Volatility shock multiplier", 1.0, 3.0, step=0.1, key="tr_vol_mult")
        st.selectbox("Borrowing base redetermination (days)", [1, 5, 20], key="tr_redet")
        st.slider("Synthetic basis volatility (% a year)", 0.0, 30.0, step=1.0, key="tr_basis_vol")
    with c4:
        st.caption("Treasury and funding")
        st.number_input("Minimum operating cash (USD M)", min_value=0.0, max_value=100.0, step=0.5, key="tr_min_cash_m")
        st.number_input("Revolver interest rate (% a year)", min_value=0.0, max_value=25.0, step=0.25, key="tr_rev_rate")
        st.number_input("Revolver drawdown delay (business days)", min_value=0, max_value=5, step=1, key="tr_delay")
        st.number_input("Additional committed line (USD M)", min_value=0.0, max_value=500.0, step=5.0,
                        key="tr_extra_line_m")
        st.number_input("Emergency funding available (USD M)", min_value=0.0, max_value=500.0, step=5.0,
                        key="tr_emergency_m")
        st.slider("Physical turnover per cycle (% of stock, every 5 days)", 0.0, 20.0, step=1.0, key="tr_turnover")
        st.radio("Credit outcomes", ["scheduled", "simulated"], key="tr_credit", horizontal=True,
                 format_func={"scheduled": "Scheduled", "simulated": "Simulated defaults"}.get)
        st.slider("Wrong-way loading (simulated credit)", 0.0, 0.9, step=0.1, key="tr_wrongway")


def _ledger_table(f: pd.DataFrame) -> None:
    cols = ["date", "cash", "available_liquidity", "shortfall", "vm_net", "im_call", "im_released", "im_posted",
            "collections", "supplier_payments", "interest", "reported_borrowing_base", "permitted",
            "revolver_drawn", "revolver_received", "revolver_repaid", "mandatory_repayment", "deficiency",
            "avg_haircut", "economic_pnl", "networth"]
    view = f[cols].copy()
    for c in cols[1:]:
        if c != "avg_haircut":
            view.loc[:, c] = view[c] / money_divisor()
    st.dataframe(view, width="stretch", hide_index=True, height=320,
                 column_config={c: st.column_config.NumberColumn(f"{c.replace('_', ' ').capitalize()} ({money_unit()})",
                                                                 format="%.2f") for c in cols[1:] if c != "avg_haircut"}
                 | {"avg_haircut": st.column_config.NumberColumn("Average haircut", format="%.3f")})


def _scenario_run(inputs, margin: MarginSettings, ts: TreasurySettings, shocks: Mapping[str, float]):
    return cached_book_scenario(inputs, margin.model_dump_json(), ts.model_dump_json(), json.dumps(dict(shocks)))


def render_treasury_tab(inputs: tuple[str, str, str, str], positions: Sequence[InventoryPosition],
                        receivables: Sequence[ReceivableItem], snapshot: Mapping[str, MarketQuote],
                        terms: FacilityTerms, margin: MarginSettings, prices_df: pd.DataFrame,
                        shocks: Mapping[str, float], scenario_label: str, synthetic_feed: bool) -> None:
    ss = _ss()
    with st.expander("Treasury simulation settings", expanded=False):
        render_treasury_settings()
        provenance_note("Defaults are illustrative and leave every existing figure unchanged: zero rates, no "
                        "minimum cash, no physical turnover, scheduled collections. Every parameter above is an "
                        "assumption; none is taken from a real facility agreement or exchange notice.")
    ts = treasury_settings()
    run = _scenario_run(inputs, margin, ts, shocks)
    f = ledger_frame(run)
    d = deterministic_summary(run)

    section_head("Daily treasury projection under the active scenario",
                 f"{scenario_label}: futures move linearly over {ts.ramp_days} business days, then hold. Each day "
                 f"starts from the previous day's closing state. Horizon {ts.horizon_days} business days.")
    render_kpi_row([
        {"label": "Minimum available liquidity", "value": fmt_signed_millions(d["min_liquidity"]),
         "context": f"on {d['min_liquidity_date']}", "status": "breach" if d["min_liquidity"] < 0 else None,
         "status_text": "Below zero"},
        {"label": "Peak unfunded requirement", "value": fmt_millions(d["peak_funding"]),
         "context": f"first breach {d['first_breach_date'] or 'none'}; {d['breach_days']} breach days",
         "status": "breach" if d["peak_funding"] > 0 else None, "status_text": "Unfunded"},
        {"label": "Peak cumulative VM paid", "value": fmt_millions(d["peak_cum_vm_paid"]),
         "context": f"peak initial margin posted {fmt_millions(d['peak_im'])}"},
        {"label": "Economic P&L over horizon", "value": fmt_signed_millions(d["economic_pnl"]),
         "context": "inventory + futures + trade margin - costs - credit losses"},
        {"label": "Peak revolver utilization", "value": fmt_pct(d["peak_utilization"], 1),
         "context": "drawn / permitted"},
    ])
    with st.container(key="panel_tr_path"):
        section_head("Liquidity trajectory and borrowing base", "Top: cash and available liquidity (cash + funding "
                     "in transit + undrawn committed availability); red bars are unfunded shortfalls. Bottom: "
                     "reported base, permitted utilization and drawn balance.")
        st.plotly_chart(localize_figure(build_treasury_path_chart(f, ts.min_operating_cash)), width="stretch",
                        theme=None, config=PLOTLY_CONFIG, key="chart_tr_path")
    col_w, col_s = st.columns([1, 1], gap="medium")
    with col_w:
        with st.container(key="panel_tr_waterfall"):
            section_head("Cash flow decomposition over the horizon",
                         "Operating, margin and financing flows from opening to closing unrestricted cash.")
            st.plotly_chart(localize_figure(build_cash_flow_waterfall(f)), width="stretch", theme=None,
                            config=PLOTLY_CONFIG, key="chart_tr_waterfall")
    with col_s:
        with st.container(key="panel_tr_haircut"):
            section_head("Haircut sensitivity", "Minimum available liquidity under alternative collateral "
                         "assumptions (illustrative lender behaviour), same market path.")
            hs = cached_haircut_sensitivity(inputs, margin.model_dump_json(), ts.model_dump_json(), json.dumps(dict(shocks)))
            st.plotly_chart(localize_figure(build_sensitivity_bars(hs)), width="stretch", theme=None,
                            config=PLOTLY_CONFIG, key="chart_tr_haircut")
    with st.expander("Haircut sensitivity table and assumptions"):
        tab = hs.assign(min_liquidity=hs["min_liquidity"] / money_divisor(),
                        change_vs_static=hs["change_vs_static"] / money_divisor(),
                        peak_funding=hs["peak_funding"] / money_divisor())
        st.dataframe(tab, width="stretch", hide_index=True, column_config={
            "assumption": st.column_config.TextColumn("Collateral assumption"),
            "min_liquidity": st.column_config.NumberColumn(f"Minimum liquidity ({money_unit()})", format="%.2f"),
            "change_vs_static": st.column_config.NumberColumn(f"Change vs static ({money_unit()})", format="%.2f"),
            "peak_funding": st.column_config.NumberColumn(f"Peak funding need ({money_unit()})", format="%.2f"),
            "first_breach": st.column_config.TextColumn("First breach"),
            "breach_days": st.column_config.NumberColumn("Breach days"),
            "peak_avg_haircut": st.column_config.NumberColumn("Peak average haircut", format="%.3f")})
        provenance_note("Static: haircut frozen at today's volatility. Dynamic: haircut = base + slope x (path EWMA "
                        "volatility - reference), clipped to each commodity's floor and cap. Tightening adds haircut "
                        "points over the ramp. Contractual limit caps the increase per redetermination. When the "
                        "borrowing base already exceeds the commitment, haircut changes do not move availability. "
                        + "; ".join(HAIRCUT_VARIANTS))
    with st.expander("Daily ledger (auditable, exportable)"):
        _ledger_table(f)
        st.download_button("Download full ledger (CSV, USD)", f.to_csv(index=False).encode(), "treasury_ledger.csv",
                           "text/csv", key="dl_tr_ledger")
        provenance_note(f"Cash reconciliation residual {run.reconciliation_residual:.2e} USD; net-worth identity "
                        f"residual {run.networth_residual:.2e} USD (tolerance 1e-4). " + " ".join(run.notes))

    # ---------------------------------------------------------------- Monte Carlo
    mc = cached_treasury_mc(inputs, margin.model_dump_json(), ts.model_dump_json(), prices_df, model_key(),
                            int(ss["tr_paths"]), int(ss["tr_seed"]), float(ss["tr_conf"]), ss["tr_credit"],
                            float(ss["tr_wrongway"]))
    m = mc.metrics
    section_head("Path-dependent Monte Carlo through the daily engine",
                 f"{mc.n_paths:,} simulated paths, {METHOD_LABELS[mc.method]}, {ss['tr_window']} calibration, "
                 f"seed {mc.seed}, horizon {mc.horizon_days} business days, credit outcomes {mc.credit_mode}.")
    lo, hi = m.prob_breach_interval
    render_kpi_row([
        {"label": "Probability of a liquidity breach", "value": fmt_pct(m.prob_breach, 1),
         "context": f"95% sampling interval {fmt_pct(lo, 1)} to {fmt_pct(hi, 1)}; threshold: cash below "
                    f"{fmt_millions(m.breach_threshold)}"},
        {"label": f"Liquidity-at-Risk {m.confidence:.0%}", "value": fmt_millions(m.lar),
         "context": f"deterioration from {fmt_millions(m.baseline_liquidity)}; interval "
                    f"{fmt_millions(m.lar_interval[0])} to {fmt_millions(m.lar_interval[1])}"},
        {"label": "Expected liquidity shortfall", "value": fmt_millions(m.els),
         "context": f"mean deterioration beyond LaR {m.confidence:.0%}"},
        {"label": "Funding need p95 / p99", "value": f"{fmt_millions(m.funding_p95)} / {fmt_millions(m.funding_p99)}",
         "context": f"expected unfunded {fmt_millions(m.expected_unfunded)}"},
        {"label": "Median minimum liquidity", "value": fmt_signed_millions(m.min_liquidity_median),
         "context": f"5th percentile {fmt_signed_millions(m.min_liquidity_p05)}"},
    ])
    days = list(range(mc.horizon_days + 1))
    col_f, col_s2 = st.columns([2.1, 1], gap="medium")
    with col_f:
        with st.container(key="panel_tr_fan"):
            section_head("Available liquidity by day", "Percentiles across paths; each path carries its own state.")
            st.plotly_chart(localize_figure(build_liquidity_fan_chart(days, mc.liquidity_fan,
                                                                      "Available liquidity (USD M)")),
                            width="stretch", theme=None, config=PLOTLY_CONFIG, key="chart_tr_fan")
    with col_s2:
        with st.container(key="panel_tr_surv"):
            section_head("Liquidity survival", "Share of paths with no breach by each day.")
            st.plotly_chart(build_survival_chart(m.survival_curve), width="stretch", theme=None, config=PLOTLY_CONFIG,
                            key="chart_tr_surv")
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        with st.container(key="panel_tr_fund"):
            section_head("Peak funding requirement distribution", "Largest unfunded shortfall on each path.")
            st.plotly_chart(localize_figure(build_distribution_chart(mc.funding_samples, m.funding_p95,
                                                                     "Peak unfunded requirement (USD M)", "p95")),
                            width="stretch", theme=None, config=PLOTLY_CONFIG, key="chart_tr_fund")
    with col_b:
        with st.container(key="panel_tr_det"):
            section_head("Liquidity deterioration distribution",
                         "Baseline available liquidity minus the path minimum; LaR is the marked quantile.")
            st.plotly_chart(localize_figure(build_distribution_chart(mc.deterioration_samples, m.lar,
                                                                     "Deterioration (USD M)", "LaR")),
                            width="stretch", theme=None, config=PLOTLY_CONFIG, key="chart_tr_det")
    if m.tail_decomposition:
        tail = pd.DataFrame([{"Driver (tail paths, at the path minimum)": k, f"Change ({money_unit()})": v / money_divisor()}
                             for k, v in m.tail_decomposition.items()])
        st.dataframe(tail, width="stretch", hide_index=True)
    notes = ("Probabilities come from the calibrated stochastic model and carry sampling error; they are model "
             "estimates, not frequencies of real events. " + " ".join(mc.notes))
    if synthetic_feed:
        notes += " The bundled feed is synthetic with near-zero correlations."
    provenance_note(notes)
    export = {"method": mc.method, "seed": mc.seed, "n_paths": mc.n_paths, "horizon_days": mc.horizon_days,
              "credit_mode": mc.credit_mode, "model_key": list(model_key()), "settings": json.loads(ts.model_dump_json()),
              "metrics": json.loads(m.model_dump_json()), "currency": "USD"}
    st.download_button("Download Monte Carlo summary (JSON)", json.dumps(export, indent=2).encode(),
                       "treasury_monte_carlo.json", "application/json", key="dl_tr_mc")

    with st.expander("Model validation (out-of-sample backtest, volatility forecasts, moments)"):
        st.checkbox("Run validation (takes a few seconds; GARCH longer)", key="tr_validate")
        if ss["tr_validate"]:
            render_validation(prices_df, inputs, synthetic_feed)


def render_validation(prices_df: pd.DataFrame, inputs: tuple[str, str, str, str], synthetic_feed: bool) -> None:
    ss = _ss()
    window = WINDOWS[ss["tr_window"]] or 250
    window = min(window, max(60, len(prices_df) - 40))
    with st.spinner("Running rolling out-of-sample validation"):
        v = cached_validation(prices_df, inputs[0], inputs[3], ss["tr_method"], int(window), float(ss["tr_conf"]))
    bt = v["backtest"]
    render_kpi_row([
        {"label": "Forecasts", "value": f"{bt.n_forecasts:,}", "context": f"rolling {bt.window}-day calibration"},
        {"label": "Exceptions", "value": f"{bt.exceptions} (expected {bt.expected:.1f})",
         "context": f"one-day hedge P&L loss beyond the {bt.level:.0%} quantile"},
        {"label": "Kupiec p-value", "value": f"{bt.kupiec_p:.3f}", "context": "H0: correct exception rate"},
        {"label": "Independence p-value", "value": f"{bt.independence_p:.3f}", "context": "H0: no clustering"},
    ])
    provenance_note(bt.note + (" The feed is synthetic, so this checks internal consistency only."
                               if synthetic_feed else ""))
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        section_head("Volatility forecast accuracy", "One-day-ahead variance vs squared return; lower is better.")
        st.dataframe(v["vol_eval"], width="stretch", hide_index=True)
    with c2:
        section_head("Simulated vs historical one-day moments",
                     f"Largest correlation gap {v['corr_gap']:.3f}. Excess kurtosis above zero means fat tails.")
        st.dataframe(v["moments"].round(3), width="stretch", hide_index=True)
    provenance_note("A more complex model is not automatically better: compare exception counts, QLIKE and tail "
                    "moments, and treat small samples (under five expected exceptions) as inconclusive.")


def render_overview_projection(positions, receivables, snapshot, terms, margin, prices_df, inputs, shocks) -> None:
    ss = _ss()
    ts = treasury_settings()
    run = _scenario_run(inputs, margin, ts, shocks)
    d = deterministic_summary(run)
    mc = cached_treasury_mc(inputs, margin.model_dump_json(), ts.model_dump_json(), prices_df, model_key(),
                            int(ss["tr_paths"]), int(ss["tr_seed"]), float(ss["tr_conf"]), ss["tr_credit"],
                            float(ss["tr_wrongway"]))
    L = run.ledger
    with st.container(key="panel_overview_treasury"):
        section_head("Daily treasury projection",
                     f"Path-dependent engine, {ts.horizon_days} business days. Restricted margin collateral is "
                     "posted initial margin; it is not available to pay obligations.")
        render_kpi_row([
            {"label": "Available liquidity today", "value": fmt_millions(L["available_liquidity"][0, 0]),
             "context": f"undrawn revolver {fmt_millions(L['revolver_undrawn'][0, 0])}; drawn "
                        f"{fmt_millions(L['revolver_drawn'][0, 0])}"},
            {"label": "Restricted margin collateral", "value": fmt_millions(L["restricted_cash"][0, 0]),
             "context": f"peak in scenario {fmt_millions(d['peak_im'])}"},
            {"label": "Minimum projected liquidity", "value": fmt_signed_millions(d["min_liquidity"]),
             "context": f"active scenario, {d['min_liquidity_date']}",
             "status": "breach" if d["min_liquidity"] < 0 else None, "status_text": "Below zero"},
            {"label": "Peak funding requirement", "value": fmt_millions(d["peak_funding"]),
             "context": f"active scenario; simulated p95 {fmt_millions(mc.metrics.funding_p95)}"},
            {"label": "Probability of a breach", "value": fmt_pct(mc.metrics.prob_breach, 1),
             "context": f"{mc.n_paths:,} simulated paths, {METHOD_LABELS[mc.method]}"},
        ])


def render_credit_tab(inputs, positions, receivables, snapshot, terms, margin, prices_df, shocks) -> None:
    ss = _ss()
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.slider("Loss given default (uninsured part)", 0.0, 1.0, step=0.05, key="cr_lgd")
    with c2:
        st.slider("Credit insurance coverage (all buyers)", 0.0, 0.95, step=0.05, key="cr_insurance")
    ts = treasury_settings()
    book = build_book(positions, receivables, terms, margin, ts, snapshot)
    exp = {c.id: inv.amount for c, inv in zip(book.counterparties, book.invoices)}
    overdue = {c.id: inv.amount if inv.days_past_due > 0 else 0.0 for c, inv in zip(book.counterparties, book.invoices)}
    table = counterparty_exposure_table(book.counterparties[:len(book.invoices)], exp, overdue, 365.0)
    total = table["exposure"].sum()
    top = table.iloc[0] if len(table) else None
    render_kpi_row([
        {"label": "Receivable exposure", "value": fmt_millions(total),
         "context": f"overdue {fmt_millions(table['overdue'].sum())}"},
        {"label": "Largest counterparty", "value": fmt_pct(top["share"], 1) if top is not None else "n/a",
         "context": top["counterparty"] if top is not None else ""},
        {"label": "Expected credit loss, one year", "value": fmt_millions(table["ecl"].sum()),
         "context": "PD x LGD x uninsured exposure (illustrative PDs)"},
        {"label": "Insured exposure", "value": fmt_millions(table["insured"].sum()),
         "context": "claims paid after the claim delay, not at default"},
    ])
    col_t, col_c = st.columns([1.4, 1], gap="medium")
    with col_t:
        with st.container(key="panel_cr_table"):
            section_head("Counterparty exposure and expected credit loss",
                         "Rating bucket is an illustrative mapping from the ledger's investment-grade flag "
                         "(BBB / BB); it is not an assessment of any named company.")
            view = table[["counterparty", "rating", "exposure", "share", "overdue", "insured", "pd_annual", "lgd", "ecl"]]
            view = view.assign(exposure=view["exposure"] / money_divisor(), overdue=view["overdue"] / money_divisor(),
                               insured=view["insured"] / money_divisor(), ecl=view["ecl"] / money_divisor())
            st.dataframe(view, width="stretch", hide_index=True, column_config={
                "share": st.column_config.NumberColumn("Share", format="%.3f"),
                "pd_annual": st.column_config.NumberColumn("PD (1 year)", format="%.4f"),
                **{c: st.column_config.NumberColumn(f"{c.capitalize()} ({money_unit()})", format="%.2f")
                   for c in ("exposure", "overdue", "insured", "ecl")}})
    with col_c:
        with st.container(key="panel_cr_chart"):
            section_head("Exposure concentration", "Insured and uninsured receivables by counterparty.")
            st.plotly_chart(localize_figure(build_exposure_chart(table)), width="stretch", theme=None,
                            config=PLOTLY_CONFIG, key="chart_cr_exposure")
    with st.container(key="panel_cr_stress"):
        section_head("Deterministic credit stress under the active market scenario",
                     "Hypothetical delays, downgrades and defaults applied to the daily engine. Delays move cash "
                     "later without a loss; defaults create claims and losses; downgrades remove eligibility.")
        stress = cached_credit_stress(inputs, margin.model_dump_json(), ts.model_dump_json(), json.dumps(dict(shocks)))
        view = stress.copy()
        for c in ("min_liquidity", "change_vs_no_stress", "peak_funding", "credit_loss", "min_receivable_contribution"):
            view.loc[:, c] = view[c] / money_divisor()
        st.dataframe(view, width="stretch", hide_index=True, column_config={
            "scenario": st.column_config.TextColumn("Scenario (hypothetical)"),
            "first_breach": st.column_config.TextColumn("First breach"),
            "breach_days": st.column_config.NumberColumn("Breach days"),
            **{c: st.column_config.NumberColumn(f"{c.replace('_', ' ').capitalize()} ({money_unit()})", format="%.2f")
               for c in ("min_liquidity", "change_vs_no_stress", "peak_funding", "credit_loss",
                         "min_receivable_contribution")}})
        provenance_note("A delay that ends after the day of minimum liquidity does not change the minimum; it can "
                        "still change the peak funding need and the borrowing base.")
    with st.container(key="panel_cr_mc"):
        section_head("Deterministic versus probabilistic credit",
                     "Same simulated market paths with scheduled collections and with simulated defaults, delays "
                     "and downgrades (one-factor model, asset correlation 0.2, wrong-way loading from the Treasury "
                     "settings).")
        args = (inputs, margin.model_dump_json(), ts.model_dump_json(), prices_df, model_key(), int(ss["tr_paths"]),
                int(ss["tr_seed"]), float(ss["tr_conf"]))
        sched = cached_treasury_mc(*args, "scheduled", 0.0)
        sim = cached_treasury_mc(*args, "simulated", float(ss["tr_wrongway"]))
        render_kpi_row([
            {"label": "P(breach), scheduled collections", "value": fmt_pct(sched.metrics.prob_breach, 1),
             "context": f"funding p95 {fmt_millions(sched.metrics.funding_p95)}"},
            {"label": "P(breach), simulated credit", "value": fmt_pct(sim.metrics.prob_breach, 1),
             "context": f"funding p95 {fmt_millions(sim.metrics.funding_p95)}"},
            {"label": "Mean realized credit loss", "value": fmt_millions(sim.metrics.credit_loss_mean),
             "context": f"over {sim.horizon_days} business days, simulated"},
        ])
        provenance_note("Default probabilities are illustrative (rating table in core/treasury_schemas.py), "
                        "converted to the horizon under constant hazard. This is not an IFRS 9 model.")


def render_mo_frontier(inputs, margin: MarginSettings, prices_df: pd.DataFrame, scope: str) -> None:
    ss = _ss()
    ts = treasury_settings()
    with st.container(key="panel_mo_frontier"):
        section_head("Economic risk against funding need (stochastic frontier)",
                     "Each hedge ratio runs the same simulated paths (common random numbers) through the daily "
                     "engine. Lower-left is better on both axes; no ratio is optimal in general.")
        st.slider("Maximum acceptable probability of a liquidity breach", 0.0, 1.0, step=0.05, key="hf_max_breach")
        df = cached_mo_frontier(inputs, margin.model_dump_json(), ts.model_dump_json(), prices_df, model_key(), scope,
                                min(500, int(ss["tr_paths"])), int(ss["tr_seed"]), float(ss["hf_max_breach"]), None)
        st.plotly_chart(localize_figure(build_pareto_frontier_chart(df)), width="stretch", theme=None,
                        config=PLOTLY_CONFIG, key="chart_mo_frontier")
        view = df.copy()
        for c in ("pnl_mean", "pnl_sd", "economic_downside", "economic_tail_mean", "funding_p95", "els", "lar",
                  "opening_im", "transaction_cost"):
            view.loc[:, c] = view[c] / money_divisor()
        st.dataframe(view, width="stretch", hide_index=True)
        provenance_note("Economically conservative: smallest 5% quantile loss among feasible ratios. Liquidity "
                        "conservative: smallest 95th percentile funding need. Balanced: smallest larger-of-two "
                        "scaled objectives (a minimax rule). These labels depend on the horizon, return model, "
                        "margin and haircut assumptions and the constraint above.")


def _eur(v: float | None) -> str:
    return "n/a" if v is None else f"{'-' if v < 0 else ''}EUR {abs(v) / 1e6:,.2f}M"


def _load_case() -> None:
    st.session_state["cs_loaded"] = True


def render_case_study_tab() -> None:
    ss = _ss()
    section_head("Case study: a hedged gas merchant in the 2022 European energy crisis",
                 "Why an economically hedged commodity merchant can still run out of cash.")
    provenance_note(DATA_NOTE + " " + COMPANY_NOTE + " Figures in this tab are in EUR (the hypothetical "
                    "merchant's functional currency) and are not converted to the reporting currency.")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.number_input("Alternative committed line (EUR M)", min_value=0.0, max_value=1000.0, step=25.0,
                        key="cs_alt_line_m")
    with c2:
        st.number_input("Opening cash (EUR M)", min_value=0.0, max_value=500.0, step=5.0, key="cs_cash_m")
    with c3:
        st.slider("Hedge ratio of stored gas", 0.0, 1.0, step=0.1, key="cs_hedge")
    if not ss.get("cs_loaded"):
        st.button("Load observed 2022 TTF prices and run the case study", key="cs_load_btn", on_click=_load_case)
        provenance_note("Prices are fetched from Yahoo Finance on request and cached for a day.")
        return
    try:
        with st.spinner("Fetching observed TTF prices"):
            observed = cached_ttf_2022()
    except Exception as exc:  # noqa: BLE001 - any vendor failure means no case study, never synthetic data
        st.info(f"Observed TTF data could not be loaded ({type(exc).__name__}). The case study needs the observed "
                "series and does not substitute synthetic prices. Try again later or use a connection with internet "
                "access.")
        return
    res = cached_case_study(observed, float(ss["cs_alt_line_m"]) * 1e6, float(ss["cs_cash_m"]) * 1e6,
                            float(ss["cs_hedge"]))
    tab: pd.DataFrame = res["table"]
    a = res["assumptions"]
    static = tab.iloc[0]
    comb = tab[tab["variant"] == "Combined adverse"].iloc[0]
    render_kpi_row([
        {"label": "Peak TTF price", "value": f"EUR {float(np.max(res['prices'])):,.0f}/MWh",
         "context": f"opening {float(res['prices'][0]):,.0f}; observed"},
        {"label": "Economic P&L, static case", "value": _eur(static["economic_pnl"]),
         "context": "hedged: inventory gain offsets futures loss"},
        {"label": "Peak cumulative VM paid", "value": _eur(static["peak_cum_vm_paid"]),
         "context": "cash paid to the clearing house"},
        {"label": "Peak funding need, combined adverse", "value": _eur(comb["peak_funding"]),
         "context": f"static financing {_eur(static['peak_funding'])}",
         "status": "breach" if comb["peak_funding"] > 0 else None, "status_text": "Unfunded"},
    ])
    with st.container(key="panel_cs_table"):
        section_head("Variants: each adds one mechanism",
                     f"Stored gas {a.storage_mwh / 1e6:,.1f} TWh, short futures in lots of {LOT_MWH:,.0f} MWh "
                     f"(illustrative lot), revolver EUR {a.commitment / 1e6:,.0f}M with EUR {a.opening_drawn / 1e6:,.0f}M "
                     f"drawn, weekly borrowing base, IM reviewed every {a.im_review_days} days.")
        view = tab[["variant", "min_liquidity", "min_liquidity_date", "peak_funding", "first_breach_date",
                    "breach_days", "economic_pnl", "peak_cum_vm_paid", "peak_im", "credit_loss", "peak_vol"]].copy()
        for c in ("min_liquidity", "peak_funding", "economic_pnl", "peak_cum_vm_paid", "peak_im", "credit_loss"):
            view.loc[:, c] = view[c] / 1e6
        st.dataframe(view, width="stretch", hide_index=True, column_config={
            **{c: st.column_config.NumberColumn(f"{c.replace('_', ' ').capitalize()} (EUR M)", format="%.2f")
               for c in ("min_liquidity", "peak_funding", "economic_pnl", "peak_cum_vm_paid", "peak_im",
                         "credit_loss")},
            "peak_vol": st.column_config.NumberColumn("Peak EWMA vol", format="%.2f")})
        st.download_button("Download variant table (CSV, EUR)", tab.to_csv(index=False).encode(),
                           "case_study_2022.csv", "text/csv", key="dl_cs_table")
    sel = st.selectbox("Variant for the price and margin chart", list(CASE_VARIANTS), index=5, key="cs_variant")
    with st.container(key="panel_cs_price"):
        section_head("Observed price, synthetic physical price and margin", "TTF=F is observed; the physical price "
                     "at the storage site uses a synthetic mean-reverting basis and is shown only for variants with "
                     "basis risk.")
        basis = res["basis_path"] if "basis" in sel.lower() or "Combined" in sel else None
        st.plotly_chart(build_case_price_chart(res["dates"], res["prices"], basis, res["ledgers"][sel]),
                        width="stretch", theme=None, config=PLOTLY_CONFIG, key="chart_cs_price")
    with st.container(key="panel_cs_liq"):
        section_head("Unrestricted cash after same-day funding", "Below zero is an unfunded obligation.")
        st.plotly_chart(build_case_liquidity_chart(res["dates"], res["ledgers"],
                                                   [CASE_VARIANTS[0], CASE_VARIANTS[2], CASE_VARIANTS[5],
                                                    CASE_VARIANTS[6]]),
                        width="stretch", theme=None, config=PLOTLY_CONFIG, key="chart_cs_liq")
    ledger = pd.DataFrame({"date": res["dates"].strftime("%Y-%m-%d"), "ttf_close": res["prices"],
                           **{k: v for k, v in res["ledgers"][sel].items()}})
    st.download_button("Download ledger for the selected variant (CSV, EUR)", ledger.to_csv(index=False).encode(),
                       "case_study_2022_ledger.csv", "text/csv", key="dl_cs_ledger")
    provenance_note(f"{res['forward_filled_days']} engine business days had no vendor quote and were forward "
                    "filled. The continuous front-month series rolls between contract months; part of the measured "
                    "volatility and variation margin reflects those rolls, which a held contract would not "
                    "experience. Margin and haircut paths are model proxies, not ICE Endex or lender data.")
