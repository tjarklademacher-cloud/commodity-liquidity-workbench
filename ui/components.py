"""Reusable Streamlit presentation pieces: header, verdict, KPI rows, panels, validated editors, exports."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from core.haircuts import compute_inventory_valuation
from core.schemas import (
    COMMODITY_SPECS,
    DEFAULT_INVENTORY_POSITIONS,
    DEFAULT_RECEIVABLES,
    BorrowingBaseResult,
    CommoditySpec,
    InventoryPosition,
    MarketLoadResult,
    MarketQuote,
    ReceivableItem,
    StressScenarioResult,
)
from core.stress_engine import build_bb_certificate_csv, build_bb_certificate_json
from ui.theme import AUTHOR, COLORS, HOUSE_NAME, STATUS, fmt_millions, fmt_pct

INVENTORY_EDITABLE = ["Physical Qty", "Hedged Qty"]
INVENTORY_CALC = ["Spot", "30d Vol", "Hedge %", "Adv Rate", "GMV", "Eligible", "Contracts"]
AR_COLUMNS = ["Buyer Name", "Commodity", "Gross Amount ($)", "Days Past Due", "Investment Grade"]


def default_inventory_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Ticker": p.ticker,
                "Commodity": COMMODITY_SPECS[p.ticker].name,
                "Unit": COMMODITY_SPECS[p.ticker].unit,
                "Physical Qty": p.physical_qty,
                "Hedged Qty": p.hedged_qty,
            }
            for p in DEFAULT_INVENTORY_POSITIONS
        ]
    )


def default_receivables_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Buyer Name": r.buyer_name,
                "Commodity": r.commodity,
                "Gross Amount ($)": r.gross_amount,
                "Days Past Due": r.days_past_due,
                "Investment Grade": r.is_investment_grade,
            }
            for r in DEFAULT_RECEIVABLES
        ]
    )


def _py(value: Any) -> Any:
    """Convert pandas/numpy scalars to plain Python, mapping missing values to None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return value
    return value.item() if hasattr(value, "item") else value


def _html(markup: str) -> str:
    """Collapse indented HTML so Markdown never mistakes it for a code block."""
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


def _status_badge(text: str, status: str, css_class: str = "badge") -> str:
    color, bg = STATUS[status]
    return f'<span class="{css_class}" style="color:{color};background:{bg};">{html.escape(text)}</span>'


def render_header(meta: Sequence[tuple[str, str]], data_status: tuple[str, str]) -> None:
    """Single-row application header: title and subtitle left, active model context right."""
    status_label, status = data_status
    dot_color = STATUS[status][0] if status != "neutral" else COLORS["accent"]
    items = [f'<span><span class="dot" style="background:{dot_color};"></span><b>{html.escape(status_label)}</b></span>']
    items += [f"<span>{html.escape(k)} <b>{html.escape(v)}</b></span>" for k, v in meta]
    st.markdown(
        '<div class="app-header"><div><div class="title">Commodity Liquidity Workbench</div>'
        f'<div class="subtitle">{html.escape(HOUSE_NAME)} (illustrative book) · Borrowing base and '
        'Liquidity-at-Risk analytics</div></div>'
        f'<div class="meta">{"".join(items)}</div></div>',
        unsafe_allow_html=True,
    )


def _kpi_html(card: dict[str, Any], primary: bool = False) -> str:
    status = card.get("status")
    border = f' style="border-left-color:{STATUS[status][0]};"' if (primary and status) else ""
    badge = _status_badge(card["status_text"], status, "status") if status and card.get("status_text") else ""
    context = f'<div class="context">{card["context"]}</div>' if card.get("context") else ""
    cls = "kpi primary" if primary else "kpi"
    return (
        f'<div class="{cls}"{border}><div class="kpi-top"><span class="eyebrow">{html.escape(card["label"])}</span>'
        f'{badge}</div><div class="value">{html.escape(card["value"])}</div>{context}</div>'
    )


def render_verdict(primary: dict[str, Any], secondary: Sequence[dict[str, Any]]) -> None:
    """Executive liquidity verdict: one emphasised metric followed by quieter supporting metrics.

    Card keys: label, value, optional context (trusted HTML built by the app), status (healthy, caution,
    breach, neutral) and status_text.
    """
    body = _kpi_html(primary, primary=True) + "".join(_kpi_html(c) for c in secondary)
    st.markdown(f'<div class="verdict">{body}</div>', unsafe_allow_html=True)


def render_kpi_row(cards: Sequence[dict[str, Any]]) -> None:
    """Compact equal-weight metric row used inside analytical tabs."""
    st.markdown(f'<div class="kpi-row">{"".join(_kpi_html(c) for c in cards)}</div>', unsafe_allow_html=True)


def section_head(title: str, subtitle: str = "") -> None:
    sub = f'<div class="sub">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(f'<div class="section-head"><div class="title">{html.escape(title)}</div>{sub}</div>',
                unsafe_allow_html=True)


def provenance_note(text: str) -> None:
    st.markdown(f'<div class="provenance">{html.escape(text)}</div>', unsafe_allow_html=True)


def methodology_expander() -> None:
    """Collapsed commercial brief explaining the problem and what each module measures."""
    with st.expander("Methodology and use case", expanded=False):
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown(
                """
**The problem.** A merchant holding physical metals, lumber and softs (cocoa, coffee, sugar) sells futures against them. In a
rally the inventory gains value on paper, but the short futures lose money in cash, every day, through
variation margin. Exchanges also raise initial margin when volatility jumps. The inventory gain only
turns into cash when the goods are sold, or as extra availability under the asset-based loan. That
availability is limited by haircuts, buyer concentration caps and the facility ceiling. A fully hedged
and solvent book can therefore run short of cash. In a sell-off the cash flows reverse: margin flows
back in, but the borrowing base shrinks and the lender can demand a deficiency paydown.

**Net liquidity buffer**, the number used throughout: unrestricted cash + revolver headroom + margin
inflows, minus margin calls, initial margin step-up and any borrowing base deficiency cure. Below zero
the desk cannot meet its cash calls.

**Who uses it.** Treasury, trading and risk teams at a physical merchant deciding how much cash, margin
funding and facility headroom to hold, and when to act. Lenders and credit committees use the same view.
                """
            )
        with col_b:
            st.markdown(
                """
**Overview.** Net liquidity buffer across parallel shocks, a briefing on where the risk sits, and the
eligible collateral mix.

**Borrowing base, Inventory, Receivables.** The lender view: volatility- and hedge-adjusted
haircuts, receivable aging and advance rates, and a single-buyer cap solved as a fixed point. CSV and
JSON certificate export.

**Stress sweep.** Parallel shocks from -20% to +20%: where margin calls exhaust liquidity and where
the base falls below the drawn balance.

**Scenario heatmap.** Metals and softs shocked independently on a 17 x 17 grid.

**Monte Carlo LaR.** Correlated price paths over 1 to 20 trading days from the recent return
covariance. LaR 95% and 99% are the 5th and 1st percentile of the simulated net liquidity buffer. The
liquidity runway follows daily paths for 20 trading days and records when each one first runs short.

**Hedging.** Sweeps the hedge ratio from 0% to 100% and shows the trade-off between margin calls in a
rally and unprotected inventory value in a fall.

**Funding plan.** A contingency funding plan: the moves that break the book (reverse stress test),
early-warning triggers, and the size of each remedial action (cash, margin financing line, revolver
accordion, hedge reduction) that restores a zero buffer.

**Replays.** Today's book run through the loaded price history, or through observed futures prices
from real episodes: the Covid-19 crash, the 2021 Brazil frost, the 2022 Russia-Ukraine shock and the
2024 cocoa squeeze.

*Assumptions:* quantities, receivables and the drawn balance are held fixed; haircuts use the current
30-day realized volatility; margin is settled in cash on the shock day.
                """
            )


def regime_explanation_text(stress: StressScenarioResult) -> str:
    """Plain-language diagnosis of the active regime, in the reporting currency (mirrors the core classifier)."""
    regime = str(stress.regime)
    if regime == "CRITICAL_SQUEEZE":
        return (f"Cash calls exceed available liquidity by {fmt_millions(-stress.net_liquidity_buffer)}: margin and "
                "cure obligations cannot be funded from cash, revolver headroom and any margin line.")
    if regime == "MARGIN_TRAP_WARNING":
        covered = min(stress.margin_line_draw, stress.vm_outflow)
        uncovered = stress.vm_outflow - covered
        line = f" after {fmt_millions(covered)} from the margin line" if covered > 0 else ""
        return (f"Futures variation margin of {fmt_millions(uncovered)}{line} outpaces the "
                f"{fmt_millions(max(0.0, stress.headroom_change))} of new revolver headroom: the inventory gain is "
                "held back by haircuts, caps and the facility ceiling while hedges drain cash.")
    if regime == "BB_DEFICIENCY_WARNING":
        return (f"The borrowing base has fallen below the drawn balance: a {fmt_millions(stress.bb_deficiency_cure)} "
                "deficiency cure is due, partly offset by futures margin inflows.")
    return "Liquid resources cover all margin and cure obligations; no deficiency and no margin trap."


def render_author_panel() -> None:
    """Compact 'built by' block for the bottom of the sidebar."""
    st.sidebar.markdown(
        f'<div class="side-section">About</div><div class="author"><div class="name">{html.escape(AUTHOR["name"])}'
        f'</div><div>{html.escape(AUTHOR["bio"])}</div><div style="margin-top:6px;">'
        f'<a href="{AUTHOR["github"]}" target="_blank">GitHub</a> · '
        f'<a href="{AUTHOR["repo"]}" target="_blank">Source code</a></div>'
        f'<div style="margin-top:6px;color:#64748B;">{html.escape(HOUSE_NAME)} is an illustrative book; '
        "positions, receivables and facility terms are not real.</div></div>",
        unsafe_allow_html=True,
    )


def render_action_list(actions: Sequence[Any]) -> None:
    """Ranked remedial actions as wrapped rows: label and size on one line, how and trade-off beneath."""
    rows = []
    for i, a in enumerate(actions, start=1):
        size = fmt_millions(a.required_usd) if a.required_usd is not None else "Not sufficient alone"
        badge = _status_badge("Sufficient alone" if a.feasible else "Not sufficient alone",
                              "healthy" if a.feasible else "neutral")
        cap = (f" Can contribute up to {fmt_millions(a.capacity_usd)}." if a.capacity_usd is not None
               and not a.feasible else "")
        rows.append(
            f'<div class="audit-row" style="display:block;"><div style="display:flex;justify-content:space-between;'
            f'gap:12px;"><span><b>{i}. {html.escape(a.label)}</b> {badge}</span>'
            f'<span class="v">{size}</span></div><div style="color:#475569;margin-top:2px;">'
            f'{html.escape(a.detail)}{html.escape(cap)}</div><div style="color:#64748B;font-size:12px;">'
            f"Trade-off: {html.escape(a.side_effect)}</div></div>"
        )
    st.markdown(f"<div>{''.join(rows)}</div>", unsafe_allow_html=True)


def render_trigger_list(triggers: Sequence[Any], label_fn: Any, status_fn: Any) -> None:
    rows = "".join(
        f'<div class="briefing-item"><span class="marker" style="background:{STATUS[status_fn(t.regime)][0]};">'
        f'</span><span><b class="num">{t.shock_pct:+.2%}</b> {html.escape(label_fn(t.regime))}: '
        f"{html.escape(t.description)}</span></div>"
        for t in triggers
    )
    st.markdown(f"<div>{rows}</div>", unsafe_allow_html=True)


def render_briefing(items: Sequence[tuple[str, str]]) -> None:
    """List of (status, trusted_html) findings; the marker colour repeats a state already named in the text."""
    rows = "".join(
        f'<div class="briefing-item"><span class="marker" style="background:{STATUS[status][0]};"></span>'
        f"<span>{text}</span></div>"
        for status, text in items
    )
    st.markdown(f"<div>{rows}</div>", unsafe_allow_html=True)


def render_audit_rows(rows: Sequence[tuple[str, str, bool]]) -> None:
    """Label/value rows. Each row: (label, trusted_html_value, is_text). Numbers render in the mono face."""
    body = "".join(
        f'<div class="audit-row"><span class="k">{html.escape(k)}</span>'
        f'<span class="v{" text" if is_text else ""}">{v}</span></div>'
        for k, v, is_text in rows
    )
    st.markdown(f"<div>{body}</div>", unsafe_allow_html=True)


def _apply_pending_edits(base_df: pd.DataFrame, editor_key: str) -> pd.DataFrame:
    """Overlay the editor's pending cell edits so calculated columns reflect them this run."""
    df = base_df.copy()
    state = st.session_state.get(editor_key)
    if not isinstance(state, Mapping):
        return df
    for row, changes in dict(state.get("edited_rows", {})).items():
        idx = int(row)
        if idx >= len(df):
            continue
        for col, val in dict(changes).items():
            if col in INVENTORY_EDITABLE:
                df.loc[df.index[idx], col] = pd.to_numeric(val, errors="coerce") if val is not None else float("nan")
    return df


def _inventory_calc_columns(
    qty_df: pd.DataFrame, snapshot: Mapping[str, MarketQuote], specs: Mapping[str, CommoditySpec]
) -> pd.DataFrame:
    rows = []
    for _, row in qty_df.iterrows():
        ticker = row["Ticker"]
        quote = snapshot[ticker]
        calc = {
            "Spot": quote.spot_price,
            "30d Vol": quote.realized_vol_30d * 100,
            "Hedge %": float("nan"),
            "Adv Rate": float("nan"),
            "GMV": float("nan"),
            "Eligible": float("nan"),
            "Contracts": float("nan"),
        }
        try:
            pos = InventoryPosition.model_validate(
                {"ticker": ticker, "physical_qty": _py(row["Physical Qty"]), "hedged_qty": _py(row["Hedged Qty"])}
            )
        except ValidationError:
            pos = None
        if pos is not None:
            v = compute_inventory_valuation(pos, quote, specs[ticker])
            calc.update(
                {
                    "Hedge %": v.hedge_ratio * 100,
                    "Adv Rate": v.advance_rate * 100,
                    "GMV": v.gross_market_value,
                    "Eligible": v.eligible_collateral,
                    "Contracts": v.futures_contracts,
                }
            )
        rows.append(calc)
    return pd.DataFrame(rows, index=qty_df.index)


def inventory_editor(
    baseline_df: pd.DataFrame,
    snapshot: Mapping[str, MarketQuote],
    nonce: int,
    specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS,
) -> tuple[list[InventoryPosition], list[str], pd.DataFrame]:
    """Validated fixed-row inventory editor; returns positions, validation errors and the edited frame."""
    key = f"inv_editor_{nonce}"
    pending = _apply_pending_edits(baseline_df, key)
    display_df = pd.concat([baseline_df, _inventory_calc_columns(pending, snapshot, specs)], axis=1)
    edited = st.data_editor(
        display_df,
        key=key,
        width="stretch",
        num_rows="fixed",
        hide_index=True,
        disabled=["Ticker", "Commodity", "Unit", *INVENTORY_CALC],
        column_config={
            "Physical Qty": st.column_config.NumberColumn("Physical qty", min_value=0.0, format="localized"),
            "Hedged Qty": st.column_config.NumberColumn("Hedged qty", min_value=0.0, format="localized"),
            "Spot": st.column_config.NumberColumn("Spot ($)", format="localized"),
            "30d Vol": st.column_config.NumberColumn("30d vol", format="%.2f%%"),
            "Hedge %": st.column_config.NumberColumn("Hedge ratio", format="%.1f%%"),
            "Adv Rate": st.column_config.NumberColumn("Advance rate", format="%.3f%%"),
            "GMV": st.column_config.NumberColumn("Market value (USD)", format="dollar"),
            "Eligible": st.column_config.NumberColumn("Eligible (USD)", format="dollar"),
            "Contracts": st.column_config.NumberColumn("Contracts", format="%.1f"),
        },
    )
    positions: list[InventoryPosition] = []
    errors: list[str] = []
    base_lookup = {p.ticker: p for p in DEFAULT_INVENTORY_POSITIONS}
    for idx, row in edited.iterrows():
        ticker = str(row["Ticker"])
        try:
            positions.append(
                InventoryPosition.model_validate(
                    {"ticker": ticker, "physical_qty": _py(row["Physical Qty"]), "hedged_qty": _py(row["Hedged Qty"])}
                )
            )
        except ValidationError as exc:
            fallback = baseline_df.loc[idx]
            errors.append(f"{ticker}: {exc.errors()[0]['msg']}. Using baseline quantities.")
            try:
                positions.append(
                    InventoryPosition.model_validate(
                        {"ticker": ticker, "physical_qty": _py(fallback["Physical Qty"]),
                         "hedged_qty": _py(fallback["Hedged Qty"])}
                    )
                )
            except ValidationError:
                positions.append(base_lookup[ticker])
    return positions, errors, edited.loc[:, list(baseline_df.columns)]


def receivables_editor(baseline_df: pd.DataFrame, nonce: int) -> tuple[list[ReceivableItem], list[str], pd.DataFrame]:
    """Validated dynamic-row AR ledger editor; invalid or incomplete rows are excluded and reported."""
    edited = st.data_editor(
        baseline_df,
        key=f"ar_editor_{nonce}",
        width="stretch",
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "Buyer Name": st.column_config.TextColumn("Buyer", required=True),
            "Commodity": st.column_config.TextColumn("Commodity", required=True),
            "Gross Amount ($)": st.column_config.NumberColumn("Gross amount (USD)", min_value=0.0, format="dollar"),
            "Days Past Due": st.column_config.NumberColumn("Days past due", min_value=0, step=1, format="%d"),
            "Investment Grade": st.column_config.CheckboxColumn("Investment grade", default=False),
        },
    )
    items: list[ReceivableItem] = []
    errors: list[str] = []
    for n, (_, row) in enumerate(edited.iterrows(), start=1):
        values = {c: _py(row.get(c)) for c in AR_COLUMNS}
        if all(v in (None, "", False) for v in values.values()):
            continue
        try:
            items.append(
                ReceivableItem.model_validate(
                    {
                        "buyer_name": values["Buyer Name"],
                        "commodity": values["Commodity"],
                        "gross_amount": values["Gross Amount ($)"],
                        "days_past_due": values["Days Past Due"],
                        "is_investment_grade": bool(values["Investment Grade"]),
                    }
                )
            )
        except ValidationError as exc:
            first = exc.errors()[0]
            field = ".".join(str(x) for x in first["loc"])
            errors.append(f"Row {n} ({values['Buyer Name'] or 'unnamed'}): {field}: {first['msg']}. Row excluded.")
    return items, errors, edited


def certificate_download_buttons(
    bb_result: BorrowingBaseResult, stress_result: StressScenarioResult, market: MarketLoadResult
) -> None:
    col_csv, col_json = st.columns(2)
    stamp = market.as_of_date.replace("-", "")
    with col_csv:
        st.download_button(
            "CSV certificate (USD)",
            data=build_bb_certificate_csv(bb_result, stress_result, market, HOUSE_NAME),
            file_name=f"borrowing_base_certificate_{stamp}.csv",
            mime="text/csv",
            width="stretch",
            key="dl_cert_csv",
        )
    with col_json:
        st.download_button(
            "JSON certificate (USD)",
            data=build_bb_certificate_json(bb_result, stress_result, market, HOUSE_NAME),
            file_name=f"borrowing_base_certificate_{stamp}.json",
            mime="application/json",
            width="stretch",
            key="dl_cert_json",
        )


def covenant_audit_rows(bb_result: BorrowingBaseResult) -> None:
    capped = ", ".join(bb_result.capped_buyer_names) or "None"
    ok = bb_result.final_borrowing_base >= bb_result.drawn_revolver
    status = _status_badge("Covered" if ok else "Deficiency", "healthy" if ok else "breach")
    render_audit_rows(
        [
            ("Base vs. drawn balance", status, True),
            ("Borrowing base", fmt_millions(bb_result.final_borrowing_base), False),
            ("Drawn balance", fmt_millions(bb_result.drawn_revolver), False),
            ("Collateral coverage", f"{bb_result.collateral_coverage_ratio:.3f}x", False),
            ("Facility utilization (drawn / limit)", fmt_pct(bb_result.facility_utilization_pct), False),
            ("Single-buyer cap", f"{fmt_pct(bb_result.concentration_cap_pct, 1)} of base", False),
            ("Cap per buyer", fmt_millions(bb_result.cap_dollar_limit), False),
            ("Capped buyers", html.escape(capped), True),
            ("Weighted inventory advance rate", fmt_pct(bb_result.weighted_inv_advance_rate), False),
            ("Weighted hedge ratio", fmt_pct(bb_result.weighted_hedge_ratio), False),
        ]
    )


def solver_audit_rows(bb_result: BorrowingBaseResult) -> None:
    method = "Sorted-suffix closed form" if bb_result.solver_used_closed_form else "Active-set fixed point"
    ok = bb_result.solver_residual < 0.01
    render_audit_rows(
        [
            ("Equation", "BB = BB<sub>inv</sub> + &Sigma; min(AR<sub>j</sub>, cap x BB)", True),
            ("Method", method, True),
            ("Iterations", str(bb_result.solver_iterations), False),
            ("Residual", f"${bb_result.solver_residual:,.6f} "
                         + _status_badge("Below $0.01" if ok else "Check", "healthy" if ok else "caution"), False),
            ("Cap dollar limit", fmt_millions(bb_result.cap_dollar_limit), False),
        ]
    )


def haircut_methodology_expander(specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS) -> None:
    with st.expander("Commodity specifications and haircut formula"):
        spec_df = pd.DataFrame(
            [
                {
                    "Ticker": s.ticker, "Commodity": s.name, "Exchange": s.exchange, "Unit": s.unit,
                    "Contract size": s.contract_multiplier, "Base haircut": fmt_pct(s.base_haircut),
                    "Ref. vol": fmt_pct(s.ref_vol), "Vol slope": s.vol_slope,
                    "Unhedged penalty": fmt_pct(s.unhedged_penalty_rate),
                    "Min haircut": fmt_pct(s.min_haircut), "Max haircut": fmt_pct(s.max_haircut),
                }
                for s in specs.values()
            ]
        )
        st.dataframe(spec_df, width="stretch", hide_index=True)
        st.markdown(
            """
            - `hedge_ratio = clip(hedged_qty / physical_qty, 0, 1)` (over-hedges still post full futures VM)
            - `vol_addon = vol_slope × max(0, realized_vol_30d − ref_vol)`
            - `unhedged_penalty = (1 − hedge_ratio) × unhedged_penalty_rate`
            - `total_haircut = clip(base + vol_addon + unhedged_penalty, min_haircut, max_haircut)`
            - `eligible = physical_qty × spot × (1 + shock) × (1 − total_haircut)`
            - Stacked bars use a sequential effective decomposition so every bar sums to exactly 100%.
            """
        )


def stress_methodology_expander() -> None:
    with st.expander("Stress methodology"):
        st.markdown(
            """
            - `headroom = max(0, min(facility_limit, BB) − drawn_revolver)`
            - `futures MtM = −Σ hedged_qty × spot × shock` (short hedges lose when prices rally)
            - `IM step-up = im_stepup_fraction × baseline IM` when `max|shock| ≥ trigger` (1e-9 tolerance)
            - `BB deficiency cure = max(0, drawn_revolver − shocked BB)`
            - `net buffer = cash + shocked headroom + VM inflow − (VM call + IM step-up + cure)`
            - Regime priority: Critical squeeze (buffer below 0), then Margin trap (VM call above new
              headroom), then BB deficiency (cure above 0), otherwise Healthy.
            """
        )
