"""Reusable Streamlit widgets: hero, KPI cards, regime banner, validated editors, exports, audits."""

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
from ui.theme import COLORS, REGIME_STYLES, TONES, fmt_millions, fmt_pct, fmt_signed_millions

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


def sparkline_svg(values: Sequence[float], color: str, width: int = 200, height: int = 30) -> str:
    """Full-width inline SVG sparkline with a soft area fill."""
    vals = [float(v) for v in values if v is not None and v == v]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    step = width / (len(vals) - 1)
    pts = [(i * step, height - 3 - (v - lo) / span * (height - 6)) for i, v in enumerate(vals)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"0,{height} " + line + f" {width},{height}"
    return (
        f'<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
        f'<polygon points="{area}" fill="{color}" fill-opacity="0.12"/>'
        f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.8" '
        f'vector-effect="non-scaling-stroke" stroke-linejoin="round" stroke-linecap="round"/></svg>'
    )


def render_kpi_cards(cards: Sequence[dict[str, Any]]) -> None:
    """Responsive grid of KPI cards. Keys: label, value, optional sub, tone, chip, chip_tone, spark."""
    parts = []
    for c in cards:
        accent, glow = TONES[c.get("tone", "blue")]
        chip = ""
        if c.get("chip"):
            chip_color, chip_bg = TONES[c.get("chip_tone", c.get("tone", "blue"))]
            chip = f'<span class="chip" style="color:{chip_color};background:{chip_bg};">{html.escape(c["chip"])}</span>'
        spark = c.get("spark")
        spark_html = f'<div class="kpi-spark">{sparkline_svg(spark, accent)}</div>' if spark else ""
        sub = f'<div class="kpi-sub">{html.escape(c["sub"])}</div>' if c.get("sub") else ""
        parts.append(
            f'<div class="kpi-card" style="--accent:{accent};--glow:{glow};">'
            f'<div class="kpi-label">{html.escape(c["label"])}</div>'
            f'<div class="kpi-value">{html.escape(c["value"])}</div>{sub}{chip}{spark_html}</div>'
        )
    st.markdown(f'<div class="kpi-grid">{"".join(parts)}</div>', unsafe_allow_html=True)


def section_head(title: str, subtitle: str = "") -> None:
    sub = f'<div class="sub">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(f'<div class="section-head"><div class="title">{html.escape(title)}</div>{sub}</div>',
                unsafe_allow_html=True)


def render_hero(market: MarketLoadResult, terms: Any, mode_label: str, status_label: str, status_tone: str) -> None:
    """Page header: title, one-line purpose and the valuation context pills (data status first)."""
    pills = [
        (TONES[status_tone][0], f"{status_label} · {market.source_label}"),
        (COLORS["blue"], f"Valuation date {market.as_of_date}"),
        (COLORS["violet"], f"Cap tier: {terms.concentration_cap_pct:.1%}"),
        (COLORS["slate"], f"Facility {fmt_millions(terms.facility_limit)} limit, "
                          f"{fmt_millions(terms.drawn_revolver)} drawn"),
        (COLORS["orange"], mode_label),
    ]
    pill_html = "".join(
        f'<span class="source-pill"><span class="dot" style="background:{c};"></span>{html.escape(t)}</span>'
        for c, t in pills
    )
    st.markdown(
        _html(
            f"""
            <div class="hero">
              <div class="eyebrow">Commodity ABL · Collateral &amp; Treasury Risk</div>
              <h1>Commodity Collateral, Borrowing Base &amp; Liquidity Stress Workbench</h1>
              <div class="subtitle">How much cash and revolver headroom a hedged Gold, Copper, Lumber and Cocoa
              book has left when prices move, and where futures margin calls outrun the borrowing base.</div>
              <div class="pills">{pill_html}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def methodology_expander() -> None:
    """Collapsed commercial brief explaining the problem and what each module measures."""
    with st.expander("Methodology & Commercial Use Case", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                """
**The problem.** A merchant holding physical metal, lumber or cocoa sells futures against it. In a
rally the inventory gains value on paper, but the short futures lose money in cash, every day, through
variation margin. Exchanges also raise initial margin when volatility jumps. The inventory gain only
turns into cash when the goods are sold, or as extra availability under the asset-based loan. That
availability is limited by haircuts, buyer concentration caps and the facility ceiling. A fully hedged
and solvent book can therefore run short of cash. In a sell-off the cash flows reverse: margin flows
back in, but the borrowing base shrinks and the lender can demand a deficiency paydown.

**Who uses it.** Treasury and trading desks at commodity merchants, trade finance and ABL lenders
sizing facilities, and risk teams setting cash buffers and hedge ratios.

**Net liquidity buffer**, the number used throughout:
unrestricted cash + revolver headroom + margin inflows, minus margin calls, initial margin step-up and
any borrowing base deficiency cure. Below zero means the desk cannot meet its cash calls.
                """
            )
        with col_b:
            st.markdown(
                """
**Overview.** Facility health at a glance: borrowing base, headroom, availability usage, the active
stress result and a short briefing on where the risk sits.

**BB Certificate, Inventory, Receivables.** The lender view. Volatility- and
hedge-adjusted inventory haircuts, receivable aging and investment-grade advance rates, and a
single-buyer cap solved as a fixed point (the cap is a share of the base it helps set). Exports CSV
and JSON certificates.

**Stress Sweep.** Parallel price shocks from -20% to +20%. Shows the shock at which margin
calls exhaust liquidity and the drop at which the base falls below the drawn balance.

**Scenario Heatmap.** Metals and softs shocked independently on a 17 x 17 grid, so you can see
which mixed moves are dangerous, not only parallel ones.

**Monte Carlo Liquidity-at-Risk.** Thousands of correlated price paths over 1 to 20 trading days,
calibrated to recent return volatility and correlation. Reports the 5th and 1st percentile buffer
(LaR 95% and 99%) and the probability of each regime.

**Historical Replay.** Today's book run through each of the past year's trading days, showing
borrowing base, daily futures margin and a 5-day stressed buffer.

*Assumptions:* quantities, receivables and the drawn balance are held fixed; haircuts use the current
30-day realized volatility; margin is settled in cash on the shock day.
                """
            )


def render_regime_banner(stress_result: StressScenarioResult, mode_label: str) -> None:
    """Semantic alert banner for the active shock's liquidity regime."""
    label, color, bg_rgba, border = REGIME_STYLES[stress_result.regime]
    st.markdown(
        _html(
            f"""
            <div class="regime-banner" style="background:{bg_rgba};border:1px solid {border};
                 border-left:5px solid {color};">
              <div class="left">
                <span class="pulse" style="background:{color};"></span>
                <div>
                  <span class="regime-label" style="color:{color};">{label}</span>
                  <span class="regime-mode">Active stress · {html.escape(mode_label)}</span>
                  <div class="regime-detail">{html.escape(stress_result.regime_explanation)}</div>
                </div>
              </div>
              <div class="right">
                <div class="nlb-label">Net Liquidity Buffer</div>
                <div class="nlb-value" style="color:{color};">
                  {fmt_signed_millions(stress_result.net_liquidity_buffer)}</div>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_insights(title: str, items: Sequence[tuple[str, str]]) -> None:
    """Card of (tone, html_text) bullet insights; text may contain <b> emphasis only."""
    rows = "".join(
        f'<div class="insight-item"><span class="bar" style="background:{TONES[tone][0]};"></span>'
        f'<span class="txt">{text}</span></div>'
        for tone, text in items
    )
    st.markdown(f'<div class="insight-card"><div class="head">{html.escape(title)}</div>{rows}</div>',
                unsafe_allow_html=True)


def _audit_rows(rows: Sequence[tuple[str, str]]) -> str:
    return "".join(
        f'<div class="row"><span class="muted">{html.escape(k)}</span><span>{v}</span></div>' for k, v in rows
    )


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
            "Physical Qty": st.column_config.NumberColumn("Physical Qty", min_value=0.0, format="localized"),
            "Hedged Qty": st.column_config.NumberColumn("Hedged Qty", min_value=0.0, format="localized"),
            "Spot": st.column_config.NumberColumn("Spot ($)", format="localized"),
            "30d Vol": st.column_config.NumberColumn("30d Vol", format="%.2f%%"),
            "Hedge %": st.column_config.NumberColumn("Hedge %", format="%.1f%%"),
            "Adv Rate": st.column_config.NumberColumn("Adv Rate", format="%.3f%%"),
            "GMV": st.column_config.NumberColumn("GMV", format="dollar"),
            "Eligible": st.column_config.NumberColumn("Eligible", format="dollar"),
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
            "Buyer Name": st.column_config.TextColumn("Buyer Name", required=True),
            "Commodity": st.column_config.TextColumn("Commodity", required=True),
            "Gross Amount ($)": st.column_config.NumberColumn("Gross Amount ($)", min_value=0.0, format="dollar"),
            "Days Past Due": st.column_config.NumberColumn("Days Past Due", min_value=0, step=1, format="%d"),
            "Investment Grade": st.column_config.CheckboxColumn("Investment Grade", default=False),
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
            "Download Certificate (CSV)",
            data=build_bb_certificate_csv(bb_result, stress_result, market),
            file_name=f"borrowing_base_certificate_{stamp}.csv",
            mime="text/csv",
            width="stretch",
            key="dl_cert_csv",
        )
    with col_json:
        st.download_button(
            "Download Certificate (JSON)",
            data=build_bb_certificate_json(bb_result, stress_result, market),
            file_name=f"borrowing_base_certificate_{stamp}.json",
            mime="application/json",
            width="stretch",
            key="dl_cert_json",
        )


def _status_chip(text: str, tone: str) -> str:
    color, bg = TONES[tone]
    return f'<span class="chip" style="margin:0;color:{color};background:{bg};">{html.escape(text)}</span>'


def covenant_audit_card(bb_result: BorrowingBaseResult) -> None:
    capped = ", ".join(bb_result.capped_buyer_names) or "None"
    ok = bb_result.final_borrowing_base >= bb_result.drawn_revolver
    rows = [
        ("BB vs. drawn revolver", _status_chip("AVAILABLE" if ok else "DEFICIENT", "emerald" if ok else "red")),
        ("Borrowing base / drawn",
         f"{fmt_millions(bb_result.final_borrowing_base)} / {fmt_millions(bb_result.drawn_revolver)}"),
        ("Collateral coverage", f"{bb_result.collateral_coverage_ratio:.3f}x"),
        ("Facility utilization", fmt_pct(bb_result.facility_utilization_pct)),
        ("Single-buyer cap",
         f"{fmt_pct(bb_result.concentration_cap_pct, 1)} × BB = {fmt_millions(bb_result.cap_dollar_limit)}"),
        ("Capped buyers", html.escape(capped)),
        ("Weighted inventory advance rate", fmt_pct(bb_result.weighted_inv_advance_rate)),
        ("Weighted hedge ratio", fmt_pct(bb_result.weighted_hedge_ratio)),
    ]
    st.markdown(f'<div class="audit-card"><div class="head">Covenant Audit</div>{_audit_rows(rows)}</div>',
                unsafe_allow_html=True)


def solver_audit_card(bb_result: BorrowingBaseResult) -> None:
    method = "Sorted-suffix closed form" if bb_result.solver_used_closed_form else "Active-set fixed point"
    ok = bb_result.solver_residual < 0.01
    rows = [
        ("Equation", "BB = BB<sub>inv</sub> + Σ min(AR<sub>j</sub>, cap × BB)"),
        ("Method", method),
        ("Iterations", str(bb_result.solver_iterations)),
        ("Residual", f"${bb_result.solver_residual:,.6f} " + _status_chip("PASS" if ok else "CHECK",
                                                                           "emerald" if ok else "amber")),
        ("Cap dollar limit", fmt_millions(bb_result.cap_dollar_limit)),
    ]
    st.markdown(
        f'<div class="audit-card"><div class="head">Fixed-Point Solver Convergence Audit</div>{_audit_rows(rows)}</div>',
        unsafe_allow_html=True,
    )


def haircut_methodology_expander(specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS) -> None:
    with st.expander("Commodity Specs & Haircut Formula Audit"):
        spec_df = pd.DataFrame(
            [
                {
                    "Ticker": s.ticker, "Commodity": s.name, "Exchange": s.exchange, "Unit": s.unit,
                    "Contract Mult.": s.contract_multiplier, "Base HC": fmt_pct(s.base_haircut),
                    "Ref Vol": fmt_pct(s.ref_vol), "Vol Slope": s.vol_slope,
                    "Unhedged Pen.": fmt_pct(s.unhedged_penalty_rate),
                    "Min HC": fmt_pct(s.min_haircut), "Max HC": fmt_pct(s.max_haircut),
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
    with st.expander("Liquidity Stress Methodology Audit"):
        st.markdown(
            """
            - `headroom = max(0, min(facility_limit, BB) − drawn_revolver)`
            - `futures MtM = −Σ hedged_qty × spot × shock` (short hedges lose when prices rally)
            - `IM step-up = im_stepup_fraction × baseline IM` when `max|shock| ≥ trigger` (1e-9 tolerance)
            - `BB deficiency cure = max(0, drawn_revolver − shocked BB)`
            - `net buffer = cash + shocked headroom + VM inflow − (VM call + IM step-up + cure)`
            - Regime priority: `CRITICAL_SQUEEZE` (buffer < 0) → `MARGIN_TRAP_WARNING`
              (VM call > new headroom) → `BB_DEFICIENCY_WARNING` (cure > 0) → `HEALTHY`.
            """
        )
