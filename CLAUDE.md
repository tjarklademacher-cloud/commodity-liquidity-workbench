# CLAUDE.md: Commodity Collateral & Liquidity Stress Workbench

Full hand-over context (architecture, decisions, limitations, backlog): see `PROJECT_CONTEXT.md`.

Streamlit + Plotly workbench for a commodity ABL facility: dynamic inventory haircuts, a fixed-point
single-buyer concentration cap, a borrowing base certificate, and a 41-point "margin trap" liquidity sweep.

## Commands

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python >= 3.11 (StrEnum)
.venv/bin/python -m core.data_feed --generate-sample                      # regenerate data/sample_prices.csv
.venv/bin/python -m pytest -v                                              # full suite (quant + AppTest)
.venv/bin/python -m pytest -v tests/test_stress_engine.py                  # one module
.venv/bin/streamlit run app.py                                             # launch UI
```

## Layout

| Path | Role |
| --- | --- |
| `core/schemas.py` | Pydantic v2 models, `LiquidityRegime`, `COMMODITY_SPECS`, defaults, presets |
| `core/data_feed.py` | seed=42 sample generator/CLI, cleaning, realized vol, yfinance loader + fallback |
| `core/haircuts.py` | Inventory haircut engine + 100%-sum effective decomposition |
| `core/borrowing_base.py` | AR eligibility, cap solver (active-set + sorted-suffix closed form), waterfall |
| `core/stress_engine.py` | Stress scenario/sweep, regime classifier, breakevens, certificate CSV/JSON |
| `core/scenario_analytics.py` | Vectorized engine, metals×softs shock grid, Monte Carlo LaR, liquidity runway, replay |
| `core/hedge_optimizer.py` | Hedge-ratio frontier against the 41-point sweep |
| `core/crisis_replay.py` | Real crisis episodes from Yahoo Finance applied to today's book |
| `core/contingency.py` | Contingency funding plan: reverse stress, triggers, sized remedial actions |
| `core/advanced_schemas.py` | Advanced model inputs/results: MarginProfile, MarginSettings, BasisAssumption, ScenarioSpec, CashflowSettings |
| `core/margin_engine.py` | Whole contracts, VM, per-contract IM, illustrative default margin settings |
| `core/basis_risk.py` | Physical vs futures prices, economic P&L identity, basis presets |
| `core/advanced_scenario.py` | Advanced scenario, legacy reconciliation, margin presets, trigger rule, sensitivities |
| `core/cashflow_engine.py` | Daily cash-flow ledger and cash-flow Monte Carlo |
| `core/market_calibration.py` | Calibration windows, PSD repair, simulation methods, factor sample, backtest |
| `core/advanced_analytics.py` | Sweep, metals x softs grid and hedge frontier through the advanced engine (legacy result types) |
| `core/scenario_comparison.py` | Scenario comparison and versioned JSON configurations (load_config validates) |
| `ui/advanced_views.py` | Advanced widgets, margin/basis editors, dashboards, comparison, config callbacks |
| `ui/report.py` | Self-contained HTML credit committee report (built on download click) |
| `ui/theme.py` | Design system: light institutional tokens (mirrored in `.streamlit/config.toml`), semantic STATUS colours, commodity palette, Plotly layout, minimal CSS for custom HTML and `panel_*` containers |
| `ui/charts.py` | The original six Plotly figures plus gauge, donut, regime ribbon, heatmap, MC, correlation, replay |
| `ui/cache.py` | `st.cache_data` wrappers (live TTL 900s, scenario suite, Monte Carlo) and data-status label |
| `ui/components.py` | Hero, KPI cards with sparklines, regime banner, insights, validated editors, exports, audits |
| `core/units.py` | Unit and currency conversion; cross-dimension conversion needs an explicit factor |
| `core/treasury_schemas.py` | Typed inputs of the daily engine (instruments, positions, counterparties, facility, policies) |
| `core/futures_curve.py` | MarketPaths: calendar, synthetic curve, rolls, EWMA path volatility, OU basis |
| `core/collateral.py` | Dynamic haircuts, haircut conventions, path-wise cap solver, borrowing base, availability |
| `core/credit_risk.py` | Horizon PD, ECL, insurance and recovery, payment outcomes, copula defaults, migration |
| `core/treasury_engine.py` | Daily treasury engine: single source of truth for path-dependent state |
| `core/liquidity_metrics.py` | LaR as deterioration, ELS, breach probability, funding percentiles, survival |
| `core/return_models.py` | Data preparation; Gaussian, Student t, bootstrap, EWMA, GARCH; matrix diagnostics |
| `core/model_validation.py` | Kupiec, Christoffersen, rolling backtest, vol forecast scores, moments |
| `core/treasury_analytics.py` | Adapter from the six-commodity book, scenarios, haircut and credit stress, Monte Carlo |
| `core/hedge_frontier.py` | Stochastic multi-objective hedge frontier |
| `core/case_study_2022.py` | Observed TTF 2022 prices through a hypothetical gas merchant |
| `ui/treasury_views.py`, `ui/treasury_charts.py` | Treasury, Credit, Case study 2022 tabs, Overview projection, frontier panel |
| `app.py` | Entry point: branded 4-section sidebar, hero + regime banner, methodology expander, 13 tabs |

## Zero-leakage boundary (enforced by `tests/test_architecture_and_app.py`)

- `core/` must never import `streamlit` or `plotly`, directly or transitively. AST scan + subprocess
  `sys.modules` check enforce this.
- All widgets, figures, CSS and session state live in `ui/` and `app.py`.
- `core/__init__.py` exports lazily (PEP 562) so `python -m core.data_feed` has no runpy double-import warning.

## Ground truth (seed=42 sample); tests assert these to $0.01M

- Default six-commodity book (`tests/test_default_book.py`): BB $144.94M (Barry capped, $2.06M cutback),
  +14% NLB -$1.79M critical, -20% cure $6.83M, breakevens +12.09% / -12.72%, sweep 13/12/8/8.
- Advanced model (default settings, +14%, trigger rule on): buffer -$2.61M, reconciliation step "Per-contract
  initial margin model" -0.82M, ledger first unfunded day 3. Advanced sweep breakevens +11.21% / -12.72%,
  sweep 13/11/8/9. Neutral advanced settings must equal legacy (scenario, sweep, grid and hedge frontier).
- Legacy four-commodity book pinned in `tests/conftest.py` (`positions`, `receivables`, `terms`, `sweep`):
  BB $114.12M, +14% NLB -$1.22M, breakevens +12.41% / -5.45%, sweep 15/7/11/8. Do not change these.
- Coffee (KC=F) and sugar (SB=F) are quoted in cents; `CommoditySpec.quote_scale=0.01` converts on load.

## 2026 library guardrails

**Streamlit (>= 1.52)**
- Never use `use_container_width`; always `width="stretch"` / `width="content"`.
- `st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONFIG, key=...)`.
- Editors: canonical frames in `st.session_state["inventory_df"]` / `["receivables_df"]`; keys
  `inv_editor_{nonce}` / `ar_editor_{nonce}`. Never write the edited frame back mid-run. Bump
  `editor_nonce` only in the Reset and Apply Preset callbacks (preset first commits current edits).
- Inventory editor is keyed + `num_rows="fixed"`, so Streamlit uses signature identity: calculated
  columns may change value each run without dropping edits.
- Widgets get state via `key` + pre-seeded `st.session_state`; do not also pass `value=`.
- No `persist_state="page"`. In `AppTest`, never `.set_value()` on `data_editor`; set
  `at.session_state[f"inv_editor_{nonce}"]` edit dicts and `run()`.

**Pandas 3 (Copy-on-Write)**
- No chained assignment; use `.loc[...] =` or `.assign`. No `applymap`, no `inplace=True`.
- Detect strings with `pd.api.types.is_string_dtype`, never `dtype == object`.

**Pydantic v2**
- `model_copy(update=...)` skips validation: only use it with internally generated values.
  Every UI/IO boundary goes through `Model.model_validate(...)`.

**Plotly 6/7**
- `title=dict(text=..., font=dict(...))`; no `titlefont`/`titleside`; colors hex or `rgba(0-255, a)`.
- No dual y-axes; use `make_subplots(rows=2, cols=1, shared_xaxes=True)`.
- `go.Waterfall` totals take `y=None`; supply explicit `$M` text labels.
- Add `add_vrect(row=..., col=...)` shapes AFTER the subplot has a trace (empty subplots are skipped).

**yfinance 1.x**
- `yf.download(tickers=..., period=..., interval="1d", auto_adjust=True, progress=False, threads=False)`.
- Never pass a custom `requests` session. 429s come back as empty frames: validate rows, tickers,
  history length and positivity; any failure falls back to `data/sample_prices.csv`.
- Live loads cached in `app.py` with `@st.cache_data(ttl=900, show_spinner=False)`.

- Model mode: `model_mode` "legacy" (default) or "advanced". Legacy outputs must not change. Advanced editors
  `margin_editor_{nonce}` / `basis_editor_{nonce}` are always rendered (collapsed expanders) so edits survive
  mode switches; callbacks that bump the nonce call `commit_pending_edits` first. Reset clears all advanced state.
- Funding cost, carry and dilution (`FacilityTerms.revolver_interest_rate`, `inventory_carry_rate`,
  `carry_horizon_days`, `ar_dilution_reserve_pct`) default to 0 so every pinned figure is unchanged; the
  carry outflow is in `run_stress_scenario`, `evaluate_shock_matrix` and `run_advanced_scenario` alike.
- Never invent exchange margins, basis, prices or rates: defaults are labelled illustrative; edited values are
  `assumption_type="user"`. Uploaded configurations are data only (`load_config`), validated before any state changes.
- `evaluate_shock_matrix` must stay identical to `run_stress_scenario`; `tests/test_scenario_analytics.py`
  checks it on random mixed shocks. If you change any stress formula, change both.
- Custom HTML goes through `_html()` / single-line strings so Markdown never renders indented HTML as code.

- UI copy and docs: plain, factual, sentence case, no em-dashes or emojis.
- Money shown in the UI goes through `fmt_millions`/`money_divisor()`/`localize_figure` (reporting currency);
  the model, inputs and certificate exports stay in USD. Margin line (`margin_line_limit`) defaults to 0.
- Visual changes go through `ui/theme.py` tokens; risk colours (`STATUS`) only for states the model defines.
  Chart titles live in the surrounding `panel()` heading, not inside the Plotly figure.

**Numerics**
- Sweep grid `np.round(np.linspace(-0.20, 0.20, 41), 6)`; IM trigger uses `>= trigger - 1e-9`.

**Treasury engine (docs/METHODOLOGY.md)**
- `run_treasury` must keep `reconciliation_residual` and `networth_residual` near zero; tests assert both.
  Never create cash to pay an obligation: negative cash is an unfunded shortfall.
- Initial margin is restricted cash, never an expense; variation margin is counted once.
- Expected values in `tests/test_treasury_engine.py` and `tests/test_credit_basis_units.py` are hand-calculated;
  never regenerate them by running the engine.
- Treasury defaults must leave legacy and advanced figures unchanged; keys are in `TREASURY_DEFAULTS`.
- The case study fetches only on its button (`cs_loaded`); tests stay offline. Case study money is EUR and is
  not converted. Never substitute synthetic data for the observed series.
- LaR in the new engine is a deterioration quantile; do not relabel the legacy buffer-level LaR.

