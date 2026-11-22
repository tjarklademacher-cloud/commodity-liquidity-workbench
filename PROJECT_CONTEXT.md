# PROJECT_CONTEXT.md

Complete hand-over context for the **Commodity Liquidity Workbench**. Read this first in a new conversation.
`CLAUDE.md` is the short rule sheet; `README.md` is the public-facing brief; this file is the full picture.

Last updated: 2026-11-22, at commit `fa0bb0c` on `main` (154 tests passing).

---

## 1. What this is

A Streamlit desk tool for the treasury, trading and risk team of a **physical commodity merchant** financed by
an asset-based revolving credit facility (ABL). It values a hedged six-commodity inventory plus a trade
receivables ledger, certifies a borrowing base, and measures how much cash and revolver headroom remain when
prices move.

**Core thesis, the "margin trap":** the merchant sells futures against its physical stock. In a rally the
futures lose cash every day (variation margin, VM) and exchanges raise initial margin (IM), while the gain on
the physical stock only becomes cash when sold or when the lender advances against it. That advance is capped
by haircuts, single-buyer concentration limits and the facility limit, so a solvent, fully hedged book can run
out of cash. In a sell-off the reverse happens: VM flows back in, but the borrowing base falls below the drawn
balance and the lender demands a deficiency cure.

- **Primary audience (chosen by the owner):** a trading-house desk (Geneva, Zug, London, Hamburg style
  treasury, trading and risk teams). Secondary: trade finance lenders and credit committees.
- **Purpose:** portfolio piece for hiring managers in physical commodities, trade finance and institutional
  sales, so it must look and behave like a production desk tool and every number must be defensible.
- **Owner / author:** Tjark Lademacher, "Commodity trade finance and risk analytics".
  GitHub: https://github.com/tjarklademacher-cloud
- **Repository:** https://github.com/tjarklademacher-cloud/commodity-liquidity-workbench (public, branch `main`)
- **Live app:** https://commodity-liquidity-workbench.streamlit.app (Streamlit Community Cloud, redeploys on
  every push to `main`)
- **Modelled firm:** "Lademacher Commodities", explicitly labelled an illustrative book. Positions,
  receivables and facility terms are not real.

---

## 2. Quick start and environment

Local folder: `~/commodity-liquidity-workbench`.

```bash
cd ~/commodity-liquidity-workbench
.venv/bin/python -m streamlit run app.py          # opens on http://localhost:8501
.venv/bin/python -m pytest                        # 154 tests, about 20 s
.venv/bin/python -m core.data_feed --generate-sample   # regenerate data/sample_prices.csv (deterministic)
```

Environment facts that matter:

- macOS system Python is 3.9, which is too old (`StrEnum` needs 3.11+, Streamlit 1.5x needs 3.10+). The
  `.venv` is **Python 3.12**, created with `uv` installed via pip: `~/Library/Python/3.9/bin/uv`.
  Rebuild if missing:
  `~/Library/Python/3.9/bin/uv venv --python 3.12 .venv && ~/Library/Python/3.9/bin/uv pip install --python .venv/bin/python -r requirements.txt`
- Installed versions at time of writing: Streamlit 1.67.0, Plotly 7.1.0, pandas 3.1.1, NumPy 2.5.5,
  Pydantic 2.14.4, yfinance 1.8.0, pytest 9.2.0.
- Always launch with `.venv/bin/python -m streamlit ...`, not `.venv/bin/streamlit` (launcher scripts embed
  absolute paths and break if the venv moves).
- `.streamlit/config.toml` sets `server.headless = true`, so there is no first-run email prompt and no auto
  browser open. Port 8501 by default; use `--server.port 8502` for a second copy.
- Streamlit does not reliably hot-reload changed modules under `ui/` or `core/` without the `watchdog`
  package. After editing those, restart the server.
- Linting used during development: `~/Library/Python/3.9/bin/uvx pyflakes app.py ui core tests` (not a project
  dependency).
- Git identity is set **repo-locally** to `Tjark Lademacher <tjarklademacher@gmail.com>` (owner's choice).
  Pushing over HTTPS uses a fine-grained GitHub token stored in the macOS keychain; it was created with an
  expiry of **2026-12-22**. After that, `git push` will ask for credentials again (username
  `tjarklademacher-cloud`, password = a new fine-grained token with Contents: read and write on this repo).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

---

## 3. Repository layout

```
app.py                   877 lines  Streamlit entry point: sidebar, header, verdict, 10 tabs (presentation only)
core/                    pure Python models; MUST NOT import streamlit or plotly (enforced by tests)
  __init__.py             lazy PEP 562 exports of the public API (avoids runpy double-import warning)
  schemas.py              Pydantic v2 models, LiquidityRegime, COMMODITY_SPECS, defaults, presets
  data_feed.py            sample generator + CLI, cleaning, realized vol, live yfinance loader, FX loader
  haircuts.py             inventory valuation and haircut decomposition
  borrowing_base.py       receivables eligibility, fixed-point concentration cap solver, waterfall, tiers
  stress_engine.py        exact single-scenario stress, 41-point sweep, regime classifier, certificate export
  scenario_analytics.py   vectorized engine, shock grid, Monte Carlo LaR, historical replay, liquidity runway
  hedge_optimizer.py      hedge-ratio frontier
  crisis_replay.py        real crisis episodes from Yahoo Finance applied to today's book
  contingency.py          contingency funding plan: reverse stress, triggers, sized remedial actions
ui/                      presentation layer
  theme.py                design tokens, STATUS colours, Plotly layout, CSS, currency context, formatters,
                          HOUSE_NAME and AUTHOR constants, bridge/regime label helpers
  charts.py               all Plotly figure builders (15)
  components.py           header, verdict, KPI rows, panels, editors, audit rows, briefing, author panel
  cache.py                st.cache_data wrappers and data-status label
  report.py               self-contained HTML credit committee report
data/sample_prices.csv   260 business days ending 2026-11-20, six tickers, generated with seed 42
tests/                   154 tests (see section 10)
.streamlit/config.toml   native light theme (mirrors ui/theme.py), headless server, error detail level
requirements.txt         bounded to tested major versions
README.md                public brief; CLAUDE.md short rules; PROJECT_CONTEXT.md this file
```

`.gitignore` excludes `.venv/`, caches, `.env*`, `.streamlit/secrets.toml`, `.claude/`, OS files, and all of
`data/` except `data/sample_prices.csv`. There are no secrets or API keys anywhere; none are needed.

---

## 4. Architecture and data flow

### 4.1 Boundary rule

`core/` contains every calculation and data loader and never imports Streamlit or Plotly, directly or
transitively. `tests/test_architecture_and_app.py` checks this with an AST scan of each `core/*.py` file and a
subprocess that imports `core` and inspects `sys.modules`. All widgets, figures, CSS, caching and session state
live in `ui/` and `app.py`.

### 4.2 What happens on each rerun (`app.py::main`)

1. `st.set_page_config` (title "Commodity Liquidity Workbench", wide layout, `initial_sidebar_state="auto"`
   so phones start with the sidebar collapsed), `inject_custom_css()`, `init_session_state()`.
2. Sidebar, in this order: Portfolio and facility (returns `FacilityTerms`), Market data (returns
   `MarketLoadResult`, plus reporting currency select), Stress scenario (parallel shock or per-commodity
   overrides), Simulation (Monte Carlo horizon and paths), Reset, About (author panel).
3. Reporting currency resolved via `cached_fx_rate` and set with `set_display_currency` (a `ContextVar`).
4. Tabs are created; the Inventory and Receivables editors render first because every other view needs the
   edited positions and receivables.
5. Calculations: `compute_portfolio_inventory` -> `calculate_borrowing_base` -> `run_stress_scenario`
   (active shock) -> `compare_concentration_cap_tiers` -> cached suite (`run_stress_sweep`,
   `run_shock_grid`, `run_historical_replay`) -> cached `run_liquidity_monte_carlo` -> cached
   `run_liquidity_runway` -> `build_contingency_plan`.
6. Header and verdict are filled into a container created above the tabs; then each tab renders.

### 4.3 Caching (`ui/cache.py`)

Cached functions take only hashable primitives plus price DataFrames. Pydantic inputs are serialized with
`model_inputs_json()` into JSON strings and re-validated with `model_validate` inside the cached function,
so the cache key is the full model input.

| Function | TTL | Notes |
| --- | --- | --- |
| `cached_live_market(period, winsorize)` | 900 s | live Yahoo prices, falls back internally |
| `cached_sample_market`, `cached_custom_market` | none | bundled sample / uploaded CSV bytes |
| `cached_scenario_suite(...)` | none | sweep, grid, replay; independent of the active shock slider |
| `cached_monte_carlo(...)`, `cached_runway(...)` | none | depend on paths, horizon, winsorize |
| `cached_hedge_frontier(..., scope)` | none | |
| `cached_episode_closes(key)` | 86,400 s | historical crisis prices |
| `cached_crisis_replay(..., key, winsorize)` | none | |
| `cached_fx_rate(code)` | 900 s | |

### 4.4 Vectorized engine and its contract

`scenario_analytics.evaluate_shock_matrix(positions, receivables, snapshot, terms, shock_matrix)` evaluates
many scenarios at once. Because a price shock does not change volatility, haircuts are fixed and eligible
inventory scales with `max(0, 1 + shock)`; only the concentration-cap fixed point is re-solved
(vectorized sorted-suffix closed form). It must stay **identical** to `stress_engine.run_stress_scenario`
(net buffer, borrowing base, margin-line draw and regime). Tests compare them on random mixed shocks with and
without a margin line. If you change any stress formula, change both places.

### 4.5 Public core API

Exported lazily from `core/__init__.py` (`from core import ...` works):

- `data_feed`: `generate_sample_csv`, `compute_realized_volatility`, `build_market_snapshot`,
  `load_sample_market_data`, `load_custom_csv_market_data`, `load_live_market_data`, `DEFAULT_SAMPLE_CSV_PATH`
- `haircuts`: `compute_inventory_valuation`, `compute_portfolio_inventory`, `inventory_valuations_to_dataframe`
- `borrowing_base`: `solve_concentration_cap_fixed_point`, `calculate_borrowing_base`,
  `compare_concentration_cap_tiers`, `receivable_valuations_to_dataframe`
- `stress_engine`: `classify_liquidity_regime`, `run_stress_scenario`, `run_stress_sweep`,
  `build_bb_certificate_csv`, `build_bb_certificate_json` (both accept an optional `borrower` name)
- `scenario_analytics`: `evaluate_shock_matrix`, `run_shock_grid`, `run_liquidity_monte_carlo`,
  `run_historical_replay`
- most schema classes and constants from `schemas.py` (the newest result models are not exported)

Not exported from the package; import from their modules:

- `core.data_feed.load_fx_rate`, `core.scenario_analytics.run_liquidity_runway`
- `core.hedge_optimizer.run_hedge_frontier` / `ALL_COMMODITIES`
- `core.crisis_replay.fetch_episode_closes` / `run_crisis_replay` / `CRISIS_EPISODES`
  (Yahoo fetch timeout 15 s, 70-day volatility lookback)
- `core.contingency.build_contingency_plan` / `breaking_moves` / `trigger_ladder`

---

## 5. Domain model

### 5.1 Commodities (`COMMODITY_SPECS` in `core/schemas.py`)

| Ticker | Name | Exchange | Unit | Contract size | Quote scale | Base haircut | Ref vol | Vol slope | Unhedged penalty | Min / max haircut |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| GC=F | Gold | COMEX | troy oz | 100 | 1 | 14% | 15% | 0.50 | 15% | 10% / 65% |
| HG=F | Copper | COMEX | lbs | 25,000 | 1 | 18% | 20% | 0.50 | 15% | 10% / 65% |
| LBR=F | Lumber | CME | mbf | 27.5 | 1 | 24% | 30% | 0.45 | 18% | 10% / 65% |
| CC=F | Cocoa | ICE | MT | 10 | 1 | 28% | 30% | 0.45 | 18% | 10% / 65% |
| KC=F | Coffee (arabica) | ICE | lbs | 37,500 | 0.01 | 26% | 30% | 0.45 | 18% | 10% / 65% |
| SB=F | Sugar No. 11 | ICE | lbs | 112,000 | 0.01 | 22% | 25% | 0.45 | 18% | 10% / 65% |

Coffee and sugar are quoted in US cents per pound on the exchange and on Yahoo. `quote_scale = 0.01` is
applied in `_clean_price_frame`, so every internal price is USD per physical unit. CSV files and live data
stay in exchange quote units.

`REQUIRED_TICKERS = ("GC=F", "HG=F", "LBR=F", "CC=F", "KC=F", "SB=F")`. Order matters for RNG
reproducibility and array columns. `METALS_TICKERS = (GC, HG)`, `SOFTS_TICKERS = (LBR, CC, KC, SB)`.

### 5.2 Default book (current, six commodities)

Positions (physical / hedged): Gold 9,000 / 9,000 oz; Copper 5.5M / 4.675M lbs; Lumber 32,000 / 19,200 mbf;
Cocoa 2,900 / 2,610 MT; Coffee 4.5M / 3.825M lbs (102 contracts); Sugar 44.8M / 35.84M lbs (320 contracts).

Receivables (gross, days past due, investment grade): Barry Callebaut AG 28.0M, 18, IG; Cargill Cocoa &
Chocolate 21.5M, 24, IG; Olam Food Ingredients 8.5M, 32, IG; West Coast Lumber Supply 5.2M, 41, non-IG;
Midwest Copper Fabricators 4.1M, 14, non-IG; Hanseatic Coffee Roasters 9.5M, 21, IG; Coastal Confectionery
Group 3.8M, 37, non-IG; Artisanal Roasters Co. 2.4M, 104 (ineligible), non-IG.

`FacilityTerms` defaults: limit 140M, drawn 133M, unrestricted cash 5.5M, baseline IM 6.0M, single-buyer cap
15%, IM step-up trigger 10%, IM step-up 20% of baseline IM, aging limit 90 days, advance rates IG 85% /
non-IG 75%, default active shock +14%, `margin_line_limit` 0 (no margin line).

Presets (`STRESS_SCENARIO_PRESETS`): Parallel +14% (default), Baseline 0%, West Africa cocoa spike
(+30% CC, +5% others), Brazil coffee frost (+30% KC, +10% SB), Broad deflation crash (-20% all), Metals
rally / lumber slump (+20% GC and HG, -10% LBR).

### 5.3 Formulas

Inventory haircuts (`haircuts.py`):
```
shocked_price    = max(0, spot * (1 + shock))
hedge_ratio      = clip(hedged_qty / physical_qty, 0, 1)        (0 if no physical)
vol_addon        = vol_slope * max(0, realized_vol_30d - ref_vol)
unhedged_penalty = (1 - hedge_ratio) * unhedged_penalty_rate
total_haircut    = clip(base + vol_addon + unhedged_penalty, min_haircut, max_haircut)
eligible         = physical_qty * shocked_price * (1 - total_haircut)
```
Over-hedging caps the hedge ratio at 1 for haircuts, but the full hedged quantity still posts VM. Stacked
bars use a sequential "effective" decomposition (`eff_*` fields) that always sums to exactly 100%.

Receivables and cap (`borrowing_base.py`): over 90 days past due is ineligible; IG 85%, non-IG 75%. The
single-buyer cap is a share of the borrowing base it helps determine:
`BB = BB_inv + sum_j min(AR_j, cap * BB)`. Solved by active-set iteration (max 25 iterations), with an exact
O(n log n) sorted-suffix closed form as fallback when `1 - k * cap <= 1e-9` or iteration does not converge.
An 8-step `WaterfallLine` bridge reconciles gross inventory to the final base within 1e-6.

Stress (`stress_engine.run_stress_scenario`):
```
headroom          = max(0, min(facility_limit, BB) - drawn)
futures_mtm       = -sum(hedged_qty * spot * shock)          (per commodity shock)
vm_outflow        = max(0, -futures_mtm);   vm_inflow = max(0, futures_mtm)
im_addon          = im_stepup_fraction * baseline_IM  if max|shock| >= trigger - 1e-9  else 0
bb_cure           = max(0, drawn - shocked_BB)
margin_line_draw  = min(margin_line_limit, vm_outflow + im_addon)       (never funds the cure)
resources         = cash + shocked_headroom + vm_inflow + margin_line_draw
outflows          = vm_outflow + im_addon + bb_cure
net_liquidity_buffer (NLB) = resources - outflows
```
Regimes, strict priority: `CRITICAL_SQUEEZE` (NLB < 0), then `MARGIN_TRAP_WARNING` (VM not covered by the
margin line exceeds the increase in headroom), then `BB_DEFICIENCY_WARNING` (cure > 0), else `HEALTHY`.
UI labels: Critical squeeze, Margin trap, BB deficiency, Healthy.

Sweep: 41 parallel shocks `np.round(np.linspace(-0.20, 0.20, 41), 6)` (rounding makes +0.10 exact).
Breakevens are linear interpolations: upside where NLB crosses zero, downside where shocked BB crosses drawn.

Scenario analytics (`scenario_analytics.py`):
- Shock grid: 17 x 17, metals shock vs softs/lumber shock, -20% to +20% in 2.5% steps.
- Monte Carlo LaR: winsorized daily log-return covariance over the last 250 observations, Cholesky,
  lognormal horizon moves (1, 5, 10, 20 days), default 5,000 paths, seed 7. LaR 95% / 99% are the 5th / 1st
  percentile of simulated NLB (a buffer level, not a loss); ES 95% is the mean of the worst 5%. Fewer than 20
  returns falls back to reference vols with zero correlation.
- Liquidity runway: 20 trading days of daily correlated paths (same covariance), seed 11, paths = sidebar
  path count. Each day applies the cumulative move since today through the engine; a path "runs short" on
  its first day with NLB < 0. Note: IM step-up follows the engine (based on the current cumulative move),
  so it is not path-dependent.
- Historical replay: today's quantities, receivables and drawn balance run through each day of the loaded
  price history with that day's 30-day vol. Stressed buffer = cash + headroom + rolling 5-day net VM - cure.

Hedge frontier (`hedge_optimizer.py`): for one commodity or all (`ALL_COMMODITIES = "ALL"`), hedge ratio
0% to 100% in 5% steps, each re-valued and run through the 41 shocks. "Liquidity-optimal" = most shocks with
NLB >= 0, ties to higher worst-case NLB, then higher ratio. It deliberately ignores price risk; the chart
shows unhedged value lost in a 20% fall next to it. For a single commodity it often says 0%; that is correct
for liquidity and is explained in the UI.

Crisis replay (`crisis_replay.py`): four episodes in `CRISIS_EPISODES`: `covid_2020` (2020-02-03 to
2020-04-30), `brazil_frost_2021` (2021-07-01 to 2021-08-31), `ukraine_2022` (2022-02-01 to 2022-04-29),
`cocoa_2024` (2024-01-02 to 2024-05-31). Prices fetched from Yahoo with a 70-day volatility lookback. Each
day's cumulative move from the episode start is applied to today's spot, with that day's observed 30-day vol
in the haircuts, through `run_stress_scenario`. Tickers with no history in the window are held at today's
price and listed as not covered.

Contingency funding plan (`contingency.py`):
- `breaking_moves`: per commodity alone, smallest rally (searched to +100% in 0.5% steps) that makes NLB < 0
  and smallest fall that makes shocked BB < drawn; linear interpolation.
- `trigger_ladder`: margin-trap onset, squeeze breakeven, deficiency breakeven and any downside critical
  point, read off the sweep.
- `build_contingency_plan`: for a target scenario (active scenario or worst sweep point), sizes four actions
  that each restore NLB = 0 on their own: raise cash (one for one), enlarge the margin line (only up to
  uncovered margin calls), raise the revolver limit (only if stressed BB exceeds the limit), cut hedges pro
  rata (bisection on a scale factor). Sorted sufficient first, then smallest size. `FundingAction` text
  fields carry no amounts; amounts are in `required_usd` / `capacity_usd` so the UI can format them in the
  reporting currency.

---

## 6. Ground truth figures (seed 42 sample, valuation date 2026-11-20)

Current default book (`tests/test_default_book.py`):

| Metric | Value |
| --- | --- |
| Gross / eligible inventory | 110.07M / 79.80M; hedged notional 93.49M |
| Gross AR / pre-cap eligible AR | 83.00M / 67.20M |
| Borrowing base at 15% cap | 144.94M; Barry Callebaut capped; cutback 2.06M; 2 iterations |
| 17.5% / 20% cap tiers | 147.00M / 147.00M (none capped) |
| Baseline headroom / NLB | 7.00M / +12.50M Healthy (base already above the 140M limit) |
| +1% | NLB +11.57M Margin trap |
| +14% (default) | BB 158.09M, VM 13.09M, IM 1.20M, NLB -1.79M Critical squeeze |
| -20% | BB 126.17M, VM inflow 18.70M, cure 6.83M, NLB +16.16M BB deficiency |
| Breakevens | squeeze +12.09%, deficiency -12.72% |
| Sweep regime counts | Healthy 13, Margin trap 12, BB deficiency 8, Critical squeeze 8 |
| Contingency plan at +14% | shortfall 1.79M; cash / margin line / limit increase 1.79M each; hedge cut 14% |
| Single-commodity breaking rallies | Gold +46.3%, Copper +55.6%, Cocoa +53.1%; Lumber, Coffee, Sugar none to +100% |

Legacy four-commodity book (original specification, pinned in `tests/conftest.py` fixtures `positions`,
`receivables`, `terms`, `sweep`; do not change): BB 114.12M (Barry and Cargill capped, 5.29M cutback),
17.5% tier 118.98M, 20% tier 119.41M, baseline NLB +9.62M, +14% NLB -1.22M, -20% cure 13.65M,
breakevens +12.41% / -5.45%, sweep 15 / 7 / 11 / 8. These prove the engines are unchanged.

Sample calibration: terminal prices GC 2710.00, HG 4.3500, LBR 565.00, CC 8150.00, KC 275.00 cents,
SB 17.10 cents; trailing 30-day vols exactly 15.5%, 23.5%, 38.5%, 51%, 36%, 27% (the last 30 draws are
standardized). Adding KC and SB left the first four series bit-identical (RNG draws appended in ticker order).

Real crisis replay results for the default book (live Yahoo data, as checked on 2026-11-22): Covid 2020 worst
NLB +6.48M (margin trap); Brazil frost 2021 +4.00M; Ukraine 2022 +6.71M; Cocoa 2024 -35.52M on 2024-04-19,
68 of 105 days critical, peak cumulative VM call 46.82M, cocoa +177.8% from episode start. Lumber is not
covered before August 2022.

---

## 7. User interface

### 7.1 Layout

- **Header** (`render_header`): title "Commodity Liquidity Workbench", subtitle "Lademacher Commodities
  (illustrative book) · Borrowing base and Liquidity-at-Risk analytics"; right side shows data status, scenario,
  cap tier, Monte Carlo paths and horizon, commodity count, price date, reporting currency (with rate, source
  and date when not USD).
- **Verdict row** (`render_verdict`): primary panel "Net liquidity buffer under active stress" with regime
  badge, explanation and no-shock baseline; four secondary panels: revolver headroom, availability used,
  LaR 95% at the chosen horizon, squeeze breakeven.
- **Methodology and use case** expander (collapsed) under the verdict.
- **Tabs** (exact labels, asserted in tests): Overview, Borrowing base, Inventory, Receivables, Stress sweep,
  Scenario heatmap, Monte Carlo LaR, Hedging, Funding plan, Replays.

| Tab | Content |
| --- | --- |
| Overview | Credit committee report button (HTML); NLB across parallel shocks chart with regime shading; treasury briefing (findings derived from current numbers); eligible collateral table |
| Borrowing base | Waterfall bridge; covenant position rows; CSV and JSON certificate downloads (USD); reconciliation ledger with bold subtotals |
| Inventory | Validated inventory editor (physical and hedged qty editable); haircut decomposition; 30-day vol history |
| Receivables | Validated receivables editor (add, edit, delete rows); concentration chart; cap tier table; solver check |
| Stress sweep | Two-panel 41-point sweep; sources and uses waterfall; active shock ledger; full sweep table and CSV; methodology |
| Scenario heatmap | KPI row; 17 x 17 metals vs softs heatmap with zero contour and active point |
| Monte Carlo LaR | KPI row (LaR 95/99, ES, probabilities); distribution with red tail; correlation matrix; horizon vol table; liquidity runway section (KPIs, fan chart, cumulative shortfall chart) |
| Hedging | Commodity selector; KPI row; hedge frontier chart (buffer in rally, crash and worst case; unprotected loss) |
| Funding plan | Target selector (active / worst sweep point); KPI row; ranked remedial actions; early-warning triggers; single-commodity reverse stress table |
| Replays | Selector: loaded price history or one of four real episodes; provenance note; KPI row; three-panel chart; daily table and CSV for loaded history |

### 7.2 Sidebar and session-state keys

Do not rename these keys; tests and callbacks depend on them.

| Key | Widget | Default |
| --- | --- | --- |
| `facility_limit_m`, `drawn_revolver_m`, `unrestricted_cash_m`, `baseline_im_m` | number inputs (USD M) | 140, 133, 5.5, 6.0 |
| `margin_line_m` | number input (USD M) | 0 |
| `cap_tier` | radio 15.0 / 17.5 / 20.0 | 15.0 |
| `im_trigger_pct`, `im_addon_pct` | sliders | 10, 20 |
| `feed_mode` | radio: bundled sample / live Yahoo / upload CSV | bundled sample |
| `winsorize` | checkbox (5-sigma return winsorization) | True |
| `report_ccy` | select USD / EUR / CHF / GBP | USD |
| `parallel_shock` | slider -20..20 | 14 |
| `preset_name` + `apply_preset_btn` | select + button (callback) | first preset |
| `override_enabled`, `shock_<TICKER>` | checkbox + six sliders -30..30 | False, 14 |
| `mc_horizon`, `mc_sims` | segmented control, select slider | 5, 5000 |
| `hedge_scope` | select (Hedging tab) | `ALL` |
| `cfp_target` | radio (Funding plan tab) | `active` |
| `replay_source` | select (Replays tab) | `loaded` |
| `reset_btn` | button (callback) | |

Other session state: `editor_nonce` (incremented only by Reset and Apply preset), `inventory_df`,
`receivables_df` (canonical editor baselines), `inventory_edited_df`, `receivables_edited_df` (latest edits;
Apply preset copies them into the baselines before bumping the nonce), `active_regime` (string, used by tests).

Editors: keys `inv_editor_{nonce}` and `ar_editor_{nonce}`. The inventory editor is keyed with
`num_rows="fixed"`, so Streamlit 1.67 keeps edits when calculated column values change; calculated columns
are recomputed from pending edits in `st.session_state[key]["edited_rows"]` each run (no two-click
reversion). Edited rows pass through `InventoryPosition.model_validate` / `ReceivableItem.model_validate`;
invalid rows fall back or are excluded with a warning.

### 7.3 Reporting currency

`ui/theme.py` holds a `ContextVar` with a `DisplayCurrency(code, symbol, usd_per_unit, source, as_of)`.
`fmt_millions`, `fmt_signed_millions`, `fmt_currency`, `money_divisor()` and `money_unit()` read it;
`localize_figure(fig)` swaps "$" and "USD M" text in figures. The model, inputs (labelled USD M), editors and
certificate exports always stay in USD. Explanations shown in the UI come from
`components.regime_explanation_text` (the core classifier's own text is in USD and is used only in exports).
Download-button callables run on another thread, so `build_report_html` receives the currency explicitly and
sets it with `using_currency`.

### 7.4 Credit committee report

`ui/report.py::build_report_html` returns one HTML file (about 5 MB because plotly.js is embedded once, so it
works offline): verdict table, borrowing base bridge and reconciliation, stress sweep, Monte Carlo chart and
table, runway fan, funding plan actions and triggers, footer with assumptions and author. Built lazily only
when the button is clicked (`st.download_button(data=callable)`).

### 7.5 Branding

`HOUSE_NAME = "Lademacher Commodities"` and `AUTHOR` (name, bio, GitHub, repo URLs) in `ui/theme.py`.
Certificates carry `"borrower": "Lademacher Commodities"` and `"currency": "USD"`. The author chose GitHub
only (no LinkedIn, no public email). Every place the house name appears says it is an illustrative book.

---

## 8. Design standards

Light institutional theme, set natively in `.streamlit/config.toml` and mirrored in `ui/theme.py`
(change both together).

- Canvas `#F6F7F9`, surfaces `#FFFFFF`, sidebar `#F0F2F5`, borders `#E2E6EC`, gridlines `#ECEFF4`;
  text `#0F172A` / `#475569` / `#64748B`; accent navy `#1B4965`, hover `#2C6E91`.
- Semantic status colours (`STATUS`), used only for states the model defines: healthy `#166534` on
  `#ECFDF3`; caution `#9A3412` on `#FFF7ED` (margin trap, BB deficiency, data fallback); breach `#991B1B` on
  `#FEF2F2` (critical squeeze, buffer below zero). Never invent thresholds to colour something.
- Commodity colours with line dashes: Gold `#9B8018` solid, Copper `#AA5C35` dash, Lumber `#386A52` dot,
  Cocoa `#6D4A3C` dashdot, Coffee `#3E2C23` longdash, Sugar `#7C8DA6` longdashdot; portfolio `#0F172A`;
  reference lines `#64748B` dashed.
- Fonts: IBM Plex Sans (interface) and IBM Plex Mono (numbers), loaded via theme font URLs with system
  fallbacks. Radius 6 px for panels, 4 px for buttons; 1 px borders; no shadows, gradients, glows, emojis,
  animations or decorative icons.
- Charts: one `apply_institutional_layout()`; titles live in the surrounding panel heading (`chart()` helper
  in `app.py`), never inside the figure; `st.plotly_chart(..., width="stretch", theme=None,
  config=PLOTLY_CONFIG)`; single y-axis per panel (use `make_subplots` rows, never dual axes); waterfall
  totals use `y=None`; add `add_vrect`/`add_hrect` to a subplot only after it has a trace (Plotly silently
  drops shapes on empty subplots).
- White analytical panels are keyed containers `st.container(key="panel_<name>")`, styled through the
  stable `st-key-panel_*` class.
- Tables: hide indices; money in the reporting currency with the unit in the header (`_money_column`, values
  like `-24.35` under "Amount (USD M)"); bold subtotals via a pandas Styler. Long text goes in wrapped HTML rows
  (`render_action_list`, `render_trigger_list`, `render_audit_rows`), not narrow dataframe columns.
- Copy: plain, factual, sentence case, no buzzwords, no em-dashes or en-dashes (a test scans `app.py`,
  `ui/*.py` and `README.md` for em-dashes). Units on every input label.
- Checked layouts: 1440 x 900 (primary), 1280 x 800, 1920 x 1080 (content capped at 1480 px), 390 px wide
  (Overview only).

Library guardrails (also in `CLAUDE.md`): never `use_container_width`; no `applymap`, `inplace=True`,
chained assignment or `dtype == object`; never pass user edits through `model_copy(update=...)` (it skips
validation; use `model_validate`); Plotly title dicts, hex or 0-255 rgba colours, no `titlefont`; yfinance
`download(..., auto_adjust=True, progress=False, threads=False, timeout=...)`, no custom requests session.

---

## 9. Data sources and resilience

| Source | Use | Fallback |
| --- | --- | --- |
| `data/sample_prices.csv` | default; offline; deterministic seed 42 synthetic series | regenerated automatically if missing |
| Yahoo Finance via yfinance (no key) | live prices (`feed_mode` live), 1 year | any failure, empty frame (how HTTP 429 shows), missing ticker, short history or 10 s timeout -> bundled sample, header shows "Bundled data (live feed unavailable)" |
| Uploaded CSV | columns Date, GC=F, HG=F, LBR=F, CC=F, KC=F, SB=F in exchange quote units | rejected file -> bundled sample with message; under 20 returns -> reference vols |
| Yahoo FX `EURUSD=X`, `GBPUSD=X`, `CHFUSD=X` | reporting currency | bundled closes from 2026-11-20: EUR 1.096972, GBP 1.308644, CHF 1.184694 USD per unit |
| Yahoo history for four crisis windows | Replays tab | without internet the tab shows a notice; the rest of the app is unaffected |

Important honesty points already in the UI: bundled prices are labelled synthetic (not observed history);
sample correlations are near zero by construction (the MC tab says so); crisis replays list uncovered tickers;
the house is illustrative.

---

## 10. Testing

Run `.venv/bin/python -m pytest`. 154 tests, no warnings:

| File | Tests | Guards |
| --- | ---: | --- |
| test_architecture_and_app.py | 30 | core import boundary, banned patterns, no em-dashes, no local paths, chart JSON serialization, Streamlit `AppTest` runs (load, regimes at +14 / 0 / -20, cap tier, presets, reset, editor edits, MC controls, per-commodity overrides, hedging scope, offline crisis replay, margin line, EUR currency, funding plan target), exact tab labels |
| test_data_feed.py | 29 | sample calibration, cents scaling, vol math, winsorization, cleaning, live loader fallbacks |
| test_borrowing_base.py | 19 | aging, advance rates, cap tiers, waterfall identity, solver edge cases, 1,000-portfolio fuzz test |
| test_stress_engine.py | 17 | legacy regimes, IM boundary symmetry, breakevens, mixed shocks, certificate export |
| test_new_analytics.py | 14 | hedge frontier, runway, crisis replay with faked Yahoo, margin line (exact and vectorized), contingency plan actions verified by re-running the engine, breaking moves |
| test_haircuts.py | 13 | haircut checkpoints, clamping, over-hedging, zero inventory |
| test_default_book.py | 10 | current six-commodity ground truth (section 6) |
| test_cache_helpers.py | 8 | cache wrappers equal direct engines, data status labels, live timeout |
| test_scenario_analytics.py | 8 | vectorized engine equals exact engine, grid, Monte Carlo, replay |
| test_presentation.py | 6 | currency context, `localize_figure`, FX fallback, certificate borrower, HTML report in USD and EUR |

No test touches the network (Yahoo calls are monkeypatched or faked). In `AppTest`, never call `.set_value()`
on a data editor; write `at.session_state["inv_editor_<nonce>"]` edit dicts instead.

Verification habit used for presentation-only changes: hash all core outputs before and after (borrowing base,
stress, sweep, tiers, grid, Monte Carlo samples, replay, certificate) and confirm they are identical; confirm
`git diff -- core/ data/` is empty.

---

## 11. Decision log (chronological, with reasons)

1. **Initial build** to a detailed specification: four commodities, exact audited checkpoints, `core/`/`ui/`
   split, Pydantic v2, CoW-safe pandas, 41-point sweep. Python 3.12 venv because system Python is 3.9.
2. **Dark visual pass, then replaced.** A first dark, gradient-heavy redesign was later replaced by the owner's
   request for a restrained light institutional design (section 8). Gauge, donut and regime ribbon were
   removed; the gauge's 85% / 95% bands were invented thresholds and violated the "model-defined states only"
   rule.
3. **Extended analytics:** vectorized engine (validated against the exact engine), 17 x 17 heatmap, Monte
   Carlo LaR, historical replay, Overview cockpit.
4. **Deployment readiness:** caching layer, live-feed timeout and fallback, data-status indicator,
   methodology expander, headless config, error details limited to type, bounded requirements, README brief.
5. **Moved out of the scratch workspace** to `~/commodity-liquidity-workbench`; `.venv` rebuilt; git initialised
   with repo-local identity; pushed to GitHub; deployed to Streamlit Cloud (Python 3.12, `app.py`, no secrets).
6. **Coffee and sugar added** (owner's request). Required a recalibrated facility (140M / 133M / 5.5M / 6.0M)
   so all four regimes still appear and +14% stays critical; Barry Callebaut raised to 28M so the concentration
   cap still binds. The original four-commodity book was pinned in tests as an engine regression.
7. **Hedge-ratio optimizer, liquidity runway, real crisis replays** (owner picked all three). Crisis replays
   apply relative moves to today's book rather than historical levels; uncovered tickers held flat and
   disclosed rather than proxied.
8. **Research-backed round** (FSB, FCA, LME Clear guidance on contingency funding plans and reverse stress
   tests; 2022 reporting on bank margin lines for traders). Owner chose: contingency funding plan, margin
   financing line, author panel and branding, credit committee report, reporting currency; audience =
   trading-house desk; bio "Commodity trade finance and risk analytics"; GitHub only; house name
   "Lademacher Commodities". The margin line defaults to 0 so ground truth stays unchanged.

---

## 12. Known limitations

- **Bundled data is synthetic**, with independent draws per commodity, so correlations are near zero and
  Monte Carlo tails on the sample are mild (for example P(critical squeeze) 0% at 5 days). Live data gives
  realistic co-movement.
- **Model simplifications:** quantities, receivables and drawn balance held fixed; price shocks do not change
  volatility (except in crisis replays, which use observed vol); IM step-up is a single fraction of baseline
  IM triggered by the largest move, not an exchange per-contract or SPAN-style model; no basis risk (futures
  and physical move together); margin settled in cash on the shock day; runway IM step-up is not
  path-dependent; no funding costs or interest.
- **Hedge optimizer** optimizes liquidity only and can recommend 0% for a single commodity; read it with the
  unprotected-loss panel.
- **Crisis replays need internet**; CME physical lumber (`LBR=F`) has no Yahoo history before August 2022, so it
  is held flat in the 2020, 2021 and 2022 episodes. No proxy exists on Yahoo (`LB=F` and `LBS=F` are delisted).
- **Reporting currency** converts outputs only; inputs and exports remain USD.
- **Visual coverage:** CHF and GBP are unit-tested but were not viewed in the browser; mobile (390 px) was
  checked for the Overview only; the report download click itself was not exercised in a browser (the same
  report was generated in code and inspected).
- **Report size** is about 5 MB because plotly.js is embedded for offline use.
- **Public commit history** contains the owner's Gmail address (owner was told; noreply rewrite was offered).
- **GitHub token** in the keychain expires 2026-12-22.
- **Streamlit-version-specific behaviour:** the inventory editor relies on Streamlit 1.67 keeping edits for
  keyed fixed-row editors when data values change; `requirements.txt` allows `>=1.52,<2`, and older 1.5x
  versions may reset edits.

---

## 13. Remaining tasks and backlog

Open items:
1. README screenshot placeholder: save an Overview capture as `docs/dashboard.png` and replace the
   placeholder line with `![Workbench overview](docs/dashboard.png)`.
2. Visual check of CHF and GBP modes, the Hedging / Funding plan / Replays tabs at 390 px and 1280 px, and a
   real click of the report download.
3. Consider raising the Streamlit floor in `requirements.txt` to the tested version (1.67) to lock in the
   editor behaviour, or test on 1.52.

Ideas offered but not yet chosen by the owner (good next candidates):
- **Exchange-style initial margin:** per-commodity IM per contract (editable, seeded from published ICE/CME
  margin notices and labelled as such), rising with volatility, replacing the flat 20% step-up. ICE raised
  cocoa margins several times in 2024 (for example a 2024-03-15 notice listed 6,910 USD initial margin for the
  May 2024 contract).
- **Basis risk:** per-commodity basis (cash minus futures) shock so hedges are imperfect.
- **Save and load your own book:** download / upload positions, receivables, facility and scenario as one file.
- Other ideas raised earlier: funding cost and carry, receivable dilution reserves (left out because it changes
  calibrated figures), LinkedIn link in the author panel if the owner provides a URL.

How to work on this project (the owner's preferred workflow):
- The owner describes changes in plain words; for open-ended requests, offer options as a click-through
  question first, then build what they pick.
- Keep financial results unchanged unless the change is the point; if a request moves the calibrated figures,
  say so, update `tests/test_default_book.py`, `README.md`, `CLAUDE.md` and this file, and keep the legacy book
  tests passing.
- After changes: run the full test suite, check affected tabs in a browser at 1440 x 900 (restart the server
  after editing `ui/` or `core/`), commit with the attribution line, `git push`, then confirm the live app
  redeployed.
