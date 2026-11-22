# CLAUDE.md: Commodity Collateral & Liquidity Stress Workbench

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
| `ui/theme.py` | Design system: light institutional tokens (mirrored in `.streamlit/config.toml`), semantic STATUS colours, commodity palette, Plotly layout, minimal CSS for custom HTML and `panel_*` containers |
| `ui/charts.py` | The original six Plotly figures plus gauge, donut, regime ribbon, heatmap, MC, correlation, replay |
| `ui/cache.py` | `st.cache_data` wrappers (live TTL 900s, scenario suite, Monte Carlo) and data-status label |
| `ui/components.py` | Hero, KPI cards with sparklines, regime banner, insights, validated editors, exports, audits |
| `app.py` | Entry point: branded 4-section sidebar, hero + regime banner, methodology expander, 8 tabs |

## Zero-leakage boundary (enforced by `tests/test_architecture_and_app.py`)

- `core/` must never import `streamlit` or `plotly`, directly or transitively. AST scan + subprocess
  `sys.modules` check enforce this.
- All widgets, figures, CSS and session state live in `ui/` and `app.py`.
- `core/__init__.py` exports lazily (PEP 562) so `python -m core.data_feed` has no runpy double-import warning.

## Ground truth (seed=42 sample); tests assert these to $0.01M

- Default six-commodity book (`tests/test_default_book.py`): BB $144.94M (Barry capped, $2.06M cutback),
  +14% NLB -$1.79M critical, -20% cure $6.83M, breakevens +12.09% / -12.72%, sweep 13/12/8/8.
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

- `evaluate_shock_matrix` must stay identical to `run_stress_scenario`; `tests/test_scenario_analytics.py`
  checks it on random mixed shocks. If you change any stress formula, change both.
- Custom HTML goes through `_html()` / single-line strings so Markdown never renders indented HTML as code.

- UI copy and docs: plain, factual, sentence case, no em-dashes or emojis.
- Visual changes go through `ui/theme.py` tokens; risk colours (`STATUS`) only for states the model defines.
  Chart titles live in the surrounding `panel()` heading, not inside the Plotly figure.

**Numerics**
- Sweep grid `np.round(np.linspace(-0.20, 0.20, 41), 6)`; IM trigger uses `>= trigger - 1e-9`.
