# Commodity Collateral, Borrowing Base & Liquidity Stress Workbench

> **Live app:** https://commodity-liquidity-workbench.streamlit.app
>
> **Screenshot:** save a capture of the Overview tab as `docs/dashboard.png`, then replace this line with
> `![Workbench overview](docs/dashboard.png)`

A Streamlit desk tool for a physical commodity merchant financed by an asset-based revolving facility. It
values a hedged Gold, Copper, Lumber and Cocoa book plus a trade receivables ledger, produces a borrowing
base certificate, and measures how much cash and revolver headroom remain when prices move.

## Commercial problem

A merchant that hedges physical inventory with short futures is protected on price but not on cash.
When prices rally, the futures lose money that must be paid in variation margin the same day, and
exchanges raise initial margin as volatility rises, while the matching gain on the physical stock
only turns into cash when the goods are sold or the lender advances against them. That advance is
capped by haircuts, single-buyer concentration limits and the committed facility size, so a solvent,
fully hedged book can still run out of liquidity in a spike. In a sell-off the borrowing base shrinks
below the drawn balance and the lender can require a deficiency paydown.

## Architecture & modules

```
core/   pure Python models (no Streamlit or Plotly imports, enforced by tests)
ui/     design system (theme.py), Plotly charts, Streamlit components, cache wrappers
app.py  Streamlit entry point
data/   bundled price history used offline and as the live-feed fallback
tests/  pytest suite, including Streamlit AppTest end-to-end runs
```

| Module | What it answers | Logic in plain terms |
| --- | --- | --- |
| **Overview** | Is the facility healthy right now? | Net liquidity buffer across parallel shocks with regime shading, a treasury briefing built from the current numbers, and the eligible collateral mix. A verdict row above all tabs shows the active-stress buffer, headroom, availability used, LaR 95% and the squeeze breakeven. |
| **Borrowing base** | What can we borrow today? | Eligible inventory plus eligible receivables, bridged step by step from gross values, with CSV and JSON certificate export. |
| **Inventory** | Why is each advance rate what it is? | Haircut = base rate + volatility add-on above a reference vol + penalty for the unhedged share, clamped to 10% to 65%. |
| **Receivables** | How much does buyer concentration cost? | Receivables over 90 days past due are excluded; investment-grade buyers advance at 85%, others at 75%. No single buyer may exceed a cap percentage of the borrowing base, which is solved as a fixed point because the cap depends on the base it helps set. |
| **Stress sweep** | At what price move do we run out of cash? | Parallel shocks from -20% to +20% in 1% steps. Net liquidity buffer = cash + headroom + margin inflows, minus margin calls, initial margin step-up and deficiency cure. Breakevens are interpolated between grid points. |
| **Scenario heatmap** | Which mixed moves are dangerous? | Metals (Gold, Copper) and softs/lumber (Lumber, Cocoa) shocked independently on a 17 x 17 grid, with the zero-buffer contour drawn. |
| **Monte Carlo LaR** | How bad is a bad week? | Thousands of correlated price paths over 1, 5, 10 or 20 trading days, drawn from the recent return covariance via Cholesky decomposition. Reports LaR 95% and 99% (5th and 1st percentile buffer), expected shortfall and regime probabilities. |
| **Historical replay** | How would today's book have fared last year? | Today's positions, receivables and drawn balance run through each past trading day's prices and 30-day volatility: borrowing base, daily futures margin and a 5-day stressed buffer. |

**Key assumptions**

- Quantities, the receivables ledger and the drawn balance are held fixed across scenarios.
- Haircuts use the current 30-day realized volatility; a price shock does not change volatility.
- Variation margin is settled in cash on the shock day; initial margin steps up by a set fraction once
  any commodity moves by more than the trigger (default 10%).
- Liquidity regimes are assigned in priority order: Critical Squeeze (buffer below zero), Margin Trap
  (margin calls exceed new headroom), BB Deficiency (base below drawn balance), Healthy.

**Default book (bundled data, valuation date 2026-11-20)**

| Metric | Value |
| --- | --- |
| Borrowing base at 15% buyer cap | $114.12M (two buyers capped, $5.29M cutback) |
| Facility limit / drawn | $115.0M / $109.0M |
| +14% parallel shock | Net buffer -$1.22M, Critical Squeeze |
| Breakevens | Squeeze at +12.41%, deficiency at -5.45% |

## Data & resilience

- **Bundled sample (default).** `data/sample_prices.csv` holds 260 business days of front-month prices
  for `GC=F`, `HG=F`, `LBR=F` and `CC=F`, generated deterministically (`seed=42`) and calibrated to
  realistic levels and 30-day volatilities. The app opens on this data with no network access, keys or
  uploads required.
- **Live feed (optional).** Yahoo Finance via `yfinance`, no API key. The response is checked for empty
  frames (how rate limits surface), missing tickers, short history and non-positive prices. Any failure
  or a 10-second timeout falls back to the bundled data, and the header shows which source is active.
- **Custom CSV (optional).** Columns `Date, GC=F, HG=F, LBR=F, CC=F`. Invalid files fall back to the
  bundled data with a message. Short files use reference volatilities when 30-day vol cannot be computed.
- **Caching.** Live loads are cached for 15 minutes; the stress sweep, heatmap, replay and Monte Carlo
  runs are cached on their full inputs, so slider moves and tab switches do not recompute unchanged work.
- **Secrets.** None are required. No credentials are read or stored.

## Local setup & testing

Requires Python 3.11 or later.

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python -m streamlit run app.py
```

```bash
.venv/bin/python -m pytest
```

To rebuild the bundled price file:

```bash
.venv/bin/python -m core.data_feed --generate-sample
```

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io), create an app from the repository with main
   file `app.py`, and pick Python 3.12 under Advanced settings.
3. No secrets are needed.
