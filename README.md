# Commodity Collateral, Borrowing Base & Liquidity Stress Workbench

> **Live app:** https://commodity-liquidity-workbench.streamlit.app
>
> **Screenshot:** save a capture of the Overview tab as `docs/dashboard.png`, then replace this line with
> `![Workbench overview](docs/dashboard.png)`

A Streamlit desk tool for the treasury, trading and risk team of a physical commodity merchant financed
by an asset-based revolving facility. The modelled book belongs to Lademacher Commodities, an illustrative
trading house. It
values a hedged Gold, Copper, Lumber, Cocoa, Coffee and Sugar book plus a trade receivables ledger, produces a borrowing
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
| **Scenario heatmap** | Which mixed moves are dangerous? | Metals (Gold, Copper) and softs/lumber (Lumber, Cocoa, Coffee, Sugar) shocked independently on a 17 x 17 grid, with the zero-buffer contour drawn. |
| **Monte Carlo LaR** | How bad is a bad week, and how long can we last? | Thousands of correlated price paths over 1, 5, 10 or 20 trading days, drawn from the recent return covariance via Cholesky decomposition. Reports LaR 95% and 99% (5th and 1st percentile buffer), expected shortfall and regime probabilities. The liquidity runway follows daily paths for 20 trading days: a percentile fan of the buffer and the share of paths that have run short by each day. |
| **Hedging** | How much should we hedge? | Sweeps the hedge ratio of one commodity, or all at once, from 0% to 100%. Each ratio is re-valued (the unhedged haircut penalty changes) and run through the 41 parallel shocks. The liquidity-optimal ratio keeps the buffer at or above zero in the most shocks; it is shown next to the inventory value left unprotected in a 20% fall. |
| **Funding plan** | What breaks us, and what fixes it? | A contingency funding plan as regulators ask of commodity traders: single-commodity reverse stress test (smallest rally that exhausts liquidity, smallest fall that opens a deficiency), early-warning triggers read off the sweep, and the size of each remedial action (cash, margin financing line, revolver accordion, pro rata hedge cut) that restores a zero buffer, each verified with the stress engine. |
| **Replays** | How would today's book have fared in real stress? | Either the loaded price history (borrowing base, daily margin, 5-day stressed buffer), or observed Yahoo Finance futures prices for four real episodes: Covid-19 crash 2020, Brazil coffee frost 2021, Russia-Ukraine shock 2022, cocoa squeeze 2024. Each day's cumulative move is applied to today's spot prices, with that day's observed 30-day volatility in the haircuts. Commodities without history in a window (CME physical lumber before August 2022) are held at today's price and flagged. |

**Desk features**

- **Margin financing line.** An optional bank line (sidebar, default none) that funds variation and
  initial margin calls but not a borrowing base deficiency cure, as banks provided to traders in 2022.
- **Reporting currency.** Outputs in USD, EUR, CHF or GBP at a live Yahoo Finance rate, with a bundled
  2026-11-20 rate as fallback. Inputs, the model and certificate exports stay in USD.
- **Credit committee report.** One-click self-contained HTML report (verdict, borrowing base, stress
  sweep, LaR, runway, funding plan) that opens without the app.

**Key assumptions**

- Quantities, the receivables ledger and the drawn balance are held fixed across scenarios.
- Haircuts use the current 30-day realized volatility; a price shock does not change volatility.
- Variation margin is settled in cash on the shock day; initial margin steps up by a set fraction once
  any commodity moves by more than the trigger (default 10%).
- Liquidity regimes are assigned in priority order: Critical Squeeze (buffer below zero), Margin Trap
  (margin calls exceed new headroom), BB Deficiency (base below drawn balance), Healthy.

**Default book (bundled data, valuation date 2026-11-20)**

Six commodities: Gold (COMEX), Copper (COMEX), Lumber (CME), Cocoa (ICE), Coffee (ICE arabica, 37,500 lbs
per contract) and Sugar No. 11 (ICE, 112,000 lbs per contract). Coffee and sugar are quoted in US cents
per pound and converted to USD per pound on load.

| Metric | Value |
| --- | --- |
| Borrowing base at 15% buyer cap | $144.94M (one buyer capped, $2.06M cutback) |
| Facility limit / drawn | $140.0M / $133.0M |
| No shock | Net buffer +$12.50M, Healthy |
| +1% parallel shock | Margin trap: the base already exceeds the limit, so a rally adds margin calls but no availability |
| +14% parallel shock | Net buffer -$1.79M, Critical squeeze |
| Breakevens | Squeeze at +12.09%, deficiency at -12.72% |

The original four-commodity book and its audited figures ($114.12M base, -$1.22M at +14%) remain pinned
in the test suite as a regression check on the engines.

## Data & resilience

- **Bundled sample (default).** `data/sample_prices.csv` holds 260 business days of front-month prices
  for `GC=F`, `HG=F`, `LBR=F`, `CC=F`, `KC=F` and `SB=F`, generated deterministically (`seed=42`) and calibrated to
  realistic levels and 30-day volatilities. The app opens on this data with no network access, keys or
  uploads required.
- **Live feed (optional).** Yahoo Finance via `yfinance`, no API key. The response is checked for empty
  frames (how rate limits surface), missing tickers, short history and non-positive prices. Any failure
  or a 10-second timeout falls back to the bundled data, and the header shows which source is active.
- **Custom CSV (optional).** Columns `Date, GC=F, HG=F, LBR=F, CC=F, KC=F, SB=F`, in exchange quote units. Invalid files fall back to the
  bundled data with a message. Short files use reference volatilities when 30-day vol cannot be computed.
- **Caching.** Live loads are cached for 15 minutes; the stress sweep, heatmap, replay and Monte Carlo
  runs are cached on their full inputs, so slider moves and tab switches do not recompute unchanged work.
- **Crisis episodes.** Fetched from Yahoo Finance on first use and cached for a day. Without internet
  the Replays tab shows a short notice and every other view keeps working.
- **Secrets.** None are required. No credentials are read or stored.

## Author

Built by **Tjark Lademacher**, commodity trade finance and risk analytics.
GitHub: [tjarklademacher-cloud](https://github.com/tjarklademacher-cloud).
Lademacher Commodities is an illustrative book; positions, receivables and facility terms are not real.

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
