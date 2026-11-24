# Implementation status: institutional model upgrade

Working document so a later session can continue. Update it after each phase.

## Phase 0 audit (2026-11-23, branch `feature/advanced-liquidity-model`, commit 649162b)

Baseline: `.venv/bin/python -m pytest -q` = 235 passed in about 60 s. No CI workflow existed.

### What already exists and is reused

| Area | Existing implementation | Gap against the upgrade brief |
| --- | --- | --- |
| Daily ledger | `core/cashflow_engine.py`: VM, IM, collections, interest, revolver with delay, margin line, cure after grace, sweep; vectorized over paths | No physical purchases or sales, no payables, no restricted cash line, no commitment fee, no dynamic haircuts, no credit events, horizon capped at 60 days, receivable collections identical on every path, no explicit cash reconciliation residual, no net-worth identity |
| Margin | `core/margin_engine.py`: whole contracts, signed VM, per-contract IM, step-up trigger | IM does not respond to volatility; no percentage-of-notional mode |
| Haircuts | `core/haircuts.py`: base + vol add-on + unhedged penalty, clamped; vol is fixed at the 30-day realized vol of the valuation date | Haircut does not evolve along a path; no liquidity, concentration, quality, age or lender-stress components |
| Haircut vs advance rate | Inventory: contribution = MV x (1 - haircut) (the haircut is the advance-rate complement). Receivables: contribution = (gross - dilution reserve) x advance rate. No double deduction in either. | Convention not selectable or documented in the UI |
| Basis | `core/basis_risk.py`: percentage basis, economic P&L identity, presets | No mean-reverting basis process, no curve, no roll, no unit or currency conversion |
| Simulation | `core/market_calibration.py`: Gaussian, stationary block bootstrap, stressed covariance; PSD repair; order-statistic intervals; margin backtest | No Student t, EWMA or GARCH; no correlation regimes; no coverage test; no convergence or seed analysis |
| LaR | `scenario_analytics.run_liquidity_monte_carlo`: "LaR 95%" is the 5th percentile of the buffer level | Convention is a level, not a deterioration; kept for legacy but the new engine reports deterioration-based LaR with the convention stated |
| Credit | Aging limit, IG / non-IG advance rates, concentration cap, dilution reserve | No PD, LGD, ECL, delays, defaults, insurance, migration or correlated defaults |
| Hedge optimizer | `hedge_optimizer.py` and `advanced_analytics.run_advanced_hedge_frontier`: liquidity frontier over 41 deterministic shocks plus a no-weights Pareto view | No stochastic economic risk, no funding distribution, no constraints |
| Crisis replay | `crisis_replay.py`: four Yahoo episodes applied to today's book | No 2022 European energy case, no hypothetical merchant |

### Hardcoded assumptions found

- `REQUIRED_TICKERS` (six commodities) is assumed by every existing engine. The new engines are instrument-agnostic
  (arrays indexed by instrument), with an adapter from the six-commodity book.
- Illustrative IM shares, haircut parameters and basis presets are labelled illustrative in code and UI.
- Legacy "LaR" is a 5th percentile buffer level; the term is redefined only in the new engine (documented).

### Discrepancies between documentation and code

- PROJECT_CONTEXT.md section 2 and 10 cite 154 and 216 tests; the suite has 235.
- PROJECT_CONTEXT.md header says last updated at `fa0bb0c` on `main`; current work is on the feature branch.

## Plan

Architecture: one new daily engine (`core/treasury_engine.py`) is the single source of truth for path-dependent
treasury state in the new analytics. Inputs are generic instruments, physical positions, futures positions,
counterparties and facility terms (`core/treasury_schemas.py`). Collateral (`core/collateral.py`), credit
(`core/credit_risk.py`), curve, basis and units (`core/futures_curve.py`, `core/units.py`) and return models
(`core/return_models.py`) are pure functions the engine calls. Metrics (`core/liquidity_metrics.py`),
validation (`core/model_validation.py`), the hedge frontier (`core/hedge_frontier.py`) and the 2022 case study
(`core/case_study_2022.py`) sit on top. The legacy and advanced engines are unchanged, so every pinned figure
is preserved; a cross-check test shows the new engine equals the existing ledger where their features overlap.

| Phase | Content | Status |
| --- | --- | --- |
| 0 | Audit, baseline, plan | done |
| 1 | State objects, daily sequence, treasury engine, reconciliation and net-worth tests | done |
| 2 | Dynamic haircuts, borrowing base conventions, vol-sensitive IM, financing waterfall | done |
| 3 | Units, currency, curve, roll accounting, OU basis | done |
| 4 | Counterparty model, ECL, delays, defaults, insurance, migration, correlated defaults | done |
| 5 | Student t, EWMA, GARCH, bootstrap, correlation regimes, path MC, risk metrics, validation | done |
| 6 | Multi-objective hedge frontier | done |
| 7 | UI tabs, exports, 2022 case study | done |
| 8 | Tests, CI, methodology, README, PROJECT_CONTEXT | done (CI not yet run on GitHub) |

## Result (2026-11-23)

- Tests: 317 passed locally (235 before; 70 new financial tests, 12 new architecture checks from the new core
  modules). Two existing AppTest assertions on tab labels were updated for the three new tabs; no other existing
  test was changed and no pinned figure moved.
- Cross-check: the new engine equals the advanced ledger (cash, debt, borrowing base, IM) on the default book
  where their features overlap.
- Benchmarks and reference figures: PROJECT_CONTEXT.md section 15.
- Not done or not verified: the GitHub Actions workflow has not run yet (ruff was not installable locally; an
  AST-based check for unused imports, unused locals and empty f-strings was run instead); the HTML credit
  committee report does not include the new sections; legacy Monte Carlo, runway and funding plan still use the
  legacy engine; the new and advanced ledgers are not yet consolidated; mobile layout of the new tabs was not
  checked.
