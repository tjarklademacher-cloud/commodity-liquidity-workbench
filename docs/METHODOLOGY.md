# Methodology: daily treasury, collateral, credit and simulation models

This document specifies the models added in the institutional upgrade (Treasury, Credit and Case study 2022
tabs, and the stochastic hedge frontier). The legacy instantaneous stress model and the advanced instantaneous
model are documented in `PROJECT_CONTEXT.md` sections 5 and 14 and are unchanged.

The workbench is a research and educational financial modelling project. It is not a validated production
trading, treasury or credit decision system, it has not been independently validated, and it makes no claim
of regulatory compliance. Every default parameter is an illustrative assumption unless stated otherwise.

## 1. Purpose and architecture

A physical commodity merchant that hedges inventory with short futures is protected against price changes in
economic terms but not in cash terms. Variation margin on the futures is paid daily; the offsetting gain on the
physical stock becomes cash only when sold and collected, or when a lender advances against it within haircuts,
eligibility rules and a committed limit. Volatility raises initial margin and haircuts at the same time, and
customers may pay late or default when markets are stressed. The models below make these timing and financing
relationships explicit.

```
market paths (core/return_models.py, core/futures_curve.py)
  -> physical and futures valuation, basis               (MarketPaths: futures curve, physical, path volatility)
  -> daily variation margin and initial margin            (core/treasury_engine.py, MarginPolicy)
  -> receivable payments, delays, defaults, insurance     (core/credit_risk.py -> CreditOutcomes)
  -> collateral eligibility and dynamic haircuts          (core/collateral.py)
  -> borrowing base, availability, deficiency             (core/collateral.py)
  -> funding waterfall and cash reconciliation            (core/treasury_engine.py)
  -> liquidity metrics                                    (core/liquidity_metrics.py)
  -> scenarios, Monte Carlo, hedge frontier, case study   (core/treasury_analytics.py, core/hedge_frontier.py,
                                                           core/case_study_2022.py)
  -> presentation                                         (ui/treasury_views.py, ui/treasury_charts.py)
```

`core/treasury_engine.run_treasury` is the single source of truth for path-dependent state in these views.
Every other module either prepares its inputs or summarizes its ledger. The engine has no iterative solver:
the only fixed point (the single-buyer concentration cap) is solved in closed form.

## 2. State variables and units

All money is in the run's functional currency (USD for the workbench book, EUR for the case study). Quantities
are in each instrument's physical unit (`core/units.py`); futures quantities are converted with the contract
size before any hedge ratio is computed. Arrays are `(paths, days + 1)`; day 0 is the opening position.

| Group | Variables (ledger field names) |
| --- | --- |
| Cash | `cash` (unrestricted), `restricted_cash` = `im_posted` + `im_release_pending` + other restricted, `shortfall` |
| Inventory | owned quantity, in-transit quantity, eligible stock, `inventory_value`, `inventory_eligible_value`, `avg_haircut` |
| Receivables | `receivables` (face outstanding), `eligible_receivables`, `overdue_receivables`, `defaulted_exposure`, `claims_receivable` |
| Payables | `payables`, `supplier_payments` |
| Financing | `revolver_drawn`, `revolver_undrawn`, `permitted`, `borrowing_base`, `reported_borrowing_base`, `deficiency`, `mandatory_repayment`, `lines_drawn`, `emergency_drawn`, `in_transit_funding` |
| Derivatives | `vm_net`, `vm_paid`, `vm_received`, `cum_vm`, `im_call`, `im_released`, `transaction_costs` |
| Results | `available_liquidity`, `liquidity_headroom`, `networth`, `economic_pnl`, `credit_loss`, `reconciliation` |

## 3. Calendar and timing

Business days run Monday to Friday from the start date (exchange holidays are not modelled). Contractual terms
in calendar days (payment terms, claim delays) are mapped to the first business day on or after the calendar
date. Interest and fees accrue over calendar days between business days (a Monday pays three days).
Drawdown and line delays, IM release lags, transit times and redetermination intervals are in business days.

## 4. Daily sequence of events

On each business day t, identically on every path:

1. Market data for t: futures settlement prices, physical prices, path volatility.
2. Variation margin on positions held over the day: VM(t) = sum over contracts of signed contracts x contract
   size x (F(t) - F(t-1)), netted across instruments. A short position pays when prices rise. Rolls and hedge
   adjustments are executed at the close at settlement prices.
3. Physical operations: storage cost on stock held, purchases (title passes and a payable is created on the
   trade day; stock is usable after transit), sales (stock leaves and an invoice is issued).
4. Receivables: defaults are settled first, then scheduled payments, partial payments and disputed remainders.
5. Payables due, operating expenses, interest on opening balances, commitment fee on undrawn commitment.
6. Initial margin re-set from close positions: calls paid, excess released after the release lag.
7. Funding requested on earlier days arrives (debt rises on arrival).
8. Collateral, borrowing base; the reported base updates on redetermination days only.
9. Deficiency and mandatory repayment after the grace period.
10. Repayments: margin-only lines from margin receipts; surplus cash above the sweep target to the revolver
    and other lines.
11. Funding waterfall for cash below the minimum operating level.
12. Close and record. An unfunded shortfall is cash below the minimum operating level after same-day funding.

## 5. Cash reconciliation and net-worth identity

Closing cash = opening cash + collections + insurance receipts + estate recoveries + VM received + IM released
+ financing received - supplier payments - storage - operating expenses - interest and fees - VM paid - IM
posted - transaction costs - mandatory repayments - voluntary repayments.

The engine records the absolute residual every day (`reconciliation`) and its maximum
(`TreasuryRun.reconciliation_residual`); tests require it below 1e-6 on hand cases and below 1e-4 on the
default book (floating-point accumulation over values of order 1e8).

Net worth NW = unrestricted cash + restricted cash + inventory at physical price + receivables + claims
receivable - payables - debt. The identity

NW(t) - NW(t-1) = inventory revaluation + futures P&L + trade margin - storage - operating expenses - interest
- fees - transaction costs - realized credit losses

is checked every day (`networth_residual`). Consequences, each covered by a test: posting initial margin moves
cash into restricted cash and leaves net worth unchanged; variation margin is counted once (as futures P&L
and as cash, never twice in P&L); inventory appreciation changes net worth but not cash; drawing debt raises cash
and debt equally.

## 6. Financing waterfall and facility mechanics

Sources in order: (1) unrestricted cash above the minimum operating level; (2) the committed revolver within
undrawn availability; (3) committed lines in configured order (a margin-only line funds only the day's VM paid
and IM calls); (4) emergency funding, only if explicitly enabled; (5) the remainder is an unfunded shortfall.

Borrowing base (`core/collateral.borrowing_base`):

- Inventory contribution = sum over eligible stock of MV x (1 - haircut) [x inventory advance rate under the
  "haircut_and_advance" convention], capped by the inventory sublimit.
- Receivable pre-cap amount per counterparty = (eligible face - dilution reserve) x advance rate (IG / non-IG;
  optionally a higher rate on the insured share). Eligible: issued, outstanding, not disputed, days past due at
  most the aging limit, counterparty rating at least the facility minimum and not defaulted.
- Concentration cap: BB = inventory + sum_j min(a_j, cap x BB), solved per path in closed form (sorted suffix);
  a randomized test checks it against the legacy solver.
- Receivable sublimit, then fixed reserves; BB floored at zero.
- Permitted utilization = min(reported BB, commitment); undrawn = max(0, permitted - drawn - requests in
  transit); deficiency = max(0, drawn - permitted). Undrawn availability is never negative, and drawn debt never
  exceeds the commitment; a base below the drawn balance is recorded as a deficiency and cured from cash after
  the grace period (which can itself create an unfunded shortfall; cash is never created to pay it).

Haircut versus advance rate: under "haircut_only" (default, the legacy convention) the haircut is the
complement of the advance rate and inventory is deducted once. Under "haircut_and_advance" the valuation
haircut and the contractual advance rate are both applied; `collateral.convention_warning` flags combinations
that look like a double deduction.

Each asset is pledged once: inventory and receivables are disjoint; a sale moves value from inventory to
receivables on the trade day (test: net worth unchanged, contributions move accordingly).

## 7. Dynamic haircuts

h(t) = clip(base + vol_coefficient x max(0, vol(t) - reference_vol) + liquidity + location + quality
+ age_per_30d x max(0, age - age_free_days) / 30 + concentration add-on (if the instrument's share of inventory
value exceeds the threshold) + unhedged_penalty x (1 - hedge ratio) + lender_stress(t), min, max).

vol(t) is the path EWMA volatility (section 9). With `dynamic=False` vol and lender stress are frozen at their
opening values. `max_increase_per_redetermination` caps how far the haircut may rise at one redetermination
(a contractual constraint). This is an illustrative specification, not a regulatory or industry-standard
formula. For the workbench book the parameters come from `CommoditySpec` (base, slope, reference vol, unhedged
penalty, 10% floor, 65% cap), so day-0 eligible inventory equals the legacy figure.

Haircut sensitivity (Treasury tab) runs the same market path under: static haircuts; dynamic (path volatility);
lender tightening +5 and +15 points; +15 points capped at +2 per redetermination; weekly redetermination.
When the base already exceeds the commitment (as in the default book), haircut changes do not change
availability until the base falls below the commitment.

## 8. Initial margin

Per contract, by `MarginPolicy.mode`: fixed per contract (a schedule value); share of notional
(pct x F x contract size); volatility-scaled proxy (z x vol(t) x sqrt(MPOR / 252) x F x contract size).
The rate is reset every `review_days`, floored at `floor_fraction` x the opening rate, multiplied by the
scenario step-up path and by (1 + step-up fraction) once the cumulative move reaches an optional trigger.
Requirement = sum |contracts| x rate. Calls are paid in cash; excess is released after `release_lag_days`, and
a new call first absorbs pending releases. Posted IM is restricted cash. These are proxies; they are not
SPAN, IRM or any clearing house's margin model, and default levels are illustrative. In the workbench book the
pct and vol-scaled modes are calibrated so the opening IM equals the per-contract margin settings, so modes
differ only in how IM responds.

## 9. Market paths, curve, basis and roll

- Futures curve: contract j priced off the front as front x (1 + slope)^j (slope = monthly carry: contango
  positive). Slope shocks model steepening and flattening; front-only shocks model front-month dislocation.
  Synthetic unless observed curve data are supplied; labelled in `MarketPaths.labels`.
- Reference contract: nearest contract not yet at its roll day (expiry minus `roll_days_before_expiry`).
- Physical price P(t) = reference futures(t) x (1 + basis(t)). Basis(t) = P(t) - F_ref(t), as a share of F_ref.
- Synthetic basis (OU): b(t+1) = b(t) + kappa (theta(t) - b(t)) dt + sigma sqrt(dt) e(t+1), dt = 1/252,
  theta(t) = mean + amplitude sin(2 pi t / period), e = rho z_benchmark + sqrt(1 - rho^2) z_own, plus
  deterministic location or quality shock paths. Half-life ln 2 / kappa years; stationary standard deviation
  sigma / sqrt(2 kappa) (tested).
- Path volatility: EWMA of reference-contract daily log returns, s2(t) = lam s2(t-1) + (1 - lam) 252 r(t)^2,
  measured within the same contract across a roll.
- Roll: positions in contract j move to j+1 at the close of the roll day. VM on contract j has been settled
  daily, so the roll creates no cash flow except transaction costs (2 legs x |contracts| x cost) and any IM
  change. The calendar spread is the new position's starting level, not an expense (test: VM continuity and
  costs only). For a short hedge in contango the spread is a roll yield earned only if the curve converges.

Economic P&L attribution for a one-to-one quantity hedge with matching units and currency: physical + futures
= Q (P1 - P0) + N (F1 - F0) with N = -Q; with P = F (1 + b), this equals Q F1 (b1 - b0) + Q b0 (F1 - F0).
With b0 = 0 the hedged P&L is exactly the change in basis (test); with a constant nonzero basis a small residual
Q b0 dF remains because the hedge is 1:1 in quantity, not in value (test).

Units: conversions only within a dimension (mass, volume, energy, board measure), with exact definitional
factors; cross-dimension conversion needs an explicit, described factor. Currency conversion goes through USD
quotes. Hedge ratios are computed in physical units after converting contract quantity.

## 10. Counterparty credit

- Horizon PD under constant hazard: PD_h = 1 - (1 - PD_1y)^(days / 365); a convenience assumption.
- Default settlement for exposure E: insured = min(limit, coverage x max(0, E - deductible)) (zero if
  excluded); uninsured = E - insured; estate recovery = (1 - LGD) x uninsured; loss = LGD x uninsured. The three
  sum to E, so recoveries cannot be double counted. Insurance cash arrives after the claim delay, recoveries
  after the recovery delay; both are claims receivable meanwhile, not cash.
- ECL = PD_h x LGD x (EAD - insured EAD). Illustrative; not IFRS 9 (no staging, lifetime ECL or discounting).
- Payment behaviour: on time, delayed (cash later, no loss), partial with a disputed remainder (ineligible
  while disputed), default, downgrade below the facility minimum rating (ineligible from that day).
- Deterministic stresses: largest buyer late, downgraded, defaulting; all buyers late; combinations.
- Correlated defaults (one-factor Gaussian copula with a commodity factor):
  X_i = sqrt(rho) (beta_i F_c + sqrt(1 - beta_i^2) M) + sqrt(1 - rho) e_i. F_c is the rank-based normal score
  of the linked commodity's simulated horizon return, so it is standard normal across paths for any return
  model, X_i is standard normal, and the unconditional PD equals PD_h for any rho and beta (tested within
  four standard errors on 40,000 paths). beta > 0 is wrong-way risk: defaults cluster when the commodity falls.
- Migration: illustrative one-year matrix built from simple rules (PD from the illustrative table, +1 notch 3%,
  -1 notch 6%, -2 notches 1%), scaled to the horizon linearly (P_h = I + (d / 365)(P - I)). Not agency data.
- Rating buckets for the workbench book: IG -> BBB, non-IG -> BB. This mapping is not an assessment of any named
  company in the illustrative ledger.

## 11. Return models and calibration (`core/return_models.py`)

| Method | Specification | Strengths | Limitations |
| --- | --- | --- | --- |
| Gaussian | r = L z, L Cholesky of the daily covariance | transparent baseline | thin tails, constant volatility |
| Student t | r = L z sqrt((nu - 2) / W), W ~ chi2(nu) shared across assets, nu > 2 | joint fat tails; covariance exactly the target | one tail parameter for all assets; symmetric |
| Block bootstrap | stationary bootstrap of synchronized historical rows, expected block length B | keeps observed co-movement, fat tails, short clustering | cannot exceed history; needs enough data |
| EWMA | Sigma(t) = lam Sigma(t-1) + (1 - lam) r r'; last value used | reacts to recent volatility | lam is a choice (0.94 is a convention, about 17-day memory) |
| GARCH(1,1) | per asset QMLE (Nelder-Mead, no SciPy), constant conditional correlation | volatility clustering in simulation | needs at least 250 observations; rejected fits (non-convergence, alpha + beta >= 0.999, alpha at bound) fall back to EWMA with a stated reason |

Correlation modes: unconditional, rolling 60 days, EWMA, crisis window, user-defined. Every matrix is checked
for symmetry, unit diagonal and positive semidefiniteness; repair (eigenvalue clipping and diagonal rescaling)
and optional shrinkage toward the identity are reported in the model notes, never applied silently.

Data preparation (`prepare_prices`): duplicates (last kept), non-finite and non-positive prices (unless allowed)
removed, gaps forward filled for at most N business days, longer gaps dropped from the joint sample, stale runs
reported but kept. Outliers are never removed. Log returns need positive prices; `return_type="absolute"` uses
price differences for series that can be zero or negative.

## 12. Liquidity risk metrics (`core/liquidity_metrics.py`)

- Available liquidity L(t) = unrestricted cash + funding in transit + undrawn permitted revolver + undrawn
  committed lines (emergency funding excluded).
- Minimum liquidity = min over t = 1..H of L(t); deterioration D = L(0) - minimum liquidity.
- LaR_c = c-quantile of D (a deterioration). The legacy "LaR" (5th percentile of a buffer level) is a different
  convention and is kept only in the legacy Monte Carlo view.
- Expected liquidity shortfall ELS_c = mean of D over paths with D >= LaR_c. Not the market-risk ES.
- Breach: unrestricted cash below the minimum operating cash after same-day funding on some day. P(breach) is
  the share of paths with a breach, with a Wilson 95% interval for sampling error.
- Peak funding requirement = max over t of the unfunded shortfall (path maximum, not terminal value; tested).
- Also: funding p95 / p99, funding given breach, median first breach day, survival curve, breach days, peak
  utilization, peak cumulative VM paid, peak IM, mean credit loss, tail decomposition at the path minimum.
- Probabilities from the Monte Carlo are model estimates under the calibrated return model. The share of
  deterministic scenarios that breach (sweep, grid, haircut and credit stress tables) is not a probability.

## 13. Hedge frontier (`core/hedge_frontier.py`)

Grid over hedge ratios (whole contracts); common random numbers across ratios. Per ratio: economic P&L mean,
standard deviation, 5% quantile loss ("economic downside") and worst-5% mean; funding p95, P(breach), ELS, LaR,
mean peak utilization, opening IM, transaction cost of rebalancing. Constraints (maximum P(breach), economic
downside, funding p95, utilization) mark infeasible ratios. Pareto efficiency on (economic downside, funding
p95). Labels: economically conservative (smallest downside), liquidity conservative (smallest funding p95),
balanced (minimax of the two objectives scaled to [0, 1]). No ratio is optimal in general; the labels depend on
every assumption above. A smaller hedge can lower margin needs while raising economic risk substantially; the
frontier shows both.

## 14. Validation (`core/model_validation.py`)

- Rolling out-of-sample one-day loss quantile of the hedge book from each return model (calibration uses data
  up to t only), with Kupiec proportion-of-failures and Christoffersen independence p-values; flagged as not
  meaningful when fewer than five exceptions are expected.
- One-day-ahead variance forecasts (Gaussian sample, EWMA, GARCH) scored by MSE and QLIKE.
- Simulated versus historical one-day moments (volatility, skew, excess kurtosis) and the largest correlation gap.
- Convergence and seed sensitivity helpers for any Monte Carlo metric.
On the bundled synthetic data these checks test internal consistency only.

## 15. Worked example (independently calculated; `tests/test_treasury_engine.py`)

Start Friday 2026-11-20. Inventory 1,000 MT at 100, short 8 contracts of 100 MT, cash 20,000, revolver
commitment 150,000 with 50,000 drawn at 3.6% (ACT/360), receivable 30,000 due Tuesday, payable 30,000 due
Wednesday, minimum cash 10,000, haircut 20% plus 10 points lender stress on day 3, IM 1,000 per contract
doubling on day 3, same-day drawdowns. Prices 100, 110, 105, 120.

| Day | Event | Cash |
| --- | --- | ---: |
| 1 (Mon) | VM -8 x 100 x 10 = -8,000; interest 50,000 x 3.6% x 3 / 360 = 15 | 11,985 |
| 2 (Tue) | VM +4,000; collection 30,000; interest 5 | 45,980 |
| 3 (Wed) | VM -12,000; IM call 8,000; payable 30,000; interest 5 -> -4,025; draw 14,025 | 10,000 |

Borrowing base: day 0 = 1,000 x 100 x 0.8 + 30,000 x 0.85 = 105,500; day 1 = 113,500; day 2 = 84,000;
day 3 = 1,000 x 120 x 0.7 = 84,000. Day 3 debt 64,025, undrawn 19,975, available liquidity 29,975. Net worth:
78,000 -> 81,975 (inventory +20,000, futures -16,000, interest -25). The test asserts every one of these.

## 16. Case study: 2022 European gas crisis (`core/case_study_2022.py`)

Observed data: Yahoo Finance `TTF=F` daily closes for 2022 (continuous front-month Dutch TTF natural gas
futures, EUR/MWh), fetched on request and cached for a day; vendor-assembled, so it contains roll jumps between
contract months. Holidays are forward filled onto the engine calendar. If the series cannot be fetched the case
study is not run; no synthetic series is substituted.

Hypothetical company (every figure is an assumption): 1.0 TWh of stored gas, short TTF futures in illustrative
lots of 720 MWh (1 MW over 30 days), opening cash EUR 25M, revolver EUR 400M with EUR 60M drawn, weekly
borrowing base, IM reviewed weekly, four hypothetical customers, a monthly operating cycle, and an optional EUR
150M alternative committed line with a two-day delay. Variants add one mechanism at a time: static financing;
operations and payment timing; dynamic haircuts and volatility-scaled IM; synthetic location basis (OU, location
shock from June); hypothetical customer delay and default; combined; combined with the alternative line.
The comparison shows a hedged merchant whose economic P&L stays small while its cash requirement peaks in
the hundreds of millions, and how each mechanism changes the funding need. Results are recomputed from the
observed series on each load and are not hard-coded.

## 17. Known limitations

- Financing agreements are simplified: no covenants beyond the deficiency cure, no cash dominion, no accordion,
  no partial redeterminations or lender discretion beyond the configured haircut paths.
- Collateral, basis, curve, margin and credit parameters are illustrative; no lender, exchange, broker or
  rating agency data are embedded. The volatility-scaled IM is a proxy, not an exchange model.
- Physical price data are not available: the physical price is futures x (1 + basis) with a synthetic basis.
- Futures curves are synthetic (constant monthly carry) unless supplied.
- Credit PDs, LGDs, transition rates and correlations are illustrative; the copula is a one-factor model.
- The bundled sample is synthetic with near-zero correlation; the live feed is a year of daily closes.
- Monte Carlo results depend on the return model; GARCH fits on short or calm samples are often rejected.
- Exchange holidays, intraday margin calls, cross-margining, taxes and FX conversion of margin are not modelled.
- Memory: a 2,000-path, 90-day run allocates about 150 MB transiently; the UI caps paths at 2,000.
- The new engine coexists with the advanced ledger (`core/cashflow_engine.py`); a test shows they agree where
  their features overlap. Consolidating them is listed as technical debt.
