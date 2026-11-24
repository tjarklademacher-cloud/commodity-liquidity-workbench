"""Validation of the return models used for path simulation.

All backtests are out of sample: the model used for the forecast at date t is calibrated only on returns up
to and including t; the realized outcome is the return after t. Rolling windows overlap, so consecutive
forecasts are not independent; with 1-day horizons the realized outcomes are non-overlapping.

Tests
- Kupiec proportion-of-failures likelihood ratio: H0 exception rate = 1 - level. Asymptotically chi-square(1).
- Christoffersen independence: H0 exceptions are serially independent (first-order Markov alternative).
- Volatility forecasts: mean squared error against squared returns, and QLIKE = mean(r^2 / s2 - ln(r^2 / s2) - 1)
  (robust to noise in the squared-return proxy; lower is better).
Sample size: with fewer than 5 expected exceptions the coverage test has very little power and the result is
flagged as not statistically meaningful. Results on the bundled synthetic sample only check internal consistency.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel

from core.return_models import ReturnMethod, calibrate_return_model, returns_from_prices, simulate_returns


def _chi2_1_sf(x: float) -> float:
    return math.erfc(math.sqrt(max(0.0, x) / 2.0))


def kupiec_pof(n: int, exceptions: int, level: float) -> tuple[float, float]:
    """(LR statistic, p-value) for H0: exception probability = 1 - level."""
    p = 1.0 - level
    x = exceptions
    if n == 0:
        return float("nan"), float("nan")
    phat = x / n

    def ll(q: float) -> float:
        a = (n - x) * math.log(1 - q) if n - x > 0 else 0.0
        b = x * math.log(q) if x > 0 else 0.0
        return a + b

    lr = -2.0 * (ll(p) - (ll(phat) if 0 < phat < 1 else 0.0))
    return lr, _chi2_1_sf(lr)


def christoffersen_independence(hits: Sequence[bool]) -> tuple[float, float]:
    h = np.asarray(hits, dtype=int)
    if len(h) < 2:
        return float("nan"), float("nan")
    a, b = h[:-1], h[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    pi0 = n01 / max(1, n00 + n01)
    pi1 = n11 / max(1, n10 + n11)
    pi = (n01 + n11) / max(1, n00 + n01 + n10 + n11)

    def term(k: int, q: float) -> float:
        return k * math.log(q) if k > 0 and q > 0 else 0.0

    l0 = term(n00 + n10, 1 - pi) + term(n01 + n11, pi)
    l1 = term(n00, 1 - pi0) + term(n01, pi0) + term(n10, 1 - pi1) + term(n11, pi1)
    lr = -2.0 * (l0 - l1)
    return lr, _chi2_1_sf(lr)


class CoverageResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    method: str
    level: float
    window: int
    n_forecasts: int
    exceptions: int
    expected: float
    rate: float
    kupiec_p: float
    independence_p: float
    meaningful: bool
    frame: pd.DataFrame
    note: str


def var_backtest(prices: pd.DataFrame, exposures: np.ndarray, method: ReturnMethod, window: int = 250,
                 level: float = 0.95, step: int = 1, n_sims: int = 4000, seed: int = 5,
                 **model_kwargs) -> CoverageResult:
    """Rolling one-day loss quantile of a linear book (exposure per asset in currency, value change =
    exposure x (exp(r) - 1)) from the chosen return model, compared with the realized next-day change."""
    rets = returns_from_prices(prices)
    rows = []
    for end in range(window, len(rets) - 1, step):
        hist_px = prices.loc[: rets.index[end]].iloc[-(window + 1):]
        model = calibrate_return_model(hist_px, method, window=None, **model_kwargs)
        sim = simulate_returns(model, 1, n_sims, seed + end)[:, 0, :]
        pnl = (np.expm1(sim) * exposures).sum(axis=1)
        var = -float(np.quantile(pnl, 1 - level))
        realized = float((np.expm1(rets.iloc[end + 1].to_numpy()) * exposures).sum())
        rows.append({"date": rets.index[end], "forecast_loss": var, "realized_change": realized,
                     "exception": -realized > var})
    frame = pd.DataFrame(rows)
    n = len(frame)
    x = int(frame["exception"].sum()) if n else 0
    lr_p = kupiec_pof(n, x, level)[1]
    ind_p = christoffersen_independence(frame["exception"].tolist())[1] if n else float("nan")
    expected = n * (1 - level)
    meaningful = expected >= 5
    return CoverageResult(
        method=method, level=level, window=window, n_forecasts=n, exceptions=x, expected=expected,
        rate=x / n if n else float("nan"), kupiec_p=lr_p, independence_p=ind_p, meaningful=meaningful, frame=frame,
        note=("Expected exceptions below 5: the coverage tests have little power and should not be read as "
              "validation." if not meaningful else "Out-of-sample, one-day, non-overlapping outcomes."))


def vol_forecast_evaluation(prices: pd.DataFrame, methods: Sequence[ReturnMethod] = ("gaussian", "ewma", "garch"),
                            window: int = 250, step: int = 5) -> pd.DataFrame:
    """Equal-weighted average across assets of one-day-ahead variance forecast errors."""
    rets = returns_from_prices(prices)
    rows = []
    for m in methods:
        se, ql, n = 0.0, 0.0, 0
        for end in range(window, len(rets) - 1, step):
            hist_px = prices.loc[: rets.index[end]].iloc[-(window + 1):]
            model = calibrate_return_model(hist_px, m, window=None)
            s2 = np.diag(model.cov_daily)
            r2 = rets.iloc[end + 1].to_numpy() ** 2
            se += float(((r2 - s2) ** 2).mean())
            ratio = np.maximum(r2, 1e-16) / s2
            ql += float((ratio - np.log(ratio) - 1).mean())
            n += 1
        rows.append({"method": m, "forecasts": n, "mse": se / n if n else float("nan"),
                     "qlike": ql / n if n else float("nan")})
    return pd.DataFrame(rows)


def moment_comparison(prices: pd.DataFrame, method: ReturnMethod, window: int | None = 250, n_paths: int = 20000,
                      seed: int = 3, **kw) -> pd.DataFrame:
    """Historical vs simulated one-day return moments per asset (sd annualized; excess kurtosis)."""
    rets = returns_from_prices(prices)
    if window is not None:
        rets = rets.iloc[-window:]
    model = calibrate_return_model(prices, method, window=window, **kw)
    sim = simulate_returns(model, 1, n_paths, seed)[:, 0, :]
    rows = []
    for i, t in enumerate(model.tickers):
        h, s = rets[t].to_numpy(), sim[:, i]
        rows.append({"asset": t, "hist_vol": h.std(ddof=1) * math.sqrt(252), "sim_vol": s.std(ddof=1) * math.sqrt(252),
                     "hist_skew": float(pd.Series(h).skew()), "sim_skew": float(pd.Series(s).skew()),
                     "hist_exkurt": float(pd.Series(h).kurt()), "sim_exkurt": float(pd.Series(s).kurt())})
    hist_c = np.corrcoef(rets.to_numpy().T)
    sim_c = np.corrcoef(sim.T)
    out = pd.DataFrame(rows)
    out.attrs["max_correlation_gap"] = float(np.abs(hist_c - sim_c).max())
    return out


def convergence(metric: Callable[[int, int], float], path_counts: Sequence[int] = (250, 500, 1000, 2000),
                seed: int = 31) -> pd.DataFrame:
    return pd.DataFrame([{"paths": n, "value": metric(n, seed)} for n in path_counts])


def seed_sensitivity(metric: Callable[[int, int], float], n_paths: int, seeds: Sequence[int] = (1, 2, 3, 4, 5)) -> dict:
    vals = np.array([metric(n_paths, s) for s in seeds])
    return {"values": vals.tolist(), "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "range": float(np.ptp(vals))}
