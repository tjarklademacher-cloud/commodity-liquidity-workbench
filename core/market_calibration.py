"""Market calibration, simulation methods and price-risk backtesting for the advanced model.

- Calibration estimates daily log-return statistics over a chosen window, reports coverage, and repairs a
  covariance matrix that is not positive semidefinite by eigenvalue clipping (the repair is reported).
- Simulation methods: "gaussian" (multivariate normal, the reference), "bootstrap" (stationary block
  bootstrap of observed joint daily returns: keeps fat tails and co-movement), "stress" (volatilities scaled
  up and correlations blended toward a stress level). None of these is a forecast; they are model scenarios.
- The factor sample is an alternative synthetic dataset with explicit common and sector factors. The
  original seed-42 sample is not modified.
- The backtest checks only the price-risk component (5-day hedge variation margin of today's book) against
  realized moves; it cannot validate treasury cash flows for which no history exists.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel

from core.advanced_schemas import SimulationMethod
from core.data_feed import SAMPLE_CALIBRATION
from core.schemas import COMMODITY_SPECS, REQUIRED_TICKERS, CommoditySpec, InventoryPosition, MarketQuote

CALIBRATION_WINDOWS: dict[str, int | None] = {"60 days": 60, "125 days": 125, "250 days": 250, "Maximum": None}
MIN_OBSERVATIONS = 20


class CalibrationResult(BaseModel):
    window_label: str
    n_obs: int
    start_date: str
    end_date: str
    annual_vols: dict[str, float]
    correlation: dict[str, dict[str, float]]
    cov_daily: list[list[float]]
    coverage: dict[str, float]
    psd_repaired: bool
    min_eigenvalue: float
    used_reference_vols: bool
    note: str


def _winsorize(rets: pd.DataFrame, sigma: float | None) -> pd.DataFrame:
    out = rets.copy()
    if sigma is not None and sigma > 0:
        for t in out.columns:
            sd = out[t].std(ddof=1)
            if pd.notna(sd) and sd > 0:
                out.loc[:, t] = out[t].clip(lower=-sigma * sd, upper=sigma * sd)
    return out


def joint_returns(prices_df: pd.DataFrame, window: int | None, winsorize_sigma: float | None = 5.0) -> pd.DataFrame:
    """Daily log returns on dates where every ticker has a return, last `window` rows (all if None)."""
    rets = np.log(prices_df.loc[:, list(REQUIRED_TICKERS)] / prices_df.loc[:, list(REQUIRED_TICKERS)].shift(1))
    rets = rets.dropna(how="any")
    if window is not None:
        rets = rets.iloc[-window:]
    return _winsorize(rets, winsorize_sigma)


def nearest_psd(cov: np.ndarray, floor: float = 1e-12) -> tuple[np.ndarray, bool, float]:
    """Symmetrize and clip negative eigenvalues to `floor`. Returns (matrix, repaired, original min eigenvalue)."""
    sym = (cov + cov.T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    min_eig = float(vals.min())
    if min_eig >= floor:
        return sym, False, min_eig
    fixed = vecs @ np.diag(np.maximum(vals, floor)) @ vecs.T
    return (fixed + fixed.T) / 2.0, True, min_eig


def calibrate(
    prices_df: pd.DataFrame,
    window_label: str = "250 days",
    winsorize_sigma: float | None = 5.0,
    specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS,
) -> CalibrationResult:
    window = CALIBRATION_WINDOWS[window_label]
    coverage = {t: float(prices_df[t].notna().mean()) if t in prices_df else 0.0 for t in REQUIRED_TICKERS}
    rets = joint_returns(prices_df, window, winsorize_sigma)
    k = len(REQUIRED_TICKERS)
    if len(rets) < MIN_OBSERVATIONS:
        vols = np.array([specs[t].ref_vol / math.sqrt(252.0) for t in REQUIRED_TICKERS])
        cov = np.diag(vols**2)
        corr = np.eye(k)
        note = (f"Only {len(rets)} joint observations (minimum {MIN_OBSERVATIONS}); reference volatilities and "
                "zero correlation are used.")
        return CalibrationResult(
            window_label=window_label, n_obs=len(rets), start_date="", end_date="",
            annual_vols={t: float(v * math.sqrt(252.0)) for t, v in zip(REQUIRED_TICKERS, vols)},
            correlation={a: {b: float(corr[i, j]) for j, b in enumerate(REQUIRED_TICKERS)}
                         for i, a in enumerate(REQUIRED_TICKERS)},
            cov_daily=cov.tolist(), coverage=coverage, psd_repaired=False, min_eigenvalue=float(vols.min() ** 2),
            used_reference_vols=True, note=note,
        )
    cov_raw = rets.cov(ddof=1).to_numpy()
    cov, repaired, min_eig = nearest_psd(cov_raw)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    note = (f"Covariance had a negative eigenvalue ({min_eig:.2e}) and was repaired by eigenvalue clipping."
            if repaired else "Covariance is positive semidefinite; no repair needed.")
    return CalibrationResult(
        window_label=window_label, n_obs=len(rets), start_date=rets.index[0].strftime("%Y-%m-%d"),
        end_date=rets.index[-1].strftime("%Y-%m-%d"),
        annual_vols={t: float(sd[i] * math.sqrt(252.0)) for i, t in enumerate(REQUIRED_TICKERS)},
        correlation={a: {b: float(corr[i, j]) for j, b in enumerate(REQUIRED_TICKERS)}
                     for i, a in enumerate(REQUIRED_TICKERS)},
        cov_daily=cov.tolist(), coverage=coverage, psd_repaired=repaired, min_eigenvalue=min_eig,
        used_reference_vols=False, note=note,
    )


def stress_covariance(cov: np.ndarray, vol_scale: float, stress_corr: float, weight: float) -> np.ndarray:
    """Scale volatilities by `vol_scale` and blend off-diagonal correlations toward `stress_corr` by `weight`."""
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    target = np.full_like(corr, stress_corr)
    np.fill_diagonal(target, 1.0)
    blended = (1.0 - weight) * corr + weight * target
    np.fill_diagonal(blended, 1.0)
    out = np.outer(sd * vol_scale, sd * vol_scale) * blended
    return nearest_psd(out)[0]


def simulate_daily_log_returns(
    calibration: CalibrationResult,
    prices_df: pd.DataFrame,
    method: SimulationMethod,
    horizon_days: int,
    n_paths: int,
    seed: int,
    winsorize_sigma: float | None = 5.0,
    block_days: int = 5,
    stress_vol_scale: float = 1.5,
    stress_corr: float = 0.6,
    stress_weight: float = 0.5,
) -> np.ndarray:
    """Array (n_paths, horizon_days, n_tickers) of daily log returns, zero drift for parametric methods."""
    rng = np.random.default_rng(seed)
    k = len(REQUIRED_TICKERS)
    cov = np.array(calibration.cov_daily)
    if method == "bootstrap" and not calibration.used_reference_vols:
        window = CALIBRATION_WINDOWS[calibration.window_label]
        hist = joint_returns(prices_df, window, winsorize_sigma).to_numpy()
        hist = hist - hist.mean(axis=0)
        n = hist.shape[0]
        out = np.empty((n_paths, horizon_days, k))
        p_new = 1.0 / max(1, block_days)
        idx = rng.integers(0, n, size=n_paths)
        for d in range(horizon_days):
            restart = rng.random(n_paths) < p_new
            idx = np.where(restart | (d == 0), rng.integers(0, n, size=n_paths), (idx + 1) % n)
            out[:, d, :] = hist[idx]
        return out
    if method == "stress":
        cov = stress_covariance(cov, stress_vol_scale, stress_corr, stress_weight)
    chol = np.linalg.cholesky(nearest_psd(cov, 1e-14)[0] + 1e-18 * np.eye(k))
    z = rng.standard_normal((n_paths, horizon_days, k))
    return z @ chol.T


def percentile_interval(samples: Sequence[float], q: float, confidence: float = 0.95) -> tuple[float, float]:
    """Distribution-free order-statistic interval for the q-quantile (Monte Carlo sampling error only)."""
    x = np.sort(np.asarray(samples, dtype=float))
    n = x.size
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-9 else float(
        math.sqrt(2.0) * _erfinv(confidence))
    half = z * math.sqrt(n * q * (1.0 - q))
    lo = int(max(0, math.floor(n * q - half)))
    hi = int(min(n - 1, math.ceil(n * q + half)))
    return float(x[lo]), float(x[hi])


def _erfinv(y: float) -> float:
    a = 0.147
    ln = math.log(1.0 - y * y)
    t = 2.0 / (math.pi * a) + ln / 2.0
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), y)


def generate_factor_sample(
    seed: int = 7,
    market_loading: float = 0.35,
    sector_loading: float = 0.35,
    n_days: int = 260,
) -> pd.DataFrame:
    """Synthetic prices with explicit factors (illustrative, not estimated): each daily shock is
    sqrt(m) x market + sqrt(s) x sector + sqrt(1 - m - s) x own, with sectors metals (gold, copper),
    softs (cocoa, coffee, sugar) and lumber on its own. Same start, end and annual volatility targets as the
    seed-42 sample; returned in exchange quote units, not written to disk."""
    if market_loading < 0 or sector_loading < 0 or market_loading + sector_loading > 0.95:
        raise ValueError("loadings must be non-negative and sum to at most 0.95")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-11-20", periods=n_days)
    steps = n_days - 1
    market = rng.standard_normal(steps)
    sectors = {"metals": rng.standard_normal(steps), "softs": rng.standard_normal(steps),
               "lumber": rng.standard_normal(steps)}
    sector_of = {"GC=F": "metals", "HG=F": "metals", "LBR=F": "lumber", "CC=F": "softs", "KC=F": "softs",
                 "SB=F": "softs"}
    data = {}
    for t in REQUIRED_TICKERS:
        cfg = SAMPLE_CALIBRATION[t]
        own = rng.standard_normal(steps)
        z = (math.sqrt(market_loading) * market + math.sqrt(sector_loading) * sectors[sector_of[t]]
             + math.sqrt(1.0 - market_loading - sector_loading) * own)
        sigma, mu, dt = cfg["annual_vol"], cfg["drift"], 1.0 / 252.0
        log_path = np.concatenate([[0.0], np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z)])
        raw = cfg["start"] * np.exp(log_path)
        data[t] = np.round(raw * cfg["target_end"] / raw[-1], 4 if t in ("HG=F", "KC=F", "SB=F") else 2)
    return pd.DataFrame(data, index=pd.DatetimeIndex(dates, name="Date"))


class BacktestResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    horizon_days: int
    window_days: int
    level: float
    n_forecasts: int
    n_breaches: int
    breach_rate: float
    expected_rate: float
    frame: pd.DataFrame
    note: str


def backtest_margin_forecast(
    prices_df: pd.DataFrame,
    positions: Sequence[InventoryPosition],
    market_snapshot: Mapping[str, MarketQuote],
    horizon_days: int = 5,
    window_days: int = 125,
    level: float = 0.95,
    winsorize_sigma: float | None = 5.0,
) -> BacktestResult:
    """Rolling Gaussian forecast of the 5-day hedge variation-margin outflow of today's book vs the realized
    outflow. A breach is a realized outflow above the forecast quantile. Counterfactual: today's positions are
    applied to past price moves; this tests the price-risk model only."""
    tickers = list(REQUIRED_TICKERS)
    exposure = np.zeros(len(tickers))
    for p in positions:
        exposure[tickers.index(p.ticker)] += p.hedged_qty * market_snapshot[p.ticker].spot_price
    rets = np.log(prices_df[tickers] / prices_df[tickers].shift(1)).dropna(how="any")
    rows = []
    z = 1.6448536269514722 if abs(level - 0.95) < 1e-9 else math.sqrt(2.0) * _erfinv(2 * level - 1)
    for end in range(window_days, len(rets) - horizon_days + 1):
        hist = _winsorize(rets.iloc[end - window_days:end], winsorize_sigma)
        cov_h = hist.cov(ddof=1).to_numpy() * horizon_days
        sd_vm = float(math.sqrt(max(0.0, exposure @ cov_h @ exposure)))
        forecast = z * sd_vm
        fwd = rets.iloc[end:end + horizon_days].sum().to_numpy()
        realized_outflow = float(exposure @ np.expm1(fwd))
        rows.append({"date": rets.index[end - 1], "forecast_outflow": forecast, "realized_outflow": realized_outflow,
                     "breach": realized_outflow > forecast})
    frame = pd.DataFrame(rows)
    n = len(frame)
    breaches = int(frame["breach"].sum()) if n else 0
    return BacktestResult(
        horizon_days=horizon_days, window_days=window_days, level=level, n_forecasts=n, n_breaches=breaches,
        breach_rate=breaches / n if n else 0.0, expected_rate=1.0 - level, frame=frame,
        note=("Overlapping 5-day windows are serially correlated, so the breach count is indicative, not a formal "
              "test. Only hedge variation margin is backtested; margin rates, debt balances and collections have "
              "no history here."),
    )
