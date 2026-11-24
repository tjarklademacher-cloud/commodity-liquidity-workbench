"""Return models for path simulation: Gaussian, multivariate Student t, block bootstrap, EWMA and GARCH(1,1).

Conventions
- Returns are daily log returns of positive price series by default. For series that can be zero or negative
  (spreads, some power prices) use `return_type="absolute"` (price differences); the simulated path is then
  P(t) = P(0) + cumulative sum, with no positivity guarantee, and the user must interpret it accordingly.
- Time scaling: daily parameters are simulated day by day; annualized figures use 252 trading days.
- Parametric methods have zero drift (a liquidity stress horizon is too short for drift estimates to matter
  and they are statistically unreliable).

Methods
- gaussian: z ~ N(0, I), r = L z with L the Cholesky factor of the daily covariance.
- student_t: multivariate t with nu > 2 degrees of freedom, built as r = L z x sqrt((nu - 2) / W) with
  W ~ chi-square(nu) shared across assets on a given day. The common scale shock W produces joint tail events;
  the factor (nu - 2) normalizes the covariance of r to exactly the target covariance (Var of t_nu is
  nu / (nu - 2)). Sampling independent univariate t variables would not give this dependence.
- bootstrap: stationary block bootstrap of synchronized historical rows (all assets on the same date), so
  cross-asset dependence, fat tails and short-run volatility clustering inside a block are preserved. Expected
  block length is configurable; returns are demeaned.
- ewma: covariance Sigma(t) = lam Sigma(t-1) + (1 - lam) r(t) r(t)', started from the sample covariance of the
  first observations; the last value is used as a constant covariance for simulation. Lambda sets the memory:
  effective sample about 1 / (1 - lam) days (0.94 is about 17 days, 0.97 about 33). The often-quoted 0.94
  for daily data is a convention, not an estimate for these commodities.
- garch: univariate GARCH(1,1) per asset (Gaussian quasi-likelihood), constant conditional correlation from
  standardized residuals; simulated variance follows the fitted recursion. Fits that fail the checks
  (sample under MIN_GARCH_OBS, non-convergence, alpha + beta >= 0.999, parameters on bounds) fall back to EWMA
  for that asset and the fallback is reported.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

ReturnMethod = Literal["gaussian", "student_t", "bootstrap", "ewma", "garch"]
CorrelationMode = Literal["unconditional", "rolling", "ewma", "crisis", "user"]
ReturnType = Literal["log", "absolute"]
MIN_OBS = 20
MIN_GARCH_OBS = 250
TRADING_DAYS = 252.0


# --------------------------------------------------------------------------- data preparation

@dataclass
class PreparationReport:
    n_input_rows: int
    n_output_rows: int
    duplicates_removed: int
    invalid_prices_removed: dict[str, int]
    stale_runs: dict[str, int]
    filled_gaps: dict[str, int]
    non_synchronous_rows_dropped: int
    notes: list[str] = field(default_factory=list)


def prepare_prices(prices: pd.DataFrame, max_fill_days: int = 2, stale_run_days: int = 5,
                   allow_non_positive: Sequence[str] = ()) -> tuple[pd.DataFrame, PreparationReport]:
    """Clean a wide price frame (index dates, columns instruments) without removing genuine extreme moves.

    - duplicate dates: the last record is kept
    - non-finite prices, and non-positive prices for columns not in `allow_non_positive`, become missing
    - gaps (trading holidays, non-synchronous calendars) are forward filled for at most `max_fill_days`
      business days; longer gaps stay missing and those dates are dropped from the joint sample
    - stale prices: runs of `stale_run_days` or more identical prices are counted and reported, not removed
    Outliers are never removed here; winsorization, if any, is an explicit modelling choice elsewhere."""
    df = prices.copy()
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index))
    n_in = len(df)
    dup = int(df.index.duplicated(keep="last").sum())
    df = df.loc[~df.index.duplicated(keep="last")].sort_index()
    invalid: dict[str, int] = {}
    for c in df.columns:
        col = pd.to_numeric(df[c], errors="coerce")
        bad = ~np.isfinite(col) | ((col <= 0) if c not in allow_non_positive else False)
        invalid[c] = int(bad.sum())
        df.loc[:, c] = col.where(~bad)
    stale = {}
    for c in df.columns:
        s = df[c]
        runs = (s != s.shift()).cumsum()
        lengths = s.groupby(runs).transform("size")
        stale[c] = int(((lengths >= stale_run_days) & s.notna()).groupby(runs).any().sum())
    before = df.isna().sum()
    df = df.ffill(limit=max_fill_days)
    filled = {c: int(before[c] - df[c].isna().sum()) for c in df.columns}
    n_before = len(df)
    df = df.dropna(how="any")
    notes = []
    if any(stale.values()):
        notes.append("Runs of unchanged prices were found; they may be stale quotes or illiquid days and are kept.")
    return df, PreparationReport(n_in, len(df), dup, invalid, stale, filled, n_before - len(df), notes)


def returns_from_prices(prices: pd.DataFrame, return_type: ReturnType = "log") -> pd.DataFrame:
    if return_type == "log":
        if (prices <= 0).any().any():
            raise ValueError("log returns need strictly positive prices; use return_type='absolute'")
        return np.log(prices / prices.shift(1)).dropna(how="any")
    return prices.diff().dropna(how="any")


# --------------------------------------------------------------------------- matrices

@dataclass
class MatrixDiagnostics:
    symmetric: bool
    unit_diagonal: bool
    min_eigenvalue: float
    condition_number: float
    repaired: bool
    shrinkage: float
    note: str


def repair_correlation(corr: np.ndarray, floor: float = 1e-10) -> tuple[np.ndarray, MatrixDiagnostics]:
    """Symmetrize, clip negative eigenvalues and rescale to a unit diagonal. Every change is reported."""
    c = np.asarray(corr, dtype=float)
    sym = bool(np.allclose(c, c.T, atol=1e-12))
    unit = bool(np.allclose(np.diag(c), 1.0, atol=1e-12))
    s = (c + c.T) / 2.0
    vals, vecs = np.linalg.eigh(s)
    min_eig = float(vals.min())
    repaired = (min_eig < floor) or not sym or not unit
    if min_eig < floor:
        s = vecs @ np.diag(np.maximum(vals, floor)) @ vecs.T
    d = np.sqrt(np.clip(np.diag(s), 1e-300, None))
    s = s / np.outer(d, d)
    np.fill_diagonal(s, 1.0)
    s = (s + s.T) / 2.0
    vals2 = np.linalg.eigvalsh(s)
    cond = float(vals2.max() / max(vals2.min(), 1e-300))
    note = ("Correlation matrix repaired (eigenvalue clipping and diagonal rescaling); minimum eigenvalue before "
            f"repair {min_eig:.2e}." if repaired else "Correlation matrix valid; no repair.")
    return s, MatrixDiagnostics(sym, unit, min_eig, cond, repaired, 0.0, note)


def shrink_correlation(corr: np.ndarray, intensity: float) -> np.ndarray:
    """Linear shrinkage toward the identity: (1 - a) C + a I. Used only when chosen explicitly."""
    if not 0.0 <= intensity <= 1.0:
        raise ValueError("shrinkage intensity must be in [0, 1]")
    return (1.0 - intensity) * np.asarray(corr) + intensity * np.eye(len(corr))


def cov_from(vols_daily: np.ndarray, corr: np.ndarray) -> np.ndarray:
    return np.outer(vols_daily, vols_daily) * corr


def ewma_covariance(rets: np.ndarray, lam: float, warmup: int = 20) -> np.ndarray:
    """Last value of the EWMA covariance recursion (daily units)."""
    if not 0.0 < lam < 1.0:
        raise ValueError("EWMA decay must be strictly between 0 and 1")
    r = np.asarray(rets, dtype=float)
    w = min(warmup, len(r))
    s = np.cov(r[:w].T, ddof=1) if w > 1 else np.outer(r[0], r[0])
    s = np.atleast_2d(s)
    for x in r[w:]:
        s = lam * s + (1.0 - lam) * np.outer(x, x)
    return s


def correlation_for_mode(rets: pd.DataFrame, mode: CorrelationMode, rolling_window: int = 60, lam: float = 0.94,
                         crisis: tuple[str, str] | None = None, user_corr: np.ndarray | None = None) -> np.ndarray:
    r = rets.to_numpy()
    if mode == "unconditional":
        c = np.corrcoef(r.T)
    elif mode == "rolling":
        c = np.corrcoef(r[-rolling_window:].T)
    elif mode == "ewma":
        s = ewma_covariance(r, lam)
        d = np.sqrt(np.diag(s))
        c = s / np.outer(d, d)
    elif mode == "crisis":
        if crisis is None:
            raise ValueError("crisis mode needs a (start, end) date range")
        sub = rets.loc[crisis[0]:crisis[1]]
        if len(sub) < MIN_OBS:
            raise ValueError(f"crisis window has {len(sub)} joint observations; at least {MIN_OBS} are needed")
        c = np.corrcoef(sub.to_numpy().T)
    else:
        if user_corr is None:
            raise ValueError("user mode needs a correlation matrix")
        c = np.asarray(user_corr, dtype=float)
        if c.shape != (r.shape[1], r.shape[1]):
            raise ValueError("user correlation matrix has the wrong shape for the asset universe")
    return np.atleast_2d(c)


# --------------------------------------------------------------------------- GARCH(1,1)

@dataclass
class GarchFit:
    omega: float
    alpha: float
    beta: float
    last_variance: float
    long_run_variance: float
    log_likelihood: float
    converged: bool
    valid: bool
    n_obs: int
    message: str


def _nelder_mead(f, x0: np.ndarray, step: float = 0.1, tol: float = 1e-9, max_iter: int = 2000) -> tuple[np.ndarray, bool]:
    """Plain Nelder-Mead simplex (no SciPy dependency); returns (argmin, converged)."""
    n = len(x0)
    pts = [x0] + [x0 + step * np.eye(n)[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(max_iter):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) <= tol * (abs(vals[0]) + tol) and np.max(np.abs(np.array(pts) - pts[0])) < 1e-7:
            return pts[0], True
        centroid = np.mean(pts[:-1], axis=0)
        xr = centroid + (centroid - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (pts[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0] + 0.5 * (p - pts[0]) for p in pts]
                vals = [f(p) for p in pts]
    return pts[int(np.argmin(vals))], False


def fit_garch11(r: np.ndarray) -> GarchFit:
    """Gaussian quasi-maximum-likelihood GARCH(1,1) on demeaned daily returns. Parameters are mapped through
    logistic transforms so omega > 0, alpha, beta >= 0 and alpha + beta < 1 hold during the search."""
    x = np.asarray(r, dtype=float)
    x = x - x.mean()
    n = len(x)
    var0 = float(x.var(ddof=1)) if n > 1 else float("nan")
    if n < MIN_GARCH_OBS or not np.isfinite(var0) or var0 <= 0:
        return GarchFit(float("nan"), float("nan"), float("nan"), var0, var0, float("nan"), False, False, n,
                        f"Not estimated: {n} observations (minimum {MIN_GARCH_OBS}).")

    def sig(v: float) -> float:
        return 1.0 / (1.0 + math.exp(-min(50.0, max(-50.0, v))))

    def unpack(p):
        persist = sig(p[0]) * 0.9999
        share = sig(p[1])
        alpha, beta = persist * share, persist * (1.0 - share)
        omega = var0 * (1.0 - persist) * math.exp(min(20.0, max(-20.0, p[2])))
        return omega, alpha, beta

    def nll(p):
        omega, alpha, beta = unpack(p)
        h = var0
        ll = 0.0
        for e in x:
            ll += math.log(h) + e * e / h
            h = omega + alpha * e * e + beta * h
            if not h > 0:
                return 1e100
        return 0.5 * ll

    p_hat, conv = _nelder_mead(nll, np.array([2.0, -2.0, 0.0]), step=0.5)
    omega, alpha, beta = unpack(p_hat)
    h = var0
    for e in x:
        h = omega + alpha * e * e + beta * h
    persist = alpha + beta
    issues = []
    if not conv:
        issues.append("optimizer did not converge")
    if persist >= 0.999:
        issues.append(f"near-integrated variance (alpha + beta = {persist:.4f})")
    if alpha < 1e-4:
        issues.append("alpha at its lower bound (no ARCH effect detected)")
    valid = not issues
    lr = omega / (1.0 - persist) if persist < 1 else float("inf")
    return GarchFit(omega, alpha, beta, h, lr, -nll(p_hat), conv, valid, n,
                    "Fit accepted." if valid else "Rejected: " + "; ".join(issues) + ". EWMA is used instead.")


# --------------------------------------------------------------------------- calibration + simulation

@dataclass
class ReturnModel:
    method: ReturnMethod
    tickers: list[str]
    return_type: ReturnType
    cov_daily: np.ndarray
    corr: np.ndarray
    diagnostics: MatrixDiagnostics
    n_obs: int
    history: np.ndarray | None = None
    nu: float | None = None
    block_days: int = 5
    garch: dict[str, GarchFit] = field(default_factory=dict)
    ewma_lambda: float = 0.94
    notes: list[str] = field(default_factory=list)

    @property
    def annual_vols(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov_daily) * TRADING_DAYS)


def calibrate_return_model(
    prices: pd.DataFrame,
    method: ReturnMethod = "gaussian",
    window: int | None = 250,
    return_type: ReturnType = "log",
    correlation_mode: CorrelationMode = "unconditional",
    ewma_lambda: float = 0.94,
    nu: float = 5.0,
    block_days: int = 5,
    rolling_window: int = 60,
    crisis: tuple[str, str] | None = None,
    user_corr: np.ndarray | None = None,
    shrinkage: float = 0.0,
    vol_scale: float = 1.0,
) -> ReturnModel:
    """Calibrate on the last `window` joint observations (all when None). Volatility comes from the method
    (sample for gaussian / student_t / bootstrap, EWMA for ewma, GARCH where valid); correlation from
    `correlation_mode`. The covariance is rebuilt from vols and a validated correlation matrix."""
    if method == "student_t" and not nu > 2.0:
        raise ValueError("Student t needs nu > 2 for a finite covariance")
    rets = returns_from_prices(prices, return_type)
    if window is not None:
        rets = rets.iloc[-window:]
    tick = list(rets.columns)
    if len(rets) < MIN_OBS:
        raise ValueError(f"{len(rets)} joint observations; at least {MIN_OBS} are needed")
    r = rets.to_numpy()
    notes: list[str] = []
    vols = r.std(axis=0, ddof=1)
    garch: dict[str, GarchFit] = {}
    if method == "ewma":
        vols = np.sqrt(np.diag(ewma_covariance(r, ewma_lambda)))
    elif method == "garch":
        ew = np.sqrt(np.diag(ewma_covariance(r, ewma_lambda)))
        for i, t in enumerate(tick):
            fit = fit_garch11(r[:, i])
            garch[t] = fit
            if fit.valid:
                vols[i] = math.sqrt(fit.last_variance)
            else:
                vols[i] = ew[i]
                notes.append(f"{t}: {fit.message}")
    corr = correlation_for_mode(rets, correlation_mode, rolling_window, ewma_lambda, crisis, user_corr)
    if shrinkage > 0:
        corr = shrink_correlation(corr, shrinkage)
        notes.append(f"Correlation shrunk toward the identity with intensity {shrinkage:.2f}.")
    corr, diag = repair_correlation(corr)
    diag.shrinkage = shrinkage
    if diag.repaired:
        notes.append(diag.note)
    vols = vols * vol_scale
    if vol_scale != 1.0:
        notes.append(f"Volatilities scaled by {vol_scale:.2f} (stress assumption).")
    hist = r - r.mean(axis=0) if method == "bootstrap" else None
    return ReturnModel(method, tick, return_type, cov_from(vols, corr), corr, diag, len(rets), hist,
                       nu if method == "student_t" else None, block_days, garch, ewma_lambda, notes)


def simulate_returns(model: ReturnModel, horizon: int, n_paths: int, seed: int) -> np.ndarray:
    """(n_paths, horizon, k) simulated daily returns (log or absolute, per the model)."""
    rng = np.random.default_rng(seed)
    k = len(model.tickers)
    chol = np.linalg.cholesky(model.cov_daily + 1e-18 * np.eye(k))
    if model.method == "bootstrap":
        hist = model.history
        n = hist.shape[0]
        p_new = 1.0 / max(1, model.block_days)
        out = np.empty((n_paths, horizon, k))
        idx = rng.integers(0, n, size=n_paths)
        for d in range(horizon):
            restart = rng.random(n_paths) < p_new
            idx = np.where(restart | (d == 0), rng.integers(0, n, size=n_paths), (idx + 1) % n)
            out[:, d, :] = hist[idx]
        return out
    z = rng.standard_normal((n_paths, horizon, k))
    if model.method == "student_t":
        w = rng.chisquare(model.nu, size=(n_paths, horizon, 1))
        return (z @ chol.T) * np.sqrt((model.nu - 2.0) / w)
    if model.method == "garch":
        corr_chol = np.linalg.cholesky(model.corr + 1e-15 * np.eye(k))
        h = np.diag(model.cov_daily).copy()
        h = np.broadcast_to(h, (n_paths, k)).copy()
        out = np.empty((n_paths, horizon, k))
        params = [model.garch.get(t) for t in model.tickers]
        for d in range(horizon):
            e = (z[:, d, :] @ corr_chol.T) * np.sqrt(h)
            out[:, d, :] = e
            for i, fit in enumerate(params):
                if fit is not None and fit.valid:
                    h[:, i] = fit.omega + fit.alpha * e[:, i] ** 2 + fit.beta * h[:, i]
                else:
                    h[:, i] = model.ewma_lambda * h[:, i] + (1 - model.ewma_lambda) * e[:, i] ** 2
        return out
    return z @ chol.T


def price_paths(spot: np.ndarray, rets: np.ndarray, return_type: ReturnType = "log") -> np.ndarray:
    """(n, h+1, k) price paths from spot (k,) and simulated returns (n, h, k)."""
    n, h, k = rets.shape
    cum = np.concatenate([np.zeros((n, 1, k)), np.cumsum(rets, axis=1)], axis=1)
    return spot * np.exp(cum) if return_type == "log" else spot + cum
