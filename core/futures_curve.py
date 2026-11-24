"""Market paths for the treasury engine: futures term structure, physical prices with basis, path volatility.

Price architecture (all prices in functional currency per physical unit):
- futures[n, t, k, j]: settlement price of contract j of instrument k on business day t of path n. Contract 0
  is the nearest contract at the start; contract j expires on business day expiry_day[k, j].
- reference futures price: the nearest contract that has not reached its roll day.
- physical[n, t, k] = reference futures x (1 + basis[n, t, k]). Basis(t) = physical(t) - reference futures(t),
  stated as a share of the reference futures price, for a stated location and quality.
- vol[n, t, k]: annualized EWMA volatility of the reference futures' daily log returns along the path, used by
  dynamic haircuts and volatility-scaled initial margin.

Synthetic curves: when no observed curve is available, contract j is priced off the front as
front x (1 + slope)^j, where `slope` is a monthly carry (positive = contango, negative = backwardation).
Curve scenarios change the slope (steepening / flattening) or the front contract only (front-month dislocation).
These are scenario values, not observed contract prices, and are labelled "synthetic" in `MarketPaths.labels`.

Roll accounting (see `roll_economics`): rolling a short hedge closes contract j and opens contract j+1 at their
settlement prices on the roll day. Variation margin on contract j has already been settled daily, so the roll
itself creates no cash flow other than transaction costs and any change in initial margin. The price difference
between the two contracts is not an expense at the roll; it is the starting level of the new position and only
turns into cash through later variation margin, as the new contract converges.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


@dataclass
class MarketPaths:
    dates: pd.DatetimeIndex
    futures: np.ndarray            # (n, h+1, k, J)
    physical: np.ndarray           # (n, h+1, k)
    vol: np.ndarray                # (n, h+1, k)
    expiry_day: np.ndarray         # (k, J) business-day index; > horizon means beyond the horizon
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def n_paths(self) -> int:
        return self.futures.shape[0]

    @property
    def horizon(self) -> int:
        return self.futures.shape[1] - 1

    @property
    def n_instruments(self) -> int:
        return self.futures.shape[2]


def business_dates(start: pd.Timestamp | str, horizon_days: int) -> pd.DatetimeIndex:
    """Start date (day 0) plus `horizon_days` following business days (Monday to Friday; exchange holidays are
    not modelled, a documented simplification)."""
    start = pd.Timestamp(start)
    return pd.DatetimeIndex([start]).append(pd.bdate_range(start + pd.Timedelta(days=1), periods=horizon_days))


def calendar_to_business_index(dates: pd.DatetimeIndex, base_index: int, calendar_days: int) -> int:
    """First business-day index whose date is at least `calendar_days` after dates[base_index]. Returns
    len(dates) (beyond the horizon) if none."""
    target = dates[base_index] + pd.Timedelta(days=int(calendar_days))
    pos = int(dates.searchsorted(target, side="left"))
    return max(pos, base_index) if pos < len(dates) else len(dates)


def synthetic_curve(front: np.ndarray, n_contracts: int, monthly_slope: np.ndarray,
                    slope_shock: np.ndarray | None = None) -> np.ndarray:
    """Contract prices (n, h+1, k, J) from front prices (n, h+1, k); slope per instrument, optional slope shock
    path (h+1, k) added to the slope (steepening if positive)."""
    slope = np.broadcast_to(np.asarray(monthly_slope, dtype=float), front.shape[2:])
    total = slope[None, None, :] + (0.0 if slope_shock is None else np.asarray(slope_shock)[None, :, :])
    j = np.arange(n_contracts)
    return front[..., None] * np.power(1.0 + total[..., None], j)


def monthly_expiries(dates: pd.DatetimeIndex, n_instruments: int, n_contracts: int,
                     first_expiry_day: int = 15, spacing_days: int = 21) -> np.ndarray:
    """Illustrative expiry schedule: contract j expires on business day first_expiry_day + j x spacing_days."""
    exp = first_expiry_day + spacing_days * np.arange(n_contracts)
    return np.tile(exp, (n_instruments, 1)).astype(int)


def reference_contract(expiry_day: np.ndarray, horizon: int, roll_days_before_expiry: int) -> np.ndarray:
    """(h+1, k) index of the nearest contract that has not reached its roll day."""
    k, n_c = expiry_day.shape
    out = np.zeros((horizon + 1, k), dtype=int)
    for t in range(horizon + 1):
        for i in range(k):
            live = np.nonzero(expiry_day[i] - roll_days_before_expiry > t)[0]
            out[t, i] = int(live[0]) if live.size else n_c - 1
    return out


def reference_prices(futures: np.ndarray, ref_idx: np.ndarray) -> np.ndarray:
    """(n, h+1, k) reference futures price picked from the curve."""
    n, h1, k, _ = futures.shape
    t_idx = np.arange(h1)[:, None]
    k_idx = np.arange(k)[None, :]
    return futures[:, t_idx, k_idx, ref_idx]


def ewma_vol_path(prices: np.ndarray, opening_vol: np.ndarray, lam: float = 0.94,
                  ref_idx: np.ndarray | None = None, futures: np.ndarray | None = None) -> np.ndarray:
    """Annualized EWMA volatility along each path: s2(t) = lam s2(t-1) + (1 - lam) 252 r(t)^2, with r the daily
    log return of the reference price and s2(0) = opening_vol^2. On a roll day the return is measured within
    the same contract (yesterday's and today's price of the contract held yesterday), so the jump between
    contract levels is not mistaken for volatility."""
    if not 0.0 < lam < 1.0:
        raise ValueError("EWMA decay must be strictly between 0 and 1")
    n, h1, k = prices.shape
    var = np.empty((n, h1, k))
    var[:, 0, :] = np.asarray(opening_vol, dtype=float) ** 2
    for t in range(1, h1):
        if futures is not None and ref_idx is not None:
            prev_c = ref_idx[t - 1]
            k_idx = np.arange(k)
            p_now = futures[:, t, k_idx, prev_c]
            p_prev = futures[:, t - 1, k_idx, prev_c]
        else:
            p_now, p_prev = prices[:, t, :], prices[:, t - 1, :]
        r = np.log(np.where((p_now > 0) & (p_prev > 0), p_now / np.where(p_prev > 0, p_prev, 1.0), 1.0))
        var[:, t, :] = lam * var[:, t - 1, :] + (1.0 - lam) * TRADING_DAYS * r**2
    return np.sqrt(var)


def ou_basis_paths(
    n_paths: int,
    horizon: int,
    opening_basis: np.ndarray,
    long_run_mean: np.ndarray,
    kappa: np.ndarray,
    sigma: np.ndarray,
    seed: int,
    benchmark_shocks: np.ndarray | None = None,
    rho_with_benchmark: np.ndarray | None = None,
    seasonal_amplitude: np.ndarray | None = None,
    seasonal_period_days: float = 252.0,
    shock_path: np.ndarray | None = None,
) -> np.ndarray:
    """Synthetic mean-reverting basis (share of futures), Euler discretization of an Ornstein-Uhlenbeck process

        b(t+1) = b(t) + kappa (theta(t) - b(t)) dt + sigma sqrt(dt) e(t+1),   dt = 1/252
        theta(t) = long_run_mean + seasonal_amplitude sin(2 pi t / period)
        e = rho z_benchmark + sqrt(1 - rho^2) z_own

    kappa is the annual mean-reversion speed (half-life ln2 / kappa years); sigma the annual basis volatility in
    basis points of price. `shock_path` (h+1, k) adds deterministic location or quality shocks. All parameters
    are user assumptions; nothing here is estimated from observed physical prices."""
    k = len(opening_basis)
    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS
    rho = np.zeros(k) if rho_with_benchmark is None else np.asarray(rho_with_benchmark, dtype=float)
    if np.any(np.abs(rho) > 1):
        raise ValueError("correlation must be within [-1, 1]")
    amp = np.zeros(k) if seasonal_amplitude is None else np.asarray(seasonal_amplitude, dtype=float)
    out = np.empty((n_paths, horizon + 1, k))
    out[:, 0, :] = opening_basis
    for t in range(horizon):
        z_own = rng.standard_normal((n_paths, k))
        z_b = benchmark_shocks[:, t, :] if benchmark_shocks is not None else np.zeros((n_paths, k))
        e = rho * z_b + np.sqrt(1.0 - rho**2) * z_own
        theta = np.asarray(long_run_mean) + amp * np.sin(2.0 * np.pi * (t + 1) / seasonal_period_days)
        out[:, t + 1, :] = out[:, t, :] + np.asarray(kappa) * (theta - out[:, t, :]) * dt + np.asarray(sigma) * np.sqrt(dt) * e
    if shock_path is not None:
        out = out + np.asarray(shock_path)[None, :, :]
    return out


def build_market_paths(
    dates: pd.DatetimeIndex,
    front: np.ndarray,
    opening_vol: np.ndarray,
    basis: np.ndarray | None = None,
    n_contracts: int = 1,
    monthly_slope: np.ndarray | float = 0.0,
    slope_shock: np.ndarray | None = None,
    expiry_day: np.ndarray | None = None,
    roll_days_before_expiry: int = 2,
    ewma_lambda: float = 0.94,
    vol_override: np.ndarray | None = None,
    labels: dict[str, str] | None = None,
) -> MarketPaths:
    """Assemble futures curve, physical prices and path volatility from front-contract paths (n, h+1, k)."""
    front = np.asarray(front, dtype=float)
    if front.ndim == 2:
        front = front[None]
    n, h1, k = front.shape
    if len(dates) != h1:
        raise ValueError("dates must have horizon + 1 entries")
    if np.any(front < 0) or not np.all(np.isfinite(front)):
        raise ValueError("futures prices must be finite and non-negative")
    futures = synthetic_curve(front, n_contracts, np.broadcast_to(np.asarray(monthly_slope, dtype=float), (k,)),
                              slope_shock)
    if expiry_day is None:
        expiry_day = np.full((k, n_contracts), 10**6, dtype=int) if n_contracts == 1 else \
            monthly_expiries(dates, k, n_contracts)
    ref = reference_contract(expiry_day, h1 - 1, roll_days_before_expiry)
    ref_px = reference_prices(futures, ref)
    b = np.zeros((n, h1, k)) if basis is None else np.broadcast_to(basis, (n, h1, k))
    physical = np.maximum(0.0, ref_px * (1.0 + b))
    vol = vol_override if vol_override is not None else ewma_vol_path(ref_px, opening_vol, ewma_lambda, ref, futures)
    lab = {"futures": "simulated or scenario", "curve": "synthetic" if n_contracts > 1 else "single contract",
           "basis": "zero" if basis is None else "synthetic", "vol": f"EWMA lambda {ewma_lambda}"}
    lab.update(labels or {})
    return MarketPaths(dates=dates, futures=futures, physical=physical, vol=np.broadcast_to(vol, (n, h1, k)).copy(),
                       expiry_day=np.asarray(expiry_day, dtype=int), labels=lab)


def roll_economics(signed_contracts: float, contract_size: float, old_price: float, new_price: float,
                   transaction_cost_per_contract: float) -> dict[str, float]:
    """Cash and economic effect of rolling `signed_contracts` from the expiring contract to the next one on one
    day. Cash at the roll = - 2 x |contracts| x cost (close and open legs); no variation margin arises from the
    roll itself. The calendar spread (new - old) x size x contracts is the change in the position's reference
    level, reported for information: for a short hedge in contango (new > old) it is the roll yield earned if
    the curve later converges to an unchanged spot, not a cash flow today."""
    cost = 2.0 * abs(signed_contracts) * transaction_cost_per_contract
    return {
        "cash_flow": -cost,
        "calendar_spread_per_unit": new_price - old_price,
        "spread_exposure_value": -signed_contracts * contract_size * (new_price - old_price),
        "transaction_cost": cost,
    }
