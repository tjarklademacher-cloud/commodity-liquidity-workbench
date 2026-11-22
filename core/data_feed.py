"""Market data ingestion: calibrated sample generator, cleaning, realized vol and live feed."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from core.schemas import (
    COMMODITY_SPECS,
    REQUIRED_TICKERS,
    CommoditySpec,
    MarketLoadResult,
    MarketQuote,
)

DEFAULT_SAMPLE_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_prices.csv"

MIN_LIVE_OBSERVATIONS = 40
LIVE_TIMEOUT_SECONDS = 10

SAMPLE_CALIBRATION: dict[str, dict[str, float]] = {
    "GC=F": {"start": 2380.0, "target_end": 2710.0, "annual_vol": 0.155, "drift": 0.10},
    "HG=F": {"start": 3.85, "target_end": 4.35, "annual_vol": 0.235, "drift": 0.08},
    "LBR=F": {"start": 495.0, "target_end": 565.0, "annual_vol": 0.385, "drift": 0.09},
    "CC=F": {"start": 6950.0, "target_end": 8150.0, "annual_vol": 0.510, "drift": 0.12},
}


def generate_sample_csv(output_path: Path | str = DEFAULT_SAMPLE_CSV_PATH, seed: int = 42) -> pd.DataFrame:
    """Generate the deterministic 260-business-day calibrated GBM price series."""
    rng = np.random.default_rng(seed=seed)
    n_days = 260
    dates = pd.bdate_range(end="2026-11-20", periods=n_days)
    data: dict[str, np.ndarray] = {}
    for ticker in REQUIRED_TICKERS:
        cfg = SAMPLE_CALIBRATION[ticker]
        mu = cfg["drift"]
        sigma = cfg["annual_vol"]
        dt = 1.0 / 252.0
        z = rng.standard_normal(n_days - 1)
        tail = z[-30:]
        tail = (tail - np.mean(tail)) / (np.std(tail, ddof=1) + 1e-12)
        z[-30:] = tail
        log_rets = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        log_path = np.concatenate([[0.0], np.cumsum(log_rets)])
        raw_prices = cfg["start"] * np.exp(log_path)
        scale_factor = cfg["target_end"] / raw_prices[-1]
        data[ticker] = np.round(raw_prices * scale_factor, 4 if ticker == "HG=F" else 2)
    df = pd.DataFrame(data, index=pd.DatetimeIndex(dates, name="Date"))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=True)
    return df


def _clean_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw price frame to a sorted, positive, gap-filled REQUIRED_TICKERS frame."""
    frame = df.copy()
    if "Date" in frame.columns:
        frame = frame.set_index(pd.DatetimeIndex(pd.to_datetime(frame["Date"]), name="Date"))
        frame = frame.drop(columns=["Date"])
    else:
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index), name="Date")
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    frame = frame.sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]

    missing = [t for t in REQUIRED_TICKERS if t not in frame.columns]
    if missing:
        raise ValueError(f"Price data is missing required tickers: {', '.join(missing)}")

    frame = frame.loc[:, list(REQUIRED_TICKERS)].copy()
    for ticker in REQUIRED_TICKERS:
        frame[ticker] = pd.to_numeric(frame[ticker], errors="coerce")
    frame = frame.where(frame > 0.0)
    frame = frame.ffill(limit=5)
    frame = frame.dropna(how="all")

    too_short = [t for t in REQUIRED_TICKERS if int(frame[t].notna().sum()) < 2]
    if too_short:
        raise ValueError(
            "Each ticker needs at least 2 positive price observations; insufficient: "
            + ", ".join(too_short)
        )
    frame.index.name = "Date"
    return frame


def compute_realized_volatility(
    prices_df: pd.DataFrame,
    window: int = 30,
    min_periods: int = 20,
    winsorize_sigma: float | None = 5.0,
) -> pd.DataFrame:
    """Rolling annualized realized volatility of daily log returns, with optional winsorization."""
    log_rets = np.log(prices_df / prices_df.shift(1))
    if winsorize_sigma is not None and winsorize_sigma > 0:
        for t in log_rets.columns:
            col_std = log_rets[t].std(ddof=1)
            if pd.notna(col_std) and col_std > 0:
                log_rets.loc[:, t] = log_rets[t].clip(
                    lower=-winsorize_sigma * col_std, upper=winsorize_sigma * col_std
                )
    return log_rets.rolling(window=window, min_periods=min_periods).std(ddof=1) * np.sqrt(252.0)


def build_market_snapshot(
    prices_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    specs: dict[str, CommoditySpec] = COMMODITY_SPECS,
) -> dict[str, MarketQuote]:
    """Latest spot, returns and realized vol per ticker (ref_vol fallback for short histories)."""
    snapshot: dict[str, MarketQuote] = {}
    for t in REQUIRED_TICKERS:
        s = prices_df[t].dropna()
        spot = float(s.iloc[-1])
        prev = float(s.iloc[-2])
        ret_30d = float(s.iloc[-1] / s.iloc[-31] - 1.0) if len(s) >= 31 else float(s.iloc[-1] / s.iloc[0] - 1.0)
        vols = vol_df[t].dropna() if t in vol_df.columns else pd.Series(dtype=float)
        if vols.empty:
            ref = specs[t].ref_vol
            vol, vmin, vmax, fallback = ref, ref, ref, True
        else:
            vol, vmin, vmax, fallback = float(vols.iloc[-1]), float(vols.min()), float(vols.max()), False
        snapshot[t] = MarketQuote(
            ticker=t,
            spot_price=spot,
            prev_price=prev,
            ret_1d=float(np.log(spot / prev)),
            ret_30d=ret_30d,
            realized_vol_30d=vol,
            vol_30d_min=vmin,
            vol_30d_max=vmax,
            as_of_date=s.index[-1].strftime("%Y-%m-%d"),
            vol_fallback_used=fallback,
        )
    return snapshot


def _build_load_result(
    raw: pd.DataFrame,
    source_label: str,
    winsorize_sigma: float | None,
    used_fallback: bool = False,
    status_message: str = "",
) -> MarketLoadResult:
    prices = _clean_price_frame(raw)
    vol = compute_realized_volatility(prices, winsorize_sigma=winsorize_sigma)
    snapshot = build_market_snapshot(prices, vol)
    return MarketLoadResult(
        prices_df=prices,
        vol_df=vol,
        snapshot=snapshot,
        source_label=source_label,
        as_of_date=prices.index[-1].strftime("%Y-%m-%d"),
        used_fallback=used_fallback,
        status_message=status_message,
    )


def load_sample_market_data(
    csv_path: Path | str = DEFAULT_SAMPLE_CSV_PATH,
    winsorize_sigma: float | None = 5.0,
) -> MarketLoadResult:
    """Load the calibrated offline sample (regenerating it deterministically if missing)."""
    path = Path(csv_path)
    if not path.exists():
        generate_sample_csv(path)
    raw = pd.read_csv(path)
    return _build_load_result(
        raw,
        source_label="Calibrated Sample (Offline)",
        winsorize_sigma=winsorize_sigma,
        status_message=f"Calibrated seed=42 sample series loaded from {path.name}.",
    )


def load_custom_csv_market_data(file_obj_or_path: Any, winsorize_sigma: float | None = 5.0) -> MarketLoadResult:
    """Load a user CSV with a Date column and the four required ticker columns."""
    raw = pd.read_csv(file_obj_or_path)
    result = _build_load_result(raw, source_label="Custom CSV Upload", winsorize_sigma=winsorize_sigma)
    fallback_tickers = [t for t, q in result.snapshot.items() if q.vol_fallback_used]
    message = f"Custom CSV loaded: {len(result.prices_df)} rows."
    if fallback_tickers:
        message += (
            " Insufficient history for 30d realized vol on "
            + ", ".join(fallback_tickers)
            + "; reference volatility used."
        )
    return result.model_copy(update={"status_message": message})


def _extract_close_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            return raw["Close"].copy()
        if "Close" in raw.columns.get_level_values(-1):
            return raw.xs("Close", axis=1, level=-1).copy()
        raise ValueError("Yahoo Finance response has no 'Close' field.")
    if all(t in raw.columns for t in REQUIRED_TICKERS):
        return raw.loc[:, list(REQUIRED_TICKERS)].copy()
    raise ValueError("Yahoo Finance response has unrecognized flat columns: " + ", ".join(map(str, raw.columns)))


def load_live_market_data(
    period: str = "1y",
    winsorize_sigma: float | None = 5.0,
    fallback_csv_path: Path | str = DEFAULT_SAMPLE_CSV_PATH,
) -> MarketLoadResult:
    """Fetch front-month futures closes from Yahoo Finance, falling back to the sample CSV."""
    try:
        raw = yf.download(
            tickers=list(REQUIRED_TICKERS),
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=LIVE_TIMEOUT_SECONDS,
        )
        if raw is None or raw.empty:
            raise ValueError("Yahoo Finance returned an empty frame (possible HTTP 429 rate limit).")
        closes = _extract_close_frame(raw)
        missing = [t for t in REQUIRED_TICKERS if t not in closes.columns or closes[t].dropna().empty]
        if missing:
            raise ValueError("Yahoo Finance returned no data for: " + ", ".join(missing))
        cleaned = _clean_price_frame(closes)
        short = [t for t in REQUIRED_TICKERS if int(cleaned[t].notna().sum()) < MIN_LIVE_OBSERVATIONS]
        if short:
            raise ValueError(
                f"Insufficient live history (<{MIN_LIVE_OBSERVATIONS} observations) for: " + ", ".join(short)
            )
        if not bool((cleaned.stack() > 0.0).all()):
            raise ValueError("Yahoo Finance returned non-positive prices.")
        vol = compute_realized_volatility(cleaned, winsorize_sigma=winsorize_sigma)
        snapshot = build_market_snapshot(cleaned, vol)
        as_of = cleaned.index[-1].strftime("%Y-%m-%d")
        return MarketLoadResult(
            prices_df=cleaned,
            vol_df=vol,
            snapshot=snapshot,
            source_label="Live Yahoo Finance (yfinance)",
            as_of_date=as_of,
            used_fallback=False,
            status_message=f"Live front-month futures loaded ({len(cleaned)} sessions, as of {as_of}).",
        )
    except Exception as exc:  # noqa: BLE001 - any live failure must degrade to offline sample
        sample = load_sample_market_data(fallback_csv_path, winsorize_sigma=winsorize_sigma)
        return sample.model_copy(
            update={
                "source_label": "Calibrated Sample (Live Fallback)",
                "used_fallback": True,
                "status_message": (
                    f"Live feed unavailable ({type(exc).__name__}: {str(exc)[:160]}). "
                    f"Showing bundled data from {Path(fallback_csv_path).name}."
                ),
            }
        )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Commodity workbench market data utilities.")
    parser.add_argument("--generate-sample", action="store_true", help="Regenerate data/sample_prices.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_SAMPLE_CSV_PATH, help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()
    if not args.generate_sample:
        parser.print_help()
        return
    generate_sample_csv(args.output, seed=args.seed)
    result = load_sample_market_data(args.output)
    print(f"Generated sample prices: {args.output}")
    print(f"{'Ticker':<8}{'Spot':>14}{'30d Vol':>10}")
    for t, q in result.snapshot.items():
        print(f"{t:<8}{q.spot_price:>14,.4f}{q.realized_vol_30d:>10.2%}")


if __name__ == "__main__":
    _main()
