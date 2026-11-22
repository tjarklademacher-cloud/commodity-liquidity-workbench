"""Market data feed: calibration, volatility math, winsorization, cleaning and fallbacks."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

import core.data_feed as data_feed
from core.data_feed import (
    _clean_price_frame,
    build_market_snapshot,
    compute_realized_volatility,
    generate_sample_csv,
    load_custom_csv_market_data,
    load_live_market_data,
    load_sample_market_data,
)
from core.schemas import COMMODITY_SPECS, REQUIRED_TICKERS

EXPECTED_SPOT = {"GC=F": 2710.00, "HG=F": 4.3500, "LBR=F": 565.00, "CC=F": 8150.00, "KC=F": 2.75, "SB=F": 0.171}
EXPECTED_VOL = {"GC=F": 0.155, "HG=F": 0.235, "LBR=F": 0.385, "CC=F": 0.510, "KC=F": 0.360, "SB=F": 0.270}


def test_sample_csv_shape_and_dates(sample_prices):
    assert len(sample_prices) == 260
    assert list(sample_prices.columns) == list(REQUIRED_TICKERS)
    assert sample_prices.index[-1] == pd.Timestamp("2026-11-20")
    assert sample_prices.index.name == "Date"


def test_generator_is_deterministic(tmp_path, sample_prices):
    regenerated = generate_sample_csv(tmp_path / "s.csv", seed=42)
    reloaded = pd.read_csv(tmp_path / "s.csv", index_col="Date", parse_dates=True)
    np.testing.assert_allclose(regenerated.to_numpy(), sample_prices.to_numpy())
    np.testing.assert_allclose(reloaded.to_numpy(), sample_prices.to_numpy())


def test_cents_quoted_contracts_are_scaled_to_usd(sample_prices, snapshot):
    assert sample_prices["KC=F"].iloc[-1] == pytest.approx(275.0)
    assert sample_prices["SB=F"].iloc[-1] == pytest.approx(17.10)
    cleaned = _clean_price_frame(sample_prices)
    assert cleaned["KC=F"].iloc[-1] == pytest.approx(2.75)
    assert cleaned["GC=F"].iloc[-1] == pytest.approx(2710.0)
    assert snapshot["SB=F"].spot_price == pytest.approx(0.171)


@pytest.mark.parametrize("ticker", REQUIRED_TICKERS)
def test_terminal_spot_calibration(snapshot, ticker):
    assert snapshot[ticker].spot_price == pytest.approx(EXPECTED_SPOT[ticker], abs=1e-9)
    assert snapshot[ticker].as_of_date == "2026-11-20"


@pytest.mark.parametrize("ticker", REQUIRED_TICKERS)
def test_terminal_30d_vol_calibration(snapshot, ticker):
    assert snapshot[ticker].realized_vol_30d == pytest.approx(EXPECTED_VOL[ticker], abs=1e-4)
    assert not snapshot[ticker].vol_fallback_used
    assert snapshot[ticker].vol_30d_min <= snapshot[ticker].realized_vol_30d <= snapshot[ticker].vol_30d_max


def test_volatility_math_matches_manual(sample_prices):
    vol = compute_realized_volatility(sample_prices, winsorize_sigma=None)
    lr = np.log(sample_prices["CC=F"] / sample_prices["CC=F"].shift(1)).iloc[-30:]
    manual = float(np.std(lr.to_numpy(), ddof=1) * np.sqrt(252.0))
    assert vol["CC=F"].iloc[-1] == pytest.approx(manual, rel=1e-12)
    assert vol.iloc[:20].isna().all().all()
    assert vol.iloc[20].notna().all()


def test_snapshot_returns(snapshot, sample_prices):
    s = sample_prices["GC=F"]
    assert snapshot["GC=F"].ret_1d == pytest.approx(np.log(s.iloc[-1] / s.iloc[-2]))
    assert snapshot["GC=F"].ret_30d == pytest.approx(s.iloc[-1] / s.iloc[-31] - 1.0)


def test_winsorization_dampens_roll_jump(sample_prices):
    jumped = sample_prices.copy()
    jump_date = jumped.index[-10]
    jumped.loc[jumped.index >= jump_date, "LBR=F"] = jumped.loc[jumped.index >= jump_date, "LBR=F"] * 1.35
    raw_vol = compute_realized_volatility(jumped, winsorize_sigma=None)["LBR=F"].iloc[-1]
    win_vol = compute_realized_volatility(jumped, winsorize_sigma=5.0)["LBR=F"].iloc[-1]
    clean_vol = compute_realized_volatility(sample_prices, winsorize_sigma=5.0)["LBR=F"].iloc[-1]
    assert raw_vol > clean_vol + 0.30
    assert win_vol < raw_vol - 0.10
    lr = np.log(jumped["LBR=F"] / jumped["LBR=F"].shift(1))
    assert np.log(1.35) > 5.0 * lr.std(ddof=1)


def test_short_csv_uses_reference_vol_fallback(sample_prices):
    buf = io.StringIO()
    sample_prices.iloc[-10:].to_csv(buf)
    buf.seek(0)
    result = load_custom_csv_market_data(buf)
    for t in REQUIRED_TICKERS:
        q = result.snapshot[t]
        assert q.vol_fallback_used
        assert q.realized_vol_30d == COMMODITY_SPECS[t].ref_vol
        assert q.vol_30d_min == q.vol_30d_max == COMMODITY_SPECS[t].ref_vol
    assert "reference volatility" in result.status_message


def test_non_positive_prices_are_cleaned():
    df = pd.DataFrame(
        {
            "Date": pd.bdate_range("2026-01-01", periods=6).strftime("%Y-%m-%d"),
            "GC=F": [2000.0, -5.0, 2010.0, 0.0, 2020.0, 2030.0],
            "HG=F": [4.0, 4.1, 4.2, 4.3, 4.4, 4.5],
            "LBR=F": [500.0, 501.0, 502.0, 503.0, 504.0, 505.0],
            "CC=F": [8000.0, 8010.0, 8020.0, 8030.0, 8040.0, 8050.0],
            "KC=F": [270.0, 271.0, 272.0, 273.0, 274.0, 275.0],
            "SB=F": [17.0, 17.1, 17.2, 17.3, 17.4, 17.5],
        }
    )
    cleaned = _clean_price_frame(df)
    assert (cleaned > 0.0).all().all()
    assert cleaned["GC=F"].tolist() == [2000.0, 2000.0, 2010.0, 2010.0, 2020.0, 2030.0]
    vol = compute_realized_volatility(cleaned)
    assert np.isfinite(np.log(cleaned / cleaned.shift(1)).iloc[1:].to_numpy()).all()
    assert vol.shape == cleaned.shape


def test_all_non_positive_ticker_is_rejected():
    df = pd.DataFrame(
        {
            "Date": ["2026-01-01", "2026-01-02", "2026-01-05"],
            "GC=F": [0.0, -1.0, -2.0],
            "HG=F": [4.0, 4.1, 4.2],
            "LBR=F": [500.0, 501.0, 502.0],
            "CC=F": [8000.0, 8010.0, 8020.0],
            "KC=F": [270.0, 271.0, 272.0],
            "SB=F": [17.0, 17.1, 17.2],
        }
    )
    with pytest.raises(ValueError, match="GC=F"):
        _clean_price_frame(df)


def test_missing_ticker_is_rejected(sample_prices):
    with pytest.raises(ValueError, match="CC=F"):
        _clean_price_frame(sample_prices.drop(columns=["CC=F"]))


def test_clean_sorts_and_dedupes(sample_prices):
    shuffled = pd.concat([sample_prices.iloc[::-1], sample_prices.iloc[[-1]] * 1.0])
    cleaned = _clean_price_frame(shuffled)
    assert cleaned.index.is_monotonic_increasing
    assert not cleaned.index.duplicated().any()
    assert len(cleaned) == 260


def test_live_failure_falls_back_to_sample(monkeypatch):
    def boom(*args, **kwargs):
        raise ConnectionError("simulated offline")

    monkeypatch.setattr(data_feed.yf, "download", boom)
    result = load_live_market_data()
    assert result.used_fallback
    assert "simulated offline" in result.status_message
    assert result.snapshot["GC=F"].spot_price == pytest.approx(2710.0)


def test_live_empty_frame_falls_back(monkeypatch):
    monkeypatch.setattr(data_feed.yf, "download", lambda *a, **k: pd.DataFrame())
    result = load_live_market_data()
    assert result.used_fallback
    assert "empty" in result.status_message


def test_live_multiindex_success(monkeypatch, sample_prices):
    closes = sample_prices.copy()
    cols = pd.MultiIndex.from_product([["Close", "Open"], list(REQUIRED_TICKERS)], names=["Price", "Ticker"])
    raw = pd.DataFrame(np.hstack([closes.to_numpy(), closes.to_numpy()]), index=closes.index, columns=cols)
    monkeypatch.setattr(data_feed.yf, "download", lambda *a, **k: raw)
    result = load_live_market_data()
    assert not result.used_fallback
    assert result.source_label.startswith("Live")
    assert result.snapshot["CC=F"].spot_price == pytest.approx(8150.0)


def test_live_missing_ticker_falls_back(monkeypatch, sample_prices):
    raw = sample_prices.drop(columns=["LBR=F"])
    monkeypatch.setattr(data_feed.yf, "download", lambda *a, **k: raw)
    result = load_live_market_data()
    assert result.used_fallback


def test_sample_loader_regenerates_missing_csv(tmp_path):
    path = tmp_path / "nested" / "missing.csv"
    result = load_sample_market_data(path)
    assert path.exists()
    assert result.snapshot["HG=F"].spot_price == pytest.approx(4.35)


def test_build_snapshot_direct(sample_prices):
    vol = compute_realized_volatility(sample_prices)
    snap = build_market_snapshot(sample_prices, vol)
    assert set(snap) == set(REQUIRED_TICKERS)
