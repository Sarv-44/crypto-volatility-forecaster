"""Unit tests for market feature engineering (Stage 3)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.market_data import generate_offline_sample_market_data
from src.features.market_features import (
    compute_bollinger_bands,
    compute_log_returns,
    compute_macd,
    compute_market_features,
    compute_rsi,
)


def test_compute_log_returns():
    prices = pd.Series([100.0, 110.0, 99.0])
    lr = compute_log_returns(prices)
    assert np.isnan(lr.iloc[0])
    assert np.isclose(lr.iloc[1], np.log(110.0 / 100.0))
    assert np.isclose(lr.iloc[2], np.log(99.0 / 110.0))


def test_compute_rsi_bounds():
    raw_df = generate_offline_sample_market_data(limit=100)
    rsi = compute_rsi(raw_df["close"], period=14)
    # Check that values are within [0, 100]
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0.0).all()
    assert (valid_rsi <= 100.0).all()


def test_compute_macd_consistency():
    raw_df = generate_offline_sample_market_data(limit=100)
    macd, signal, hist = compute_macd(raw_df["close"])
    diff = (macd - signal) - hist
    assert np.isclose(diff.abs().max(), 0.0)


def test_compute_bollinger_bands_ordering():
    raw_df = generate_offline_sample_market_data(limit=100)
    middle, upper, lower, width, pct_b = compute_bollinger_bands(raw_df["close"], period=20)
    valid_idx = ~(upper.isna() | lower.isna() | middle.isna())
    assert (upper[valid_idx] >= middle[valid_idx]).all()
    assert (middle[valid_idx] >= lower[valid_idx]).all()
    assert (width[valid_idx] >= 0.0).all()


def test_compute_market_features_end_to_end(tmp_path: Path):
    raw_df = generate_offline_sample_market_data(limit=80)
    features_df = compute_market_features(
        raw_df,
        drop_warmup=True,
        output_dir=tmp_path,
        save_filename="test_features.csv",
    )

    # Verify no NaNs remain after dropping warmup
    assert features_df.isnull().sum().sum() == 0

    # Verify all expected columns are present
    expected_cols = [
        "timestamp",
        "close",
        "log_return",
        "volatility_12h",
        "volatility_24h",
        "vol_change_pct",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_middle",
        "bb_upper",
        "bb_lower",
        "bb_width",
        "bb_pct_b",
        "sma_7",
        "sma_24",
        "ema_7",
        "ema_24",
        "momentum_4h",
        "momentum_10h",
    ]
    for col in expected_cols:
        assert col in features_df.columns

    # Verify file saved
    assert (tmp_path / "test_features.csv").exists()
