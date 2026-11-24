"""Unit tests for market data fetching and validation (Stage 2)."""

from pathlib import Path
import pandas as pd
import pytest

from src.data.market_data import (
    OHLCV_COLUMNS,
    fetch_market_data,
    generate_offline_sample_market_data,
    validate_market_data,
)


def test_generate_offline_sample_market_data():
    limit = 50
    df = generate_offline_sample_market_data(symbol="BTC/USDT", timeframe="1h", limit=limit)

    assert len(df) == limit
    for col in OHLCV_COLUMNS:
        assert col in df.columns
    assert "data_source" in df.columns
    assert (df["data_source"] == "sample_offline").all()
    assert (df["open"] > 0).all()
    assert (df["close"] > 0).all()
    assert (df["high"] >= df["low"]).all()
    assert (df["volume"] >= 0).all()


def test_validate_market_data_handles_nans():
    df = generate_offline_sample_market_data(limit=20)
    # Inject NaN in close and volume
    df.loc[5, "close"] = None
    df.loc[10, "volume"] = None

    cleaned = validate_market_data(df)
    assert cleaned["close"].isnull().sum() == 0
    assert cleaned["volume"].isnull().sum() == 0


def test_validate_market_data_missing_column():
    df = pd.DataFrame({"timestamp": [1, 2], "open": [10, 20]})
    with pytest.raises(ValueError, match="Missing mandatory column"):
        validate_market_data(df)


def test_fetch_market_data_fallback_and_save(tmp_path: Path):
    df = fetch_market_data(
        symbol="BTC/USDT",
        timeframe="1h",
        limit=30,
        output_dir=tmp_path,
        force_offline=True,
    )

    assert len(df) == 30
    saved_file = tmp_path / "btc_usdt_1h.csv"
    assert saved_file.exists()

    loaded = pd.read_csv(saved_file)
    assert len(loaded) == 30
    assert loaded["data_source"].iloc[0] == "sample_offline"
