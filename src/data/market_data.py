"""Cryptocurrency market data retrieval module using CCXT with offline fallback.

Fetches OHLCV hourly data for BTC/USDT from Binance, validates integrity,
and saves the result to CSV in data/raw/market/.
"""

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import DEFAULT_LIMIT, DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, RAW_MARKET_DIR
from src.utils.logger import get_logger

logger = get_logger("MarketData")

# Standard expected columns for OHLCV
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def generate_offline_sample_market_data(
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    limit: int = DEFAULT_LIMIT,
    start_price: float = 65000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates realistic synthetic OHLCV data when API access is unavailable.

    Clearly labeled as 'sample_offline' in the 'data_source' column.

    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT').
        timeframe: Candle duration (e.g., '1h').
        limit: Number of hourly candles to generate.
        start_price: Starting price for synthetic simulation.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame containing synthetic OHLCV candles labeled as sample data.
    """
    logger.warning(
        "[DATA SOURCE: SAMPLE DATA] Generating offline synthetic market data for %s (%s). "
        "THIS IS NOT LIVE EXCHANGE DATA.",
        symbol,
        timeframe,
    )
    np.random.seed(seed)

    # Hourly timestamps up to the current hour
    end_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    timestamps = [end_time - timedelta(hours=(limit - 1 - i)) for i in range(limit)]

    # Simulate realistic price walk with stochastic volatility
    returns = np.random.normal(loc=0.0001, scale=0.008, size=limit)
    prices = [start_price]
    for r in returns[1:]:
        prices.append(prices[-1] * math.exp(r))

    opens, highs, lows, closes, volumes = [], [], [], [], []
    for p in prices:
        # Generate intra-candle variation
        noise = np.random.uniform(0.001, 0.005)
        c = p * (1 + np.random.normal(0, 0.002))
        h = max(p, c) * (1 + noise)
        l = min(p, c) * (1 - noise)
        vol = float(np.random.gamma(shape=5.0, scale=120.0))

        opens.append(round(p, 2))
        highs.append(round(h, 2))
        lows.append(round(l, 2))
        closes.append(round(c, 2))
        volumes.append(round(vol, 4))

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "data_source": "sample_offline",
        }
    )
    return df


def validate_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validates structure, checks for missing data, and cleans OHLCV records.

    Args:
        df: Input DataFrame with OHLCV data.

    Returns:
        Cleaned pd.DataFrame.

    Raises:
        ValueError: If mandatory columns are missing or data is empty.
    """
    if df.empty:
        raise ValueError("Market DataFrame is empty.")

    for col in OHLCV_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing mandatory column: {col}")

    # Ensure timestamp is datetime and sort chronologically
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # Check for NaN values
    null_counts = df[OHLCV_COLUMNS].isnull().sum()
    if null_counts.any():
        logger.warning("Detected missing values in market data: %s. Applying forward fill.", null_counts.to_dict())
        df[OHLCV_COLUMNS] = df[OHLCV_COLUMNS].ffill().bfill()

    # Ensure prices and volumes are positive
    for col in ["open", "high", "low", "close", "volume"]:
        if (df[col] <= 0).any():
            logger.warning("Found non-positive values in column %s. Clipping to minimum positive.", col)
            df[col] = df[col].clip(lower=1e-4)

    return df


def fetch_from_binance_ccxt(
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    limit: int = DEFAULT_LIMIT,
) -> pd.DataFrame:
    """Attempts to fetch OHLCV data from Binance using CCXT.

    Args:
        symbol: Market symbol (e.g. 'BTC/USDT').
        timeframe: Interval (e.g. '1h').
        limit: Number of candles.

    Returns:
        pd.DataFrame with OHLCV data and 'data_source' set to 'binance_live'.
    """
    try:
        import ccxt
    except ImportError as e:
        raise ImportError("CCXT library is not installed. Please install it with `pip install ccxt`.") from e

    logger.info("Connecting to Binance public API via CCXT for %s (%s)...", symbol, timeframe)
    exchange = ccxt.binance(
        {
            "enableRateLimit": True,
            "timeout": 15000,
        }
    )

    try:
        raw_ohlcv = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
    finally:
        # Clean up any open connections if applicable
        if hasattr(exchange, "close"):
            exchange.close()

    if not raw_ohlcv:
        raise RuntimeError(f"No OHLCV records returned from Binance for {symbol}")

    df = pd.DataFrame(raw_ohlcv, columns=OHLCV_COLUMNS)
    # Convert epoch millisecond timestamps to UTC datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    df["data_source"] = "binance_live"
    return df


def fetch_market_data(
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    limit: int = DEFAULT_LIMIT,
    output_dir: Optional[Path] = None,
    force_offline: bool = False,
) -> pd.DataFrame:
    """Retrieves market data from Binance CCXT, falling back to offline sample data if unavailable.

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT').
        timeframe: Candle interval (e.g., '1h').
        limit: Total number of records.
        output_dir: Directory to save raw CSV file (defaults to data/raw/market).
        force_offline: If True, skips API call and directly generates sample data.

    Returns:
        pd.DataFrame containing validated OHLCV records.
    """
    df: Optional[pd.DataFrame] = None

    if not force_offline:
        try:
            df = fetch_from_binance_ccxt(symbol=symbol, timeframe=timeframe, limit=limit)
            logger.info("Successfully retrieved %d live candles from Binance.", len(df))
        except Exception as err:
            logger.warning(
                "Failed to fetch live data from Binance CCXT (%s). Activating offline fallback.",
                str(err),
            )
            df = None

    if df is None:
        df = generate_offline_sample_market_data(symbol=symbol, timeframe=timeframe, limit=limit)

    # Validate and clean data
    cleaned_df = validate_market_data(df)

    # Persist raw CSV
    save_dir = Path(output_dir) if output_dir else RAW_MARKET_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    symbol_slug = symbol.replace("/", "_").lower()
    file_name = f"{symbol_slug}_{timeframe}.csv"
    output_path = save_dir / file_name

    cleaned_df.to_csv(output_path, index=False)
    logger.info("Saved raw market data to: %s", output_path)

    return cleaned_df
