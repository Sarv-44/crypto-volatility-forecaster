"""Market feature engineering module for cryptocurrency time-series data.

Computes technical indicators, momentum, volatility, and volume statistics
from raw OHLCV price series and saves cleaned features to data/processed/.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import PROCESSED_DIR
from src.utils.logger import get_logger

logger = get_logger("MarketFeatures")


def compute_log_returns(series: pd.Series) -> pd.Series:
    """Computes log returns: r_t = ln(P_t / P_{t-1})."""
    return np.log(series / series.shift(1))


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI) using exponential moving averages.

    Args:
        close: Pandas series of closing prices.
        period: RSI lookback window (default 14).

    Returns:
        pd.Series of RSI values bounded [0, 100].
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use Wilder's style exponential smoothing (alpha = 1 / period)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When loss is 0, RSI is 100; when gain is 0, RSI is 0
    rsi = rsi.fillna(50.0)
    return rsi


def compute_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculates MACD line, Signal line, and MACD Histogram.

    Returns:
        (macd_line, macd_signal, macd_histogram)
    """
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def compute_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculates Bollinger Bands (Middle, Upper, Lower, Bandwidth, %B).

    Returns:
        (bb_middle, bb_upper, bb_lower, bb_width, bb_pct_b)
    """
    bb_middle = close.rolling(window=period).mean()
    bb_std = close.rolling(window=period).std()
    bb_upper = bb_middle + (num_std * bb_std)
    bb_lower = bb_middle - (num_std * bb_std)

    # Normalized bandwidth = (Upper - Lower) / Middle
    bb_width = (bb_upper - bb_lower) / bb_middle.replace(0, np.nan)

    # %B = (Close - Lower) / (Upper - Lower)
    band_range = (bb_upper - bb_lower).replace(0, np.nan)
    bb_pct_b = (close - bb_lower) / band_range

    return bb_middle, bb_upper, bb_lower, bb_width, bb_pct_b


def compute_market_features(
    df: pd.DataFrame,
    drop_warmup: bool = True,
    output_dir: Optional[Path] = None,
    save_filename: str = "market_features.csv",
) -> pd.DataFrame:
    """Computes full suite of market technical indicators and volatility features.

    Features generated:
        - log_return: 1-hour log return
        - vol_change_pct: percentage change in hourly volume
        - volatility_12h: 12-hour rolling standard deviation of log returns
        - volatility_24h: 24-hour rolling standard deviation of log returns
        - rsi_14: 14-hour Relative Strength Index
        - macd, macd_signal, macd_hist: 12/26/9 MACD indicators
        - bb_middle, bb_upper, bb_lower, bb_width, bb_pct_b: 20-hour Bollinger Bands
        - sma_7, sma_24: Simple Moving Averages
        - ema_7, ema_24: Exponential Moving Averages
        - momentum_4h, momentum_10h: Percentage price change over 4h and 10h

    Args:
        df: Raw DataFrame containing OHLCV columns and timestamp.
        drop_warmup: If True, drops initial NaN rows caused by rolling windows.
        output_dir: Optional directory to save output CSV (defaults to data/processed).
        save_filename: Filename for the output CSV.

    Returns:
        pd.DataFrame containing engineered market features.
    """
    df = df.copy()

    # Verify required columns
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Input DataFrame missing required column: {col}")

    # Ensure chronological sort
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1. Log Returns
    df["log_return"] = compute_log_returns(df["close"])

    # 2. Rolling Historical Volatility (std of log returns)
    df["volatility_12h"] = df["log_return"].rolling(window=12).std()
    df["volatility_24h"] = df["log_return"].rolling(window=24).std()

    # 3. Volume Percentage Change
    vol_prev = df["volume"].shift(1).replace(0, np.nan)
    df["vol_change_pct"] = (df["volume"] - df["volume"].shift(1)) / vol_prev
    # Clip extreme volume spikes to [-1.0, 10.0]
    df["vol_change_pct"] = df["vol_change_pct"].clip(lower=-1.0, upper=10.0)

    # 4. RSI
    df["rsi_14"] = compute_rsi(df["close"], period=14)

    # 5. MACD (12, 26, 9)
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(
        df["close"], fast_period=12, slow_period=26, signal_period=9
    )

    # 6. Bollinger Bands (20, 2)
    (
        df["bb_middle"],
        df["bb_upper"],
        df["bb_lower"],
        df["bb_width"],
        df["bb_pct_b"],
    ) = compute_bollinger_bands(df["close"], period=20, num_std=2.0)

    # 7. Simple Moving Averages (7h, 24h)
    df["sma_7"] = df["close"].rolling(window=7).mean()
    df["sma_24"] = df["close"].rolling(window=24).mean()

    # 8. Exponential Moving Averages (7h, 24h)
    df["ema_7"] = df["close"].ewm(span=7, adjust=False).mean()
    df["ema_24"] = df["close"].ewm(span=24, adjust=False).mean()

    # 9. Momentum (4h, 10h price change percentage)
    c_prev4 = df["close"].shift(4).replace(0, np.nan)
    df["momentum_4h"] = (df["close"] - df["close"].shift(4)) / c_prev4

    c_prev10 = df["close"].shift(10).replace(0, np.nan)
    df["momentum_10h"] = (df["close"] - df["close"].shift(10)) / c_prev10

    # Handle missing values
    if drop_warmup:
        initial_len = len(df)
        df = df.dropna().reset_index(drop=True)
        dropped_count = initial_len - len(df)
        logger.info(
            "Dropped %d initial warmup rows containing indicator NaNs. Clean rows: %d.",
            dropped_count,
            len(df),
        )
    else:
        # Backward fill any remaining NaNs for continuity
        df = df.bfill().ffill()

    # Save to data/processed/
    save_path = Path(output_dir) if output_dir else PROCESSED_DIR
    save_path.mkdir(parents=True, exist_ok=True)
    out_file = save_path / save_filename
    df.to_csv(out_file, index=False)
    logger.info("Saved engineered market features to: %s", out_file)

    return df
