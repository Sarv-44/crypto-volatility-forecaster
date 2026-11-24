"""Multimodal feature fusion and target engineering module.

Aligns hourly numerical market indicators with hourly NLP sentiment features,
handles missing sentiment hours with zero future information leakage,
and computes the forward 4-hour realized volatility target.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.features.market_features import compute_market_features
from src.nlp.sentiment_aggregation import aggregate_hourly_sentiment
from src.utils.config import PROCESSED_DIR, VOLATILITY_HORIZON_HOURS
from src.utils.logger import get_logger

logger = get_logger("FeatureFusion")


def compute_future_realized_volatility(
    log_returns: pd.Series,
    horizon_hours: int = VOLATILITY_HORIZON_HOURS,
) -> pd.Series:
    """Computes the target representing next-N-hour realized volatility.

    Formula:
        future_volatility_t = sqrt( sum_{k=1}^{horizon} r_{t+k}^2 )

    CRITICAL LEAKAGE PREVENTION:
        The target at time t strictly sums returns from (t+1) to (t+horizon).
        It does NOT include r_t. Input features at time t must only use information
        up to time t.

    Args:
        log_returns: Hourly log return series (r_t).
        horizon_hours: Forward window in hours (default: 4).

    Returns:
        pd.Series containing the forward realized volatility.
        The last `horizon_hours` entries will be NaN.
    """
    squared_returns = log_returns**2

    # Sum of squared returns from t+1 to t+horizon
    forward_sq_sum = pd.Series(0.0, index=log_returns.index)
    for shift_k in range(1, horizon_hours + 1):
        forward_sq_sum += squared_returns.shift(-shift_k)

    target_volatility = np.sqrt(forward_sq_sum)
    return target_volatility


def _normalize_timestamp_series(series: pd.Series) -> pd.Series:
    """Standardizes timestamp series to timezone-naive UTC hourly timestamps."""
    ts = pd.to_datetime(series)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return ts.dt.floor("h")


def align_market_and_sentiment(
    df_market: pd.DataFrame,
    df_sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """Aligns hourly market indicators with aggregated sentiment features.

    Handles missing sentiment hours by filling neutral defaults and adding a
    'sentiment_missing' binary mask to prevent lookahead leakage.

    Args:
        df_market: Cleaned market features indexed by hourly 'timestamp'.
        df_sentiment: Aggregated hourly sentiment features.

    Returns:
        pd.DataFrame merged chronologically without forward-looking bias.
    """
    df_m = df_market.copy()
    df_s = df_sentiment.copy()

    df_m["timestamp"] = _normalize_timestamp_series(df_m["timestamp"])
    df_s["timestamp"] = _normalize_timestamp_series(df_s["timestamp"])

    # Drop potential duplicates
    df_m = df_m.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df_s = df_s.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    # Left join to preserve complete market timeline
    merged = pd.merge(df_m, df_s, on="timestamp", how="left")

    # Identify and impute missing sentiment hours
    sentiment_cols = [
        "avg_sentiment",
        "pct_positive",
        "pct_negative",
        "pct_neutral",
        "sentiment_variance",
        "post_count",
        "sentiment_shock",
    ]

    # Binary flag indicating whether social sentiment was recorded for this hour
    merged["sentiment_missing"] = merged["avg_sentiment"].isnull().astype(int)

    # Impute missing hours with neutral baseline values
    impute_defaults = {
        "avg_sentiment": 0.0,
        "pct_positive": 0.0,
        "pct_negative": 0.0,
        "pct_neutral": 1.0,
        "sentiment_variance": 0.0,
        "post_count": 0,
        "sentiment_shock": 0.0,
    }

    for col, default_val in impute_defaults.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(default_val)

    return merged


def fuse_market_and_sentiment_features(
    df_market: Optional[pd.DataFrame] = None,
    df_sentiment: Optional[pd.DataFrame] = None,
    market_features_path: Optional[Path] = None,
    hourly_sentiment_path: Optional[Path] = None,
    horizon_hours: int = VOLATILITY_HORIZON_HOURS,
    drop_unlabeled_rows: bool = True,
    output_dir: Optional[Path] = None,
    save_filename: str = "aligned_features.csv",
) -> pd.DataFrame:
    """Orchestrates end-to-end data alignment and target volatility generation.

    Saves the final multimodal dataset ready for machine learning model training
    into data/processed/aligned_features.csv.

    Args:
        df_market: Optional market DataFrame in memory.
        df_sentiment: Optional sentiment DataFrame in memory.
        market_features_path: Path to market_features.csv if df_market not provided.
        hourly_sentiment_path: Path to hourly_sentiment.csv if df_sentiment not provided.
        horizon_hours: Forecast window (default 4 hours).
        drop_unlabeled_rows: If True, drops the final horizon rows with NaN target.
        output_dir: Directory to save aligned CSV.
        save_filename: Output filename.

    Returns:
        pd.DataFrame ready for time-series / sequence ML modeling.
    """
    save_dir = Path(output_dir) if output_dir else PROCESSED_DIR

    # Load market features if not supplied
    if df_market is None:
        m_path = market_features_path or (save_dir / "market_features.csv")
        if not m_path.exists():
            raise FileNotFoundError(f"Market features not found at {m_path}. Run market pipeline first.")
        df_market = pd.read_csv(m_path)

    # Load sentiment features if not supplied
    if df_sentiment is None:
        s_path = hourly_sentiment_path or (save_dir / "hourly_sentiment.csv")
        if not s_path.exists():
            raise FileNotFoundError(f"Hourly sentiment not found at {s_path}. Run sentiment pipeline first.")
        df_sentiment = pd.read_csv(s_path)

    logger.info("Aligning %d market records with %d sentiment periods...", len(df_market), len(df_sentiment))
    aligned_df = align_market_and_sentiment(df_market, df_sentiment)

    # Compute target: 4-hour forward realized volatility
    aligned_df["target_future_volatility"] = compute_future_realized_volatility(
        aligned_df["log_return"], horizon_hours=horizon_hours
    )

    if drop_unlabeled_rows:
        n_before = len(aligned_df)
        aligned_df = aligned_df.dropna(subset=["target_future_volatility"]).reset_index(drop=True)
        dropped_count = n_before - len(aligned_df)
        logger.info(
            "Dropped %d trailing boundary rows lacking %d-hour forward target. Clean rows: %d.",
            dropped_count,
            horizon_hours,
            len(aligned_df),
        )

    # Persist final multimodal dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / save_filename
    aligned_df.to_csv(out_file, index=False)
    logger.info("Successfully saved aligned multimodal training dataset to: %s", out_file)

    return aligned_df
