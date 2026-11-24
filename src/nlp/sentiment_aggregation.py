"""Hourly sentiment aggregation module for financial NLP signals.

Aggregates individual headline/post sentiment scores into hourly metrics:
- Average sentiment score
- Positive percentage
- Negative percentage
- Neutral percentage
- Sentiment variance
- Post count
- Sentiment shock (deviation from 24h rolling mean)
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import PROCESSED_DIR, RAW_SOCIAL_DIR
from src.utils.logger import get_logger

logger = get_logger("SentimentAggregation")

# Predefined realistic sample headlines for testing and offline development
SAMPLE_HEADLINES = [
    ("Bitcoin surges to new weekly high as institutional inflows accelerate", "positive"),
    ("Whale moves 10,000 BTC to exchange, market fears imminent sell-off", "negative"),
    ("Federal Reserve maintains benchmark rates; crypto markets trade sideways", "neutral"),
    ("SEC approves new crypto exchange-traded product in historic regulatory shift", "positive"),
    ("Major DeFi protocol suffers smart contract exploit, 40M liquidated", "negative"),
    ("Bitcoin hash rate reaches all-time high amid network upgrades", "positive"),
    ("Exchange announces scheduled maintenance window for spot trading pairs", "neutral"),
    ("Crypto derivatives liquidations surpass 300M in abrupt market flush", "negative"),
    ("Analysts cite strong on-chain accumulation patterns for Bitcoin holders", "positive"),
    ("Global inflation report shows mild cooling, tech and crypto equities rally", "positive"),
    ("Regulatory scrutiny intensifies for offshore cryptocurrency platforms", "negative"),
    ("Mining difficulty adjusts upward by 2.5 percent this cycle", "neutral"),
    ("Major investment bank expands cryptocurrency custody offering for clients", "positive"),
    ("Technical indicators signal overbought momentum, pullback anticipated", "negative"),
    ("Bitcoin 30-day volatility index drops to multi-month consolidation lows", "neutral"),
]


def generate_sample_social_posts(
    csv_path: Optional[str] = None,
    num_hours: int = 72,
    posts_per_hour: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Loads social data from CSV or generates realistic sample posts labeled as sample data.

    Args:
        csv_path: Optional path to existing raw social CSV.
        num_hours: Number of past hours to cover if generating sample data.
        posts_per_hour: Average number of posts per hour.
        seed: Random seed for deterministic generation.

    Returns:
        pd.DataFrame with 'timestamp', 'text', and 'data_source'.
    """
    if csv_path and Path(csv_path).exists():
        logger.info("Loading existing social posts from: %s", csv_path)
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    logger.warning(
        "[DATA SOURCE: SAMPLE DATA] Generating offline synthetic social/news headlines. "
        "NO LIVE SOCIAL MEDIA API ACCESSED.",
    )
    np.random.seed(seed)
    end_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    records = []
    num_samples = len(SAMPLE_HEADLINES)

    for h in range(num_hours):
        hour_stamp = end_time - timedelta(hours=(num_hours - 1 - h))
        # Vary post count slightly per hour
        count = max(1, int(np.random.poisson(lam=posts_per_hour)))
        for _ in range(count):
            headline_idx = np.random.randint(0, num_samples)
            text, _ = SAMPLE_HEADLINES[headline_idx]
            # Random minute within the hour
            minute = np.random.randint(0, 60)
            second = np.random.randint(0, 60)
            post_time = hour_stamp + timedelta(minutes=minute, seconds=second)
            records.append(
                {
                    "timestamp": post_time,
                    "text": text,
                    "data_source": "sample_social_offline",
                }
            )

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Save raw sample social data
    RAW_SOCIAL_DIR.mkdir(parents=True, exist_ok=True)
    raw_sample_file = RAW_SOCIAL_DIR / "sample_social_posts.csv"
    df.to_csv(raw_sample_file, index=False)
    logger.info("Saved raw sample social posts to: %s (%d records)", raw_sample_file, len(df))

    return df


def aggregate_hourly_sentiment(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    score_col: str = "sentiment_score",
    pos_col: str = "sentiment_positive",
    neg_col: str = "sentiment_negative",
    neu_col: str = "sentiment_neutral",
    output_dir: Optional[Path] = None,
    save_filename: str = "hourly_sentiment.csv",
) -> pd.DataFrame:
    """Aggregates post-level sentiment scores into hourly signals.

    Features generated per hour:
        - avg_sentiment: Mean sentiment score (-1 to +1)
        - pct_positive: Proportion of posts with predominantly positive sentiment
        - pct_negative: Proportion of posts with predominantly negative sentiment
        - pct_neutral: Proportion of posts with predominantly neutral sentiment
        - sentiment_variance: Variance of sentiment scores within the hour
        - post_count: Total posts scored in that hour
        - sentiment_shock: Hourly avg_sentiment minus 24h rolling mean

    Args:
        df: Scored DataFrame with timestamps and sentiment columns.
        timestamp_col: Column containing datetimes.
        score_col: Column with sentiment score.
        pos_col: Positive probability column.
        neg_col: Negative probability column.
        neu_col: Neutral probability column.
        output_dir: Optional path to save aggregated CSV (defaults to data/processed).
        save_filename: Output CSV filename.

    Returns:
        pd.DataFrame containing hourly aggregated sentiment metrics.
    """
    if df.empty:
        raise ValueError("Input DataFrame for sentiment aggregation is empty.")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df[timestamp_col])
    # Floor to hour for hourly grouping
    df["hourly_timestamp"] = df["timestamp"].dt.floor("h")

    # Determine dominant sentiment category for each post
    conditions = [
        (df[pos_col] >= df[neg_col]) & (df[pos_col] >= df[neu_col]),
        (df[neg_col] > df[pos_col]) & (df[neg_col] >= df[neu_col]),
    ]
    choices = ["positive", "negative"]
    df["dominant_sentiment"] = np.select(conditions, choices, default="neutral")

    # Group by hourly timestamp
    grouped = df.groupby("hourly_timestamp")

    hourly_rows = []
    for hour, group in grouped:
        n_posts = len(group)
        avg_score = float(group[score_col].mean())
        var_score = float(group[score_col].var(ddof=1)) if n_posts > 1 else 0.0
        if np.isnan(var_score):
            var_score = 0.0

        pct_pos = float((group["dominant_sentiment"] == "positive").mean())
        pct_neg = float((group["dominant_sentiment"] == "negative").mean())
        pct_neu = float((group["dominant_sentiment"] == "neutral").mean())

        hourly_rows.append(
            {
                "timestamp": hour,
                "avg_sentiment": round(avg_score, 4),
                "pct_positive": round(pct_pos, 4),
                "pct_negative": round(pct_neg, 4),
                "pct_neutral": round(pct_neu, 4),
                "sentiment_variance": round(var_score, 6),
                "post_count": int(n_posts),
            }
        )

    hourly_df = pd.DataFrame(hourly_rows)
    hourly_df = hourly_df.sort_values("timestamp").reset_index(drop=True)

    # Calculate sentiment shock: avg_sentiment - 24h rolling mean
    rolling_24h_mean = hourly_df["avg_sentiment"].rolling(window=24, min_periods=1).mean()
    hourly_df["sentiment_shock"] = (hourly_df["avg_sentiment"] - rolling_24h_mean).round(4)

    # Save to data/processed/
    save_path = Path(output_dir) if output_dir else PROCESSED_DIR
    save_path.mkdir(parents=True, exist_ok=True)
    out_file = save_path / save_filename
    hourly_df.to_csv(out_file, index=False)
    logger.info("Saved hourly aggregated sentiment features to: %s", out_file)

    return hourly_df
