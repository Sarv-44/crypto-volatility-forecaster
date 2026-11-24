"""Unit tests for feature fusion, data alignment, and volatility target (Stages 6 & 7)."""

from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.feature_fusion import (
    align_market_and_sentiment,
    compute_future_realized_volatility,
    fuse_market_and_sentiment_features,
)


def test_future_realized_volatility_formula():
    # Returns for 6 consecutive hours
    # r0, r1, r2, r3, r4, r5
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02, 0.01])
    target = compute_future_realized_volatility(returns, horizon_hours=4)

    # For index 0, target should be sqrt(r1^2 + r2^2 + r3^2 + r4^2)
    expected_0 = np.sqrt(0.02**2 + (-0.01)**2 + 0.03**2 + (-0.02)**2)
    assert np.isclose(target.iloc[0], expected_0)

    # For index 1, target should be sqrt(r2^2 + r3^2 + r4^2 + r5^2)
    expected_1 = np.sqrt((-0.01)**2 + 0.03**2 + (-0.02)**2 + 0.01**2)
    assert np.isclose(target.iloc[1], expected_1)

    # The last 4 entries (indices 2, 3, 4, 5) cannot look 4 steps ahead without exceeding bounds
    assert np.isnan(target.iloc[-1])
    assert np.isnan(target.iloc[-2])
    assert np.isnan(target.iloc[-3])
    assert np.isnan(target.iloc[-4])


def test_zero_leakage_in_target():
    """Verify target does NOT depend on current or past returns."""
    base_returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    t_base = compute_future_realized_volatility(base_returns, horizon_hours=4)

    # Alter return at time t=0
    altered_returns = base_returns.copy()
    altered_returns.iloc[0] = 0.999  # Massive shock at t=0

    t_altered = compute_future_realized_volatility(altered_returns, horizon_hours=4)

    # Target at t=0 must NOT change because it looks strictly at t=1..t=4
    assert np.isclose(t_base.iloc[0], t_altered.iloc[0])


def test_missing_sentiment_hours_alignment():
    timestamps_m = [datetime(2026, 1, 1, h) for h in range(5)]
    df_market = pd.DataFrame(
        {
            "timestamp": timestamps_m,
            "close": [65000, 65200, 64900, 65100, 65300],
            "log_return": [0.001, 0.002, -0.003, 0.002, 0.003],
        }
    )

    # Sentiment only exists for hours 1 and 3 (hours 0, 2, 4 are missing)
    timestamps_s = [datetime(2026, 1, 1, 1), datetime(2026, 1, 1, 3)]
    df_sentiment = pd.DataFrame(
        {
            "timestamp": timestamps_s,
            "avg_sentiment": [0.45, -0.25],
            "pct_positive": [0.8, 0.1],
            "pct_negative": [0.1, 0.7],
            "pct_neutral": [0.1, 0.2],
            "sentiment_variance": [0.02, 0.01],
            "post_count": [5, 3],
            "sentiment_shock": [0.1, -0.2],
        }
    )

    aligned = align_market_and_sentiment(df_market, df_sentiment)

    assert len(aligned) == 5
    # Missing hour 0
    assert aligned.loc[0, "sentiment_missing"] == 1
    assert aligned.loc[0, "avg_sentiment"] == 0.0
    assert aligned.loc[0, "pct_neutral"] == 1.0
    assert aligned.loc[0, "post_count"] == 0

    # Present hour 1
    assert aligned.loc[1, "sentiment_missing"] == 0
    assert np.isclose(aligned.loc[1, "avg_sentiment"], 0.45)
    assert aligned.loc[1, "post_count"] == 5


def test_fuse_market_and_sentiment_features_end_to_end(tmp_path: Path):
    hours = 30
    base_time = datetime(2026, 1, 1, 0)
    timestamps = [base_time + timedelta(hours=i) for i in range(hours)]

    df_market = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": np.linspace(60000, 65000, hours),
            "log_return": np.random.normal(0, 0.01, hours),
            "volatility_24h": [0.015] * hours,
            "rsi_14": [55.0] * hours,
        }
    )

    df_sentiment = pd.DataFrame(
        {
            "timestamp": timestamps[:25],  # last 5 hours missing
            "avg_sentiment": [0.2] * 25,
            "pct_positive": [0.5] * 25,
            "pct_negative": [0.1] * 25,
            "pct_neutral": [0.4] * 25,
            "sentiment_variance": [0.01] * 25,
            "post_count": [4] * 25,
            "sentiment_shock": [0.05] * 25,
        }
    )

    fused = fuse_market_and_sentiment_features(
        df_market=df_market,
        df_sentiment=df_sentiment,
        horizon_hours=4,
        drop_unlabeled_rows=True,
        output_dir=tmp_path,
        save_filename="test_aligned.csv",
    )

    # 30 initial - 4 trailing rows lacking target = 26 rows
    assert len(fused) == 26
    assert "target_future_volatility" in fused.columns
    assert fused["target_future_volatility"].isnull().sum() == 0
    assert (tmp_path / "test_aligned.csv").exists()
