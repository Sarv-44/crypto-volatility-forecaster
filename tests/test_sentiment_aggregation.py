"""Unit tests for hourly sentiment aggregation (Stage 5)."""

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.nlp.sentiment_aggregation import (
    aggregate_hourly_sentiment,
    generate_sample_social_posts,
)


def test_generate_sample_social_posts():
    posts = generate_sample_social_posts(num_hours=10, posts_per_hour=3)
    assert not posts.empty
    assert "timestamp" in posts.columns
    assert "text" in posts.columns
    assert "data_source" in posts.columns
    assert (posts["data_source"] == "sample_social_offline").all()


def test_aggregate_hourly_sentiment_metrics(tmp_path: Path):
    # Create synthetic scored records for 2 distinct hours
    h1 = datetime(2026, 1, 1, 10, 15)
    h1_2 = datetime(2026, 1, 1, 10, 45)
    h2 = datetime(2026, 1, 1, 11, 20)

    df = pd.DataFrame(
        [
            {
                "timestamp": h1,
                "sentiment_positive": 0.8,
                "sentiment_negative": 0.1,
                "sentiment_neutral": 0.1,
                "sentiment_score": 0.7,
            },
            {
                "timestamp": h1_2,
                "sentiment_positive": 0.6,
                "sentiment_negative": 0.2,
                "sentiment_neutral": 0.2,
                "sentiment_score": 0.4,
            },
            {
                "timestamp": h2,
                "sentiment_positive": 0.1,
                "sentiment_negative": 0.8,
                "sentiment_neutral": 0.1,
                "sentiment_score": -0.7,
            },
        ]
    )

    hourly = aggregate_hourly_sentiment(
        df,
        output_dir=tmp_path,
        save_filename="test_hourly.csv",
    )

    assert len(hourly) == 2

    # Check hour 1
    row1 = hourly.iloc[0]
    assert np.isclose(row1["avg_sentiment"], 0.55)
    assert row1["post_count"] == 2
    assert row1["pct_positive"] == 1.0
    assert row1["pct_negative"] == 0.0
    assert row1["sentiment_variance"] > 0.0

    # Check hour 2
    row2 = hourly.iloc[1]
    assert np.isclose(row2["avg_sentiment"], -0.7)
    assert row2["post_count"] == 1
    assert row2["pct_negative"] == 1.0
    assert row2["sentiment_variance"] == 0.0

    # Check shock calculation
    assert "sentiment_shock" in hourly.columns

    # Check file was saved
    assert (tmp_path / "test_hourly.csv").exists()


def test_aggregate_empty_df_raises():
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame for sentiment aggregation is empty"):
        aggregate_hourly_sentiment(empty_df)
