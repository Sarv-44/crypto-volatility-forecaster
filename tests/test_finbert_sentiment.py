"""Unit tests for FinBERT sentiment scoring and caching (Stage 4)."""

from pathlib import Path
import pandas as pd
import pytest

from src.nlp.finbert_sentiment import FinBERTSentimentAnalyzer


def test_missing_and_empty_text_handling():
    analyzer = FinBERTSentimentAnalyzer(enable_cache=False)

    for invalid_text in [None, "", "   ", float("nan")]:
        result = analyzer.predict_single(invalid_text)
        assert result["positive"] == 0.0
        assert result["negative"] == 0.0
        assert result["neutral"] == 1.0
        assert result["sentiment_score"] == 0.0


def test_sentiment_score_formula():
    analyzer = FinBERTSentimentAnalyzer(enable_cache=False)
    text = "Bitcoin surges past 70000 with strong institutional volume"
    result = analyzer.predict_single(text)

    assert "positive" in result
    assert "negative" in result
    assert "neutral" in result
    assert "sentiment_score" in result
    # Check that score = positive - negative within floating tolerance
    expected_score = round(result["positive"] - result["negative"], 4)
    assert abs(result["sentiment_score"] - expected_score) < 1e-4


def test_caching_mechanism(tmp_path: Path):
    cache_path = tmp_path / "test_cache.json"
    analyzer = FinBERTSentimentAnalyzer(cache_file=cache_path, enable_cache=True)

    text = "Ethereum layer-2 network experiences high network activity"
    # First call fills cache
    res1 = analyzer.predict_single(text)
    assert cache_path.exists()

    # Verify key is in memory cache
    key = analyzer._get_cache_key(text)
    assert key in analyzer.memory_cache

    # Second call uses cache
    res2 = analyzer.predict_single(text)
    assert res1 == res2


def test_batch_processing_and_dataframe(tmp_path: Path):
    cache_path = tmp_path / "batch_cache.json"
    analyzer = FinBERTSentimentAnalyzer(cache_file=cache_path, enable_cache=True)

    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "text": [
                "Crypto market rallies on positive regulatory clarity",
                "Major exchange suffers severe security breach and liquidation",
                None,
                "Federal Reserve holds interest rates steady",
            ],
        }
    )

    scored_df = analyzer.analyze_dataframe(df, text_col="text", batch_size=2)

    assert len(scored_df) == 4
    for col in ["sentiment_positive", "sentiment_negative", "sentiment_neutral", "sentiment_score"]:
        assert col in scored_df.columns

    # None row must be neutral
    assert scored_df.loc[2, "sentiment_neutral"] == 1.0
    assert scored_df.loc[2, "sentiment_score"] == 0.0
