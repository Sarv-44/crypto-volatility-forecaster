"""Unit tests for multimodal ablation study and permutation feature importance."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.ablation import (
    MARKET_ONLY_FEATURES,
    MULTIMODAL_FEATURES,
    SENTIMENT_ONLY_FEATURES,
    calculate_permutation_importance,
    evaluate_feature_subset,
    run_ablation_study,
)


@pytest.fixture
def dummy_aligned_df():
    N = 80
    data = {"target_future_volatility": np.abs(np.random.normal(0.01, 0.005, N))}
    for col in MULTIMODAL_FEATURES:
        data[col] = np.random.normal(0, 1, N)
    data["volatility_24h"] = np.abs(np.random.normal(0.01, 0.002, N))
    return pd.DataFrame(data)


def test_evaluate_feature_subset(dummy_aligned_df):
    metrics, y_test, y_pred = evaluate_feature_subset(dummy_aligned_df, MARKET_ONLY_FEATURES)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert len(y_test) == len(y_pred)
    assert (y_pred >= 0.0).all()


def test_calculate_permutation_importance(dummy_aligned_df):
    imp_df = calculate_permutation_importance(dummy_aligned_df, feature_cols=MULTIMODAL_FEATURES[:6], n_repeats=2)
    assert len(imp_df) == 6
    assert "feature" in imp_df.columns
    assert "importance_score" in imp_df.columns
    assert "relative_importance_pct" in imp_df.columns
    assert (imp_df["importance_score"] >= 0.0).all()
    assert np.isclose(imp_df["relative_importance_pct"].sum(), 100.0, atol=1e-1)


def test_run_ablation_study_end_to_end(dummy_aligned_df, tmp_path: Path):
    ablation_df, feat_imp_df = run_ablation_study(dummy_aligned_df, output_dir=tmp_path)
    assert len(ablation_df) == 3
    assert "Market Features Only" in ablation_df["configuration"].values
    assert "FinBERT Sentiment Only" in ablation_df["configuration"].values
    assert "Multimodal Fusion (Market + NLP)" in ablation_df["configuration"].values

    assert (tmp_path / "ablation_study_results.csv").exists()
    assert (tmp_path / "feature_importance.csv").exists()
