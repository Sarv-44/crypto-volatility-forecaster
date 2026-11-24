"""Ablation study and feature importance evaluation module.

Quantifies the incremental value added by FinBERT NLP sentiment signals
over pure market numerical indicators to prove multimodal efficacy for research vivas.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.models.evaluation import compute_volatility_metrics
from src.utils.config import PROCESSED_DIR
from src.utils.logger import get_logger

logger = get_logger("AblationStudy")

MARKET_ONLY_FEATURES = [
    "log_return",
    "volatility_12h",
    "volatility_24h",
    "vol_change_pct",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_width",
    "bb_pct_b",
    "momentum_4h",
    "momentum_10h",
]

SENTIMENT_ONLY_FEATURES = [
    "avg_sentiment",
    "pct_positive",
    "pct_negative",
    "sentiment_variance",
    "sentiment_shock",
    "sentiment_missing",
]

MULTIMODAL_FEATURES = MARKET_ONLY_FEATURES + SENTIMENT_ONLY_FEATURES
TARGET_COL = "target_future_volatility"


def evaluate_feature_subset(
    df: pd.DataFrame,
    feature_cols: List[str],
    test_ratio: float = 0.20,
    alpha: float = 1.0,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Trains and evaluates a model on a specific feature subset using strict chronological split."""
    clean_df = df.dropna(subset=feature_cols + [TARGET_COL]).reset_index(drop=True)
    X = clean_df[feature_cols].values
    y = clean_df[TARGET_COL].values

    split_idx = int(len(clean_df) * (1.0 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

    ref_vol = clean_df["volatility_24h"].values[split_idx:] if "volatility_24h" in clean_df.columns else None
    metrics = compute_volatility_metrics(y_test, y_pred, ref_volatility=ref_vol)
    return metrics, y_test, y_pred


def calculate_permutation_importance(
    df: pd.DataFrame,
    feature_cols: List[str] = MULTIMODAL_FEATURES,
    test_ratio: float = 0.20,
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Computes permutation importance: measures increase in RMSE when each feature is shuffled.

    Args:
        df: Aligned features DataFrame.
        feature_cols: Feature columns to rank.
        test_ratio: Held-out test fraction.
        n_repeats: Number of shuffles per feature.
        seed: Random seed.

    Returns:
        pd.DataFrame ranking features by importance.
    """
    np.random.seed(seed)
    clean_df = df.dropna(subset=feature_cols + [TARGET_COL]).reset_index(drop=True)
    X = clean_df[feature_cols].values
    y = clean_df[TARGET_COL].values

    split_idx = int(len(clean_df) * (1.0 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = Ridge(alpha=1.0, random_state=seed)
    model.fit(X_train_s, y_train)

    base_preds = np.clip(model.predict(X_test_s), a_min=0.0, a_max=None)
    base_rmse = np.sqrt(np.mean((y_test - base_preds) ** 2))

    importance_scores = []
    for col_idx, col_name in enumerate(feature_cols):
        perm_losses = []
        for _ in range(n_repeats):
            X_perm = X_test_s.copy()
            shuffled_idx = np.random.permutation(len(X_perm))
            X_perm[:, col_idx] = X_perm[shuffled_idx, col_idx]

            perm_preds = np.clip(model.predict(X_perm), a_min=0.0, a_max=None)
            perm_rmse = np.sqrt(np.mean((y_test - perm_preds) ** 2))
            # Increase in RMSE due to destroying feature signal
            perm_losses.append(perm_rmse - base_rmse)

        mean_increase = float(np.mean(perm_losses))
        category = "NLP Sentiment" if col_name in SENTIMENT_ONLY_FEATURES else "Market Technical"
        importance_scores.append(
            {
                "feature": col_name,
                "category": category,
                "importance_score": round(max(0.0, mean_increase), 7),
            }
        )

    imp_df = pd.DataFrame(importance_scores)
    total_score = imp_df["importance_score"].sum()
    if total_score > 0:
        imp_df["relative_importance_pct"] = (
            (imp_df["importance_score"] / total_score) * 100.0
        ).round(2)
    else:
        imp_df["relative_importance_pct"] = round(100.0 / len(imp_df), 2)

    imp_df = imp_df.sort_values("importance_score", ascending=False).reset_index(drop=True)
    return imp_df


def run_ablation_study(
    df: Optional[pd.DataFrame] = None,
    csv_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs full ablation study and saves results.

    Configurations compared:
    1. Market Indicators Only
    2. Sentiment Signals Only
    3. Multimodal Fusion (Market + Sentiment)

    Returns:
        (ablation_df, feature_importance_df)
    """
    logger.info("=== Starting Multimodal Ablation Study ===")
    if df is None:
        data_file = csv_path or (PROCESSED_DIR / "aligned_features.csv")
        df = pd.read_csv(data_file)

    # 1. Market Only
    m_metrics, _, _ = evaluate_feature_subset(df, MARKET_ONLY_FEATURES)
    m_metrics["configuration"] = "Market Features Only"
    m_metrics["feature_count"] = len(MARKET_ONLY_FEATURES)

    # 2. Sentiment Only
    s_metrics, _, _ = evaluate_feature_subset(df, SENTIMENT_ONLY_FEATURES)
    s_metrics["configuration"] = "FinBERT Sentiment Only"
    s_metrics["feature_count"] = len(SENTIMENT_ONLY_FEATURES)

    # 3. Multimodal Fusion
    f_metrics, _, _ = evaluate_feature_subset(df, MULTIMODAL_FEATURES)
    f_metrics["configuration"] = "Multimodal Fusion (Market + NLP)"
    f_metrics["feature_count"] = len(MULTIMODAL_FEATURES)

    # Calculate Lift / Incremental Gain of Multimodal over Market Only
    rmse_lift = ((m_metrics["rmse"] - f_metrics["rmse"]) / m_metrics["rmse"]) * 100.0
    qlike_lift = ((m_metrics["qlike_loss"] - f_metrics["qlike_loss"]) / m_metrics["qlike_loss"]) * 100.0
    dir_acc_gain = f_metrics["directional_accuracy_pct"] - m_metrics["directional_accuracy_pct"]

    ablation_rows = [m_metrics, s_metrics, f_metrics]
    ablation_df = pd.DataFrame(ablation_rows)
    cols = ["configuration", "feature_count", "rmse", "mae", "qlike_loss", "directional_accuracy_pct"]
    ablation_df = ablation_df[cols]

    # Feature Importance
    logger.info("Computing permutation feature importance...")
    feat_imp_df = calculate_permutation_importance(df, feature_cols=MULTIMODAL_FEATURES)

    # Save to data/processed/
    save_dir = Path(output_dir) if output_dir else PROCESSED_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    ablation_file = save_dir / "ablation_study_results.csv"
    feat_imp_file = save_dir / "feature_importance.csv"

    ablation_df.to_csv(ablation_file, index=False)
    feat_imp_df.to_csv(feat_imp_file, index=False)
    logger.info("Saved ablation study results to: %s", ablation_file)
    logger.info("Saved feature importance to: %s", feat_imp_file)

    logger.info(
        "Ablation Summary: Multimodal RMSE Lift = %.2f%%, Directional Accuracy Change = %+.2f%%",
        rmse_lift,
        dir_acc_gain,
    )
    return ablation_df, feat_imp_df
