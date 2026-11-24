"""Baseline volatility forecasting model using scikit-learn.

Trains a regularized Ridge regression and Random Forest baseline on aligned
multimodal features to predict next-4-hour realized cryptocurrency volatility.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.utils.config import PROCESSED_DIR, SAVED_MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger("BaselineModel")

DEFAULT_FEATURE_COLS = [
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
    "avg_sentiment",
    "pct_positive",
    "pct_negative",
    "sentiment_variance",
    "sentiment_shock",
    "sentiment_missing",
]

TARGET_COL = "target_future_volatility"


class CryptoVolatilityForecaster:
    """Baseline machine learning forecaster combining market and sentiment features."""

    def __init__(self, model_type: str = "ridge", alpha: float = 1.0):
        self.model_type = model_type
        self.alpha = alpha
        self.scaler = StandardScaler()
        if model_type == "rf":
            self.model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        else:
            self.model = Ridge(alpha=alpha, random_state=42)
        self.feature_cols: List[str] = DEFAULT_FEATURE_COLS
        self.is_trained = False

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        test_ratio: float = 0.2,
    ) -> Dict[str, float]:
        """Trains baseline model on aligned features using temporal train/test split.

        Args:
            df: DataFrame containing input features and target.
            feature_cols: List of column names to use as features.
            test_ratio: Ratio of chronologically latest data reserved for test.

        Returns:
            Dict of evaluation metrics (RMSE, MAE, R2).
        """
        if feature_cols:
            self.feature_cols = feature_cols

        # Ensure required columns are present
        missing = [c for c in self.feature_cols + [TARGET_COL] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in dataset: {missing}")

        clean_df = df.dropna(subset=self.feature_cols + [TARGET_COL]).reset_index(drop=True)
        if len(clean_df) < 10:
            raise ValueError(f"Insufficient samples for training: {len(clean_df)}")

        X = clean_df[self.feature_cols].values
        y = clean_df[TARGET_COL].values

        # Strict chronological temporal split (no future shuffle)
        split_idx = int(len(clean_df) * (1.0 - test_ratio))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Fit scaler on train set only to prevent leakage
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        metrics = {
            "test_rmse": round(rmse, 6),
            "test_mae": round(mae, 6),
            "test_r2": round(r2, 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }
        logger.info("Trained %s baseline model. Metrics: %s", self.model_type, metrics)
        return metrics

    def predict(self, feature_vector: np.ndarray) -> float:
        """Predicts next-4-hour volatility for a single feature vector."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before predicting.")
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)
        scaled = self.scaler.transform(feature_vector)
        pred = float(self.model.predict(scaled)[0])
        # Volatility must be non-negative
        return max(0.0, pred)

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Saves trained model and scaler to disk."""
        target_path = filepath or (SAVED_MODELS_DIR / f"baseline_{self.model_type}.joblib")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "model_type": self.model_type,
            },
            target_path,
        )
        logger.info("Saved baseline model to %s", target_path)
        return target_path

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "CryptoVolatilityForecaster":
        """Loads trained model and scaler from disk."""
        target_path = filepath or (SAVED_MODELS_DIR / "baseline_ridge.joblib")
        if not target_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {target_path}")
        payload = joblib.load(target_path)
        instance = cls(model_type=payload.get("model_type", "ridge"))
        instance.model = payload["model"]
        instance.scaler = payload["scaler"]
        instance.feature_cols = payload["feature_cols"]
        instance.is_trained = True
        logger.info("Loaded baseline model from %s", target_path)
        return instance


def train_and_save_baseline(dataset_path: Optional[Path] = None) -> Tuple[CryptoVolatilityForecaster, Dict[str, float]]:
    """Convenience function to train baseline model on aligned dataset and persist it."""
    data_file = dataset_path or (PROCESSED_DIR / "aligned_features.csv")
    if not data_file.exists():
        raise FileNotFoundError(f"Aligned dataset not found at {data_file}")
    df = pd.read_csv(data_file)
    forecaster = CryptoVolatilityForecaster(model_type="ridge")
    metrics = forecaster.train(df)
    forecaster.save()
    return forecaster, metrics
