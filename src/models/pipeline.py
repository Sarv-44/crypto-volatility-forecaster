"""End-to-end model training, checkpointing, and benchmarking orchestrator.

Trains Ridge, LSTM, and GRU models on aligned features and benchmarks them
on the held-out chronological test split.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

from src.models.baseline import CryptoVolatilityForecaster
from src.models.deep_models import CryptoGRU, CryptoLSTM
from src.models.evaluation import benchmark_models, compute_volatility_metrics
from src.models.sequence_dataset import get_dataloaders, prepare_sequence_data
from src.models.trainer import DeepModelTrainer
from src.utils.config import PROCESSED_DIR, SAVED_MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger("ModelPipeline")


def train_all_models(
    seq_len: int = 12,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 2e-3,
    loss_type: str = "mse",
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Trains Ridge baseline, LSTM, and GRU forecasters, then benchmarks them.

    Args:
        seq_len: Lookback window length in hours (default: 12).
        epochs: Max training epochs per deep model.
        batch_size: Batch size.
        lr: Learning rate.
        loss_type: 'mse' or 'qlike'.

    Returns:
        (benchmark_df, summary_dict)
    """
    logger.info("=== Starting Multi-Model Training & Benchmarking Pipeline ===")

    # 1. Prepare temporal sequence data
    train_data, val_data, test_data, scaler, feature_cols = prepare_sequence_data(
        seq_len=seq_len, train_ratio=0.70, val_ratio=0.15
    )

    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    # Save sequence scaler and feature list for inference
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = SAVED_MODELS_DIR / "sequence_scaler.joblib"
    joblib.dump(
        {"scaler": scaler, "feature_cols": feature_cols, "seq_len": seq_len},
        scaler_path,
    )
    logger.info("Saved sequence scaler metadata to: %s", scaler_path)

    # 2. Train Ridge Baseline (using most recent timestep features)
    logger.info("Training Ridge regression baseline...")
    ridge_forecaster = CryptoVolatilityForecaster(model_type="ridge", alpha=1.0)
    aligned_df = pd.read_csv(PROCESSED_DIR / "aligned_features.csv")
    ridge_metrics = ridge_forecaster.train(aligned_df, feature_cols=feature_cols)
    ridge_forecaster.save()

    # Create Ridge predictions on the sequence test set
    # X_test has shape (N, seq_len, num_features); take the last timestep of each sequence
    X_test_last = X_test[:, -1, :]
    ridge_preds = np.array([ridge_forecaster.model.predict(feat.reshape(1, -1))[0] for feat in X_test_last])
    ridge_preds = np.clip(ridge_preds, a_min=0.0, a_max=None)

    # 3. Train PyTorch Deep Models
    train_loader, val_loader, test_loader = get_dataloaders(
        train_data, val_data, test_data, batch_size=batch_size
    )

    trainer = DeepModelTrainer()
    in_dim = len(feature_cols)

    # Model A: CryptoLSTM
    logger.info("--- Training CryptoLSTM Model ---")
    lstm_model = CryptoLSTM(input_size=in_dim, hidden_size=64, num_layers=2, dropout=0.2)
    lstm_res = trainer.fit(
        model=lstm_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        loss_type=loss_type,
        model_name="crypto_lstm",
    )

    # Model B: CryptoGRU
    logger.info("--- Training CryptoGRU Model ---")
    gru_model = CryptoGRU(input_size=in_dim, hidden_size=64, num_layers=2, dropout=0.2)
    gru_res = trainer.fit(
        model=gru_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        loss_type=loss_type,
        model_name="crypto_gru",
    )

    # 4. Generate Predictions on Held-Out Test Set
    device = trainer.device
    lstm_model.eval()
    gru_model.eval()

    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        lstm_preds = lstm_model(X_test_t).cpu().numpy().flatten()
        gru_preds = gru_model(X_test_t).cpu().numpy().flatten()

    # 5. Naive Persistence Baseline (predicting future vol = current 24h rolling vol)
    # The 24h rolling vol feature index:
    vol_24h_idx = feature_cols.index("volatility_24h") if "volatility_24h" in feature_cols else 1
    # Note: features in X_test were scaled; retrieve unscaled 24h vol from test_df
    total_samples = len(aligned_df)
    train_end = int(total_samples * 0.70)
    val_end = int(total_samples * 0.85)
    test_slice = aligned_df.iloc[val_end + seq_len - 1 :].reset_index(drop=True)
    naive_preds = test_slice["volatility_24h"].values[: len(y_test)]

    # If length mismatch due to window bounds, adjust
    min_len = min(len(y_test), len(naive_preds), len(lstm_preds), len(gru_preds), len(ridge_preds))
    y_test_eval = y_test[:min_len]
    ref_vol = naive_preds[:min_len]

    predictions_dict = {
        "Naive Persistence (24h Vol)": naive_preds[:min_len],
        "Ridge Regressor": ridge_preds[:min_len],
        "CryptoLSTM": lstm_preds[:min_len],
        "CryptoGRU": gru_preds[:min_len],
    }

    # 6. Benchmark Evaluation
    benchmark_df = benchmark_models(
        test_actuals=y_test_eval,
        predictions_dict=predictions_dict,
        ref_volatility=ref_vol,
    )

    summary = {
        "test_samples": min_len,
        "feature_count": in_dim,
        "lstm_checkpoint": lstm_res["checkpoint_path"],
        "gru_checkpoint": gru_res["checkpoint_path"],
        "benchmark_table": benchmark_df.to_dict(orient="records"),
    }

    logger.info("=== Model Benchmarking Complete ===")
    return benchmark_df, summary
