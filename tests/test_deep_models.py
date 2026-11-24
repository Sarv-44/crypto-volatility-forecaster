"""Unit tests for sequence dataset, deep models, QLIKE loss, trainer, and evaluation (Phase 2)."""

from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.models.deep_models import CryptoGRU, CryptoLSTM, CryptoMultimodalAttention
from src.models.evaluation import compute_directional_accuracy, compute_qlike_numpy, compute_volatility_metrics
from src.models.sequence_dataset import CryptoVolatilityDataset, create_sliding_windows
from src.models.trainer import DeepModelTrainer, QLIKELoss


def test_create_sliding_windows():
    T, D, L = 50, 5, 10
    features = np.random.randn(T, D)
    targets = np.random.rand(T)

    seqs, tgts = create_sliding_windows(features, targets, seq_len=L)
    expected_samples = T - L + 1

    assert seqs.shape == (expected_samples, L, D)
    assert tgts.shape == (expected_samples,)
    # Verify the last timestep in the first sequence window corresponds to window[L-1]
    assert np.allclose(seqs[0][-1], features[L - 1])
    assert np.isclose(tgts[0], targets[L - 1])


def test_lstm_forward_pass_and_nonnegativity():
    batch_size, seq_len, in_dim = 8, 12, 10
    model = CryptoLSTM(input_size=in_dim, hidden_size=32, num_layers=1)
    dummy_input = torch.randn(batch_size, seq_len, in_dim)

    output = model(dummy_input)
    assert output.shape == (batch_size, 1)
    # Output must strictly be non-negative for volatility
    assert (output >= 0.0).all()


def test_gru_forward_pass_and_nonnegativity():
    batch_size, seq_len, in_dim = 8, 12, 10
    model = CryptoGRU(input_size=in_dim, hidden_size=32, num_layers=1)
    dummy_input = torch.randn(batch_size, seq_len, in_dim)

    output = model(dummy_input)
    assert output.shape == (batch_size, 1)
    assert (output >= 0.0).all()


def test_attention_forward_pass():
    batch_size, seq_len, in_dim = 8, 12, 10
    model = CryptoMultimodalAttention(input_size=in_dim, hidden_size=32, num_heads=2)
    dummy_input = torch.randn(batch_size, seq_len, in_dim)

    output = model(dummy_input)
    assert output.shape == (batch_size, 1)
    assert (output >= 0.0).all()


def test_qlike_loss_behavior():
    loss_fn = QLIKELoss()

    # Exact prediction should yield 0 loss
    y_true = torch.tensor([[0.02], [0.03]], dtype=torch.float32)
    y_pred_exact = torch.tensor([[0.02], [0.03]], dtype=torch.float32)
    loss_exact = loss_fn(y_pred_exact, y_true)
    assert torch.isclose(loss_exact, torch.tensor(0.0), atol=1e-5)

    # QLIKE asymmetry: under-predicting volatility incurs higher penalty than over-predicting
    y_target = torch.tensor([[0.02]], dtype=torch.float32)
    under_pred = torch.tensor([[0.01]], dtype=torch.float32)  # 50% under
    over_pred = torch.tensor([[0.03]], dtype=torch.float32)   # 50% over

    loss_under = loss_fn(under_pred, y_target)
    loss_over = loss_fn(over_pred, y_target)
    assert loss_under > loss_over


def test_deep_model_trainer_step(tmp_path: Path):
    trainer = DeepModelTrainer(device="cpu", save_dir=tmp_path)
    model = CryptoLSTM(input_size=4, hidden_size=16, num_layers=1)

    # Create dummy DataLoader
    X_train = np.random.randn(20, 8, 4).astype(np.float32)
    y_train = (np.abs(np.random.randn(20, 1)) * 0.02).astype(np.float32)
    train_loader = DataLoader(CryptoVolatilityDataset(X_train, y_train), batch_size=4)
    val_loader = DataLoader(CryptoVolatilityDataset(X_train[:8], y_train[:8]), batch_size=4)

    results = trainer.fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        lr=0.01,
        loss_type="mse",
        model_name="test_lstm",
    )

    assert "best_epoch" in results
    assert "best_val_loss" in results
    assert len(results["history"]["train_loss"]) == 3
    assert (tmp_path / "best_test_lstm.pt").exists()


def test_volatility_evaluation_metrics():
    y_true = np.array([0.01, 0.02, 0.03, 0.04])
    y_pred = np.array([0.012, 0.018, 0.031, 0.039])
    ref_vol = np.array([0.015, 0.015, 0.025, 0.035])

    metrics = compute_volatility_metrics(y_true, y_pred, ref_volatility=ref_vol)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "mape_pct" in metrics
    assert "qlike_loss" in metrics
    assert "directional_accuracy_pct" in metrics
    assert metrics["rmse"] > 0.0
    assert 0.0 <= metrics["directional_accuracy_pct"] <= 100.0
