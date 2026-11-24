"""Model training engine and financial loss functions for volatility forecasting.

Implements:
- QLIKELoss: Quasi-Likelihood loss function for volatility models.
- DeepModelTrainer: Comprehensive training loop with early stopping, LR scheduling,
  gradient clipping, and checkpoint management.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import SAVED_MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger("ModelTrainer")


class QLIKELoss(nn.Module):
    """Quasi-Likelihood (QLIKE) loss function for volatility forecasting.

    Formula:
        L(y, y_hat) = (y / y_hat) - ln(y / y_hat) - 1

    Properties:
        - Strictly non-negative.
        - Asymmetric penalty: heavily penalizes under-forecasting volatility.
        - Robust standard in financial econometrics (Patton, 2011).
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Clamp inputs to avoid division by zero or log of non-positive numbers
        y_p = torch.clamp(y_pred, min=self.eps)
        y_t = torch.clamp(y_true, min=self.eps)

        ratio = y_t / y_p
        loss = ratio - torch.log(ratio) - 1.0
        return torch.mean(loss)


class DeepModelTrainer:
    """Trainer for PyTorch time-series volatility models."""

    def __init__(
        self,
        device: Optional[str] = None,
        save_dir: Optional[Path] = None,
    ):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.save_dir = Path(save_dir) if save_dir else SAVED_MODELS_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _get_loss_fn(self, loss_type: str) -> nn.Module:
        loss_type = loss_type.lower()
        if loss_type == "qlike":
            return QLIKELoss()
        elif loss_type == "mae" or loss_type == "l1":
            return nn.L1Loss()
        else:
            return nn.MSELoss()

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        clip_grad: float = 1.0,
    ) -> float:
        """Runs a single training epoch."""
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)

            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        return total_loss / max(1, n_batches)

    def evaluate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> float:
        """Evaluates model on validation DataLoader."""
        model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                preds = model(x_batch)
                loss = loss_fn(preds, y_batch)
                total_loss += float(loss.item())
                n_batches += 1

        return total_loss / max(1, n_batches)

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 7,
        loss_type: str = "mse",
        model_name: str = "crypto_lstm",
    ) -> Dict[str, any]:
        """Full training loop with early stopping and checkpointing.

        Args:
            model: PyTorch nn.Module instance.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs: Max training epochs.
            lr: Initial learning rate.
            weight_decay: L2 regularization penalty.
            patience: Early stopping patience epochs.
            loss_type: 'mse', 'qlike', or 'mae'.
            model_name: Checkpoint identifier.

        Returns:
            Dict containing training history and best metrics.
        """
        model.to(self.device)
        loss_fn = self._get_loss_fn(loss_type)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
        )

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        best_weights_path = self.save_dir / f"best_{model_name}.pt"

        logger.info(
            "Starting training for %s (%d epochs, lr=%.4f, loss=%s, device=%s)",
            model_name,
            epochs,
            lr,
            loss_type,
            self.device,
        )

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(model, train_loader, optimizer, loss_fn)
            val_loss = self.evaluate(model, val_loader, loss_fn)
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            logger.info(
                "Epoch %02d/%02d - Train Loss: %.6f | Val Loss: %.6f",
                epoch,
                epochs,
                train_loss,
                val_loss,
            )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_loss": val_loss,
                        "model_name": model_name,
                        "loss_type": loss_type,
                    },
                    best_weights_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered at epoch %d. Best epoch: %d", epoch, best_epoch)
                    break

        # Load best weights back into model
        if best_weights_path.exists():
            checkpoint = torch.load(best_weights_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Restored best model weights from epoch %d (Val Loss: %.6f)", best_epoch, best_val_loss)

        return {
            "model_name": model_name,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "checkpoint_path": str(best_weights_path),
            "history": history,
        }
