"""Hyperparameter optimization and model search module for deep volatility forecasters.

Performs grid search over sequence length, recurrent unit types, hidden sizes,
dropout rates, and learning rates to identify optimal validation performance.
"""

import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

from src.models.deep_models import CryptoGRU, CryptoLSTM
from src.models.sequence_dataset import get_dataloaders, prepare_sequence_data
from src.models.trainer import DeepModelTrainer
from src.utils.config import PROCESSED_DIR, SAVED_MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger("HyperparameterTuning")


def run_grid_search(
    df: Optional[pd.DataFrame] = None,
    csv_path: Optional[Path] = None,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    epochs_per_trial: int = 8,
    save_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluates hyperparameter configurations and selects the best model.

    Args:
        df: Input aligned features DataFrame.
        csv_path: Optional CSV path if df is not provided.
        param_grid: Dictionary of parameter lists to search.
        epochs_per_trial: Max epochs per candidate trial.
        save_dir: Directory to save results.

    Returns:
        (best_params, trial_results_df)
    """
    logger.info("=== Starting Hyperparameter Grid Search ===")
    if df is None:
        file_path = csv_path or (PROCESSED_DIR / "aligned_features.csv")
        df = pd.read_csv(file_path)

    default_grid = {
        "model_type": ["gru", "lstm"],
        "seq_len": [6, 12],
        "hidden_size": [32, 64],
        "lr": [0.002, 0.001],
        "dropout": [0.2],
    }
    grid = param_grid or default_grid

    # Generate all candidate combinations
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    logger.info("Total hyperparameter trials to evaluate: %d", len(combinations))

    trainer = DeepModelTrainer()
    results: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_params: Optional[Dict[str, Any]] = None

    for idx, params in enumerate(combinations, 1):
        seq_len = params["seq_len"]
        model_type = params["model_type"]
        hidden_size = params["hidden_size"]
        lr = params["lr"]
        dropout = params["dropout"]

        try:
            train_data, val_data, test_data, _, feature_cols = prepare_sequence_data(
                df=df, seq_len=seq_len, train_ratio=0.70, val_ratio=0.15
            )
            train_loader, val_loader, _ = get_dataloaders(train_data, val_data, test_data, batch_size=32)

            in_dim = len(feature_cols)
            if model_type == "gru":
                model = CryptoGRU(input_size=in_dim, hidden_size=hidden_size, dropout=dropout)
            else:
                model = CryptoLSTM(input_size=in_dim, hidden_size=hidden_size, dropout=dropout)

            trial_name = f"trial_{idx}_{model_type}_s{seq_len}_h{hidden_size}"
            fit_res = trainer.fit(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs_per_trial,
                lr=lr,
                patience=4,
                model_name=trial_name,
            )

            val_loss = fit_res["best_val_loss"]
            record = {
                "trial": idx,
                "model_type": model_type,
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "lr": lr,
                "dropout": dropout,
                "best_epoch": fit_res["best_epoch"],
                "best_val_loss": round(val_loss, 6),
            }
            results.append(record)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = record
                logger.info("New best configuration found: %s", best_params)

        except Exception as e:
            logger.warning("Trial %d failed with error: %s", idx, str(e))

    results_df = pd.DataFrame(results).sort_values("best_val_loss").reset_index(drop=True)

    # Save results and best config
    out_dir = Path(save_dir) if save_dir else SAVED_MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    best_config_file = out_dir / "best_hyperparams.json"
    tuning_log_file = PROCESSED_DIR / "hyperparameter_tuning_results.csv"

    if best_params:
        with open(best_config_file, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)
        logger.info("Saved best hyperparameters to: %s", best_config_file)

    results_df.to_csv(tuning_log_file, index=False)
    logger.info("Saved hyperparameter tuning results table to: %s", tuning_log_file)

    return best_params, results_df
