"""Unit tests for hyperparameter tuning and model search."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.hyperparameter_tuning import run_grid_search
from src.models.sequence_dataset import DEFAULT_FEATURE_COLUMNS, TARGET_COLUMN


@pytest.fixture
def dummy_tuning_df():
    N = 60
    data = {TARGET_COLUMN: np.abs(np.random.normal(0.01, 0.003, N)).astype(np.float32)}
    for col in DEFAULT_FEATURE_COLUMNS:
        data[col] = np.random.normal(0, 1, N).astype(np.float32)
    return pd.DataFrame(data)


def test_run_grid_search_mini(dummy_tuning_df, tmp_path: Path):
    mini_grid = {
        "model_type": ["gru"],
        "seq_len": [6],
        "hidden_size": [16],
        "lr": [0.01],
        "dropout": [0.1],
    }

    best_params, results_df = run_grid_search(
        df=dummy_tuning_df,
        param_grid=mini_grid,
        epochs_per_trial=2,
        save_dir=tmp_path,
    )

    assert best_params is not None
    assert best_params["model_type"] == "gru"
    assert best_params["seq_len"] == 6
    assert "best_val_loss" in best_params
    assert len(results_df) == 1
    assert (tmp_path / "best_hyperparams.json").exists()
