"""Sliding window sequence dataset generator for PyTorch deep models.

Creates temporal sequence windows of length L from multimodal features,
applies chronological train/val/test splits, and normalizes features with zero leakage.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.config import PROCESSED_DIR
from src.utils.logger import get_logger

logger = get_logger("SequenceDataset")

DEFAULT_FEATURE_COLUMNS = [
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

TARGET_COLUMN = "target_future_volatility"


class CryptoVolatilityDataset(Dataset):
    """PyTorch Dataset for cryptocurrency volatility sequences."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """Args:
            sequences: Array of shape (N, seq_len, num_features).
            targets: Array of shape (N,) or (N, 1).
        """
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


def create_sliding_windows(
    data_matrix: np.ndarray,
    target_vector: np.ndarray,
    seq_len: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates sliding sequence windows of length seq_len.

    For each index i (from seq_len to len), sequence is data[i-seq_len : i]
    and target is target_vector[i - 1].

    Args:
        data_matrix: 2D array of features (T, num_features).
        target_vector: 1D array of targets (T,).
        seq_len: Lookback window length (in hours).

    Returns:
        (sequences, targets) where sequences has shape (N, seq_len, D)
        and targets has shape (N,).
    """
    num_samples = len(data_matrix) - seq_len + 1
    if num_samples <= 0:
        raise ValueError(
            f"Data length ({len(data_matrix)}) is shorter than sequence length ({seq_len})."
        )

    sequences = []
    targets = []

    for i in range(num_samples):
        # Window of past seq_len hours
        window = data_matrix[i : i + seq_len]
        # Target associated with the last hour of this window
        target = target_vector[i + seq_len - 1]
        sequences.append(window)
        targets.append(target)

    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)


def prepare_sequence_data(
    df: Optional[pd.DataFrame] = None,
    csv_path: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    target_col: str = TARGET_COLUMN,
    seq_len: int = 12,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    StandardScaler,
    List[str],
]:
    """Prepares chronological Train / Val / Test sequence splits with zero lookahead bias.

    Steps:
    1. Chronological splitting before windowing or scaling.
    2. Fit StandardScaler strictly on the training partition.
    3. Transform train, validation, and test partitions with the fitted scaler.
    4. Generate sliding windows for each partition.

    Args:
        df: Input DataFrame containing aligned features.
        csv_path: Optional CSV file path if df is not provided.
        feature_cols: List of feature column names.
        target_col: Target column name.
        seq_len: Sequence lookback length (hours).
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.

    Returns:
        (train_data, val_data, test_data, scaler, feature_cols)
        where each data tuple is (X_sequences, y_targets).
    """
    if df is None:
        file_path = csv_path or (PROCESSED_DIR / "aligned_features.csv")
        df = pd.read_csv(file_path)

    cols = feature_cols or DEFAULT_FEATURE_COLUMNS
    missing = [c for c in cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    clean_df = df.dropna(subset=cols + [target_col]).reset_index(drop=True)
    total_len = len(clean_df)
    if total_len < seq_len * 2:
        raise ValueError(f"Dataset has {total_len} rows, need at least {seq_len * 2}.")

    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))

    train_df = clean_df.iloc[:train_end]
    val_df = clean_df.iloc[train_end:val_end]
    test_df = clean_df.iloc[val_end:]

    # Fit scaler strictly on train set
    scaler = StandardScaler()
    train_feat_scaled = scaler.fit_transform(train_df[cols].values)
    val_feat_scaled = scaler.transform(val_df[cols].values)
    test_feat_scaled = scaler.transform(test_df[cols].values)

    train_targets = train_df[target_col].values
    val_targets = val_df[target_col].values
    test_targets = test_df[target_col].values

    # Create sliding sequence windows
    X_train, y_train = create_sliding_windows(train_feat_scaled, train_targets, seq_len=seq_len)
    X_val, y_val = create_sliding_windows(val_feat_scaled, val_targets, seq_len=seq_len)
    X_test, y_test = create_sliding_windows(test_feat_scaled, test_targets, seq_len=seq_len)

    logger.info(
        "Sequence data prepared (seq_len=%d). Train: %d, Val: %d, Test: %d samples. Features: %d",
        seq_len,
        len(X_train),
        len(X_val),
        len(X_test),
        len(cols),
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, cols


def get_dataloaders(
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    batch_size: int = 32,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Wraps sequence numpy arrays into PyTorch DataLoaders."""
    train_ds = CryptoVolatilityDataset(train_data[0], train_data[1])
    val_ds = CryptoVolatilityDataset(val_data[0], val_data[1])
    test_ds = CryptoVolatilityDataset(test_data[0], test_data[1])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
