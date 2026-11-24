"""Quantitative evaluation and benchmarking suite for crypto volatility models.

Calculates standard econometric and machine learning metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- QLIKE (Quasi-Likelihood Loss)
- Directional Accuracy (Expansion vs. Compression)
- Pearson & Spearman Rank Correlation
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.utils.config import PROCESSED_DIR, SAVED_MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger("Evaluation")


def compute_qlike_numpy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Computes Quasi-Likelihood loss for volatility series using NumPy."""
    y_t = np.clip(y_true, a_min=eps, a_max=None)
    y_p = np.clip(y_pred, a_min=eps, a_max=None)
    ratio = y_t / y_p
    qlike = ratio - np.log(ratio) - 1.0
    return float(np.mean(qlike))


def compute_directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ref_volatility: np.ndarray,
) -> float:
    """Calculates percentage of correct directional calls (volatility expansion vs contraction).

    Args:
        y_true: Actual future realized volatility.
        y_pred: Forecasted future realized volatility.
        ref_volatility: Current baseline volatility (e.g. 24h historical vol).

    Returns:
        Directional accuracy as a float percentage (0.0 to 100.0).
    """
    actual_direction = (y_true - ref_volatility) > 0
    pred_direction = (y_pred - ref_volatility) > 0
    correct = np.sum(actual_direction == pred_direction)
    return float(round((correct / max(1, len(y_true))) * 100.0, 2))


def compute_volatility_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ref_volatility: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculates full battery of quantitative volatility metrics.

    Args:
        y_true: True realized volatility values.
        y_pred: Predicted volatility values.
        ref_volatility: Optional baseline volatility for directional accuracy.

    Returns:
        Dict containing RMSE, MAE, MAPE, QLIKE, Pearson correlation, and Directional Accuracy.
    """
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.asarray(y_pred, dtype=np.float64).flatten()

    mse = mean_squared_error(y_t, y_p)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_t, y_p))

    # MAPE with epsilon protection
    eps = 1e-6
    mape = float(np.mean(np.abs((y_t - y_p) / (y_t + eps))) * 100.0)

    # QLIKE
    qlike = compute_qlike_numpy(y_t, y_p, eps=eps)

    # Correlations
    if len(y_t) > 2 and np.std(y_p) > 1e-9:
        p_corr, _ = pearsonr(y_t, y_p)
        s_corr, _ = spearmanr(y_t, y_p)
    else:
        p_corr, s_corr = 0.0, 0.0

    metrics = {
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "mape_pct": round(mape, 2),
        "qlike_loss": round(qlike, 6),
        "pearson_corr": round(float(p_corr), 4),
        "spearman_corr": round(float(s_corr), 4),
    }

    if ref_volatility is not None:
        ref_v = np.asarray(ref_volatility, dtype=np.float64).flatten()
        da = compute_directional_accuracy(y_t, y_p, ref_v)
        metrics["directional_accuracy_pct"] = da

    return metrics


def benchmark_models(
    test_actuals: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    ref_volatility: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
    save_filename: str = "model_benchmark_results.csv",
) -> pd.DataFrame:
    """Evaluates multiple models side-by-side and returns a formatted benchmark table.

    Args:
        test_actuals: Ground truth test targets.
        predictions_dict: Mapping of model_name -> predicted targets array.
        ref_volatility: Reference volatility for directional accuracy.
        output_dir: Directory to save benchmark CSV.
        save_filename: Output filename.

    Returns:
        pd.DataFrame ranking models across all metrics.
    """
    rows = []
    for model_name, preds in predictions_dict.items():
        m = compute_volatility_metrics(test_actuals, preds, ref_volatility=ref_volatility)
        m["model"] = model_name
        rows.append(m)

    df_results = pd.DataFrame(rows)
    # Order columns
    cols = ["model", "rmse", "mae", "mape_pct", "qlike_loss", "pearson_corr"]
    if "directional_accuracy_pct" in df_results.columns:
        cols.append("directional_accuracy_pct")
    df_results = df_results[cols].sort_values("rmse").reset_index(drop=True)

    save_path = Path(output_dir) if output_dir else PROCESSED_DIR
    save_path.mkdir(parents=True, exist_ok=True)
    out_file = save_path / save_filename
    df_results.to_csv(out_file, index=False)
    logger.info("Saved benchmark evaluation results to: %s", out_file)

    return df_results
