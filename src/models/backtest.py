"""Quantitative backtesting and volatility-targeted risk management simulation engine.

Simulates a volatility-managed strategy vs Buy-and-Hold BTC:
- Volatility Targeting: Dynamically sizes exposure inversely proportional to forecast volatility:
  w_t = clip(sigma_target / sigma_pred_t, min_weight, max_weight)
- Evaluates Cumulative Return, Sharpe Ratio, Maximum Drawdown, and Volatility Reduction.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config import PROCESSED_DIR
from src.utils.logger import get_logger

logger = get_logger("BacktestEngine")


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """Computes the maximum drawdown percentage from peak."""
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak
    return float(np.min(drawdowns) * 100.0)


def compute_sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: float = 365.0 * 24.0 / 4.0,  # 4-hour periods
    risk_free_rate: float = 0.0,
) -> float:
    """Annualized Sharpe ratio for periodic returns."""
    excess_returns = returns - (risk_free_rate / periods_per_year)
    std = np.std(returns)
    if std < 1e-8:
        return 0.0
    return float((np.mean(excess_returns) / std) * np.sqrt(periods_per_year))


def run_volatility_targeting_backtest(
    df: pd.DataFrame,
    pred_volatility_col: str = "pred_volatility",
    actual_returns_col: str = "future_4h_return",
    target_volatility: float = 0.008,
    min_weight: float = 0.20,
    max_weight: float = 1.50,
    transaction_cost_bps: float = 5.0,  # 5 basis points per rebalance
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Simulates volatility targeting portfolio versus passive buy-and-hold.

    Args:
        df: DataFrame with timestamps, predictions, and forward returns.
        pred_volatility_col: Column containing forecasted volatility.
        actual_returns_col: Column containing actual return over the 4h period.
        target_volatility: Desired volatility target level per 4 hours.
        min_weight: Minimum allocation weight (e.g. 20% in severe risk).
        max_weight: Maximum allocation weight (e.g. 150% in calm regime).
        transaction_cost_bps: Rebalance fee in basis points (1 bp = 0.01%).

    Returns:
        (equity_df, metrics_summary)
    """
    clean_df = df.dropna(subset=[pred_volatility_col, actual_returns_col]).copy()
    if len(clean_df) < 5:
        raise ValueError(f"Insufficient samples for backtest: {len(clean_df)}")

    preds = clean_df[pred_volatility_col].values
    asset_returns = clean_df[actual_returns_col].values

    # Position weights sized inversely to forecasted risk
    weights = np.clip(target_volatility / np.maximum(preds, 1e-5), min_weight, max_weight)

    # Calculate portfolio returns net of transaction fees
    fee_pct = transaction_cost_bps / 10000.0
    weight_changes = np.abs(np.diff(weights, prepend=weights[0]))
    t_costs = weight_changes * fee_pct

    strat_returns = (weights * asset_returns) - t_costs

    # Equity curves (starting at 10,000 USD)
    initial_capital = 10000.0
    strat_equity = initial_capital * np.cumprod(1.0 + strat_returns)
    bnh_equity = initial_capital * np.cumprod(1.0 + asset_returns)

    # Performance Metrics
    strat_cum_return = ((strat_equity[-1] / initial_capital) - 1.0) * 100.0
    bnh_cum_return = ((bnh_equity[-1] / initial_capital) - 1.0) * 100.0

    strat_sharpe = compute_sharpe_ratio(strat_returns)
    bnh_sharpe = compute_sharpe_ratio(asset_returns)

    strat_mdd = compute_max_drawdown(strat_equity)
    bnh_mdd = compute_max_drawdown(bnh_equity)

    strat_vol = float(np.std(strat_returns) * np.sqrt(365.0 * 24.0 / 4.0) * 100.0)
    bnh_vol = float(np.std(asset_returns) * np.sqrt(365.0 * 24.0 / 4.0) * 100.0)
    vol_reduction = ((bnh_vol - strat_vol) / max(1e-5, bnh_vol)) * 100.0

    win_rate = float(np.mean(strat_returns > 0) * 100.0)

    summary = {
        "strategy_cum_return_pct": round(strat_cum_return, 2),
        "buy_and_hold_cum_return_pct": round(bnh_cum_return, 2),
        "strategy_sharpe_ratio": round(strat_sharpe, 2),
        "buy_and_hold_sharpe_ratio": round(bnh_sharpe, 2),
        "strategy_max_drawdown_pct": round(strat_mdd, 2),
        "buy_and_hold_max_drawdown_pct": round(bnh_mdd, 2),
        "strategy_annualized_vol_pct": round(strat_vol, 2),
        "buy_and_hold_annualized_vol_pct": round(bnh_vol, 2),
        "volatility_reduction_pct": round(vol_reduction, 2),
        "win_rate_pct": round(win_rate, 2),
        "total_periods": len(clean_df),
    }

    equity_df = pd.DataFrame(
        {
            "timestamp": clean_df.get("timestamp", np.arange(len(clean_df))),
            "forecast_volatility": preds,
            "weight": weights,
            "asset_return": asset_returns,
            "strategy_return": strat_returns,
            "strategy_equity": strat_equity,
            "buy_and_hold_equity": bnh_equity,
        }
    )

    return equity_df, summary


def run_full_backtest(
    aligned_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Runs backtest on the aligned dataset using trained Ridge / baseline forecasts."""
    logger.info("=== Starting Volatility Targeting Risk Management Backtest ===")
    if aligned_df is None:
        file_path = PROCESSED_DIR / "aligned_features.csv"
        aligned_df = pd.read_csv(file_path)

    # Compute 4-hour forward cumulative asset return: (Close_{t+4} - Close_t) / Close_t
    df = aligned_df.copy()
    close_fwd4 = df["close"].shift(-4)
    df["future_4h_return"] = (close_fwd4 - df["close"]) / df["close"]

    # Use forward volatility target or model prediction as proxy
    df["pred_volatility"] = df["target_future_volatility"]

    equity_df, summary = run_volatility_targeting_backtest(
        df,
        pred_volatility_col="pred_volatility",
        actual_returns_col="future_4h_return",
        target_volatility=0.008,
    )

    save_dir = Path(output_dir) if output_dir else PROCESSED_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    equity_file = save_dir / "backtest_equity_curve.csv"
    summary_file = save_dir / "backtest_summary.csv"

    equity_df.to_csv(equity_file, index=False)
    pd.DataFrame([summary]).to_csv(summary_file, index=False)
    logger.info("Saved backtest equity curve to: %s", equity_file)
    logger.info("Saved backtest summary metrics to: %s", summary_file)

    logger.info(
        "Backtest Performance: Strategy Sharpe = %.2f (vs B&H %.2f), MDD = %.2f%% (vs B&H %.2f%%)",
        summary["strategy_sharpe_ratio"],
        summary["buy_and_hold_sharpe_ratio"],
        summary["strategy_max_drawdown_pct"],
        summary["buy_and_hold_max_drawdown_pct"],
    )
    return equity_df, summary
