"""Unit tests for quantitative backtesting and volatility-targeted position sizing."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.backtest import (
    compute_max_drawdown,
    compute_sharpe_ratio,
    run_full_backtest,
    run_volatility_targeting_backtest,
)


def test_compute_max_drawdown():
    # Curve starts at 100, drops to 80, recovers to 110
    equity = np.array([100.0, 90.0, 80.0, 100.0, 110.0])
    mdd = compute_max_drawdown(equity)
    # Peak is 100, trough is 80 -> -20%
    assert np.isclose(mdd, -20.0)


def test_compute_sharpe_ratio():
    returns = np.array([0.01, 0.02, 0.01, 0.015, -0.005])
    sharpe = compute_sharpe_ratio(returns)
    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)

    # Constant returns should return 0
    zero_var = np.array([0.01, 0.01, 0.01])
    assert compute_sharpe_ratio(zero_var) == 0.0


def test_volatility_targeting_backtest_weights_and_equity():
    N = 30
    df = pd.DataFrame(
        {
            "pred_volatility": np.linspace(0.004, 0.020, N),
            "future_4h_return": np.random.normal(0.001, 0.01, N),
        }
    )

    equity_df, summary = run_volatility_targeting_backtest(
        df,
        target_volatility=0.008,
        min_weight=0.20,
        max_weight=1.50,
    )

    assert len(equity_df) == N
    assert (equity_df["weight"] >= 0.20).all()
    assert (equity_df["weight"] <= 1.50).all()

    # When volatility is low (0.004), weight should be higher than when vol is high (0.020)
    assert equity_df["weight"].iloc[0] > equity_df["weight"].iloc[-1]

    for key in [
        "strategy_cum_return_pct",
        "buy_and_hold_cum_return_pct",
        "strategy_sharpe_ratio",
        "strategy_max_drawdown_pct",
        "volatility_reduction_pct",
    ]:
        assert key in summary


def test_run_full_backtest_saves_csv(tmp_path: Path):
    N = 25
    dummy_df = pd.DataFrame(
        {
            "close": np.linspace(60000, 65000, N),
            "target_future_volatility": np.abs(np.random.normal(0.01, 0.002, N)),
            "timestamp": pd.date_range("2026-01-01", periods=N, freq="h"),
        }
    )

    equity_df, summary = run_full_backtest(aligned_df=dummy_df, output_dir=tmp_path)
    assert (tmp_path / "backtest_equity_curve.csv").exists()
    assert (tmp_path / "backtest_summary.csv").exists()
