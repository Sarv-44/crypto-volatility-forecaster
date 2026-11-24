"""Unit tests for final academic report and figure generation."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.utils.report_generator import (
    compile_academic_report,
    generate_backtest_chart,
    generate_benchmark_chart,
    generate_feature_importance_chart,
)


def test_generate_benchmark_chart(tmp_path: Path):
    df = pd.DataFrame(
        {
            "model": ["Ridge", "CryptoGRU", "CryptoLSTM"],
            "rmse": [0.0041, 0.0046, 0.0058],
            "directional_accuracy_pct": [86.7, 88.3, 43.3],
        }
    )
    chart_path = tmp_path / "test_benchmark.png"
    out = generate_benchmark_chart(df, chart_path)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_generate_feature_importance_chart(tmp_path: Path):
    df = pd.DataFrame(
        {
            "feature": ["bb_pct_b", "volatility_24h", "rsi_14"],
            "relative_importance_pct": [60.0, 25.0, 15.0],
            "category": ["Market Technical", "Market Technical", "Market Technical"],
        }
    )
    chart_path = tmp_path / "test_feat_imp.png"
    out = generate_feature_importance_chart(df, chart_path)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_generate_backtest_chart(tmp_path: Path):
    df = pd.DataFrame(
        {
            "strategy_equity": np.linspace(10000, 11000, 20),
            "buy_and_hold_equity": np.linspace(10000, 9000, 20),
        }
    )
    chart_path = tmp_path / "test_bt.png"
    out = generate_backtest_chart(df, chart_path)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_compile_academic_report(tmp_path: Path):
    report_file = tmp_path / "FINAL_PROJECT_REPORT.md"
    out = compile_academic_report(output_file=report_file)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "Executive Summary" in content
    assert "Theoretical & Mathematical Foundations" in content
    assert "Undergraduate Team Responsibilities" in content
