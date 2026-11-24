"""Integration tests for FastAPI REST endpoints."""

from fastapi.testclient import TestClient
import pytest
from api.main import app

client = TestClient(app)


def test_api_status_endpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["trading_pair"] == "BTC/USDT"
    assert data["forecast_horizon"] == "4 hours"


def test_api_report_endpoint():
    response = client.get("/api/report")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "available_charts" in data
    assert "content_markdown" in data
    assert len(data["content_markdown"]) > 0


def test_api_benchmark_endpoint():
    response = client.get("/api/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "records" in data


def test_api_ablation_endpoint():
    response = client.get("/api/ablation")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ablation" in data
    assert "feature_importance" in data


def test_api_backtest_endpoint():
    response = client.get("/api/backtest")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "summary" in data


def test_api_predict_volatility():
    response = client.get("/api/predict/volatility")
    assert response.status_code == 200
    data = response.json()
    assert "forecast_next_4h_volatility" in data
    assert "volatility_regime" in data
    assert data["forecast_next_4h_volatility"] >= 0.0
