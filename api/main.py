"""FastAPI web server and interactive dashboard for Crypto Volatility Forecaster.

Provides RESTful endpoints and serves a local interactive UI on http://localhost:8000.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from src.data.market_data import fetch_market_data
from src.features.feature_fusion import fuse_market_and_sentiment_features
from src.features.market_features import compute_market_features
from src.models.baseline import CryptoVolatilityForecaster
from src.nlp.finbert_sentiment import FinBERTSentimentAnalyzer
from src.nlp.sentiment_aggregation import aggregate_hourly_sentiment, generate_sample_social_posts
from src.utils.config import PROCESSED_DIR, RAW_MARKET_DIR, SAVED_MODELS_DIR, PROJECT_ROOT
from src.utils.logger import get_logger

import joblib
import torch
from src.models.deep_models import CryptoGRU, CryptoLSTM

logger = get_logger("APIServer")

# Global instances loaded on startup
sentiment_analyzer: Optional[FinBERTSentimentAnalyzer] = None
forecaster_model: Optional[CryptoVolatilityForecaster] = None
lstm_model: Optional[CryptoLSTM] = None
gru_model: Optional[CryptoGRU] = None
seq_scaler_meta: Optional[Dict] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes models and resources on server startup."""
    global sentiment_analyzer, forecaster_model, lstm_model, gru_model, seq_scaler_meta
    logger.info("Initializing FinBERT analyzer and forecasting models...")
    try:
        sentiment_analyzer = FinBERTSentimentAnalyzer(enable_cache=True)
    except Exception as e:
        logger.warning("Could not initialize FinBERT analyzer: %s", str(e))

    # Load Ridge baseline
    model_path = SAVED_MODELS_DIR / "baseline_ridge.joblib"
    if model_path.exists():
        try:
            forecaster_model = CryptoVolatilityForecaster.load(model_path)
            logger.info("Forecaster Ridge model loaded successfully.")
        except Exception as e:
            logger.warning("Failed to load baseline model: %s", str(e))

    # Load sequence scaler
    scaler_path = SAVED_MODELS_DIR / "sequence_scaler.joblib"
    if scaler_path.exists():
        try:
            seq_scaler_meta = joblib.load(scaler_path)
            in_dim = len(seq_scaler_meta["feature_cols"])

            # Load LSTM
            lstm_path = SAVED_MODELS_DIR / "best_crypto_lstm.pt"
            if lstm_path.exists():
                lstm_model = CryptoLSTM(input_size=in_dim, hidden_size=64, num_layers=2)
                ckpt = torch.load(lstm_path, map_location="cpu")
                lstm_model.load_state_dict(ckpt["model_state_dict"])
                lstm_model.eval()
                logger.info("Loaded PyTorch CryptoLSTM model.")

            # Load GRU
            gru_path = SAVED_MODELS_DIR / "best_crypto_gru.pt"
            if gru_path.exists():
                gru_model = CryptoGRU(input_size=in_dim, hidden_size=64, num_layers=2)
                ckpt = torch.load(gru_path, map_location="cpu")
                gru_model.load_state_dict(ckpt["model_state_dict"])
                gru_model.eval()
                logger.info("Loaded PyTorch CryptoGRU model.")
        except Exception as e:
            logger.warning("Error loading deep models: %s", str(e))

    yield


app = FastAPI(
    title="Algorithmic Crypto Volatility Forecaster API",
    description="Combines OHLCV market indicators with FinBERT NLP sentiment to forecast short-term BTC/USDT volatility.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HeadlineRequest(BaseModel):
    text: str


class PipelineRunRequest(BaseModel):
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    limit: int = 100


@app.get("/api/status")
def get_system_status() -> Dict:
    """Returns system status, active models, and dataset shapes."""
    aligned_file = PROCESSED_DIR / "aligned_features.csv"
    aligned_rows = 0
    if aligned_file.exists():
        try:
            aligned_rows = len(pd.read_csv(aligned_file))
        except Exception:
            pass

    return {
        "status": "online",
        "model_loaded": forecaster_model is not None and forecaster_model.is_trained,
        "finbert_active": sentiment_analyzer is not None,
        "aligned_dataset_rows": aligned_rows,
        "exchange": "Binance CCXT",
        "trading_pair": "BTC/USDT",
        "timeframe": "1h",
        "forecast_horizon": "4 hours",
    }


@app.get("/api/market/latest")
def get_latest_market_data(limit: int = 48) -> Dict:
    """Returns recent market candles and technical features."""
    market_file = PROCESSED_DIR / "market_features.csv"
    if not market_file.exists():
        market_file = RAW_MARKET_DIR / "btc_usdt_1h.csv"
        if not market_file.exists():
            raise HTTPException(status_code=404, detail="Market data not found. Run pipeline first.")

    df = pd.read_csv(market_file)
    recent = df.tail(limit).to_dict(orient="records")
    return {"count": len(recent), "records": recent}


@app.post("/api/sentiment/analyze")
def analyze_headline(req: HeadlineRequest) -> Dict:
    """Performs real-time FinBERT sentiment analysis on input text."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    global sentiment_analyzer
    if sentiment_analyzer is None:
        sentiment_analyzer = FinBERTSentimentAnalyzer()

    result = sentiment_analyzer.predict_single(req.text)
    return {
        "text": req.text,
        "probabilities": {
            "positive": result["positive"],
            "negative": result["negative"],
            "neutral": result["neutral"],
        },
        "sentiment_score": result["sentiment_score"],
        "label": (
            "Bullish (Positive)"
            if result["sentiment_score"] > 0.15
            else "Bearish (Negative)"
            if result["sentiment_score"] < -0.15
            else "Neutral"
        ),
    }


@app.get("/api/predict/volatility")
def predict_current_volatility(model: str = "gru") -> Dict:
    """Forecasts next-4-hour realized volatility using latest aligned features."""
    global forecaster_model
    aligned_file = PROCESSED_DIR / "aligned_features.csv"
    if not aligned_file.exists():
        raise HTTPException(status_code=404, detail="Aligned feature dataset not found.")

    df = pd.read_csv(aligned_file)
    if df.empty:
        raise HTTPException(status_code=400, detail="Aligned feature dataset is empty.")

    latest_row = df.iloc[-1]
    hist_vol = float(latest_row.get("volatility_24h", 0.0))
    current_price = float(latest_row.get("close", 0.0))

    # Determine which model to use
    selected_model_name = model.lower()
    forecast_val = None

    if selected_model_name in ("gru", "lstm") and seq_scaler_meta is not None:
        seq_len = seq_scaler_meta.get("seq_len", 12)
        feature_cols = seq_scaler_meta.get("feature_cols", [])
        if len(df) >= seq_len:
            seq_window = df[feature_cols].tail(seq_len).values
            scaler = seq_scaler_meta["scaler"]
            seq_scaled = scaler.transform(seq_window)
            tensor_in = torch.tensor(seq_scaled.reshape(1, seq_len, -1), dtype=torch.float32)

            with torch.no_grad():
                if selected_model_name == "gru" and gru_model is not None:
                    forecast_val = float(gru_model(tensor_in).cpu().numpy().flatten()[0])
                    active_model = "CryptoGRU (Deep Learning)"
                elif selected_model_name == "lstm" and lstm_model is not None:
                    forecast_val = float(lstm_model(tensor_in).cpu().numpy().flatten()[0])
                    active_model = "CryptoLSTM (Deep Learning)"

    if forecast_val is None:
        # Fallback to Ridge or latest volatility
        feature_cols = (
            forecaster_model.feature_cols
            if (forecaster_model and forecaster_model.is_trained)
            else seq_scaler_meta.get("feature_cols", []) if seq_scaler_meta else []
        )
        if forecaster_model and forecaster_model.is_trained and feature_cols:
            feat_vals = np.array([float(latest_row[col]) for col in feature_cols]).reshape(1, -1)
            forecast_val = forecaster_model.predict(feat_vals)
            active_model = "Ridge Regressor (Baseline)"
        else:
            forecast_val = hist_vol
            active_model = "Historical Rolling Persistence"

    forecast_val = max(0.0, forecast_val)

    return {
        "timestamp": str(latest_row.get("timestamp", "")),
        "model_used": active_model,
        "current_price": current_price,
        "historical_24h_volatility": round(hist_vol, 6),
        "forecast_next_4h_volatility": round(forecast_val, 6),
        "volatility_regime": (
            "High Volatility"
            if forecast_val > 0.015
            else "Moderate Volatility"
            if forecast_val > 0.006
            else "Low / Compression"
        ),
        "latest_sentiment_score": float(latest_row.get("avg_sentiment", 0.0)),
        "rsi_14": float(latest_row.get("rsi_14", 50.0)),
    }


@app.get("/api/benchmark")
def get_model_benchmarks() -> Dict:
    """Returns test set evaluation metrics across all trained models."""
    bench_file = PROCESSED_DIR / "model_benchmark_results.csv"
    if not bench_file.exists():
        return {"status": "pending", "records": []}
    df = pd.read_csv(bench_file)
    return {"status": "available", "records": df.to_dict(orient="records")}


@app.get("/api/ablation")
def get_ablation_results() -> Dict:
    """Returns multimodal ablation study results and feature importance ranking."""
    abl_file = PROCESSED_DIR / "ablation_study_results.csv"
    imp_file = PROCESSED_DIR / "feature_importance.csv"
    abl_records = pd.read_csv(abl_file).to_dict(orient="records") if abl_file.exists() else []
    imp_records = pd.read_csv(imp_file).to_dict(orient="records") if imp_file.exists() else []
    return {"status": "available", "ablation": abl_records, "feature_importance": imp_records}


@app.get("/api/backtest")
def get_backtest_results() -> Dict:
    """Returns volatility targeting portfolio backtest metrics and equity curve."""
    sum_file = PROCESSED_DIR / "backtest_summary.csv"
    eq_file = PROCESSED_DIR / "backtest_equity_curve.csv"
    summary = pd.read_csv(sum_file).iloc[0].to_dict() if sum_file.exists() else {}
    equity = pd.read_csv(eq_file).tail(30).to_dict(orient="records") if eq_file.exists() else []
    return {"status": "available", "summary": summary, "equity_curve": equity}


@app.get("/api/report")
def get_academic_report() -> Dict:
    """Returns the compiled final academic capstone project report and figure metadata."""
    report_file = PROJECT_ROOT / "FINAL_PROJECT_REPORT.md"
    if not report_file.exists():
        from src.utils.report_generator import compile_academic_report
        compile_academic_report()

    report_text = report_file.read_text(encoding="utf-8") if report_file.exists() else ""
    reports_dir = PROCESSED_DIR / "reports"
    available_charts = [f.name for f in reports_dir.glob("*.png")] if reports_dir.exists() else []

    return {
        "status": "available" if report_file.exists() else "not_generated",
        "report_path": str(report_file),
        "available_charts": available_charts,
        "content_markdown": report_text,
    }


@app.get("/api/report/charts/{chart_name}")
def get_report_chart(chart_name: str):
    """Serves high-resolution figures generated for the academic report."""
    reports_dir = PROCESSED_DIR / "reports"
    chart_path = reports_dir / chart_name
    if not chart_path.exists() or not chart_name.endswith(".png"):
        raise HTTPException(status_code=404, detail="Chart not found.")
    return FileResponse(chart_path, media_type="image/png")


@app.post("/api/pipeline/run")
def trigger_pipeline_refresh(req: PipelineRunRequest) -> Dict:
    """Refreshes market data, FinBERT sentiment, and feature fusion."""
    try:
        raw_df = fetch_market_data(symbol=req.symbol, timeframe=req.timeframe, limit=req.limit)
        features_df = compute_market_features(raw_df)

        global sentiment_analyzer
        if sentiment_analyzer is None:
            sentiment_analyzer = FinBERTSentimentAnalyzer()

        posts_df = generate_sample_social_posts()
        scored_posts = sentiment_analyzer.analyze_dataframe(posts_df, text_col="text")
        hourly_sentiment = aggregate_hourly_sentiment(scored_posts)

        aligned_df = fuse_market_and_sentiment_features(
            df_market=features_df,
            df_sentiment=hourly_sentiment,
        )

        global forecaster_model
        if forecaster_model is None or not forecaster_model.is_trained:
            from src.models.baseline import train_and_save_baseline

            forecaster_model, _ = train_and_save_baseline()

        return {
            "success": True,
            "candles_fetched": len(raw_df),
            "market_features_rows": len(features_df),
            "sentiment_periods": len(hourly_sentiment),
            "aligned_features_shape": list(aligned_df.shape),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serves the interactive single-page dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantitative Cryptocurrency Volatility Forecaster | Terminal</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --bg: #090b10;
            --surface: #10131a;
            --surface-subtle: #141822;
            --surface-inset: #0d1017;
            --surface-hover: #181d2a;
            --border: #1d2330;
            --border-subtle: #161a24;
            --border-active: #2f384d;
            --border-focus: #3b82f6;
            --text-primary: #f0f3f8;
            --text-secondary: #86909f;
            --text-tertiary: #525c6c;
            --accent: #2563eb;
            --accent-hover: #1d4ed8;
            --accent-subtle: rgba(37, 99, 235, 0.12);
            --positive: #10b981;
            --negative: #f43f5e;
            --warning: #f59e0b;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg);
            color: var(--text-primary);
            background-image: 
                linear-gradient(to right, rgba(255, 255, 255, 0.018) 1px, transparent 1px),
                linear-gradient(to bottom, rgba(255, 255, 255, 0.018) 1px, transparent 1px);
            background-size: 32px 32px;
            letter-spacing: -0.011em;
            -webkit-font-smoothing: antialiased;
        }

        .font-mono {
            font-family: 'JetBrains Mono', Consolas, monospace;
            font-feature-settings: "tnum" 1;
        }

        .tabular-nums {
            font-variant-numeric: tabular-nums;
        }

        /* Subtle hairline panels */
        .panel {
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-radius: 4px;
        }

        .panel-inset {
            background-color: var(--surface-inset);
            border: 1px solid var(--border-subtle);
            border-radius: 3px;
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 5px;
            height: 5px;
        }
        ::-webkit-scrollbar-track {
            background: var(--bg);
        }
        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 2px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--border-active);
        }

        /* Input / select focus rings */
        input:focus, select:focus, textarea:focus {
            outline: 2px solid var(--border-focus);
            outline-offset: -1px;
            border-color: var(--border-focus) !important;
        }

        /* Micro transitions */
        .transition-fast {
            transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
        }
    </style>
</head>
<body class="min-h-screen flex flex-col">

    <!-- Top Institutional Ticker & Navigation Bar -->
    <header class="border-b border-[#1d2330] bg-[#0c0f16]/95 backdrop-blur-sm sticky top-0 z-50 px-5 py-2.5 flex items-center justify-between">
        <div class="flex items-center space-x-4">
            <div class="flex items-center gap-2.5">
                <div class="h-6 px-1.5 rounded-[2px] bg-blue-950 border border-blue-800/80 flex items-center justify-center text-blue-400 font-mono text-[10px] font-bold tracking-widest">
                    QVF
                </div>
                <div class="flex items-baseline gap-2">
                    <span class="text-xs font-semibold tracking-tight text-white uppercase">Crypto Volatility Forecaster</span>
                    <span class="text-[10px] text-[#86909f] hidden sm:inline">| Multimodal Econometrics & NLP</span>
                </div>
            </div>

            <div class="hidden lg:flex items-center gap-1.5 pl-3 border-l border-[#1d2330] text-[11px] font-mono">
                <span class="text-[#86909f]">PAIR:</span>
                <span class="text-[#f0f3f8] font-semibold">BTC/USDT</span>
                <span class="text-[#525c6c]">•</span>
                <span class="text-[#86909f]">BAR:</span>
                <span class="text-[#f0f3f8]">1-HOUR</span>
                <span class="text-[#525c6c]">•</span>
                <span class="text-[#86909f]">HORIZON:</span>
                <span class="text-blue-400 font-semibold">FORWARD 4H</span>
            </div>
        </div>

        <div class="flex items-center space-x-2.5">
            <div class="flex items-center gap-1.5 px-2 py-1 rounded-[2px] bg-[#141822] border border-[#1d2330] text-[10px] font-mono text-[#86909f]">
                <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block"></span>
                <span class="hidden md:inline">BINANCE CCXT:</span>
                <span class="text-emerald-400 font-medium">LIVE</span>
            </div>

            <button onclick="refreshData()" class="px-2.5 py-1 text-xs font-medium rounded-[3px] bg-[#141822] hover:bg-[#1b212e] text-[#f0f3f8] border border-[#1d2330] hover:border-[#2f384d] transition-fast flex items-center gap-1.5">
                <svg class="w-3 h-3 text-[#86909f]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
                <span>Refresh</span>
            </button>

            <a href="/docs" target="_blank" class="px-2.5 py-1 text-xs font-medium rounded-[3px] bg-blue-600 hover:bg-blue-500 text-white transition-fast flex items-center gap-1">
                <span>API Docs</span>
                <span class="text-blue-200 text-[10px]">&nearr;</span>
            </a>
        </div>
    </header>

    <!-- Main Workspace -->
    <main class="flex-1 p-5 max-w-[1400px] mx-auto w-full space-y-5">

        <!-- Top Asymmetric Metric Strip: Historical Context + Target Volatility Focal Hero -->
        <div class="grid grid-cols-1 md:grid-cols-12 gap-3.5">
            
            <!-- 1. Spot Price (Col 1-3) -->
            <div class="md:col-span-3 panel p-3.5 flex flex-col justify-between">
                <div>
                    <div class="flex items-center justify-between text-[11px] font-medium text-[#86909f] uppercase tracking-wider">
                        <span>BTC/USDT Spot</span>
                        <span class="text-[10px] font-mono text-[#525c6c]">T-0</span>
                    </div>
                    <div class="text-2xl font-bold font-mono text-white mt-1 tabular-nums" id="stat-price">Loading...</div>
                </div>
                <div class="text-[11px] text-[#525c6c] mt-2 flex items-center gap-1 font-mono">
                    <span>FEED:</span>
                    <span class="text-[#86909f]">CONTINUOUS 1H OHLCV</span>
                </div>
            </div>

            <!-- 2. Historical Realized Volatility (Col 4-6) -->
            <div class="md:col-span-3 panel p-3.5 flex flex-col justify-between">
                <div>
                    <div class="flex items-center justify-between text-[11px] font-medium text-[#86909f] uppercase tracking-wider">
                        <span>Historical Volatility (24H)</span>
                        <span class="text-[10px] font-mono text-[#525c6c]">&sigma;<sub>24h</sub></span>
                    </div>
                    <div class="text-2xl font-bold font-mono text-amber-300 mt-1 tabular-nums" id="stat-hist-vol">Loading...</div>
                </div>
                <div class="text-[11px] text-[#525c6c] mt-2 font-mono">
                    Sample standard deviation of log returns
                </div>
            </div>

            <!-- 3. Primary Forecast Hero (Col 7-9: FOCAL POINT) -->
            <div class="md:col-span-4 panel p-3.5 border-l-2 border-l-blue-500 bg-gradient-to-r from-blue-950/20 to-transparent flex flex-col justify-between">
                <div>
                    <div class="flex items-center justify-between text-[11px] font-medium uppercase tracking-wider">
                        <span class="text-blue-400 font-semibold">Forecast Volatility (Next 4H)</span>
                        <select id="modelSelect" onchange="fetchVolatilityMetrics()" class="bg-[#0d1017] border border-[#1d2330] text-[10px] text-[#f0f3f8] rounded-[2px] px-1.5 py-0.5 focus:outline-none font-mono">
                            <option value="gru" selected>CryptoGRU (Deep)</option>
                            <option value="lstm">CryptoLSTM (Deep)</option>
                            <option value="ridge">Ridge (Baseline)</option>
                        </select>
                    </div>
                    <div class="text-2xl font-bold font-mono text-blue-300 mt-1 tabular-nums" id="stat-forecast-vol">Loading...</div>
                </div>
                <div class="text-[11px] mt-2 text-[#86909f] flex items-center justify-between font-mono">
                    <span id="stat-regime" class="font-medium text-[#f0f3f8]">Regime: Evaluating...</span>
                    <span id="stat-model-badge" class="text-[10px] text-blue-400"></span>
                </div>
            </div>

            <!-- 4. FinBERT Sentiment Composite (Col 10-12) -->
            <div class="md:col-span-2 panel p-3.5 flex flex-col justify-between">
                <div>
                    <div class="flex items-center justify-between text-[11px] font-medium text-[#86909f] uppercase tracking-wider">
                        <span>FinBERT Sentiment</span>
                        <span class="text-[10px] font-mono text-[#525c6c]">NLP</span>
                    </div>
                    <div class="text-2xl font-bold font-mono mt-1 tabular-nums text-white" id="stat-sentiment">Loading...</div>
                </div>
                <div class="text-[10px] text-[#86909f] mt-2 font-mono truncate" id="stat-sentiment-sub">
                    P(+) - P(-) Spread
                </div>
            </div>

        </div>

        <!-- Analytical Workbench: 65% Chart & Indicator Canvas + 35% Live FinBERT Inference -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-3.5">
            
            <!-- Left Stage: Hourly Time-Series Canvas (Col 1-8) -->
            <div class="lg:col-span-8 panel p-4 space-y-3">
                <div class="flex items-center justify-between">
                    <div>
                        <h2 class="text-xs font-semibold text-white uppercase tracking-wider">Price History & Bollinger Volatility Bands</h2>
                        <p class="text-[11px] text-[#86909f]">20-period simple moving average with &plusmn;2&sigma; volatility envelope boundaries</p>
                    </div>
                    <span class="text-[11px] font-mono text-[#86909f] bg-[#141822] px-2 py-0.5 rounded-[2px] border border-[#1d2330]" id="chart-timestamp-range">-</span>
                </div>
                <div class="h-64 w-full">
                    <canvas id="marketChart"></canvas>
                </div>
            </div>

            <!-- Right Stage: Live FinBERT Inference Sandbox (Col 9-12) -->
            <div class="lg:col-span-4 panel p-4 flex flex-col justify-between space-y-3">
                <div>
                    <div class="flex items-center justify-between">
                        <h2 class="text-xs font-semibold text-white uppercase tracking-wider">Headline Sentiment Scorer</h2>
                        <span class="text-[10px] font-mono text-blue-400 bg-blue-950/60 border border-blue-900 px-1.5 py-0.2 rounded-[2px]">ProsusAI/finbert</span>
                    </div>
                    <p class="text-[11px] text-[#86909f] mt-0.5">Evaluates probability distribution using pretrained financial BERT transformer.</p>
                    
                    <div class="mt-2.5">
                        <textarea id="headlineInput" rows="3" class="w-full panel-inset p-2 text-xs text-[#f0f3f8] placeholder-[#525c6c] focus:outline-none font-sans" placeholder="e.g. Bitcoin holds support as institutional spot exchange volumes reach weekly peak..."></textarea>
                    </div>

                    <button onclick="analyzeCustomHeadline()" id="analyzeBtn" class="mt-2 w-full py-1.5 px-3 bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white rounded-[3px] text-xs font-medium transition-fast flex items-center justify-center gap-1.5">
                        <span>Evaluate Sentiment</span>
                    </button>
                </div>

                <!-- Structured Probability Breakdown Panel -->
                <div id="sentimentResult" class="panel-inset p-2.5 text-xs space-y-2">
                    <div class="flex justify-between items-center text-[11px]">
                        <span class="text-[#86909f]">Classification:</span>
                        <span id="resLabel" class="font-mono font-medium text-white">Ready</span>
                    </div>
                    <div class="flex justify-between items-center text-[11px]">
                        <span class="text-[#86909f]">Continuous Score [-1, +1]:</span>
                        <span id="resScore" class="font-mono text-white">0.0000</span>
                    </div>
                    
                    <!-- Class Probability Bars -->
                    <div class="space-y-1 pt-1 border-t border-[#1d2330]">
                        <div class="flex justify-between text-[10px] font-mono text-[#86909f]">
                            <span>Pos: <b id="resPos" class="text-[#f0f3f8]">0%</b></span>
                            <span>Neu: <b id="resNeu" class="text-[#f0f3f8]">0%</b></span>
                            <span>Neg: <b id="resNeg" class="text-[#f0f3f8]">0%</b></span>
                        </div>
                        <div class="w-full bg-[#1b212e] h-1.5 rounded-[1px] overflow-hidden flex">
                            <div id="barPos" class="bg-emerald-500 h-full transition-fast" style="width: 33%"></div>
                            <div id="barNeu" class="bg-slate-500 h-full transition-fast" style="width: 34%"></div>
                            <div id="barNeg" class="bg-rose-500 h-full transition-fast" style="width: 33%"></div>
                        </div>
                    </div>
                </div>
            </div>

        </div>

        <!-- Model Benchmark Evaluation Section -->
        <div class="panel p-4 space-y-2.5">
            <div class="flex items-center justify-between">
                <div>
                    <h2 class="text-xs font-semibold text-white uppercase tracking-wider">Model Benchmark Evaluation (Held-Out 15% Test Split)</h2>
                    <p class="text-[11px] text-[#86909f]">Chronologically strictly unobserved sequence performance across loss metrics and directional accuracy</p>
                </div>
                <span class="text-[10px] bg-[#141822] text-[#86909f] border border-[#1d2330] px-2 py-0.5 rounded-[2px] font-mono">SPLIT: 70% TRAIN / 15% VAL / 15% TEST</span>
            </div>

            <div class="overflow-x-auto border border-[#1d2330] rounded-[3px]">
                <table class="w-full text-left text-xs font-mono">
                    <thead class="bg-[#0d1017] text-[#86909f] border-b border-[#1d2330]">
                        <tr>
                            <th class="p-2 font-medium">Model Architecture</th>
                            <th class="p-2 font-medium">RMSE &darr;</th>
                            <th class="p-2 font-medium">MAE &darr;</th>
                            <th class="p-2 font-medium">MAPE (%)</th>
                            <th class="p-2 font-medium">QLIKE Loss &darr;</th>
                            <th class="p-2 font-medium">Pearson Corr &uarr;</th>
                            <th class="p-2 font-medium text-blue-400">Directional Accuracy (%) &uarr;</th>
                        </tr>
                    </thead>
                    <tbody id="benchmarkTableBody" class="divide-y divide-[#1d2330] text-[#f0f3f8]">
                        <tr><td colspan="7" class="p-3 text-center text-[#525c6c]">Loading benchmark metrics...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Factor Ablation & Volatility-Targeted Risk Simulation -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-3.5">
            
            <!-- Factor Ablation Card -->
            <div class="panel p-4 space-y-3">
                <div class="flex items-center justify-between">
                    <div>
                        <h2 class="text-xs font-semibold text-white uppercase tracking-wider">Multimodal Factor Ablation</h2>
                        <p class="text-[11px] text-[#86909f]">Quantifying predictive contribution of FinBERT NLP over numerical technical signals</p>
                    </div>
                    <span class="text-[10px] bg-[#141822] text-[#86909f] border border-[#1d2330] px-1.5 py-0.5 rounded-[2px] font-mono">Permutation Analysis</span>
                </div>
                
                <div class="overflow-x-auto border border-[#1d2330] rounded-[3px]">
                    <table class="w-full text-left text-xs font-mono">
                        <thead class="bg-[#0d1017] text-[#86909f] border-b border-[#1d2330]">
                            <tr>
                                <th class="p-2 font-medium">Feature Configuration</th>
                                <th class="p-2 font-medium">RMSE &darr;</th>
                                <th class="p-2 font-medium">QLIKE &darr;</th>
                                <th class="p-2 font-medium text-blue-400">Dir Acc (%)</th>
                            </tr>
                        </thead>
                        <tbody id="ablationTableBody" class="divide-y divide-[#1d2330] text-[#f0f3f8]">
                            <tr><td colspan="4" class="p-3 text-center text-[#525c6c]">Loading ablation study...</td></tr>
                        </tbody>
                    </table>
                </div>

                <!-- Top Feature Drivers -->
                <div class="pt-1">
                    <span class="text-[10px] font-semibold text-[#86909f] uppercase tracking-wider">Top Permutation Feature Drivers:</span>
                    <div id="topFeaturesContainer" class="flex flex-wrap gap-1.5 mt-1.5">
                        <span class="text-xs text-[#525c6c]">Evaluating feature weights...</span>
                    </div>
                </div>
            </div>

            <!-- Quantitative Risk Management Simulation Card -->
            <div class="panel p-4 space-y-3">
                <div class="flex items-center justify-between">
                    <div>
                        <h2 class="text-xs font-semibold text-white uppercase tracking-wider">Volatility-Targeted Risk Simulation</h2>
                        <p class="text-[11px] text-[#86909f]">Dynamic exposure scaling: <code class="text-[#f0f3f8] font-mono">w_t = clip(&sigma;_target / &sigma;&#770;_t, 0.2, 1.5)</code></p>
                    </div>
                    <span class="text-[10px] bg-[#141822] text-[#86909f] border border-[#1d2330] px-1.5 py-0.5 rounded-[2px] font-mono">Risk Management</span>
                </div>

                <div class="grid grid-cols-2 gap-2.5 pt-0.5">
                    <div class="panel-inset p-2.5">
                        <span class="text-[10px] text-[#86909f] uppercase font-mono">Vol Reduction</span>
                        <div class="text-xl font-bold font-mono text-emerald-400 mt-0.5 tabular-nums" id="btVolRed">-</div>
                        <div class="text-[10px] text-[#525c6c]">vs Passive Buy & Hold</div>
                    </div>
                    <div class="panel-inset p-2.5">
                        <span class="text-[10px] text-[#86909f] uppercase font-mono">Strategy Max Drawdown</span>
                        <div class="text-xl font-bold font-mono text-blue-300 mt-0.5 tabular-nums" id="btMDD">-</div>
                        <div class="text-[10px] text-[#525c6c]" id="btBnhMDD">B&H MDD: -</div>
                    </div>
                    <div class="panel-inset p-2.5">
                        <span class="text-[10px] text-[#86909f] uppercase font-mono">Sharpe Ratio</span>
                        <div class="text-xl font-bold font-mono text-white mt-0.5 tabular-nums" id="btSharpe">-</div>
                        <div class="text-[10px] text-[#525c6c]" id="btBnhSharpe">B&H: -</div>
                    </div>
                    <div class="panel-inset p-2.5">
                        <span class="text-[10px] text-[#86909f] uppercase font-mono">Periods Evaluated</span>
                        <div class="text-xl font-bold font-mono text-[#f0f3f8] mt-0.5 tabular-nums" id="btPeriods">-</div>
                        <div class="text-[10px] text-[#525c6c]">4-Hour Forward Steps</div>
                    </div>
                </div>
            </div>

        </div>

        <!-- Aligned Multimodal Data Matrix -->
        <div class="panel p-4 space-y-2.5">
            <div class="flex items-center justify-between">
                <div>
                    <h2 class="text-xs font-semibold text-white uppercase tracking-wider">Synchronized Multimodal Training Matrix</h2>
                    <p class="text-[11px] text-[#86909f]">Aligned numerical indicators with FinBERT sentiment shock and zero lookahead leakage</p>
                </div>
                <div class="text-[11px] text-[#86909f] font-mono">
                    Target: <code class="text-blue-400 font-mono">&radic;(&Sigma; r<sub>t+1..t+4</sub>&sup2;)</code>
                </div>
            </div>

            <div class="overflow-x-auto border border-[#1d2330] rounded-[3px]">
                <table class="w-full text-left text-xs font-mono">
                    <thead class="bg-[#0d1017] text-[#86909f] border-b border-[#1d2330]">
                        <tr>
                            <th class="p-2 font-medium">Timestamp</th>
                            <th class="p-2 font-medium">Close</th>
                            <th class="p-2 font-medium">Log Return</th>
                            <th class="p-2 font-medium">RSI (14)</th>
                            <th class="p-2 font-medium">MACD Hist</th>
                            <th class="p-2 font-medium">Avg Sentiment</th>
                            <th class="p-2 font-medium">Sentiment Shock</th>
                            <th class="p-2 font-medium text-blue-400">4H Target Vol</th>
                        </tr>
                    </thead>
                    <tbody id="featureTableBody" class="divide-y divide-[#1d2330] text-[#f0f3f8]">
                        <tr><td colspan="8" class="p-3 text-center text-[#525c6c]">Loading aligned dataset...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Academic Capstone Deliverables & Publication Figures -->
        <div class="panel p-4 space-y-3">
            <div class="flex items-center justify-between">
                <div>
                    <h2 class="text-xs font-semibold text-white uppercase tracking-wider">Academic Capstone Deliverables & Research Artifacts</h2>
                    <p class="text-[11px] text-[#86909f]">Formal capstone paper, methodology validation, and high-resolution publication charts</p>
                </div>
                <div class="flex items-center gap-2">
                    <button onclick="toggleReportViewer()" id="reportToggleBtn" class="px-2.5 py-1 text-xs font-medium rounded-[3px] bg-[#141822] hover:bg-[#1b212e] text-[#f0f3f8] border border-[#1d2330] hover:border-[#2f384d] transition-fast flex items-center gap-1.5">
                        <svg class="w-3.5 h-3.5 text-[#86909f]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                        <span>View Capstone Report</span>
                    </button>
                    <a href="/api/report" target="_blank" class="px-2.5 py-1 text-xs font-medium rounded-[3px] bg-[#141822] hover:bg-[#1b212e] text-[#86909f] hover:text-white border border-[#1d2330] transition-fast">
                        <span>Raw JSON</span> &rarr;
                    </a>
                </div>
            </div>

            <!-- High-Resolution Publication Plates -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3.5 pt-1">
                <div class="panel-inset p-2.5 space-y-2">
                    <div class="flex justify-between items-center text-[11px]">
                        <span class="font-medium text-[#f0f3f8]">Figure 1: Benchmark Comparison</span>
                        <a href="/api/report/charts/benchmark_comparison.png" target="_blank" class="text-[10px] text-blue-400 hover:underline font-mono">Full Res &nearr;</a>
                    </div>
                    <img src="/api/report/charts/benchmark_comparison.png" alt="Benchmark Comparison" class="w-full h-36 object-cover rounded-[2px] border border-[#1d2330] hover:opacity-90 transition-fast cursor-pointer" onclick="window.open(this.src)">
                    <p class="text-[10px] text-[#86909f]">RMSE & Directional Accuracy across Ridge, LSTM, & GRU architectures.</p>
                </div>

                <div class="panel-inset p-2.5 space-y-2">
                    <div class="flex justify-between items-center text-[11px]">
                        <span class="font-medium text-[#f0f3f8]">Figure 2: Permutation Importance</span>
                        <a href="/api/report/charts/feature_importance.png" target="_blank" class="text-[10px] text-blue-400 hover:underline font-mono">Full Res &nearr;</a>
                    </div>
                    <img src="/api/report/charts/feature_importance.png" alt="Feature Importance" class="w-full h-36 object-cover rounded-[2px] border border-[#1d2330] hover:opacity-90 transition-fast cursor-pointer" onclick="window.open(this.src)">
                    <p class="text-[10px] text-[#86909f]">Relative QLIKE degradation quantifying market vs. NLP factor ranking.</p>
                </div>

                <div class="panel-inset p-2.5 space-y-2">
                    <div class="flex justify-between items-center text-[11px]">
                        <span class="font-medium text-[#f0f3f8]">Figure 3: Portfolio Backtest</span>
                        <a href="/api/report/charts/backtest_equity_curve.png" target="_blank" class="text-[10px] text-blue-400 hover:underline font-mono">Full Res &nearr;</a>
                    </div>
                    <img src="/api/report/charts/backtest_equity_curve.png" alt="Backtest Equity Curve" class="w-full h-36 object-cover rounded-[2px] border border-[#1d2330] hover:opacity-90 transition-fast cursor-pointer" onclick="window.open(this.src)">
                    <p class="text-[10px] text-[#86909f]">Volatility-targeting equity curve demonstrating drawdown compression.</p>
                </div>
            </div>

            <!-- Collapsible Report Markdown Drawer -->
            <div id="reportMarkdownContainer" class="hidden border-t border-[#1d2330] pt-3 mt-1">
                <div class="bg-[#0b0e15] border border-[#1d2330] rounded-[3px] p-4 max-h-96 overflow-y-auto">
                    <pre id="reportMarkdownText" class="text-xs font-mono text-[#f0f3f8] whitespace-pre-wrap">Loading capstone report markdown...</pre>
                </div>
            </div>
        </div>

    </main>

    <!-- Institutional Footer -->
    <footer class="border-t border-[#1d2330] py-3.5 px-6 text-center text-[11px] text-[#525c6c] font-mono">
        QUANTITATIVE CRYPTOCURRENCY VOLATILITY FORECASTER &bull; UNDERGRADUATE CAPSTONE RESEARCH &bull; INSTITUTIONAL TIME-SERIES ARCHITECTURE
    </footer>

    <!-- Frontend Logic -->
    <script>
        let marketChart = null;

        async function initDashboard() {
            await Promise.all([
                fetchVolatilityMetrics(),
                fetchMarketHistory(),
                fetchBenchmarkMetrics(),
                fetchAblationMetrics(),
                fetchBacktestMetrics()
            ]);
        }

        async function fetchVolatilityMetrics() {
            try {
                const model = document.getElementById('modelSelect') ? document.getElementById('modelSelect').value : 'gru';
                const res = await fetch('/api/predict/volatility?model=' + model);
                if (!res.ok) throw new Error('Failed to fetch prediction');
                const data = await res.json();

                document.getElementById('stat-price').innerText = '$' + data.current_price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                document.getElementById('stat-hist-vol').innerText = (data.historical_24h_volatility * 100).toFixed(3) + '%';
                document.getElementById('stat-forecast-vol').innerText = (data.forecast_next_4h_volatility * 100).toFixed(3) + '%';
                document.getElementById('stat-regime').innerText = 'Regime: ' + data.volatility_regime;
                if (document.getElementById('stat-model-badge')) {
                    document.getElementById('stat-model-badge').innerText = 'Model: ' + data.model_used;
                }

                const sentEl = document.getElementById('stat-sentiment');
                const score = data.latest_sentiment_score;
                sentEl.innerText = score.toFixed(3);
                if (score > 0.1) {
                    sentEl.className = 'text-2xl font-bold font-mono mt-1 tabular-nums text-emerald-400';
                    document.getElementById('stat-sentiment-sub').innerText = 'Bullish Sentiment (+)';
                } else if (score < -0.1) {
                    sentEl.className = 'text-2xl font-bold font-mono mt-1 tabular-nums text-rose-400';
                    document.getElementById('stat-sentiment-sub').innerText = 'Bearish Sentiment (-)';
                } else {
                    sentEl.className = 'text-2xl font-bold font-mono mt-1 tabular-nums text-white';
                    document.getElementById('stat-sentiment-sub').innerText = 'Neutral Sentiment (=)';
                }
            } catch (err) {
                console.error('Error fetching volatility metrics:', err);
            }
        }

        async function fetchBenchmarkMetrics() {
            try {
                const res = await fetch('/api/benchmark');
                if (!res.ok) return;
                const json = await res.json();
                const records = json.records || [];
                const tbody = document.getElementById('benchmarkTableBody');
                if (!tbody) return;
                tbody.innerHTML = '';

                records.forEach(r => {
                    const tr = document.createElement('tr');
                    tr.className = 'hover:bg-[#161a24] transition-fast';
                    const isBest = r.model.includes('GRU') || r.model.includes('Ridge');
                    tr.innerHTML = `
                        <td class="p-2 font-medium ${isBest ? 'text-blue-400 font-semibold' : 'text-[#f0f3f8]'}">${r.model}</td>
                        <td class="p-2 text-[#86909f]">${r.rmse.toFixed(6)}</td>
                        <td class="p-2 text-[#86909f]">${r.mae.toFixed(6)}</td>
                        <td class="p-2 text-[#86909f]">${r.mape_pct.toFixed(2)}%</td>
                        <td class="p-2 text-[#86909f]">${r.qlike_loss.toFixed(6)}</td>
                        <td class="p-2 ${r.pearson_corr > 0 ? 'text-emerald-400' : 'text-rose-400'}">${r.pearson_corr.toFixed(4)}</td>
                        <td class="p-2 font-semibold ${r.directional_accuracy_pct > 70 ? 'text-emerald-400' : 'text-[#86909f]'}">${r.directional_accuracy_pct.toFixed(1)}%</td>
                    `;
                    tbody.appendChild(tr);
                });
            } catch (err) {
                console.error('Error fetching benchmarks:', err);
            }
        }

        async function fetchAblationMetrics() {
            try {
                const res = await fetch('/api/ablation');
                if (!res.ok) return;
                const json = await res.json();
                const ablRecords = json.ablation || [];
                const impRecords = json.feature_importance || [];

                const tbody = document.getElementById('ablationTableBody');
                if (tbody) {
                    tbody.innerHTML = '';
                    ablRecords.forEach(r => {
                        const tr = document.createElement('tr');
                        tr.className = 'hover:bg-[#161a24] transition-fast';
                        const isMulti = r.configuration.includes('Multimodal');
                        tr.innerHTML = `
                            <td class="p-2 font-medium ${isMulti ? 'text-blue-400 font-semibold' : 'text-[#f0f3f8]'}">${r.configuration}</td>
                            <td class="p-2 text-[#86909f]">${r.rmse.toFixed(6)}</td>
                            <td class="p-2 text-[#86909f]">${r.qlike_loss.toFixed(6)}</td>
                            <td class="p-2 font-semibold ${r.directional_accuracy_pct > 80 ? 'text-emerald-400' : 'text-[#f0f3f8]'}">${r.directional_accuracy_pct.toFixed(1)}%</td>
                        `;
                        tbody.appendChild(tr);
                    });
                }

                const featContainer = document.getElementById('topFeaturesContainer');
                if (featContainer && impRecords.length > 0) {
                    featContainer.innerHTML = '';
                    impRecords.slice(0, 5).forEach(f => {
                        const span = document.createElement('span');
                        const isNlp = f.category.includes('NLP');
                        span.className = `px-2 py-0.5 rounded-[2px] text-[10px] font-mono border ${isNlp ? 'bg-emerald-950/40 text-emerald-300 border-emerald-900/60' : 'bg-[#141822] text-blue-300 border-[#1d2330]'}`;
                        span.innerText = `${f.feature} (${f.relative_importance_pct}%)`;
                        featContainer.appendChild(span);
                    });
                }
            } catch (err) {
                console.error('Error fetching ablation metrics:', err);
            }
        }

        async function fetchBacktestMetrics() {
            try {
                const res = await fetch('/api/backtest');
                if (!res.ok) return;
                const json = await res.json();
                const s = json.summary || {};

                if (document.getElementById('btVolRed') && s.volatility_reduction_pct !== undefined) {
                    document.getElementById('btVolRed').innerText = (s.volatility_reduction_pct >= 0 ? '+' : '') + s.volatility_reduction_pct.toFixed(1) + '%';
                }
                if (document.getElementById('btMDD') && s.strategy_max_drawdown_pct !== undefined) {
                    document.getElementById('btMDD').innerText = s.strategy_max_drawdown_pct.toFixed(1) + '%';
                    document.getElementById('btBnhMDD').innerText = `B&H MDD: ${s.buy_and_hold_max_drawdown_pct.toFixed(1)}%`;
                }
                if (document.getElementById('btSharpe') && s.strategy_sharpe_ratio !== undefined) {
                    document.getElementById('btSharpe').innerText = s.strategy_sharpe_ratio.toFixed(2);
                    document.getElementById('btBnhSharpe').innerText = `B&H: ${s.buy_and_hold_sharpe_ratio.toFixed(2)}`;
                }
                if (document.getElementById('btPeriods') && s.total_periods !== undefined) {
                    document.getElementById('btPeriods').innerText = s.total_periods;
                }
            } catch (err) {
                console.error('Error fetching backtest metrics:', err);
            }
        }

        async function fetchMarketHistory() {
            try {
                const res = await fetch('/api/market/latest?limit=30');
                if (!res.ok) throw new Error('Failed to fetch market data');
                const json = await res.json();
                const records = json.records;

                renderChart(records);
                renderTable(records.slice(-8).reverse());
            } catch (err) {
                console.error('Error fetching market history:', err);
            }
        }

        function renderChart(records) {
            const labels = records.map(r => r.timestamp.slice(11, 16));
            const closes = records.map(r => r.close);
            const bbUpper = records.map(r => r.bb_upper);
            const bbLower = records.map(r => r.bb_lower);

            if (records.length > 0) {
                document.getElementById('chart-timestamp-range').innerText = 
                    `${records[0].timestamp.slice(5, 16)} → ${records[records.length-1].timestamp.slice(5, 16)}`;
            }

            const ctx = document.getElementById('marketChart').getContext('2d');
            if (marketChart) marketChart.destroy();

            marketChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'BTC Close Price',
                            data: closes,
                            borderColor: '#38bdf8',
                            backgroundColor: 'rgba(56, 189, 248, 0.04)',
                            borderWidth: 1.5,
                            tension: 0.1,
                            pointRadius: 0,
                            pointHoverRadius: 3,
                            fill: false
                        },
                        {
                            label: 'Bollinger Upper (+2σ)',
                            data: bbUpper,
                            borderColor: 'rgba(148, 163, 184, 0.3)',
                            borderDash: [3, 3],
                            borderWidth: 1,
                            pointRadius: 0,
                            fill: false
                        },
                        {
                            label: 'Bollinger Lower (-2σ)',
                            data: bbLower,
                            borderColor: 'rgba(148, 163, 184, 0.3)',
                            borderDash: [3, 3],
                            borderWidth: 1,
                            pointRadius: 0,
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { 
                            position: 'top',
                            align: 'end',
                            labels: { color: '#86909f', boxWidth: 10, font: { family: 'JetBrains Mono', size: 10 } } 
                        }
                    },
                    scales: {
                        x: { 
                            grid: { color: 'rgba(29, 35, 48, 0.6)' }, 
                            ticks: { color: '#525c6c', font: { family: 'JetBrains Mono', size: 10 } } 
                        },
                        y: { 
                            grid: { color: 'rgba(29, 35, 48, 0.6)' }, 
                            ticks: { color: '#525c6c', font: { family: 'JetBrains Mono', size: 10 } } 
                        }
                    }
                }
            });
        }

        function renderTable(rows) {
            const tbody = document.getElementById('featureTableBody');
            tbody.innerHTML = '';
            rows.forEach(r => {
                const tr = document.createElement('tr');
                tr.className = 'hover:bg-[#161a24] transition-fast';
                const rsi = r.rsi_14 ? r.rsi_14.toFixed(1) : '-';
                const logR = r.log_return ? (r.log_return * 100).toFixed(2) + '%' : '-';
                const macdHist = r.macd_hist ? r.macd_hist.toFixed(2) : '-';
                const avgSent = r.avg_sentiment !== undefined ? r.avg_sentiment.toFixed(3) : '-';
                const shock = r.sentiment_shock !== undefined ? r.sentiment_shock.toFixed(3) : '-';
                const volTarget = r.target_future_volatility ? (r.target_future_volatility * 100).toFixed(3) + '%' : 'Pending';

                tr.innerHTML = `
                    <td class="p-2 text-[#86909f]">${r.timestamp.slice(5, 16)}</td>
                    <td class="p-2 font-medium text-white">$${r.close.toLocaleString()}</td>
                    <td class="p-2 ${parseFloat(r.log_return) >= 0 ? 'text-emerald-400' : 'text-rose-400'}">${logR}</td>
                    <td class="p-2 text-[#86909f]">${rsi}</td>
                    <td class="p-2 text-[#86909f]">${macdHist}</td>
                    <td class="p-2 ${parseFloat(avgSent) > 0 ? 'text-emerald-400' : parseFloat(avgSent) < 0 ? 'text-rose-400' : 'text-[#86909f]'}">${avgSent}</td>
                    <td class="p-2 text-[#86909f]">${shock}</td>
                    <td class="p-2 text-blue-400 font-semibold">${volTarget}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        async function analyzeCustomHeadline() {
            const input = document.getElementById('headlineInput');
            const text = input.value.trim();
            if (!text) return;

            const btn = document.getElementById('analyzeBtn');
            btn.innerHTML = '<span>Evaluating FinBERT...</span>';
            btn.disabled = true;

            try {
                const res = await fetch('/api/sentiment/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });
                if (!res.ok) throw new Error('Analysis failed');
                const data = await res.json();

                const pos = (data.probabilities.positive * 100).toFixed(1);
                const neg = (data.probabilities.negative * 100).toFixed(1);
                const neu = (data.probabilities.neutral * 100).toFixed(1);

                document.getElementById('resLabel').innerText = data.label;
                document.getElementById('resScore').innerText = data.sentiment_score.toFixed(4);
                document.getElementById('resPos').innerText = pos + '%';
                document.getElementById('resNeg').innerText = neg + '%';
                document.getElementById('resNeu').innerText = neu + '%';

                document.getElementById('barPos').style.width = pos + '%';
                document.getElementById('barNeg').style.width = neg + '%';
                document.getElementById('barNeu').style.width = neu + '%';

                if (data.sentiment_score > 0.15) {
                    document.getElementById('resLabel').className = 'font-mono font-semibold text-emerald-400';
                } else if (data.sentiment_score < -0.15) {
                    document.getElementById('resLabel').className = 'font-mono font-semibold text-rose-400';
                } else {
                    document.getElementById('resLabel').className = 'font-mono font-medium text-[#f0f3f8]';
                }

            } catch (err) {
                alert('Error analyzing headline: ' + err.message);
            } finally {
                btn.innerHTML = '<span>Evaluate Sentiment</span>';
                btn.disabled = false;
            }
        }

        async function refreshData() {
            await fetch('/api/pipeline/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol: 'BTC/USDT', timeframe: '1h', limit: 100 })
            });
            await initDashboard();
        }

        async function toggleReportViewer() {
            const container = document.getElementById('reportMarkdownContainer');
            const btn = document.getElementById('reportToggleBtn');
            if (container.classList.contains('hidden')) {
                container.classList.remove('hidden');
                btn.innerHTML = '<span>Hide Capstone Report</span>';
                try {
                    const res = await fetch('/api/report');
                    const data = await res.json();
                    document.getElementById('reportMarkdownText').innerText = data.content_markdown || 'Report content not available.';
                } catch (e) {
                    document.getElementById('reportMarkdownText').innerText = 'Failed to load report: ' + e.message;
                }
            } else {
                container.classList.add('hidden');
                btn.innerHTML = '<svg class="w-3.5 h-3.5 text-[#86909f]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg><span>View Capstone Report</span>';
            }
        }

        window.onload = initDashboard;
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
