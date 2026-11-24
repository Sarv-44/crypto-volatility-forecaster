"""Main CLI entry point for the Algorithmic Crypto Volatility Forecaster pipeline.

Usage:
    python run.py --mode all
    python run.py --mode market
    python run.py --mode sentiment
    python run.py --mode fuse
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for easy module imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger("PipelineRunner")


def run_market_pipeline(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 500):
    """Executes Stage 2 (Market Data) and Stage 3 (Market Features)."""
    from src.data.market_data import fetch_market_data
    from src.features.market_features import compute_market_features

    logger.info(f"=== Starting Market Pipeline ({symbol}, {timeframe}) ===")
    df_raw = fetch_market_data(symbol=symbol, timeframe=timeframe, limit=limit)
    logger.info(f"Market data ready with {len(df_raw)} candles.")

    df_features = compute_market_features(df_raw)
    logger.info(f"Market features calculated with {len(df_features)} rows and {df_features.shape[1]} columns.")
    return df_features


def run_sentiment_pipeline(sample_csv_path: str = None):
    """Executes Stage 4 (FinBERT Sentiment) and Stage 5 (Hourly Aggregation)."""
    from src.nlp.finbert_sentiment import FinBERTSentimentAnalyzer
    from src.nlp.sentiment_aggregation import aggregate_hourly_sentiment, generate_sample_social_posts

    logger.info("=== Starting NLP Sentiment Pipeline ===")
    analyzer = FinBERTSentimentAnalyzer()

    # Load or generate sample social headlines
    posts_df = generate_sample_social_posts(sample_csv_path)
    logger.info(f"Evaluating sentiment for {len(posts_df)} texts...")

    scored_df = analyzer.analyze_dataframe(posts_df, text_col="text")
    logger.info("FinBERT scoring completed.")

    hourly_sentiment = aggregate_hourly_sentiment(scored_df)
    logger.info(f"Hourly sentiment aggregated into {len(hourly_sentiment)} hourly periods.")
    return hourly_sentiment


def run_fusion_pipeline():
    """Executes Stage 6 (Data Alignment) and Stage 7 (Target Realized Volatility)."""
    from src.features.feature_fusion import fuse_market_and_sentiment_features

    logger.info("=== Starting Feature Fusion & Target Volatility Pipeline ===")
    dataset = fuse_market_and_sentiment_features()
    logger.info(f"Multimodal dataset created: {dataset.shape[0]} rows, {dataset.shape[1]} columns.")
    return dataset


def run_all(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 500):
    """Executes all stages sequentially."""
    logger.info(">>> Running Complete Pipeline (Stages 2 -> 7) <<<")
    run_market_pipeline(symbol=symbol, timeframe=timeframe, limit=limit)
    run_sentiment_pipeline()
    final_df = run_fusion_pipeline()
    logger.info(">>> Pipeline execution complete! <<<")
    return final_df


def run_training_pipeline(epochs: int = 15, seq_len: int = 12, loss_type: str = "mse"):
    """Executes Deep Learning & Baseline Training and comparative evaluation."""
    from src.models.pipeline import train_all_models

    logger.info("=== Starting Model Training & Benchmarking ===")
    benchmark_df, summary = train_all_models(seq_len=seq_len, epochs=epochs, loss_type=loss_type)
    print("\n" + "=" * 65)
    print("             MODEL BENCHMARK RESULTS (TEST SPLIT)")
    print("=" * 65)
    print(benchmark_df.to_string(index=False))
    print("=" * 65 + "\n")
    return benchmark_df


def run_ablation_pipeline():
    """Runs ablation study to quantify FinBERT NLP vs Market technical features."""
    from src.models.ablation import run_ablation_study

    logger.info("=== Starting Ablation Study Pipeline ===")
    abl_df, imp_df = run_ablation_study()
    print("\n" + "=" * 65)
    print("           MULTIMODAL ABLATION STUDY RESULTS")
    print("=" * 65)
    print(abl_df.to_string(index=False))
    print("\nTop 5 Most Important Features:")
    print(imp_df.head(5).to_string(index=False))
    print("=" * 65 + "\n")
    return abl_df, imp_df


def run_backtest_pipeline():
    """Runs volatility-targeted risk management portfolio simulation."""
    from src.models.backtest import run_full_backtest

    logger.info("=== Starting Portfolio Backtest Pipeline ===")
    eq_df, summary = run_full_backtest()
    print("\n" + "=" * 65)
    print("    QUANTITATIVE VOLATILITY-TARGETING BACKTEST SUMMARY")
    print("=" * 65)
    for k, v in summary.items():
        print(f"  {k:<35}: {v}")
    print("=" * 65 + "\n")
    return eq_df, summary


def run_tuning_pipeline(epochs: int = 4):
    """Executes hyperparameter grid search across deep model architectures."""
    from src.models.hyperparameter_tuning import run_grid_search

    logger.info("=== Starting Hyperparameter Optimization ===")
    best_params, results_df = run_grid_search(epochs_per_trial=epochs)
    print("\n" + "=" * 65)
    print("         OPTIMAL HYPERPARAMETER CONFIGURATION")
    print("=" * 65)
    for k, v in best_params.items():
        print(f"  {k:<20}: {v}")
    print("=" * 65 + "\n")
    return best_params


def run_report_pipeline():
    """Generates the academic capstone project report and high-resolution figures."""
    from src.utils.report_generator import compile_academic_report

    logger.info("=== Starting Academic Report & Figures Pipeline ===")
    report_path = compile_academic_report()
    logger.info(f"Report compiled successfully at: {report_path}")
    print("\n" + "=" * 65)
    print("      ACADEMIC CAPSTONE REPORT & FIGURES COMPILED")
    print("=" * 65)
    print(f"  Report File : {report_path}")
    print("  Charts      : data/processed/reports/*.png")
    print("=" * 65 + "\n")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Algorithmic Crypto Volatility Forecaster: Complete End-to-End Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "market", "sentiment", "fuse", "train", "benchmark", "ablation", "backtest", "tune", "report"],
        default="all",
        help="Pipeline mode to execute (default: all)",
    )
    parser.add_argument("--symbol", default="BTC/USDT", help="Cryptocurrency trading pair")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe")
    parser.add_argument("--limit", type=int, default=500, help="Number of candles")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs for deep model training")
    parser.add_argument("--loss", choices=["mse", "qlike"], default="mse", help="Loss function for training")

    args = parser.parse_args()

    if args.mode == "market":
        run_market_pipeline(symbol=args.symbol, timeframe=args.timeframe, limit=args.limit)
    elif args.mode == "sentiment":
        run_sentiment_pipeline()
    elif args.mode == "fuse":
        run_fusion_pipeline()
    elif args.mode in ("train", "benchmark"):
        run_training_pipeline(epochs=args.epochs, loss_type=args.loss)
    elif args.mode == "ablation":
        run_ablation_pipeline()
    elif args.mode == "backtest":
        run_backtest_pipeline()
    elif args.mode == "tune":
        run_tuning_pipeline(epochs=args.epochs)
    elif args.mode == "report":
        run_report_pipeline()
    elif args.mode == "all":
        run_all(symbol=args.symbol, timeframe=args.timeframe, limit=args.limit)
        run_training_pipeline(epochs=args.epochs, loss_type=args.loss)
        run_ablation_pipeline()
        run_backtest_pipeline()
        run_report_pipeline()


if __name__ == "__main__":
    main()
