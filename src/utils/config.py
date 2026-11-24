from pathlib import Path

# Base project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_MARKET_DIR = DATA_DIR / "raw" / "market"
RAW_SOCIAL_DIR = DATA_DIR / "raw" / "social"
PROCESSED_DIR = DATA_DIR / "processed"

# Model directory
MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"

# Default trading parameters
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAME = "1h"
DEFAULT_LIMIT = 500

# Forecast horizon for target volatility (in hours)
VOLATILITY_HORIZON_HOURS = 4
