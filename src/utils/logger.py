import logging
import sys
from pathlib import Path


def get_logger(name: str = "crypto_volatility", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standard logger for console output.
    
    Args:
        name: Name of the logger.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
