"""Logging infrastructure for Qdrant retrieval validation tool."""

import logging
import sys
from typing import Optional
from logging import Logger


def setup_logging(level: Optional[str] = None) -> Logger:
    """Set up logging infrastructure with console and file handlers."""

    if level is None:
        from .env_config import EnvironmentConfig
        level = EnvironmentConfig.get_log_level()

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger("qdrant_validation")
    logger.setLevel(numeric_level)

    # Prevent duplicate handlers
    if logger.handlers:
        logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(f"qdrant_validation.{name}")


def log_validation_result(query_text: str, num_results: int, execution_time: float, success: bool):
    """Log the result of a validation operation."""
    logger = get_logger("validation")

    if success:
        logger.info(
            f"Validation successful: Query '{query_text[:50]}...' returned {num_results} results in {execution_time:.2f}ms"
        )
    else:
        logger.error(
            f"Validation failed: Query '{query_text[:50]}...' failed after {execution_time:.2f}ms"
        )