"""Centralized logging setup for AuditLens."""

import logging
import os
from typing import Optional

from core.config import DEFAULT_CONFIG


def setup_logger(
    name: str = "auditlens",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configures and returns a logger with console and file handlers.

    Args:
        name: Logger name.
        log_file: Path to output log file. Defaults to DEFAULT_CONFIG.log_file_path.
        level: Logging level (e.g. logging.INFO or logging.DEBUG).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Root filter accepts all; handlers filter

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    target_path = log_file or DEFAULT_CONFIG.log_file_path
    log_dir = os.path.dirname(target_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(target_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


logger = setup_logger()
