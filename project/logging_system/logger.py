"""
logging_system/logger.py

Central logging configuration. Every module in this project should do:

    from logging_system.logger import get_logger
    logger = get_logger(__name__)

instead of using print() or configuring logging itself. This replaces
every bare print() statement that existed in the notebook (Cells 5, 8,
9, 10, etc.) with structured, leveled, file+console logging.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from project.config.settings import settings

_CONFIGURED = False


def _configure_root_logger() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings.ensure_dirs()
    log_dir = Path(settings.LOG_DIR)
    log_file = log_dir / f"run_{datetime.now().strftime('%Y%m%d')}.log"

    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger. Root logger is configured on first call."""
    _configure_root_logger()
    return logging.getLogger(name)
