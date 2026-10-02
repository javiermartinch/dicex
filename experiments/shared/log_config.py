# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Logging configuration shared by the experiment scripts."""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiments.shared.persistence import RunLayout

LOGGER_NAME = "experiments"
_DEFAULT_FORMAT = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``experiments`` namespace.

    Args:
        name (str): Sub-logger name (e.g. ``"training"`` -> ``"experiments.training"``).

    Returns:
        logging.Logger: A :class:`logging.Logger` instance.
    """
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(
    layout: RunLayout | None = None,
    script_name: str = "general",
    log_dir: str | Path | None = None,
) -> None:
    """Configure structured logging for the current execution.

    Sets up basicConfig to log to both sys.stdout and a dedicated log file.

    Args:
        layout (RunLayout | None): The current execution layout containing logs_root.
        script_name (str): The name of the executing script (e.g., ``"training"``).
        log_dir (str | Path | None): Optional directory path to save logs. If provided, overrides layout and default.
    """
    timestamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S")

    if log_dir is not None:
        logs_dir = Path(log_dir)
    elif layout:
        logs_dir = layout.logs_root
    else:
        logs_dir = Path(__file__).parent.parent / "logs"

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{script_name}_{timestamp}.log"

    # Configure root/parent logging level and handlers
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if configure_logging is called multiple times
    logger.handlers = []

    # StreamHandler for stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    logger.addHandler(stream_handler)

    # FileHandler to persist logs
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    logger.addHandler(file_handler)

    # Keep external library logs at WARNING level except dicex which goes to the file
    logging.getLogger("optuna").setLevel(logging.WARNING)
    dicex_logger = logging.getLogger("dicex")
    dicex_logger.setLevel(logging.INFO)
    dicex_logger.addHandler(file_handler)
