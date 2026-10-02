# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Logging configuration for the DiCEx package.

All DiCEx loggers live under the ``"dicex"`` namespace.
Users can control verbosity via:

1. The ``verbose`` flag on :class:`~dicex.explainer.Dicex`::

       explainer = Dicex(model, verbose=True)  # sets dicex logger to DEBUG

2. The standard Python logging API::

       import logging

       logging.getLogger("dicex").setLevel(logging.DEBUG)

By default the library follows the Python logging best practice of
adding a :class:`logging.NullHandler` to the root package logger,
so no output is produced unless the user explicitly configures a handler.
"""

import logging
from pathlib import Path

LOGGER_NAME = "dicex"

# Attach a NullHandler so that library consumers who do not configure
# logging do not see "No handlers could be found" warnings.
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())

_DEFAULT_FORMAT = "%(name)s — %(levelname)s — %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``dicex`` namespace.

    Args:
        name (str): Sub-logger name (e.g. ``"optim"`` → ``"dicex.optim"``).

    Returns:
        logging.Logger: A child logger under the ``dicex`` namespace.
    """
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_verbosity(verbose: bool | str | int, log_file: str | Path | None = None) -> None:  # noqa: FBT001, C901, PLR0912
    """Configure the dicex logger for interactive use.

    Args:
        verbose (bool | str | int): Verbosity configuration. Can be:
            - ``"info"``: Set console level to ``INFO`` and attach a stream handler.
            - ``"debug"`` or ``True``: Set console level to ``DEBUG`` and attach a stream handler.
            - ``"none"`` or ``False``: Set console level to ``WARNING`` and do not attach a stream handler.
            - An integer logging level constant (e.g. ``logging.INFO``).
        log_file (str | Path | None): If provided, attaches a :class:`logging.FileHandler` to save
            logs to this file.
    """
    logger = logging.getLogger(LOGGER_NAME)

    attach_stream = False
    if isinstance(verbose, bool):
        if verbose:
            console_level = logging.DEBUG
            attach_stream = True
        else:
            console_level = logging.WARNING
            attach_stream = False
    elif isinstance(verbose, str):
        v_lower = verbose.lower()
        if v_lower == "debug":
            console_level = logging.DEBUG
            attach_stream = True
        elif v_lower == "info":
            console_level = logging.INFO
            attach_stream = True
        elif v_lower == "none":
            console_level = logging.WARNING
            attach_stream = False
        else:
            msg = f"Unknown verbose string value: '{verbose}'"
            raise ValueError(msg)
    elif type(verbose) is int:
        console_level = verbose
        if verbose <= logging.DEBUG:
            attach_stream = True
    else:
        msg = f"Invalid type for verbose: {type(verbose)}"
        raise ValueError(msg)

    if log_file is not None:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(console_level)

    # Add a StreamHandler only if one is requested and none is attached yet.
    has_stream_handler = any(type(h) is logging.StreamHandler for h in logger.handlers)
    if attach_stream and not has_stream_handler:
        handler = logging.StreamHandler()
        handler.setLevel(console_level)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(handler)

    if log_file is not None:
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == Path(log_file).resolve()
            for h in logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
            logger.addHandler(file_handler)
