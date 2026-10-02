# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Tests for the logging configuration."""

import logging
import pathlib
from typing import Any, TextIO

import pytest

from dicex.logging import LOGGER_NAME, configure_verbosity, get_logger


def _stream_handlers(logger: logging.Logger) -> list[logging.StreamHandler[TextIO]]:
    return [handler for handler in logger.handlers if isinstance(handler, logging.StreamHandler)]


def test_get_logger() -> None:
    """Verify get_logger returns a child of the root dicex logger."""
    logger = get_logger("test")
    assert logger.name == f"{LOGGER_NAME}.test"


def test_root_logger_has_null_handler() -> None:
    """Verify the root logger has a NullHandler to prevent warnings."""
    logger = logging.getLogger(LOGGER_NAME)
    has_null_handler = any(isinstance(h, logging.NullHandler) for h in logger.handlers)
    assert has_null_handler


def test_configure_verbosity_true() -> None:
    """Verify verbose=True sets level to DEBUG and adds a StreamHandler."""
    logger = logging.getLogger(LOGGER_NAME)
    # Clear handlers and reset level for clean test
    logger.handlers = []
    logger.setLevel(logging.NOTSET)

    configure_verbosity(verbose=True)

    assert logger.level == logging.DEBUG
    # Should have added exactly one StreamHandler
    stream_handlers = _stream_handlers(logger)
    assert len(stream_handlers) == 1

    # Cleanup
    logger.handlers = []
    logger.setLevel(logging.NOTSET)


def test_configure_verbosity_false() -> None:
    """Verify verbose=False sets level to WARNING and doesn't add StreamHandler."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers = []
    logger.setLevel(logging.NOTSET)

    configure_verbosity(verbose=False)

    assert logger.level == logging.WARNING
    stream_handlers = _stream_handlers(logger)
    assert len(stream_handlers) == 0

    # Cleanup
    logger.setLevel(logging.NOTSET)


def test_configure_verbosity_level_constant() -> None:
    """Verify configure_verbosity accepts standard logging integer constants."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers = []
    logger.setLevel(logging.NOTSET)

    configure_verbosity(verbose=logging.INFO)

    assert logger.level == logging.INFO
    stream_handlers = _stream_handlers(logger)
    assert len(stream_handlers) == 0

    # Cleanup
    logger.setLevel(logging.NOTSET)


def test_configure_verbosity_strings() -> None:
    """Verify configure_verbosity accepts 'info', 'debug', and 'none' strings."""
    logger = logging.getLogger(LOGGER_NAME)

    # 1. Test "info"
    logger.handlers = []
    logger.setLevel(logging.NOTSET)
    configure_verbosity(verbose="info")
    assert logger.level == logging.INFO
    assert len(_stream_handlers(logger)) == 1

    # 2. Test "debug"
    logger.handlers = []
    logger.setLevel(logging.NOTSET)
    configure_verbosity(verbose="debug")
    assert logger.level == logging.DEBUG
    assert len(_stream_handlers(logger)) == 1

    # 3. Test "none"
    logger.handlers = []
    logger.setLevel(logging.NOTSET)
    configure_verbosity(verbose="none")
    assert logger.level == logging.WARNING
    assert len(_stream_handlers(logger)) == 0

    # Cleanup
    logger.handlers = []
    logger.setLevel(logging.NOTSET)


def test_configure_verbosity_exceptions() -> None:
    """Verify configure_verbosity raises errors for invalid inputs."""
    with pytest.raises(ValueError, match="Unknown verbose string value: 'magic'"):
        configure_verbosity("magic")

    with pytest.raises(ValueError, match="Invalid type for verbose"):
        configure_verbosity(3.14)  # pyright: ignore[reportArgumentType]


def test_configure_verbosity_int_attach_stream() -> None:
    """Verify int verbose <= logging.DEBUG attaches a stream handler."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers = []

    # 5 is lower than DEBUG (10)
    configure_verbosity(5)
    assert logger.level == 5
    assert len(_stream_handlers(logger)) == 1

    logger.handlers = []


def test_configure_verbosity_log_file(tmp_path: pathlib.Path) -> None:
    """Verify log_file attaches a FileHandler."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers = []

    log_file: Any = tmp_path / "test.log"
    configure_verbosity(verbose=False, log_file=log_file)

    assert logger.level == logging.DEBUG
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1

    # Second call should not attach duplicate
    configure_verbosity(verbose=False, log_file=log_file)
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1

    # Cleanup and close handlers
    for h in logger.handlers:
        h.close()
    logger.handlers = []
