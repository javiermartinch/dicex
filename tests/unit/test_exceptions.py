# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Tests for the custom exception hierarchy."""

import warnings

import pytest

from dicex.exceptions import (
    ConvergenceWarning,
    DicexError,
    DimensionMismatchError,
    InvalidModelError,
    InvalidParameterError,
    NotFittedError,
    NumericalError,
    UnsupportedPerturbationError,
)


def test_dicex_error_base() -> None:
    """All custom exceptions should inherit from DicexError."""
    exceptions = [
        InvalidModelError,
        DimensionMismatchError,
        InvalidParameterError,
        NotFittedError,
        UnsupportedPerturbationError,
        NumericalError,
    ]
    for exc in exceptions:
        assert issubclass(exc, DicexError)


def test_dual_inheritance_from_builtin_exceptions() -> None:
    """Verify that selected errors also derive from the matching built-in exception types."""
    assert issubclass(InvalidParameterError, ValueError)
    assert issubclass(NotFittedError, ValueError)
    assert issubclass(UnsupportedPerturbationError, TypeError)


def test_exception_messages() -> None:
    """Verify exceptions store and display messages correctly."""
    msg = "Test message"
    with pytest.raises(InvalidParameterError, match=msg) as exc_info:
        raise InvalidParameterError(msg)
    assert isinstance(exc_info.value, DicexError)


def test_convergence_warning() -> None:
    """Verify ConvergenceWarning is a UserWarning."""
    assert issubclass(ConvergenceWarning, UserWarning)

    with pytest.warns(ConvergenceWarning, match="test warning"):
        warnings.warn("test warning", ConvergenceWarning, stacklevel=1)
