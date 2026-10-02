# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Tests for validation functions."""

import numpy as np
import pytest

from dicex.exceptions import DicexError, InvalidModelError
from dicex.validation import validate_alpha, validate_direction, validate_model


def test_validate_direction_valid() -> None:
    """Test validate_direction with a valid unit vector."""
    c = np.array([1.0, 0.0])
    validate_direction(c)  # Should not raise


def test_validate_direction_invalid() -> None:
    """Test validate_direction raises DicexError for non-unit vector."""
    c = np.array([1.0, 1.0])
    with pytest.raises(DicexError):
        validate_direction(c)


def test_validate_alpha_valid() -> None:
    """Test validate_alpha with valid values."""
    validate_alpha(0.1)
    validate_alpha(0.5)
    validate_alpha(0.99)  # Should not raise
    validate_alpha(1.0)  # Mean improvement is also supported


def test_validate_alpha_invalid() -> None:
    """Test validate_alpha raises ValueError for invalid values."""
    for invalid_alpha in [0.0, -0.5, 1.5]:
        with pytest.raises(ValueError, match="Alpha must be strictly greater than 0 and less than or equal to 1"):
            validate_alpha(invalid_alpha)


def test_validate_model_valid() -> None:
    """Test validate_model accepts a valid model."""

    class ValidModel:
        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.zeros(len(x))

    model = ValidModel()
    validate_model(model, d=5)  # Should not raise


def test_validate_model_invalid_shape() -> None:
    """Test validate_model raises InvalidModelError for wrong output shape."""

    class InvalidShapeModel:
        def predict(self, x: np.ndarray) -> np.ndarray:
            # Returns wrong shape: (n, 1) instead of (n,)
            return np.zeros((len(x), 1))

    model = InvalidShapeModel()
    with pytest.raises(InvalidModelError):
        validate_model(model, d=5)


def test_validate_model_not_protocol() -> None:
    """Test validate_model raises InvalidModelError when protocol is not satisfied."""

    class EmptyModel:
        pass

    model = EmptyModel()
    with pytest.raises(InvalidModelError):
        validate_model(model, d=5)


def test_validate_model_classifier_valid() -> None:
    """Test validate_model accepts a valid classifier model."""

    class ValidClassifier:
        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            return np.zeros((len(x), 2))

    model = ValidClassifier()
    validate_model(model, d=5)  # Should not raise


def test_validate_model_classifier_invalid_shape() -> None:
    """Test validate_model raises InvalidModelError for wrong classifier output shape."""

    class InvalidClassifier:
        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            # Returns 1D array instead of 2D
            return np.zeros(len(x))

    model = InvalidClassifier()
    with pytest.raises(InvalidModelError, match="Classifier output shape"):
        validate_model(model, d=5)


def test_validate_model_exception() -> None:
    """Test validate_model handles exceptions during model evaluation."""

    class CrashingModel:
        def predict(self, x: np.ndarray) -> np.ndarray:
            msg = "Boom!"
            raise RuntimeError(msg)

    model = CrashingModel()
    with pytest.raises(InvalidModelError, match="Failed to evaluate model"):
        validate_model(model, d=5)
