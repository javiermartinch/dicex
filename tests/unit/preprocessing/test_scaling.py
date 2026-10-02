# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Unit tests for the DicexScaler and model wrappers."""

import numpy as np
import pytest

from dicex.core.scoring import ModelScorer
from dicex.exceptions import NotFittedError
from dicex.preprocessing.scaling import DicexScaler


def test_scaler_round_trip() -> None:
    """Verify that transform and inverse_transform are consistent."""
    data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    scaler = DicexScaler()
    scaler.fit(data)

    scaled = scaler.transform(data)
    # Check zero mean and unit variance
    assert np.allclose(np.mean(scaled, axis=0), 0.0)
    assert np.allclose(np.std(scaled, axis=0), 1.0)

    # Round trip
    inverted = scaler.inverse_transform(scaled)
    assert np.allclose(inverted, data)


def test_transform_direction() -> None:
    """Verify that direction transformation accounts for feature scales."""
    # x1 varies much more than x2
    data = np.array([[0.0, 0.0], [10.0, 1.0]])
    scaler = DicexScaler()
    scaler.fit(data)

    # Direction in scaled space: equal importance to both
    c_scaled = np.array([1.0, 1.0]) / np.sqrt(2)

    c_orig = scaler.transform_direction(c_scaled)

    # In original space, the direction should point more towards x1
    # because it has a larger scale (std_1 = 5, std_2 = 0.5)
    assert c_orig[0] > c_orig[1]
    assert np.isclose(np.linalg.norm(c_orig), 1.0)


def test_wrap_model_regression() -> None:
    """Verify model wrapping for regression."""

    class MockModel:
        def predict(self, x: np.ndarray) -> np.ndarray:
            # f(x1, x2) = x1 + x2
            return np.sum(x, axis=1)

    data = np.array([[0.0, 0.0], [10.0, 10.0]])
    scaler = DicexScaler().fit(data)
    wrapped = scaler.wrap_model(MockModel())

    # Point in scaled space [1, 1] corresponds to [10, 10] in raw space
    x_scaled = np.array([[1.0, 1.0]])
    scorer = ModelScorer(wrapped)
    y = scorer.score(x_scaled)

    assert np.isclose(y[0], 20.0)


def test_model_scorer_classifier() -> None:
    """Verify scorer creation for classifiers."""

    class MockClassifier:
        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            # Return a simple two-class probability vector.
            p = np.clip(x[:, 0], 0, 1)
            return np.column_stack([1 - p, p])

    model = MockClassifier()
    scorer = ModelScorer(model, target_class=1)

    x = np.array([[0.8, 0.0]])
    scores = scorer.score(x)

    assert np.isclose(scores[0], 0.8)


def test_scaler_not_fitted_errors() -> None:
    """Verify NotFittedError is raised when methods are called before fit."""

    scaler = DicexScaler()
    x = np.array([[1.0, 2.0]])
    c = np.array([1.0, 0.0])

    with pytest.raises(NotFittedError):
        scaler.transform(x)

    with pytest.raises(NotFittedError):
        scaler.inverse_transform(x)

    with pytest.raises(NotFittedError):
        scaler.transform_direction(c)


def test_wrap_classifier_predict_proba() -> None:
    """Verify ScaledModel.predict_proba works for wrapped classifiers."""
    data = np.array([[0.0, 0.0], [1.0, 1.0]])
    scaler = DicexScaler().fit(data)

    class MockClassifier:
        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.sum(x, axis=1)

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            # Simple two-class: probability proportional to first feature
            p = np.clip(x[:, 0] / 10.0, 0, 1)
            return np.column_stack([1 - p, p])

    wrapped = scaler.wrap_model(MockClassifier())
    x_scaled = np.array([[0.5, 0.5]])
    proba = wrapped.predict_proba(x_scaled)  # type: ignore[union-attr]
    assert proba.shape == (1, 2)  # type: ignore[reportUnknownMemberType]
    assert 0.0 <= proba[0, 0] <= 1.0


def test_transform_direction_zero_vector() -> None:
    """Verify transform_direction handles zero vector (norm < eps)."""
    data = np.array([[0.0, 0.0], [10.0, 2.0]])
    scaler = DicexScaler().fit(data)

    c_zero = np.array([0.0, 0.0])
    result = scaler.transform_direction(c_zero)
    # Should return the zero direction unchanged (norm < eps branch)
    assert np.allclose(result, 0.0)
