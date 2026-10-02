# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Unit tests for Scikit-Learn adapters."""

from typing import Any, cast

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from dicex.adapters.sklearn import wrap_sklearn_classifier, wrap_sklearn_regressor


def test_wrap_sklearn_regressor() -> None:
    """Verify that wrap_sklearn_regressor correctly maps predict."""
    x = np.array([[1.0], [2.0], [3.0]])
    y = np.array([10.0, 20.0, 30.0])

    estimator = cast("Any", LinearRegression())
    model = estimator.fit(x, y)
    wrapped = wrap_sklearn_regressor(model)

    test_x = np.array([[1.5]])
    preds = wrapped.predict(test_x)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (1,)
    assert np.isclose(preds[0], 15.0)


def test_wrap_sklearn_classifier() -> None:
    """Verify that wrap_sklearn_classifier correctly maps predict_proba."""
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    # Simple binary classification
    y = np.array([0, 0, 1, 1])

    estimator = cast("Any", LogisticRegression())
    model = estimator.fit(x, y)
    wrapped = wrap_sklearn_classifier(model)

    test_x = np.array([[2.5]])
    probs = wrapped.predict_proba(test_x)

    assert isinstance(probs, np.ndarray)
    assert probs.shape == (1, 2)  # 2 classes
    assert np.isclose(np.sum(probs), 1.0)
