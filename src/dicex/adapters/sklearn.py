# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Adapters for scikit-learn models.

These wrappers ensure that sklearn models satisfy the internal
RegressorModel and ClassifierModel protocols.
"""

from typing import Any

import numpy as np

from dicex.core.model_protocols import ClassifierModel, RegressorModel


def wrap_sklearn_regressor(model: Any) -> RegressorModel:  # noqa: ANN401
    """Wrap a scikit-learn regressor.

    Args:
        model (Any): A fitted sklearn regressor (must have a predict method).

    Returns:
        RegressorModel: The wrapped model.
    """

    class SklearnRegressorWrapper:
        def __init__(self, original_model: Any) -> None:  # noqa: ANN401
            self.model = original_model

        def predict(self, x: np.ndarray) -> np.ndarray:
            # Flatten the output to shape (n,).
            return self.model.predict(x).flatten()

    return SklearnRegressorWrapper(model)


def wrap_sklearn_classifier(model: Any) -> ClassifierModel:  # noqa: ANN401
    """Wrap a scikit-learn classifier.

    Args:
        model (Any): A fitted sklearn classifier (must have a predict_proba method).

    Returns:
        ClassifierModel: The wrapped model.
    """

    class SklearnClassifierWrapper:
        def __init__(self, original_model: Any) -> None:  # noqa: ANN401
            self.model = original_model

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            # scikit-learn predict_proba returns an array of shape (n, k).
            return self.model.predict_proba(x)

    return SklearnClassifierWrapper(model)
