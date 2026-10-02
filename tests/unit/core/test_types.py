# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Tests for base types and model protocols."""

import numpy as np

from dicex.core.model_protocols import ClassifierModel, RegressorModel


def test_regressor_model_protocol() -> None:
    """Verify that an object with ``predict`` satisfies ``RegressorModel``."""

    class ValidModel:
        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.zeros(len(x))

    model = ValidModel()
    # RegressorModel is runtime-checkable, so structural conformance suffices.
    assert isinstance(model, RegressorModel)


def test_classifier_model_protocol() -> None:
    """Verify that an object with ``predict_proba`` satisfies ``ClassifierModel``."""

    class ValidClassifier:
        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            return np.zeros((len(x), 2))

    model = ValidClassifier()
    # ClassifierModel is runtime-checkable, so structural conformance suffices.
    assert isinstance(model, ClassifierModel)


def test_invalid_model_protocol() -> None:
    """Verify that an object without prediction methods satisfies neither protocol."""

    class InvalidModel:
        pass

    model = InvalidModel()
    assert not isinstance(model, RegressorModel)
    assert not isinstance(model, ClassifierModel)
