# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Model protocols for the DiCEx package.

This module defines the runtime-checkable protocols that external models
must satisfy to be used with DiCEx.
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RegressorModel(Protocol):
    """Protocol for black-box regression models."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the output for the given inputs.

        Args:
            x (np.ndarray): Input array of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predictions of shape (n_samples,).
        """
        ...


@runtime_checkable
class ClassifierModel(Protocol):
    """Protocol for classifier models."""

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities for the given inputs.

        Args:
            x (np.ndarray): Input array of shape (n_samples, n_features).

        Returns:
            np.ndarray: Class probabilities of shape (n_samples, n_classes).
        """
        ...
