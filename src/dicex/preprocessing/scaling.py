# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Data scaling and model wrapping for DiCEx."""

from typing import cast

import numpy as np

from dicex.core.model_protocols import ClassifierModel, RegressorModel
from dicex.exceptions import NotFittedError

SCALER_EPS = 1e-8


class DicexScaler:
    """Standardization (z-score) of the features used by `Dicex`.

    The optimizer searches and perturbs in standardized units; the scaler wraps the
    model so that it receives the original units, and maps the resulting direction
    back to the original feature space.

    Attributes:
        mean_ (np.ndarray | None): Mean of each feature, set by `fit`.
        std_ (np.ndarray | None): Standard deviation of each feature, set by `fit`
            (constant features get 1).
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.eps = SCALER_EPS

    def _check_is_fitted(self) -> None:
        """Ensure the scaler has been fitted."""
        if self.mean_ is None or self.std_ is None:
            msg = "Scaler must be fitted before this operation."
            raise NotFittedError(msg)

    def fit(self, x: np.ndarray) -> "DicexScaler":
        """Learn mean and std from data.

        Args:
            x (np.ndarray): Input data array of shape (n_samples, n_features).

        Returns:
            DicexScaler: The fitted scaler instance (self).
        """
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        # Avoid division by zero for constant features.
        std[std < self.eps] = 1.0
        self.mean_ = mean
        self.std_ = std
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Scale data to zero mean and unit variance.

        Args:
            x (np.ndarray): Input data array of shape (n_samples, n_features).

        Returns:
            np.ndarray: The scaled data array.
        """
        self._check_is_fitted()
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x_scaled: np.ndarray) -> np.ndarray:
        """Invert the scaling transformation.

        Args:
            x_scaled (np.ndarray): Scaled data array.

        Returns:
            np.ndarray: The data array in the original space.
        """
        self._check_is_fitted()
        return x_scaled * self.std_ + self.mean_

    def wrap_model(self, model: RegressorModel | ClassifierModel) -> RegressorModel | ClassifierModel:
        """Wrap a model to accept scaled input.

        Args:
            model (RegressorModel | ClassifierModel): The original model expecting raw data.

        Returns:
            RegressorModel | ClassifierModel: A model wrapper that composes model(inverse_transform(x)).
        """

        class ScaledModel:
            def __init__(self, parent_scaler: "DicexScaler", original_model: RegressorModel | ClassifierModel) -> None:
                self.scaler = parent_scaler
                self.model = original_model

            def predict(self, x_scaled: np.ndarray) -> np.ndarray:
                x_raw = self.scaler.inverse_transform(x_scaled)
                reg_model = cast("RegressorModel", self.model)
                return reg_model.predict(x_raw)

            def predict_proba(self, x_scaled: np.ndarray) -> np.ndarray:
                x_raw = self.scaler.inverse_transform(x_scaled)
                cls_model = cast("ClassifierModel", self.model)
                return cls_model.predict_proba(x_raw)

        return cast("RegressorModel | ClassifierModel", ScaledModel(self, model))

    def transform_direction(self, c_scaled: np.ndarray) -> np.ndarray:
        """Convert a direction from scaled space back to original space.

        If x_scaled = (x - mu) / sigma, then a change Delta x_scaled
        corresponds to Delta x = Delta x_scaled * sigma.

        Args:
            c_scaled (np.ndarray): Direction vector in scaled space (unit norm).

        Returns:
            np.ndarray: Corresponding unit direction in original space.
        """
        self._check_is_fitted()

        # Scale the direction by the original standard deviations.
        c_orig = c_scaled * self.std_

        # Renormalize to unit length in the original space.
        norm = np.linalg.norm(c_orig)
        if norm < self.eps:
            return c_orig
        return c_orig / norm
