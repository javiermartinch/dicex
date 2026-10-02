# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Validation utilities for inputs, models, and constraints."""

import numpy as np

from dicex.core.model_protocols import ClassifierModel, RegressorModel
from dicex.exceptions import DicexError, InvalidModelError, InvalidParameterError


def validate_direction(c: np.ndarray) -> None:
    """Validate that the direction vector is a unit vector.

    Args:
        c (np.ndarray): Direction vector.

    Raises:
        DicexError: If the vector norm is not close to 1.
    """
    norm = np.linalg.norm(c)
    if not np.isclose(norm, 1.0, atol=1e-6):
        msg = f"Direction vector must be a unit vector. Got norm {norm}"
        raise DicexError(msg)


def validate_alpha(alpha: float) -> None:
    """Validate that alpha is within the interval (0, 1].

    Args:
        alpha (float): Risk level of the lower-tail CVaR.

    Raises:
        InvalidParameterError: If alpha is not in (0, 1].
    """
    if not (0.0 < alpha <= 1.0):
        msg = f"Alpha must be strictly greater than 0 and less than or equal to 1. Got {alpha}"
        raise InvalidParameterError(msg)


def validate_model(model: object, d: int) -> None:
    """Validate that the model satisfies the required protocol and returns valid shapes.

    Args:
        model (object): Black-box model to validate.
        d (int): Expected feature dimension.

    Raises:
        InvalidModelError: If the model does not satisfy the protocols or returns bad shapes.
    """
    is_regressor = isinstance(model, RegressorModel)
    is_classifier = isinstance(model, ClassifierModel)

    if not (is_regressor or is_classifier):
        msg = "Model must satisfy RegressorModel or ClassifierModel protocol."
        raise InvalidModelError(msg)

    n_samples = 2
    x_dummy = np.zeros((n_samples, d))

    try:
        if isinstance(model, RegressorModel):  # noqa: SIM108
            out = model.predict(x_dummy)
        else:
            out = model.predict_proba(x_dummy)
    except Exception as e:
        msg = f"Failed to evaluate model with dummy input: {e}"
        raise InvalidModelError(msg) from e

    if isinstance(model, RegressorModel):
        if out.shape != (n_samples,):
            msg = f"Regressor output shape should be (n,). Got {out.shape}"
            raise InvalidModelError(msg)
    else:
        expected_ndim = 2
        if out.ndim != expected_ndim or out.shape[0] != n_samples:
            msg = f"Classifier output shape should be (n, K). Got {out.shape}"
            raise InvalidModelError(msg)
