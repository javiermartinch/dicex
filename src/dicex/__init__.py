# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""DiCEx: Directional Counterfactual Explanations.

Public API exports.
"""

from importlib.metadata import PackageNotFoundError, version

from dicex.distributions.custom import CustomPerturbation
from dicex.distributions.gaussian import GaussianPerturbation
from dicex.distributions.uniform import UniformPerturbation
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
from dicex.explainer import Dicex
from dicex.results import ExplanationResult

try:
    __version__ = version("dicex")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = [
    "ConvergenceWarning",
    "CustomPerturbation",
    "Dicex",
    "DicexError",
    "DimensionMismatchError",
    "ExplanationResult",
    "GaussianPerturbation",
    "InvalidModelError",
    "InvalidParameterError",
    "NotFittedError",
    "NumericalError",
    "UniformPerturbation",
    "UnsupportedPerturbationError",
    "__version__",
]
