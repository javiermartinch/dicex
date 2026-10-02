# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Custom exceptions and warnings for the DiCEx package.

Exception hierarchy:

    DicexError (base)
    ├── InvalidModelError          — model does not satisfy required protocol
    ├── DimensionMismatchError     — input shapes mismatch
    ├── InvalidParameterError      — bad hyperparameter (also a ValueError)
    ├── NotFittedError             — object not fitted yet (also a ValueError)
    ├── UnsupportedPerturbationError — perturbation type not supported (also a TypeError)
    └── NumericalError             — NaN, overflow, or degenerate computation

    ConvergenceWarning (UserWarning) — optimization failed to improve
"""


class DicexError(Exception):
    """Base class for all DiCEx errors."""


class InvalidModelError(DicexError):
    """Raised when a provided model is invalid or incompatible."""


class DimensionMismatchError(DicexError):
    """Raised when input dimensions do not match the expected shapes."""


class InvalidParameterError(DicexError, ValueError):
    """Raised when a hyperparameter value is invalid.

    Inherits from both DicexError and ValueError so that
    ``except ValueError`` in user code still catches it.
    """


class NotFittedError(DicexError, ValueError):
    """Raised when an operation requires a fitted object.

    Inherits from both DicexError and ValueError, following the
    scikit-learn convention for ``NotFittedError``.
    """


class UnsupportedPerturbationError(DicexError, TypeError):
    """Raised when a perturbation type is not supported for an operation.

    Inherits from both DicexError and TypeError so that
    ``except TypeError`` in user code still catches it.
    """


class NumericalError(DicexError):
    """Raised when a numerical issue is detected (NaN, overflow, degeneracy)."""


class ConvergenceWarning(UserWarning):
    """Warning raised when the optimization fails to improve from the initial direction."""
