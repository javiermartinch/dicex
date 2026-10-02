# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
# pyright: reportUnknownVariableType=false, reportPossiblyUnboundVariable=false, reportConstantRedefinition=false
"""Lower-tail Conditional Value at Risk (CVaR) computations.

Provides an empirical CVaR estimator (with optional Rust acceleration).
"""

import math

import numpy as np

from dicex.exceptions import InvalidParameterError
from dicex.logging import get_logger
from dicex.validation import validate_alpha

_logger = get_logger(__name__)

try:
    from dicex._rust_ext import lower_tail_cvar_empirical_fast

    HAS_RUST_CVAR = True
except ImportError:  # pragma: no cover
    HAS_RUST_CVAR = False


def lower_tail_cvar_empirical(samples: np.ndarray, alpha: float) -> float:
    """Compute the empirical lower-tail CVaR of a sample.

    Averages the ``ceil(alpha * n)`` smallest order statistics.

    Args:
        samples (np.ndarray): 1D array of sample values.
        alpha (float): Risk level of the lower-tail CVaR, in (0, 1].

    Returns:
        float: The empirical CVaR_alpha.
    """
    validate_alpha(alpha)

    n = len(samples)
    if n == 0:
        msg = "Samples array cannot be empty."
        raise InvalidParameterError(msg)

    if HAS_RUST_CVAR:
        return float(lower_tail_cvar_empirical_fast(np.ascontiguousarray(samples, dtype=np.float64), alpha))  # pyright: ignore[reportUnknownArgumentType]

    _logger.debug("Using pure-Python CVaR path (Rust extension unavailable)")
    k = math.ceil(alpha * n)
    k = max(1, k)

    # A partial partition suffices because only the k smallest values are needed.
    partitioned = np.partition(samples, k - 1)
    return float(np.mean(partitioned[:k]))
