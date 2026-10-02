# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Bootstrap acceptance tests for directional optimization."""

from typing import Literal, cast, overload

import numpy as np

from dicex.logging import get_logger

try:
    from dicex._rust_ext import (
        paired_percentile_bootstrap_lci_fast,  # pyright: ignore[reportUnknownVariableType]
    )

    HAS_RUST_EXT = True
except ImportError:  # pragma: no cover
    HAS_RUST_EXT = False  # pyright: ignore[reportConstantRedefinition]

_logger = get_logger("bootstrap")


@overload
def paired_percentile_bootstrap_lci(
    z_c: np.ndarray,
    z_cprime: np.ndarray,
    alpha: float,
    eta: float,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
    *,
    return_deltas: Literal[False] = False,
) -> float: ...


@overload
def paired_percentile_bootstrap_lci(
    z_c: np.ndarray,
    z_cprime: np.ndarray,
    alpha: float,
    eta: float,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
    *,
    return_deltas: Literal[True],
) -> tuple[float, np.ndarray]: ...


def paired_percentile_bootstrap_lci(  # noqa: PLR0913
    z_c: np.ndarray,
    z_cprime: np.ndarray,
    alpha: float,
    eta: float,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
    *,
    return_deltas: bool = False,
) -> float | tuple[float, np.ndarray]:
    """Compute the lower confidence interval limit for the difference in CVaR.

    Uses paired percentile bootstrap to estimate the eta-percentile of the
    distribution of (CVaR(z_c) - CVaR(z_cprime)).

    Args:
        z_c (np.ndarray): Improvements for candidate direction c.
        z_cprime (np.ndarray): Improvements for incumbent direction c'.
        alpha (float): Risk level for CVaR.
        eta (float): Significance level used as the bootstrap percentile (e.g., 0.05 for 95% confidence).
        n_bootstrap (int): Number of bootstrap resamples.
        rng (np.random.Generator | None): Random number generator.
        return_deltas (bool): If True, return both LCI and the bootstrap_deltas
            array.

    Returns:
        float | tuple[float, np.ndarray]: The lower confidence limit and, if
            ``return_deltas`` is True, the bootstrap differences.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples = len(z_c)
    if len(z_cprime) != n_samples:
        msg = "z_c and z_cprime must have the same length for paired bootstrap."
        raise ValueError(msg)

    k = max(1, int(np.ceil(alpha * n_samples)))

    if HAS_RUST_EXT:
        seed = int(rng.integers(0, 2**63 - 1))
        # The Rust extension requires contiguous float64 arrays.
        z_c_contiguous = np.ascontiguousarray(z_c, dtype=np.float64)
        z_cprime_contiguous = np.ascontiguousarray(z_cprime, dtype=np.float64)
        rust_func = paired_percentile_bootstrap_lci_fast  # pyright: ignore[reportPossiblyUnboundVariable, reportUnknownVariableType]
        rust_res = rust_func(z_c_contiguous, z_cprime_contiguous, alpha, eta, n_bootstrap, seed)  # pyright: ignore[reportUnknownVariableType]
        lci_raw, bootstrap_deltas_raw = cast("tuple[float, list[float]]", rust_res)
        lci = float(lci_raw)
        if return_deltas:
            return lci, np.array(bootstrap_deltas_raw, dtype=float)
        return lci

    # Draw all bootstrap indices at once, with shape (n_bootstrap, n_samples).
    indices = rng.integers(0, n_samples, size=(n_bootstrap, n_samples))

    # Apply the same indices to z_c and z_cprime, with shape (n_bootstrap, n_samples).
    z_c_samples = z_c[indices]
    z_cprime_samples = z_cprime[indices]

    # A partial partition suffices because only the k smallest values are needed.
    partitioned_c = np.partition(z_c_samples, k - 1, axis=1)
    partitioned_cprime = np.partition(z_cprime_samples, k - 1, axis=1)

    # Compute the mean of the k smallest values in each row, with shape (n_bootstrap,).
    cvar_c = np.mean(partitioned_c[:, :k], axis=1)
    cvar_cprime = np.mean(partitioned_cprime[:, :k], axis=1)

    bootstrap_deltas = cvar_c - cvar_cprime

    # Return the eta-percentile (lower confidence limit); np.percentile expects a percentage.
    lci = float(np.percentile(bootstrap_deltas, eta * 100))
    if return_deltas:
        return lci, bootstrap_deltas
    return lci


@overload
def acceptance_test(
    z_c: np.ndarray,
    z_cprime: np.ndarray,
    alpha: float,
    eta: float,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
    *,
    return_details: Literal[False] = False,
) -> bool: ...


@overload
def acceptance_test(
    z_c: np.ndarray,
    z_cprime: np.ndarray,
    alpha: float,
    eta: float,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
    *,
    return_details: Literal[True],
) -> tuple[bool, float, np.ndarray]: ...


def acceptance_test(  # noqa: PLR0913
    z_c: np.ndarray,
    z_cprime: np.ndarray,
    alpha: float,
    eta: float,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
    *,
    return_details: bool = False,
) -> bool | tuple[bool, float, np.ndarray]:
    """Decide whether to accept the new direction c over the incumbent c'.

    Args:
        z_c (np.ndarray): Improvements for candidate direction c.
        z_cprime (np.ndarray): Improvements for incumbent direction c'.
        alpha (float): Risk level for CVaR.
        eta (float): Significance level used as the bootstrap percentile.
        n_bootstrap (int): Number of bootstrap resamples.
        rng (np.random.Generator | None): Random number generator.
        return_details (bool): If True, return (accepted, lci,
            bootstrap_deltas).

    Returns:
        bool | tuple[bool, float, np.ndarray]: Whether the move is accepted and,
            if ``return_details`` is True, the lower confidence limit and the
            bootstrap differences.
    """
    res = paired_percentile_bootstrap_lci(
        z_c,
        z_cprime,
        alpha,
        eta,
        n_bootstrap,
        rng,
        return_deltas=return_details,
    )

    if return_details:
        lci, bootstrap_deltas = cast("tuple[float, np.ndarray]", res)
        accepted = lci > 0
        _logger.debug("Confirm phase: LCI=%+.4f, η_bonf=%.4f → %s", lci, eta, "ACCEPT" if accepted else "REJECT")
        return accepted, lci, bootstrap_deltas
    lci = cast("float", res)
    accepted = lci > 0
    _logger.debug("Confirm phase: LCI=%+.4f, η_bonf=%.4f → %s", lci, eta, "ACCEPT" if accepted else "REJECT")
    return accepted
