# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Geodesic interpolation and refinement on the sphere.

Provides SLERP (spherical linear interpolation) and geodesic refinement
for searching optimal directions along great-circle arcs. Supports
additive noise ε in the evaluation M(x₀ + T·c + ε).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dicex.core.cvar import lower_tail_cvar_empirical
from dicex.core.geometry import orthogonal_unit_vector
from dicex.logging import get_logger

if TYPE_CHECKING:
    from dicex.core.scoring import ModelScorer

SMALL_ANGLE = 1e-6
DIMENSION_ONE = 1
BINARY_SWITCH_T = 0.5

_logger = get_logger("slerp")


def _binary_slerp(m: np.ndarray, u: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Interpolate on S^0, where only the endpoints exist.

    Args:
        m (np.ndarray): Starting unit vector.
        u (np.ndarray): Ending unit vector.
        t (float | np.ndarray): Interpolation parameter in [0, 1].

    Returns:
        np.ndarray: Interpolated unit vector(s).
    """
    if isinstance(t, np.ndarray):
        selectors = (t >= BINARY_SWITCH_T).astype(float)[:, np.newaxis]
        return (1.0 - selectors) * m + selectors * u
    return u if t >= BINARY_SWITCH_T else m


def _antipodal_slerp(m: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Interpolate along a deterministic great circle for antipodal endpoints.

    Args:
        m (np.ndarray): Starting unit vector.
        t (float | np.ndarray): Interpolation parameter in [0, 1].

    Returns:
        np.ndarray: Interpolated unit vector(s).
    """
    if len(m) == DIMENSION_ONE:
        return _binary_slerp(m, -m, t)  # pragma: no cover

    ortho = orthogonal_unit_vector(m)
    if isinstance(t, np.ndarray):
        t_col = t[:, np.newaxis] if t.ndim == 1 else t
        res = np.cos(np.pi * t_col) * m + np.sin(np.pi * t_col) * ortho
        return res / np.linalg.norm(res, axis=1, keepdims=True)

    res = np.cos(np.pi * t) * m + np.sin(np.pi * t) * ortho
    return res / np.linalg.norm(res)


def slerp(m: np.ndarray, u: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Interpolate between m and u along the great circle (SLERP).

    Args:
        m (np.ndarray): Starting unit vector.
        u (np.ndarray): Ending unit vector.
        t (float | np.ndarray): Interpolation parameter in [0, 1].

    Returns:
        np.ndarray: Interpolated unit vector(s).
    """
    # Compute the angle between the vectors.
    dot = np.clip(np.dot(m, u), -1.0, 1.0)
    theta = np.arccos(dot)

    if len(m) == DIMENSION_ONE:
        return _binary_slerp(m, u, t)

    # Handle very small angles to avoid division by zero.
    if theta < SMALL_ANGLE:
        # Linear interpolation followed by renormalization is stable and accurate for small theta.
        if isinstance(t, np.ndarray):
            t_col = t[:, np.newaxis] if t.ndim == 1 else t
            res = (1.0 - t_col) * m + t_col * u
            return res / np.linalg.norm(res, axis=1, keepdims=True)
        res = (1.0 - t) * m + t * u
        return res / np.linalg.norm(res)

    if np.abs(np.pi - theta) < SMALL_ANGLE:
        return _antipodal_slerp(m, t)

    sin_theta = np.sin(theta)

    if isinstance(t, np.ndarray):
        # Vectorized evaluation for multiple values of t.
        t_col = t[:, np.newaxis] if t.ndim == 1 else t
        res = (np.sin((1.0 - t_col) * theta) / sin_theta) * m + (np.sin(t_col * theta) / sin_theta) * u
        return res / np.linalg.norm(res, axis=1, keepdims=True)

    # Scalar evaluation.
    res = (np.sin((1.0 - t) * theta) / sin_theta) * m + (np.sin(t * theta) / sin_theta) * u
    return res / np.linalg.norm(res)


def geodesic_refinement(  # noqa: PLR0913
    scorer: ModelScorer,
    x0: np.ndarray,
    m: np.ndarray,
    u_tilde: np.ndarray,
    noise_samples: np.ndarray,
    alpha: float,
    grid_t: np.ndarray,
    env_noise_samples: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Search for the best direction along the geodesic arc between m and u_tilde.

    Args:
        scorer (ModelScorer): A scorer used for predictions.
        x0 (np.ndarray): Current feature vector.
        m (np.ndarray): Starting direction (usually incumbent).
        u_tilde (np.ndarray): Target direction (usually the new candidate).
        noise_samples (np.ndarray): Scalar samples T_i from the perturbation distribution.
        alpha (float): Risk level for CVaR.
        grid_t (np.ndarray): Grid of interpolation parameters in [0, 1].
        env_noise_samples (np.ndarray | None): Additive noise samples ε_i (n, d).
            If None, no additive noise is applied.

    Returns:
        tuple[np.ndarray, float]: (best_direction, best_cvar_value).
    """
    # Ensure that the grid includes both endpoints.
    grid = np.unique(np.concatenate([grid_t, [0.0, 1.0]]))
    n_grid = len(grid)
    n_samples = len(noise_samples)
    d = len(x0)

    # Compute all grid directions with a single vectorized SLERP call.
    directions = slerp(m, u_tilde, grid)  # shape (n_grid, d)

    # Build the perturbed points for all directions in one array.
    if noise_samples.ndim == 2 and noise_samples.shape[1] == d:  # noqa: PLR2004
        directional_term = noise_samples[:, np.newaxis, :] * directions[np.newaxis, :, :]
    else:
        directional_term = noise_samples[:, np.newaxis, np.newaxis] * directions[np.newaxis, :, :]

    points = x0 + directional_term  # shape (n_samples, n_grid, d)
    if env_noise_samples is not None:
        points = points + env_noise_samples[:, np.newaxis, :]

    # Flatten the points to evaluate the model in a single call.
    flat_points = points.reshape(n_samples * n_grid, d)

    # Evaluate all points in one batched call.
    scores = scorer.score(flat_points).reshape(n_samples, n_grid)
    y0 = scorer.score_baseline(x0)
    improvements = scores - y0

    # Compute the CVaR for each grid direction.
    best_val = -np.inf
    best_direction = m

    for j, t in enumerate(grid):
        cvar_val = lower_tail_cvar_empirical(improvements[:, j], alpha)
        if cvar_val > best_val:
            best_val = cvar_val
            best_direction = directions[j]

        _logger.debug("    t=%.2f: CVaR=%.6f%s", t, cvar_val, " (best)" if cvar_val == best_val else "")

    return best_direction, float(best_val)
