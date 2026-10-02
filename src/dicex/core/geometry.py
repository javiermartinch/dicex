# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Geometric utilities for directional sampling.

Provides common functions for moving points along directions with noise:
    x_perturbed = x₀ + T ⊙ c + ε
"""

import numpy as np

_NOISE_2D = 2
_ORTHO_EPS = 1e-10


def apply_directional_noise(
    x0: np.ndarray,
    direction: np.ndarray,
    noise_samples: np.ndarray,
    env_noise_samples: np.ndarray | None = None,
) -> np.ndarray:
    """Apply directional and additive noise to a base point.

    Compute points = x0 + noise ⊙ direction + env_noise.

    Args:
        x0 (np.ndarray): Base point (d,).
        direction (np.ndarray): Unit direction vector (d,).
        noise_samples (np.ndarray): Directional noise samples T_i of shape (n, d).
        env_noise_samples (np.ndarray | None): Additive noise samples ε_i (n, d).
            If None, no additive noise is applied.

    Returns:
        np.ndarray: Array of points (n, d).
    """
    noise = np.asarray(noise_samples, dtype=float)
    if noise.ndim == 1:
        noise = noise.reshape(-1, 1)

    if noise.ndim != _NOISE_2D:
        msg = f"noise_samples must have shape (n, d). Got array with shape {noise.shape}"
        raise ValueError(msg)

    if (noise.shape[1] == 1 and len(x0) == 1) or noise.shape[1] == len(x0):
        points = x0 + noise * direction.reshape(1, -1)
    else:
        msg = f"noise_samples second dimension must match len(x0)={len(x0)}. Got shape {noise.shape}"
        raise ValueError(msg)

    # Add additive environmental noise ε
    if env_noise_samples is not None:
        points = points + env_noise_samples

    return points


def orthogonal_unit_vector(v: np.ndarray) -> np.ndarray:
    """Find a unit vector orthogonal to *v* using Gram-Schmidt.

    Picks the standard basis vector with the smallest projection onto *v*
    and orthogonalises it, guaranteeing numerical stability.

    Args:
        v (np.ndarray): Non-zero input vector.

    Returns:
        np.ndarray: A unit vector orthogonal to *v*.
    """
    idx = int(np.argmin(np.abs(v)))
    e = np.zeros_like(v)
    e[idx] = 1.0
    ortho = e - (np.dot(e, v) / np.dot(v, v)) * v
    norm = np.linalg.norm(ortho)
    if norm < _ORTHO_EPS:
        msg = "Cannot construct a stable orthogonal direction."
        raise ValueError(msg)
    return ortho / norm
