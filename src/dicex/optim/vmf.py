# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""von Mises-Fisher distribution utilities for directional sampling.

Provides sampling from the von Mises-Fisher distribution (Wood 1994 rejection
sampler), uniform sphere sampling (Muller method), Bessel ratio computation,
angular radius calculation and inversion.
"""

from typing import cast

import numpy as np
from scipy.optimize import brentq  # type: ignore[reportUnknownVariableType]
from scipy.special import ive

KAPPA_EPS = 1e-6
ORTHO_EPS = 1e-10
DIMENSION_ONE = 1


def sample_uniform_sphere(d: int, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample n points uniformly from the (d-1)-sphere using Muller's method.

    Args:
        d (int): Dimensionality of the sphere.
        n (int): Number of samples.
        rng (np.random.Generator | None): Optional random number generator.

    Returns:
        np.ndarray: Array of shape (n, d) containing the sampled points.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Muller's method: sample from the standard normal distribution and normalize.
    z = rng.standard_normal((n, d))
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    # Guard against division by zero, which occurs with probability zero.
    norms[norms == 0] = 1.0
    return z / norms


def vmf_A_d(kappa: float, d: int) -> float:  # noqa: N802
    """Compute the ratio of modified Bessel functions A_d(kappa) = I_{d/2}(kappa) / I_{d/2 - 1}(kappa).

    Args:
        kappa (float): Concentration parameter.
        d (int): Dimensionality.

    Returns:
        float: The evaluated ratio A_d(kappa).
    """
    if kappa < KAPPA_EPS:
        return 0.0
    nu = d / 2.0 - 1.0
    # Use ive (exponentially scaled modified Bessel function) for numerical stability.
    # ratio = (iv(nu+1, k) * exp(-k)) / (iv(nu, k) * exp(-k)) = iv(nu+1, k) / iv(nu, k)
    # For large kappa, both ive values can underflow to 0.0, producing 0/0 = NaN.
    # The limit as kappa->infinity is 1.0, handled below.
    with np.errstate(invalid="ignore"):
        res = float(ive(nu + 1, kappa) / ive(nu, kappa))
    # If ive yields 0/0 (NaN), return the limit 1.0 as kappa -> infinity.
    return 1.0 if np.isnan(res) else res


def angular_radius(kappa: float, d: int) -> float:
    """Compute the expected angular radius rho(kappa) = arccos(A_d(kappa)).

    Args:
        kappa (float): Concentration parameter.
        d (int): Dimensionality.

    Returns:
        float: The expected angular radius in radians.
    """
    a_d = vmf_A_d(kappa, d)
    return float(np.arccos(np.clip(a_d, -1.0, 1.0)))


def kappa_from_angular_radius(rho: float, d: int) -> float:
    """Find kappa such that angular_radius(kappa, d) = rho using numerical inversion.

    Args:
        rho (float): Target angular radius in radians.
        d (int): Dimensionality.

    Returns:
        float: The calculated concentration parameter kappa.
    """
    if rho >= np.pi / 2.0:
        return 0.0
    target_a_d = np.cos(rho)

    def objective(k: float) -> float:
        return vmf_A_d(k, d) - target_a_d

    # A_d is monotonic in kappa, so brentq applies on the bracket [0, 1e5],
    # which covers all angular radii of practical interest.
    try:
        res = brentq(objective, 0.0, 1e5)  # type: ignore[reportUnknownVariableType]
        return float(cast("float", res))
    except (ValueError, RuntimeError):
        return 0.0


def _get_orthogonal_basis(mu: np.ndarray) -> np.ndarray:
    """Construct an orthonormal basis where the first vector is mu.

    Args:
        mu (np.ndarray): Direction vector used as the first basis element.

    Returns:
        np.ndarray: Matrix of shape (d, d) whose columns form an orthonormal basis.
    """
    d = len(mu)
    x = mu / np.linalg.norm(mu)
    if d == 1:
        return x.reshape(1, 1)  # pragma: no cover

    v = x.copy()
    sgn = 1.0 if x[0] >= 0 else -1.0
    v[0] += sgn

    v_norm = np.linalg.norm(v)
    if v_norm < ORTHO_EPS:  # pragma: no cover
        basis = np.eye(d)  # pragma: no cover
        basis[0, 0] = x[0]  # pragma: no cover
        return basis  # pragma: no cover

    v = v / v_norm
    basis = np.eye(d) - 2.0 * np.outer(v, v)
    basis[:, 0] = x
    return basis


def sample_vmf(mu: np.ndarray, kappa: float, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample n points from vMF(mu, kappa) using Wood (1994) rejection sampling.

    Args:
        mu (np.ndarray): Mean direction vector.
        kappa (float): Concentration parameter.
        n (int): Number of samples.
        rng (np.random.Generator | None): Optional random number generator.

    Returns:
        np.ndarray: Array of shape (n, d) containing the sampled points.

    Raises:
        ValueError: If `mu` is a zero vector.
    """
    if rng is None:
        rng = np.random.default_rng()

    d = len(mu)
    mu_norm = np.linalg.norm(mu)
    if mu_norm == 0.0:
        msg = "mu must be a non-zero vector."
        raise ValueError(msg)
    mu_unit = mu / mu_norm

    if kappa < KAPPA_EPS:
        return sample_uniform_sphere(d, n, rng)

    if d == DIMENSION_ONE:
        align_prob = float(1.0 / (1.0 + np.exp(-2.0 * kappa)))
        signs = np.where(rng.uniform(size=n) < align_prob, 1.0, -1.0)
        return (signs[:, np.newaxis] * mu_unit[np.newaxis, :]).astype(float, copy=False)

    # Wood's algorithm for W, the projection onto mu.
    b = (d - 1.0) / (2.0 * kappa + np.sqrt(4.0 * kappa**2 + (d - 1.0) ** 2))
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (d - 1.0) * np.log(1.0 - x0**2)

    w_samples = np.zeros(n)
    for i in range(n):
        while True:
            # Sample Z ~ Beta((d-1)/2, (d-1)/2).
            z = rng.beta((d - 1.0) / 2.0, (d - 1.0) / 2.0)
            w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
            u = rng.uniform(0, 1)
            if kappa * w + (d - 1.0) * np.log(1.0 - x0 * w) - c >= np.log(u):
                w_samples[i] = w
                break

    # Sample v uniformly from the (d-2)-sphere orthogonal to mu.
    v_samples = sample_uniform_sphere(d - 1, n, rng)

    # Rotate the samples so that they are centered at mu; basis[:, 0] is mu
    # and basis[:, 1:] spans its orthogonal complement.
    basis = _get_orthogonal_basis(mu_unit)
    v_ortho = v_samples @ basis[:, 1:].T
    return w_samples[:, np.newaxis] * basis[:, 0] + np.sqrt(1.0 - w_samples**2)[:, np.newaxis] * v_ortho
