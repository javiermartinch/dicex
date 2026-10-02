# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Angular and robustness metrics for comparing explanation directions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dicex.core.cvar import lower_tail_cvar_empirical

if TYPE_CHECKING:
    from collections.abc import Callable


FloatArray = NDArray[np.float64]
EPSILON = 1e-8


def normalize(direction: FloatArray) -> FloatArray:
    """Normalize a direction vector to unit norm.

    Args:
        direction (FloatArray): Direction vector as a 1D array.

    Returns:
        FloatArray: The normalized unit vector as a 1D array.

    Raises:
        ValueError: If the vector has a norm of zero.
    """
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        msg = "Direction norm is zero; cannot normalize."
        raise ValueError(msg)
    return np.asarray(direction, dtype=float) / norm


def angle_deg(c1: FloatArray, c2: FloatArray) -> float:
    """Compute the angle in degrees between two direction vectors.

    Args:
        c1 (FloatArray): First vector.
        c2 (FloatArray): Second vector.

    Returns:
        float: The angle between the vectors in degrees, bounded in [0, 180].
    """
    n1 = float(np.linalg.norm(c1))
    n2 = float(np.linalg.norm(c2))
    if n1 <= 0.0 or n2 <= 0.0:
        return 0.0
    u1 = c1 / n1
    u2 = c2 / n2
    dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def minimal_angle_to_projection_set_deg(direction: FloatArray, beta: FloatArray, t_star: float) -> float:
    """Compute the minimum angle in degrees between a direction and the projection set.

    The projection set is defined as the intersection of the hyperplane
    {c : beta^T c = t_star} and the unit hypersphere {c : ||c|| = 1}.

    Args:
        direction (FloatArray): Candidate direction vector.
        beta (FloatArray): Direction vector of the model coefficients.
        t_star (float): Target inner product value.

    Returns:
        float: The minimal angular distance in degrees.
    """
    beta_norm = float(np.linalg.norm(beta))
    if beta_norm <= 0.0:
        return 0.0
    direction_unit = normalize(direction)
    beta_unit = beta / beta_norm
    a_coeff = float(np.clip(np.dot(direction_unit, beta_unit), -1.0, 1.0))
    b_coeff = float(np.clip(t_star / beta_norm, -1.0, 1.0))
    a_perp = float(np.sqrt(max(0.0, 1.0 - a_coeff**2)))
    b_perp = float(np.sqrt(max(0.0, 1.0 - b_coeff**2)))
    best_dot = float(np.clip(a_coeff * b_coeff + a_perp * b_perp, -1.0, 1.0))
    return float(np.degrees(np.arccos(best_dot)))


def evaluate_metrics(
    predict_fn: Callable[[FloatArray], FloatArray],
    x0: FloatArray,
    c: FloatArray,
    mu: float,
    sigma: float,
    env_std: float,
    alpha: float,
    n_fresh_samples: int,
    seed: int,
) -> dict[str, float]:
    """Evaluate external robustness metrics for a given direction.

    Draws fresh samples of action and environment noise to calculate the
    robust gain (CVaR), mean gain, failure rate, volatility, and other metrics
    for the specified candidate direction.

    Args:
        predict_fn (Callable[[FloatArray], FloatArray]): Prediction function of the target model.
        x0 (FloatArray): Baseline query point.
        c (FloatArray): Direction vector to evaluate (unit norm).
        mu (float): Nominal step size.
        sigma (float): Standard deviation of user action noise.
        env_std (float): Standard deviation of environmental noise.
        alpha (float): Confidence level for CVaR.
        n_fresh_samples (int): Number of fresh Monte Carlo noise samples to draw.
        seed (int): Random seed for reproducible noise generation.

    Returns:
        dict[str, float]: A dictionary containing computed metrics: 'robust_gain', 'mean_gain',
            'failure_rate', 'volatility', 'safety_floor', 'nominal_gain', and
            'action_magnitude'.
    """
    rng = np.random.default_rng(seed)
    d = len(x0)

    action_noise = rng.normal(scale=sigma, size=(n_fresh_samples, d))
    env_noise = rng.normal(scale=env_std, size=(n_fresh_samples, d))

    perturbed_x = x0 + (mu + action_noise) * c + env_noise
    y_pred = predict_fn(perturbed_x)
    y_base = predict_fn(x0[None, :])[0]

    gain = y_pred - y_base

    # Metrics
    robust_gain = float(lower_tail_cvar_empirical(gain, alpha))
    mean_gain = float(np.mean(gain))
    failure_rate = float(np.mean(gain < 0.0))
    volatility = float(np.std(gain))
    safety_floor = float(np.quantile(gain, 0.01))
    nominal_gain = float(predict_fn((x0 + mu * c)[None, :])[0] - y_base)
    action_magnitude = float(np.linalg.norm(mu * c))

    return {
        "robust_gain": robust_gain,
        "cvar_gain": robust_gain,
        "mean_gain": mean_gain,
        "failure_rate": failure_rate,
        "volatility": volatility,
        "safety_floor": safety_floor,
        "nominal_gain": nominal_gain,
        "action_magnitude": action_magnitude,
    }
