# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Exact analytical evaluators for linear models.

For multidimensional directional noise T,

    Z(c) = (beta odot c)^T T + beta^T epsilon

and, for the iid Gaussian specification T ~ N(mu 1_d, sigma^2 I_d),

    Delta_alpha(c) = m_{beta,epsilon} + mu (beta^T c)
                   - (phi(q_alpha) / alpha) * sqrt(v_{beta,epsilon} + sigma^2 ||beta odot c||^2).
"""

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import scipy.stats

from dicex.analytical.profiler import build_variance_profiler
from dicex.distributions.base import BasePerturbation
from dicex.distributions.gaussian import GaussianPerturbation
from dicex.exceptions import UnsupportedPerturbationError

STANDARD_NORMAL = scipy.stats.norm


def directional_gaussian_mean_var(
    perturbation: GaussianPerturbation,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Return mean and variance of ``weights^T T`` for Gaussian directional noise.

    Args:
        perturbation (GaussianPerturbation): Gaussian directional perturbation.
        weights (np.ndarray): Weight vector.

    Returns:
        tuple[float, float]: Mean and variance of the directional projection.

    Raises:
        ValueError: If neither sigma nor cov is provided.
    """
    mu = perturbation.mu
    mean = float(mu * np.sum(weights)) if np.isscalar(mu) else float(np.dot(weights, mu))

    cov = perturbation.cov
    if cov is None:
        sigma = perturbation.sigma
        if sigma is None:  # pragma: no cover -- guaranteed by GaussianPerturbation.__init__
            msg = "GaussianPerturbation must define sigma or cov."
            raise ValueError(msg)
        if np.isscalar(sigma):
            sigma_scalar = cast("float", sigma)
            var = float((sigma_scalar**2) * np.sum(weights**2))
        else:
            var = float(np.dot(weights**2, sigma**2))
        return mean, var

    var = float(np.dot(weights**2, cov)) if cov.ndim == 1 else float(weights @ cov @ weights)
    return mean, var


def linear_directional_functional(
    beta: np.ndarray,
    perturbation: BasePerturbation,
    alpha: float,
    c: np.ndarray,
) -> float:
    """Compute the exact CVaR for a linear model under Gaussian directional noise.

    Args:
        beta (np.ndarray): Model coefficients.
        perturbation (BasePerturbation): Gaussian directional perturbation with additive noise.
        alpha (float): Risk level of the lower-tail CVaR.
        c (np.ndarray): Direction vector (unit norm).

    Returns:
        float: The exact robust directional value.

    Raises:
        UnsupportedPerturbationError: If perturbation is not Gaussian.
    """
    weights = beta * c
    m_beta_env = perturbation.projected_env_mean(beta)
    v_beta_env = perturbation.projected_env_var(beta)

    if isinstance(perturbation, GaussianPerturbation):
        directional_mean, directional_var = directional_gaussian_mean_var(perturbation, weights)
        q_alpha = float(STANDARD_NORMAL.ppf(alpha))
        phi_q = float(STANDARD_NORMAL.pdf(q_alpha))
        lambda_alpha = phi_q / alpha

        return float(m_beta_env + directional_mean - lambda_alpha * np.sqrt(v_beta_env + directional_var))

    msg = f"Unsupported perturbation type: {type(perturbation)}"
    raise UnsupportedPerturbationError(msg)


def is_isotropic(perturbation: GaussianPerturbation) -> bool:
    """Check if a Gaussian perturbation has an isotropic covariance structure.

    Args:
        perturbation (GaussianPerturbation): Gaussian perturbation to check.

    Returns:
        bool: True if both mu and sigma are scalars and ``cov`` is None.
    """
    if perturbation.cov is not None:
        return False
    if not (np.isscalar(perturbation.mu) or np.asarray(perturbation.mu).ndim == 0):
        return False
    return np.isscalar(perturbation.sigma) or np.asarray(perturbation.sigma).ndim == 0


def _isotropic_linear_optimal_direction(  # noqa: PLR0915
    beta: np.ndarray,
    perturbation: GaussianPerturbation,
    alpha: float,
    *,
    optimizer_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Solve the optimal direction for an isotropic linear model via 1-D profiling.

    Args:
        beta (np.ndarray): Model coefficients.
        perturbation (GaussianPerturbation): Isotropic Gaussian perturbation.
        alpha (float): Risk level of the lower-tail CVaR.
        optimizer_kwargs (dict[str, Any] | None): Optional grid-search tuning options
            (see ``linear_optimal_direction``).

    Returns:
        np.ndarray: Optimal unit direction vector.
    """
    opts: dict[str, Any] = {} if optimizer_kwargs is None else optimizer_kwargs
    if np.allclose(beta, 0.0):
        c = np.zeros_like(beta)
        c[0] = 1.0
        return c

    beta_norm = float(np.linalg.norm(beta))
    mu = float(perturbation.mu)
    sigma = float(cast("float", perturbation.sigma))
    m_beta_env = float(perturbation.projected_env_mean(beta))
    v_beta_env = float(perturbation.projected_env_var(beta))
    q_alpha = float(STANDARD_NORMAL.ppf(alpha))
    phi_q = float(STANDARD_NORMAL.pdf(q_alpha))
    lambda_alpha = phi_q / alpha

    n_grid = opts.get("n_grid_points", max(501, min(2000, 2 * int(beta_norm))))
    n_top_candidates = opts.get("n_top_candidates", 5)
    search_margin = 4.0 * beta_norm / (n_grid - 1) if n_grid > 1 else 0.1

    get_extrema = build_variance_profiler(beta)

    def _dense_grid_search(obj_func: Callable[[float], float]) -> float:
        t_grid = np.linspace(-beta_norm, beta_norm, n_grid)
        f_grid = np.array([obj_func(float(t)) for t in t_grid])
        top_indices = np.argsort(f_grid)[:n_top_candidates]
        best_f = np.inf
        best_t = 0.0

        for idx in top_indices:
            t_guess = float(t_grid[idx])
            t_lo = max(-beta_norm, t_guess - search_margin)
            t_hi = min(beta_norm, t_guess + search_margin)

            # First zoom: dense grid around the candidate.
            fine_t = np.linspace(t_lo, t_hi, 500)
            fine_f = np.array([obj_func(float(t)) for t in fine_t])
            best_idx = int(np.argmin(fine_f))
            if fine_f[best_idx] < best_f:
                best_f = float(fine_f[best_idx])
                best_t = float(fine_t[best_idx])

            # Second zoom: micro-grid around the current best.
            micro_margin = (t_hi - t_lo) / 250.0
            m_lo = max(-beta_norm, float(fine_t[best_idx]) - micro_margin)
            m_hi = min(beta_norm, float(fine_t[best_idx]) + micro_margin)
            micro_t = np.linspace(m_lo, m_hi, 500)
            micro_f = np.array([obj_func(float(t)) for t in micro_t])
            m_best_idx = int(np.argmin(micro_f))
            if micro_f[m_best_idx] < best_f:
                best_f = float(micro_f[m_best_idx])
                best_t = float(micro_t[m_best_idx])

        # Full-range fallback search to escape local minima.
        fallback_t = np.linspace(-beta_norm, beta_norm, 500)
        fallback_f = np.array([obj_func(float(t)) for t in fallback_t])
        f_best_idx = int(np.argmin(fallback_f))

        # Micro-grid zoom around the fallback minimizer.
        micro_margin = 2.0 * beta_norm / 250.0
        m_lo = max(-beta_norm, float(fallback_t[f_best_idx]) - micro_margin)
        m_hi = min(beta_norm, float(fallback_t[f_best_idx]) + micro_margin)
        micro_t = np.linspace(m_lo, m_hi, 500)
        micro_f = np.array([obj_func(float(t)) for t in micro_t])
        m_best_idx = int(np.argmin(micro_f))

        if micro_f[m_best_idx] < best_f:  # pragma: no cover
            best_f = float(micro_f[m_best_idx])  # pragma: no cover
            best_t = float(micro_t[m_best_idx])  # pragma: no cover

        return best_t

    def outer_objective(t: float) -> float:
        smin, _, _, _ = get_extrema(t)
        var_term = max(0.0, v_beta_env + sigma**2 * smin)
        da = m_beta_env + mu * t - lambda_alpha * np.sqrt(var_term)
        return -da

    t_opt = _dense_grid_search(outer_objective)

    _, _, c_opt, _ = get_extrema(t_opt)

    c_norm = float(np.linalg.norm(c_opt))
    if c_norm > 0:  # pragma: no branch
        c_opt /= c_norm
    return c_opt


def linear_optimal_direction(
    beta: np.ndarray,
    perturbation: BasePerturbation,
    alpha: float,
    **kwargs: Any,  # noqa: ANN401
) -> np.ndarray:
    """Return an optimal direction for the exact linear objective on the sphere.

    Args:
        beta (np.ndarray): Model coefficients.
        perturbation (BasePerturbation): Gaussian directional perturbation.
        alpha (float): Risk level of the lower-tail CVaR.
        **kwargs (Any): Tuning options forwarded to the grid search. Supported keys:

            - ``n_grid_points``: number of points of the outer 1-D grid over t
              (default: ``max(501, min(2000, 2 * int(||beta||)))``).
            - ``n_top_candidates``: number of best grid points refined by local
              zooming (default: 5).

    Returns:
        np.ndarray: The optimal unit direction vector.

    Raises:
        UnsupportedPerturbationError: If perturbation is not isotropic Gaussian.
    """
    if not isinstance(perturbation, GaussianPerturbation):
        msg = f"Unsupported perturbation type: {type(perturbation)}"
        raise UnsupportedPerturbationError(msg)

    if not is_isotropic(perturbation):
        msg = "Exact linear optimization is only supported for isotropic Gaussian perturbations."
        raise UnsupportedPerturbationError(msg)

    return _isotropic_linear_optimal_direction(beta, perturbation, alpha, optimizer_kwargs=kwargs)
