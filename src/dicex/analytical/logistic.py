# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false
"""Exact analytical evaluators for logistic regression models.

Uses numerical quadrature for high precision and stability.

With multidimensional Gaussian directional noise T,

    m(c) = η₀ + E[(β ⊙ c)^T T] + m_{β,ε}
    s(c) = sqrt(Var[(β ⊙ c)^T T] + v_{β,ε}).
"""

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from scipy.integrate import quad
from scipy.special import expit
import scipy.stats

from dicex.analytical.linear import directional_gaussian_mean_var, is_isotropic
from dicex.analytical.profiler import build_variance_profiler
from dicex.distributions.base import BasePerturbation
from dicex.distributions.gaussian import GaussianPerturbation
from dicex.exceptions import UnsupportedPerturbationError

STANDARD_NORMAL = scipy.stats.norm


def _lambda(x: float) -> float:
    """Evaluate the logistic sigmoid function.

    Args:
        x (float): Input value.

    Returns:
        float: ``1 / (1 + exp(-x))``.
    """
    return float(expit(x))


def _logistic_integrand(z: float, m: float, s: float) -> float:
    """Build the logistic-normal integrand ``Lambda(m + s*z) * phi(z)``.

    Args:
        z (float): Integration variable (standard normal quantile).
        m (float): Mean parameter.
        s (float): Standard deviation parameter.

    Returns:
        float: The integrand value at ``z``.
    """
    return _lambda(m + s * z) * float(STANDARD_NORMAL.pdf(z))


def _logistic_h_alpha_quad(m: float, s: float, alpha: float) -> float:
    """Compute the logistic CVaR functional using numerical quadrature.

    Args:
        m (float): Mean parameter.
        s (float): Standard deviation parameter.
        alpha (float): Risk level of the lower-tail CVaR.

    Returns:
        float: The CVaR value, or zero for degenerate cases.
    """
    if alpha <= 0:
        return 0.0

    if np.isclose(s, 0):
        return _lambda(m)

    if alpha >= 1.0:
        # Expected value over the whole real line
        res, _ = quad(_logistic_integrand, -8.0, 8.0, args=(m, s))
        return float(cast("float", res))

    q_alpha = float(STANDARD_NORMAL.ppf(alpha))
    if q_alpha <= -8.0:  # noqa: PLR2004
        return 0.0
    # Integrate over the non-negligible normal range [-8.0, q_alpha]
    res, _ = quad(_logistic_integrand, -8.0, q_alpha, args=(m, s))
    return float(cast("float", res) / alpha)


def logistic_h_alpha(m: float, s: float, alpha: float) -> float:
    """Compute the logistic CVaR functional H_alpha(m, s).

    Evaluated by adaptive numerical quadrature, which is accurate to machine
    precision for all (m, s, alpha). The analytical series expansion
    (``logistic_h_alpha_series``) converges slowly, with terms decaying like 1/k,
    and can be inaccurate by more than 1e-2 in parts of the parameter space, so it
    is not used for the exact benchmarks.

    Args:
        m (float): Mean parameter.
        s (float): Standard deviation parameter.
        alpha (float): Risk level of the lower-tail CVaR.

    Returns:
        float: The evaluated functional H_alpha.
    """
    return _logistic_h_alpha_quad(m, s, alpha)


def _log_diff_cdf(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    """Compute log(Phi(a) - Phi(b)) stably for a > b.

    Uses the ``logcdf`` trick and ``log1p`` to avoid catastrophic cancellation
    when ``a`` and ``b`` are close.

    Args:
        a (np.ndarray | float): Upper limit (must be >= b element-wise).
        b (np.ndarray | float): Lower limit.

    Returns:
        np.ndarray | float: ``log(Phi(a) - Phi(b))``, or ``-inf`` when ``a <= b``.
    """
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)

    mask = (a_arr > 0) & (b_arr > 0)
    a_new = np.where(mask, -b_arr, a_arr)
    b_new = np.where(mask, -a_arr, b_arr)

    log_phi_a = STANDARD_NORMAL.logcdf(a_new)
    log_phi_b = STANDARD_NORMAL.logcdf(b_new)

    with np.errstate(invalid="ignore", divide="ignore"):
        res = log_phi_a + np.log1p(-np.exp(log_phi_b - log_phi_a))

    res = np.where(a_arr <= b_arr, -np.inf, res)
    return float(res) if res.ndim == 0 else res


def _logistic_integral_series(m: float, s: float, q: float, n_terms: int = 100) -> float:
    """Compute the integral of Lambda(m+sz)phi(z) using series expansion.

    Splits the integration range at ``z0 = -m/s``: the left branch uses a
    direct expansion, and the right branch uses a ``log``-space expansion
    for numerical stability.

    Args:
        m (float): Mean parameter.
        s (float): Standard deviation parameter.
        q (float): Upper integration limit.
        n_terms (int): Number of series terms (default 100).

    Returns:
        float: The integral value.
    """
    if np.isclose(s, 0):
        return float(STANDARD_NORMAL.cdf(q) * _lambda(m))
    z0 = -m / s
    k = np.arange(1, n_terms + 1)

    if q <= z0:
        # Left branch expansion
        log_phi = STANDARD_NORMAL.logcdf(q - k * s)
        log_val = k * m + 0.5 * (k * s) ** 2 + log_phi
        terms = ((-1.0) ** (k + 1)) * np.exp(log_val)
        return float(np.sum(terms))

    # Right branch split
    i_z0 = _logistic_integral_series(m, s, z0, n_terms)
    term_mid = float(STANDARD_NORMAL.cdf(q) - STANDARD_NORMAL.cdf(z0))

    log_val = -k * m + 0.5 * (k * s) ** 2 + _log_diff_cdf(q + k * s, z0 + k * s)
    terms = ((-1.0) ** (k + 1)) * np.exp(log_val)
    res_right = float(np.sum(terms))

    return i_z0 + term_mid - res_right


def logistic_h_alpha_series(m: float, s: float, alpha: float, n_terms: int = 100) -> float:
    """Compute H_alpha(m, s) using the series expansion.

    Warning: Can be numerically unstable for large s or n_terms.

    Args:
        m (float): Mean parameter.
        s (float): Standard deviation parameter.
        alpha (float): Risk level of the lower-tail CVaR.
        n_terms (int): Number of series terms (default 100).

    Returns:
        float: The CVaR value.
    """
    if alpha <= 0:
        return 0.0
    if np.isclose(s, 0):
        return _lambda(m)
    if alpha >= 1.0:
        return _logistic_h_alpha_quad(m, s, 1.0)
    q_alpha = float(STANDARD_NORMAL.ppf(alpha))
    return _logistic_integral_series(m, s, q_alpha, n_terms) / alpha


def logistic_directional_functional(
    beta: np.ndarray,
    b: float,
    x0: np.ndarray,
    perturbation: BasePerturbation,
    alpha: float,
    c: np.ndarray,
) -> float:
    """Compute the exact CVaR for a logistic model improvement.

    Exact lower-tail functional under multidimensional Gaussian directional noise.

    Args:
        beta (np.ndarray): Model coefficients.
        b (float): Model intercept.
        x0 (np.ndarray): Baseline point.
        perturbation (BasePerturbation): Directional perturbation distribution.
        alpha (float): Risk level of the lower-tail CVaR.
        c (np.ndarray): Direction vector.

    Returns:
        float: The exact CVaR_alpha relative to the baseline ``Lambda(eta0)``.

    Raises:
        UnsupportedPerturbationError: If perturbation is not Gaussian.
    """
    eta0 = float(np.dot(beta, x0) + b)
    weights = beta * c
    m_beta_env = perturbation.projected_env_mean(beta)
    v_beta_env = perturbation.projected_env_var(beta)

    if not isinstance(perturbation, GaussianPerturbation):
        msg = f"Unsupported perturbation type: {type(perturbation)}"
        raise UnsupportedPerturbationError(msg)

    directional_mean, directional_var = directional_gaussian_mean_var(perturbation, weights)
    m = eta0 + m_beta_env + directional_mean
    s = float(np.sqrt(v_beta_env + directional_var))
    h_val = logistic_h_alpha(m, s, alpha)

    baseline = _lambda(eta0)

    return float(h_val - baseline)


def _logistic_optimal_direction_base(  # noqa: C901, PLR0913, PLR0915
    beta: np.ndarray,
    b: float,
    x0: np.ndarray,
    perturbation: BasePerturbation,
    alpha: float,
    h_func: Callable[[float, float, float], float],
    *,
    optimizer_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Optimize the exact logistic objective using 1D/2D analytical reductions.

    Args:
        beta (np.ndarray): Model coefficients.
        b (float): Model intercept.
        x0 (np.ndarray): Baseline point.
        perturbation (BasePerturbation): Gaussian directional perturbation.
        alpha (float): Risk level of the lower-tail CVaR.
        h_func (Callable[[float, float, float], float]): Logistic CVaR functional.
        optimizer_kwargs (dict[str, Any] | None): Optional grid-search tuning options.
            Supported keys:

            - ``n_grid_points``: number of points of the outer 1-D grid over t
              (default: ``max(501, min(2000, 2 * int(||beta||)))``).
            - ``n_top_candidates``: number of best grid points refined by local
              zooming (default: 5).

    Returns:
        np.ndarray: The optimal unit direction vector.
    """
    opts: dict[str, Any] = {} if optimizer_kwargs is None else optimizer_kwargs

    if not isinstance(perturbation, GaussianPerturbation):
        msg = f"Unsupported perturbation type: {type(perturbation)}"
        raise UnsupportedPerturbationError(msg)

    if not is_isotropic(perturbation):
        msg = "Exact logistic optimization is only supported for isotropic Gaussian perturbations."
        raise UnsupportedPerturbationError(msg)

    eta0 = float(np.dot(beta, x0) + b)
    m_beta_env = float(perturbation.projected_env_mean(beta))
    v_beta_env = float(perturbation.projected_env_var(beta))
    baseline = _lambda(eta0)

    mu = float(cast("Any", perturbation.mu))
    sigma = float(cast("Any", perturbation.sigma))

    beta_norm = float(np.linalg.norm(beta))
    if beta_norm == 0:
        c = np.zeros_like(beta)
        c[0] = 1.0
        return c

    get_extrema = build_variance_profiler(beta)

    # Tuning parameters shared by both outer searches.
    n_grid = opts.get("n_grid_points", max(501, min(2000, 2 * int(beta_norm))))
    n_top_candidates = opts.get("n_top_candidates", 5)

    def outer_objective_smin(t: float) -> float:
        smin, _, _, _ = get_extrema(t)
        m = eta0 + m_beta_env + mu * t
        s_var = max(0.0, v_beta_env + sigma**2 * smin)
        s = float(np.sqrt(s_var))
        return -(h_func(m, s, alpha) - baseline)

    def outer_objective_smax(t: float) -> float:
        _, smax, _, _ = get_extrema(t)
        m = eta0 + m_beta_env + mu * t
        s_var = max(0.0, v_beta_env + sigma**2 * smax)
        s = float(np.sqrt(s_var))
        return -(h_func(m, s, alpha) - baseline)

    t_grid = np.linspace(-beta_norm, beta_norm, n_grid)
    search_margin = 4.0 * beta_norm / (n_grid - 1) if n_grid > 1 else 0.1

    def _dense_grid_search(obj_func: Callable[[float], float]) -> tuple[float, float]:
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

        if micro_f[m_best_idx] < best_f:  # pragma: no branch
            best_f = float(micro_f[m_best_idx])
            best_t = float(micro_t[m_best_idx])  # pragma: no branch

        return best_f, best_t

    best_f_smin, best_t_smin = _dense_grid_search(outer_objective_smin)

    if alpha <= 0.5:  # noqa: PLR2004
        # For alpha <= 0.5, H_alpha is non-increasing in s (see the monotonicity lemma for
        # H_alpha in the paper), so the optimum is attained at the minimum variance s_min.
        t_opt = best_t_smin
        _, _, c_opt, _ = get_extrema(t_opt)
    else:
        best_f_smax, best_t_smax = _dense_grid_search(outer_objective_smax)
        if best_f_smax < best_f_smin:
            t_opt = best_t_smax
            _, _, _, c_opt = get_extrema(t_opt)
        else:
            t_opt = best_t_smin
            _, _, c_opt, _ = get_extrema(t_opt)

    c_norm = float(np.linalg.norm(c_opt))
    if c_norm > 0:  # pragma: no branch
        c_opt /= c_norm

    return c_opt


def logistic_optimal_direction(
    beta: np.ndarray,
    b: float,
    x0: np.ndarray,
    perturbation: BasePerturbation,
    alpha: float,
    **kwargs: Any,  # noqa: ANN401
) -> np.ndarray:
    """Compute the optimal direction c* using the exact logistic objective.

    Args:
        beta (np.ndarray): Model coefficients.
        b (float): Model intercept.
        x0 (np.ndarray): Baseline point.
        perturbation (BasePerturbation): Gaussian directional perturbation.
        alpha (float): Risk level of the lower-tail CVaR.
        **kwargs (Any): Grid-search tuning options forwarded to the base optimizer
            (``n_grid_points`` and ``n_top_candidates``).

    Returns:
        np.ndarray: The optimal unit direction vector.

    Raises:
        UnsupportedPerturbationError: If perturbation is not Gaussian.
    """
    if isinstance(perturbation, GaussianPerturbation):
        return _logistic_optimal_direction_base(
            beta, b, x0, perturbation, alpha, logistic_h_alpha, optimizer_kwargs=kwargs
        )

    msg = f"Unsupported perturbation type: {type(perturbation)}"
    raise UnsupportedPerturbationError(msg)
