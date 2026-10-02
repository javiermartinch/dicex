# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Tests for exact logistic evaluators."""

import numpy as np
import pytest
from scipy.integrate import quad  # type: ignore[reportUnknownVariableType]
from scipy.stats import norm

from dicex.analytical.logistic import (
    _log_diff_cdf,  # pyright: ignore[reportPrivateUsage]
    _logistic_h_alpha_quad,  # pyright: ignore[reportPrivateUsage]
    _logistic_integral_series,  # pyright: ignore[reportPrivateUsage]
    logistic_directional_functional,
    logistic_h_alpha,
    logistic_h_alpha_series,
    logistic_optimal_direction,
)
from dicex.core.cvar import lower_tail_cvar_empirical
from dicex.core.functional import directional_improvement
from dicex.distributions.gaussian import GaussianPerturbation
from dicex.distributions.uniform import UniformPerturbation
from dicex.exceptions import UnsupportedPerturbationError


def test_logistic_h_alpha_vs_quad() -> None:
    """Verify series expansion against numerical quadrature (Left Branch)."""
    m, s, alpha = 0.5, 0.2, 0.1

    # Series expansion
    val_series = logistic_h_alpha(m, s, alpha)

    # Numerical integration
    q_alpha = float(norm.ppf(alpha))  # type: ignore[reportUnknownMemberType]

    def integrand(z: float) -> float:
        return (1.0 / (1.0 + np.exp(-(m + s * z)))) * float(norm.pdf(z))  # type: ignore[reportUnknownMemberType]

    val_quad, _ = quad(integrand, -np.inf, q_alpha)  # type: ignore[reportUnknownVariableType]
    val_quad = float(val_quad) / alpha  # type: ignore[reportUnknownArgumentType]

    assert np.isclose(val_series, val_quad, atol=1e-3)


def test_logistic_h_alpha_right_branch() -> None:
    """Verify series expansion against numerical quadrature (Right Branch)."""
    m, s, alpha = -0.5, 1.0, 0.8
    val_series = logistic_h_alpha(m, s, alpha)

    q_alpha = float(norm.ppf(alpha))  # type: ignore[reportUnknownMemberType]

    def integrand(z: float) -> float:
        # Use a safe exp to avoid overflow in quad
        val = m + s * z
        if val > 700:
            return float(norm.pdf(z))  # type: ignore[reportUnknownMemberType]
        if val < -700:
            return 0.0
        return (1.0 / (1.0 + np.exp(-val))) * float(norm.pdf(z))  # type: ignore[reportUnknownMemberType]

    val_quad, _ = quad(integrand, -np.inf, q_alpha)  # type: ignore[reportUnknownVariableType]
    val_quad = float(val_quad) / alpha  # type: ignore[reportUnknownArgumentType]

    assert np.isclose(val_series, val_quad, atol=1e-5)


def test_logistic_functional_vs_mc() -> None:
    """Verify exact logistic functional against large-sample Monte Carlo (no env noise)."""
    beta = np.array([0.5, -0.2])
    b = 0.1
    x0 = np.array([1.0, 2.0])
    c = np.array([1.0, 0.0])
    mu, sigma = 0.1, 0.3
    perturbation = GaussianPerturbation(mu=mu, sigma=sigma, seed=42)
    alpha = 0.2

    # Exact calculation
    val_exact = logistic_directional_functional(beta, b, x0, perturbation, alpha, c)

    # Monte Carlo approximation
    class LogisticModel:
        def predict(self, x: np.ndarray) -> np.ndarray:
            logits = np.dot(x, beta) + b
            return 1.0 / (1.0 + np.exp(-logits))

    model = LogisticModel()

    rng = np.random.default_rng(42)
    delta_samples = perturbation.sample(500_000, d=len(beta), rng=rng)
    z_samples = directional_improvement(model, x0, c, delta_samples)
    val_mc = lower_tail_cvar_empirical(z_samples, alpha)

    assert np.isclose(val_exact, val_mc, rtol=0.03)


def test_logistic_functional_vs_mc_with_env_noise() -> None:
    """Verify exact logistic functional against MC with additive noise ε."""
    beta = np.array([0.5, -0.2])
    b = 0.1
    x0 = np.array([1.0, 2.0])
    c = np.array([1.0, 0.0])
    mu, sigma = 0.1, 0.3
    env_mean = np.array([0.05, -0.1])
    env_cov = np.array([[0.3, 0.05], [0.05, 0.2]])
    perturbation = GaussianPerturbation(mu=mu, sigma=sigma, env_mean=env_mean, env_cov=env_cov, seed=42)
    alpha = 0.2

    val_exact = logistic_directional_functional(beta, b, x0, perturbation, alpha, c)

    class LogisticModel:
        def predict(self, x: np.ndarray) -> np.ndarray:
            logits = np.dot(x, beta) + b
            return 1.0 / (1.0 + np.exp(-logits))

    model = LogisticModel()

    rng = np.random.default_rng(42)
    n_mc = 500_000
    scalar_noise = perturbation.sample(n_mc, d=len(beta), rng=rng)
    env_noise = rng.multivariate_normal(env_mean, env_cov, n_mc)
    z_samples = directional_improvement(model, x0, c, scalar_noise, env_noise)
    val_mc = lower_tail_cvar_empirical(z_samples, alpha)

    assert np.isclose(val_exact, val_mc, rtol=0.02)


def test_logistic_uniform_not_supported_exactly() -> None:
    """Uniform exact formulas were removed in the multidimensional rewrite."""

    with pytest.raises(UnsupportedPerturbationError):
        logistic_directional_functional(
            np.array([0.5, -0.2]),
            0.1,
            np.array([1.0, 2.0]),
            UniformPerturbation(mu=0.1, delta=0.3, seed=42),
            0.2,
            np.array([1.0, 0.0]),
        )


def test_logistic_optimal_direction() -> None:
    """Verify optimal direction via 1D reduction."""
    beta = np.array([1.0, 0.0])
    b = 0.0
    x0 = np.array([0.0, 0.0])
    # Clear improvement in beta direction
    perturbation = GaussianPerturbation(mu=5.0, sigma=0.1)
    alpha = 0.5

    c_star = logistic_optimal_direction(beta, b, x0, perturbation, alpha)
    assert np.allclose(c_star, [1.0, 0.0], atol=1e-2)
    assert np.isclose(np.linalg.norm(c_star), 1.0)


def test_logistic_optimal_direction_with_env_noise() -> None:
    """Verify optimal direction with additive noise."""
    beta = np.array([1.0, 0.0])
    b = 0.0
    x0 = np.array([0.0, 0.0])
    env_cov = np.eye(2) * 0.1
    perturbation = GaussianPerturbation(mu=5.0, sigma=0.1, env_cov=env_cov)
    alpha = 0.5

    c_star = logistic_optimal_direction(beta, b, x0, perturbation, alpha)
    # With strong positive mu, direction should still be close to beta
    assert np.dot(c_star, beta / np.linalg.norm(beta)) > 0.9
    assert np.isclose(np.linalg.norm(c_star), 1.0)


def test_logistic_optimal_direction_uniform() -> None:
    """Uniform exact optimization is unsupported after the Gaussian-only rewrite."""

    with pytest.raises(UnsupportedPerturbationError):
        logistic_optimal_direction(
            np.array([1.0, 0.0]),
            0.0,
            np.array([0.0, 0.0]),
            UniformPerturbation(mu=5.0, delta=0.1),
            0.5,
        )


def test_log_diff_cdf_branches() -> None:
    """Verify the log CDF difference for reversed, positive, and negative arguments."""
    # Reversed arguments imply a negative difference, whose logarithm is -inf.
    assert _log_diff_cdf(1.0, 2.0) == -np.inf

    # By symmetry of the normal distribution, both positive and both negative
    # arguments yield the same value.
    res1 = _log_diff_cdf(2.0, 1.0)
    res2 = _log_diff_cdf(-1.0, -2.0)
    assert np.isclose(res1, res2)


def test_logistic_h_alpha_series_branches() -> None:
    """Verify the series evaluation of H_alpha across its case analysis."""
    # Nonpositive alpha.
    assert logistic_h_alpha_series(0.0, 1.0, 0.0) == 0.0

    # Scale s close to zero.
    assert logistic_h_alpha_series(0.0, 0.0, 0.5) == 0.5

    # Alpha of at least one.
    assert logistic_h_alpha_series(0.0, 1.0, 1.5) > 0.0

    # Left branch only (q <= z0): with m=0 and s=1, z0 = 0 and q_alpha = -1.28 for alpha = 0.1.
    res_left = logistic_h_alpha_series(0.0, 1.0, 0.1, n_terms=5)
    assert res_left > 0.0

    # Split at z0 (q > z0): with m=0 and s=1, z0 = 0 and q_alpha = 1.28 for alpha = 0.9.
    res_right = logistic_h_alpha_series(0.0, 1.0, 0.9, n_terms=5)
    assert res_right > 0.0


def test_logistic_alpha_greater_than_half() -> None:
    """Verify 2D reduction branch when alpha > 0.5."""
    beta = np.array([1.0, 0.5])
    b = 0.1
    x0 = np.array([0.0, 0.0])
    perturbation = GaussianPerturbation(mu=2.0, sigma=0.5)
    alpha = 0.8  # triggers 2D optimization

    c_star = logistic_optimal_direction(beta, b, x0, perturbation, alpha)
    assert np.isclose(np.linalg.norm(c_star), 1.0)
    # The optimal direction should still be reasonably aligned with beta
    assert np.dot(c_star, beta) > 0


def test_logistic_with_non_isotropic_env_noise() -> None:
    """Verify that isotropic directional noise with non-isotropic environmental noise works."""
    beta = np.array([1.0, -1.0])
    b = 0.0
    x0 = np.array([0.5, 0.5])
    env_cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    perturbation = GaussianPerturbation(mu=1.0, sigma=0.1, env_cov=env_cov)
    alpha = 0.2

    c_star = logistic_optimal_direction(beta, b, x0, perturbation, alpha)
    assert np.isclose(np.linalg.norm(c_star), 1.0)


def test_logistic_non_isotropic_directional_noise_raises() -> None:
    """Verify that non-isotropic Gaussian directional noise raises UnsupportedPerturbationError."""
    beta = np.array([1.0, -1.0])
    b = 0.0
    x0 = np.array([0.5, 0.5])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    perturbation = GaussianPerturbation(mu=1.0, cov=cov)
    alpha = 0.2

    with pytest.raises(
        UnsupportedPerturbationError,
        match="Exact logistic optimization is only supported for isotropic Gaussian perturbations",
    ):
        logistic_optimal_direction(beta, b, x0, perturbation, alpha)


def test_logistic_alpha_greater_than_one() -> None:
    """Verify behavior for alpha >= 1.0 in h_alpha and series."""
    # alpha=1.0 should map to expectation over the whole line
    val_1 = logistic_h_alpha(0.0, 1.0, 1.0)
    val_series_1 = logistic_h_alpha_series(0.0, 1.0, 1.0)
    assert val_1 > 0
    assert val_series_1 == val_1


def test_logistic_beta_zero() -> None:
    """Verify behavior when beta is identically zero."""
    beta = np.array([0.0, 0.0])
    b = 0.0
    x0 = np.array([0.0, 0.0])
    perturbation = GaussianPerturbation(mu=1.0, sigma=1.0)
    alpha = 0.5

    c_star = logistic_optimal_direction(beta, b, x0, perturbation, alpha)
    assert np.isclose(np.linalg.norm(c_star), 1.0)


def test_logistic_h_alpha_quad_early_returns() -> None:
    """Verify the degenerate-case values returned by the quadrature evaluation of H_alpha."""
    # alpha <= 0 branch
    assert _logistic_h_alpha_quad(1.0, 1.0, 0.0) == 0.0
    assert _logistic_h_alpha_quad(1.0, 1.0, -0.5) == 0.0

    # s close to 0 branch: returns _lambda(m)
    val = _logistic_h_alpha_quad(0.5, 0.0, 0.3)
    expected = 1.0 / (1.0 + np.exp(-0.5))
    assert np.isclose(val, expected)

    # alpha >= 1.0 branch: whole-line quadrature
    val = _logistic_h_alpha_quad(0.0, 1.0, 1.0)
    assert val > 0.0

    # Extreme alpha where q_alpha <= -8 (alpha very small)
    val = _logistic_h_alpha_quad(0.0, 1.0, 1e-100)
    assert val == 0.0


def test_logistic_integral_series_s_zero() -> None:
    """Verify that the series integral reduces to Phi(q) * Lambda(m) when s is zero."""
    # s = 0 early return: cdf(q) * _lambda(m)
    val = _logistic_integral_series(1.0, 0.0, 0.0, n_terms=5)
    expected = float(norm.cdf(0.0)) / (1.0 + np.exp(-1.0))  # type: ignore[reportUnknownMemberType]
    assert np.isclose(val, expected)


def test_logistic_h_alpha_series_edge_alphas() -> None:
    """Verify the series evaluation of H_alpha at boundary values of alpha and s."""
    # alpha <= 0 branch
    assert logistic_h_alpha_series(1.0, 1.0, 0.0) == 0.0

    # s close to 0 branch: returns _lambda(m)
    val = logistic_h_alpha_series(0.5, 0.0, 0.3)
    expected = 1.0 / (1.0 + np.exp(-0.5))
    assert np.isclose(val, expected)

    # alpha >= 1.0 branch: delegates to whole-line quad
    val = logistic_h_alpha_series(0.0, 1.0, 1.0)
    assert val > 0.0
