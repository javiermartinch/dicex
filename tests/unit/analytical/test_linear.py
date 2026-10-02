# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Tests for exact linear evaluators."""

import numpy as np
import pytest

from dicex.analytical.linear import (
    directional_gaussian_mean_var,
    is_isotropic,
    linear_directional_functional,
    linear_optimal_direction,
)
from dicex.core.cvar import lower_tail_cvar_empirical
from dicex.core.functional import directional_improvement
from dicex.distributions.base import BasePerturbation
from dicex.distributions.gaussian import GaussianPerturbation
from dicex.distributions.uniform import UniformPerturbation
from dicex.exceptions import UnsupportedPerturbationError


class MockLinearModel:
    """Mock linear model for Monte Carlo comparison."""

    def __init__(self, beta: np.ndarray):
        self.beta = beta

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Linear prediction."""
        return np.dot(x, self.beta)


def test_linear_functional_vs_mc_gaussian() -> None:
    """Verify exact Gaussian formula against large-sample Monte Carlo (no env noise)."""
    beta = np.array([1.0, -0.5])
    c = np.array([0.6, 0.8])
    perturbation = GaussianPerturbation(mu=0.1, sigma=0.2, seed=42)
    alpha = 0.5

    # Exact value
    val_exact = linear_directional_functional(beta, perturbation, alpha, c)

    # Monte Carlo approximation
    model = MockLinearModel(beta)
    x0 = np.zeros(2)
    noise = perturbation.sample(200_000, d=len(beta))
    z_samples = directional_improvement(model, x0, c, noise)
    val_mc = lower_tail_cvar_empirical(z_samples, alpha)

    # Use 2% tolerance for MC noise
    assert np.isclose(val_exact, val_mc, rtol=0.02)


def test_linear_functional_vs_mc_gaussian_with_env_noise() -> None:
    """Verify exact Gaussian formula against MC with additive noise ε."""
    beta = np.array([1.0, -0.5])
    c = np.array([0.6, 0.8])
    env_mean = np.array([0.1, -0.2])
    env_cov = np.array([[0.5, 0.1], [0.1, 0.3]])
    perturbation = GaussianPerturbation(mu=0.1, sigma=0.2, env_mean=env_mean, env_cov=env_cov, seed=42)
    alpha = 0.5

    # Exact value
    val_exact = linear_directional_functional(beta, perturbation, alpha, c)

    # Monte Carlo: Z_i = M(x0 + T_i*c + ε_i) - M(x0)
    model = MockLinearModel(beta)
    x0 = np.zeros(2)
    rng = np.random.default_rng(42)
    n_mc = 500_000
    noise = perturbation.sample(n_mc, d=len(beta), rng=rng)
    env_noise = perturbation.sample_env(n_mc, 2, rng=rng)
    z_samples = directional_improvement(model, x0, c, noise, env_noise)
    val_mc = lower_tail_cvar_empirical(z_samples, alpha)

    assert np.isclose(val_exact, val_mc, rtol=0.02)


def test_linear_uniform_not_supported_exactly() -> None:
    """Uniform exact formulas were removed in the multidimensional rewrite."""

    with pytest.raises(UnsupportedPerturbationError):
        linear_directional_functional(
            np.array([1.0, -0.5]),
            UniformPerturbation(mu=0.4, delta=0.2, seed=42),
            0.5,
            np.array([0.6, 0.8]),
        )


def test_optimal_direction_norm() -> None:
    """Ensure the analytical optimal direction is always a unit vector."""
    beta = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    perturbation = GaussianPerturbation(0, 1)
    c_star = linear_optimal_direction(beta, perturbation, 0.5)
    assert np.isclose(np.linalg.norm(c_star), 1.0)


def test_optimal_direction_norm_with_env_noise() -> None:
    """Ensure the analytical optimal direction is a unit vector with env noise."""
    beta = np.array([1.0, 2.0, 3.0])
    env_cov = np.eye(3) * 0.5
    perturbation = GaussianPerturbation(0.1, 0.5, env_cov=env_cov)
    c_star = linear_optimal_direction(beta, perturbation, 0.5)
    assert np.isclose(np.linalg.norm(c_star), 1.0)


def test_optimal_direction_cases() -> None:
    """Verify optimal direction cases with different (mu, sigma) regimes."""
    beta = np.array([1.0, 0.0])

    # mu > lambda_alpha * sigma: c* points in the direction of beta.
    p1 = GaussianPerturbation(mu=10.0, sigma=0.1)
    c1 = linear_optimal_direction(beta, p1, 0.5)
    assert np.allclose(c1, [1.0, 0.0])

    # mu < -lambda_alpha * sigma: c* points in the direction of -beta.
    p2 = GaussianPerturbation(mu=-10.0, sigma=0.1)
    c2 = linear_optimal_direction(beta, p2, 0.5)
    assert np.allclose(c2, [-1.0, 0.0])

    # |mu| < lambda_alpha * sigma: c* is orthogonal to beta.
    p3 = GaussianPerturbation(mu=0.0, sigma=10.0)
    c3 = linear_optimal_direction(beta, p3, 0.5)
    assert np.isclose(np.dot(c3, beta), 0.0, atol=1e-7)
    assert np.isclose(np.linalg.norm(c3), 1.0)


def test_optimal_direction_with_env_noise() -> None:
    """Verify optimal direction with v_{beta,eps} > 0 (strictly concave f_alpha)."""
    beta = np.array([1.0, 0.0])
    env_cov = np.eye(2) * 1.0  # v_{β,ε} = 1.0

    # Strong positive mu → t* should be near ‖β‖
    p1 = GaussianPerturbation(mu=10.0, sigma=0.1, env_cov=env_cov)
    c1 = linear_optimal_direction(beta, p1, 0.5)
    assert np.dot(c1, beta) > 0.9  # Close to beta direction
    assert np.isclose(np.linalg.norm(c1), 1.0)

    # mu = 0 with env noise → t*_unc = 0, so direction is orthogonal to beta
    p2 = GaussianPerturbation(mu=0.0, sigma=0.5, env_cov=env_cov)
    c2 = linear_optimal_direction(beta, p2, 0.5)
    assert np.isclose(np.dot(c2, beta), 0.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(c2), 1.0)


def test_optimal_direction_zero_beta() -> None:
    """Edge case: flat model."""
    beta = np.zeros(3)
    perturbation = GaussianPerturbation(0, 1)
    c_star = linear_optimal_direction(beta, perturbation, 0.5)
    assert np.isclose(np.linalg.norm(c_star), 1.0)


def test_uniform_optimal_direction_case() -> None:
    """Verify Uniform perturbations raise UnsupportedPerturbationError."""
    beta = np.array([1.0, 0.0])

    with pytest.raises(UnsupportedPerturbationError):
        linear_optimal_direction(beta, UniformPerturbation(mu=1.0, delta=0.1), 0.5)


def test_unsupported_perturbation() -> None:
    """Verify error handling for custom perturbations."""

    class Custom(BasePerturbation):
        mu = 0.0

        def sample(
            self,
            n: int,
            d: int | None = None,
            rng: np.random.Generator | None = None,
        ) -> np.ndarray:
            del n, d, rng
            return np.array([])

        def sample_env(self, n: int, d: int, rng: np.random.Generator | None = None) -> np.ndarray:
            del rng
            return np.zeros((n, d))

        def pdf(self, t: np.ndarray | float) -> np.ndarray | float:
            del t
            return 0.0

        def cdf(self, t: np.ndarray | float) -> np.ndarray | float:
            del t
            return 0.0

        def quantile(self, u: np.ndarray | float) -> np.ndarray | float:
            del u
            return 0.0

        def projected_env_mean(self, beta: np.ndarray) -> float:
            return 0.0

        def projected_env_var(self, beta: np.ndarray) -> float:
            return 0.0

        @property
        def params(self) -> dict[str, float]:
            return {}

    with pytest.raises(UnsupportedPerturbationError, match="Unsupported perturbation type"):
        linear_directional_functional(np.ones(1), Custom(), 0.5, np.ones(1))


def test_directional_gaussian_mean_var_array_sigma() -> None:
    """Verify the projected mean and variance for a per-feature sigma vector."""
    beta = np.array([1.0, 0.5])
    c = np.array([0.6, 0.8])
    weights = beta * c
    pert = GaussianPerturbation(mu=0.0, sigma=np.array([0.2, 0.3]))
    mean, var = directional_gaussian_mean_var(pert, weights)
    assert isinstance(mean, float)
    assert isinstance(var, float)
    assert var > 0


def test_directional_gaussian_mean_var_vector_mu() -> None:
    """Verify the projected mean and variance for a vector-valued mean."""
    beta = np.array([1.0, 0.5])
    c = np.array([0.6, 0.8])
    weights = beta * c
    pert = GaussianPerturbation(mu=np.array([0.1, 0.2]), sigma=0.3)
    mean, var = directional_gaussian_mean_var(pert, weights)
    assert isinstance(mean, float)
    assert isinstance(var, float)


def test_directional_gaussian_mean_var_diag_cov() -> None:
    """Verify the projected mean and variance for a diagonal covariance given as a vector."""
    beta = np.array([1.0, 0.5])
    c = np.array([0.6, 0.8])
    weights = beta * c
    pert = GaussianPerturbation(mu=0.0, cov=np.array([0.04, 0.09]))
    mean, var = directional_gaussian_mean_var(pert, weights)
    assert isinstance(mean, float)
    assert isinstance(var, float)
    assert var > 0


def test_directional_gaussian_mean_var_full_cov() -> None:
    """Verify the projected mean and variance for a full covariance matrix."""
    beta = np.array([1.0, 0.5])
    c = np.array([0.6, 0.8])
    weights = beta * c
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    pert = GaussianPerturbation(mu=0.0, cov=cov)
    mean, var = directional_gaussian_mean_var(pert, weights)
    assert isinstance(mean, float)
    assert isinstance(var, float)
    assert var > 0


def test_is_isotropic_returns_false() -> None:
    """Verify that a vector mean or an explicit covariance is not isotropic."""
    # Vector mu → not isotropic
    pert1 = GaussianPerturbation(mu=np.array([0.1, 0.2]), sigma=0.5)
    assert not is_isotropic(pert1)
    # Non-None cov → not isotropic
    pert2 = GaussianPerturbation(mu=0.0, cov=np.array([0.04, 0.09]))
    assert not is_isotropic(pert2)


def test_linear_optimal_direction_non_isotropic() -> None:
    """Verify error for non-isotropic Gaussian perturbations."""
    beta = np.array([1.0, 0.5])
    # Vector mu → non-isotropic
    pert = GaussianPerturbation(mu=np.array([0.1, 0.2]), sigma=0.5)
    with pytest.raises(UnsupportedPerturbationError, match="isotropic"):
        linear_optimal_direction(beta, pert, 0.5)
    # Diagonal cov → non-isotropic
    pert2 = GaussianPerturbation(mu=0.0, cov=np.array([0.04, 0.09]))
    with pytest.raises(UnsupportedPerturbationError, match="isotropic"):
        linear_optimal_direction(beta, pert2, 0.5)


def test_optimal_direction_left_bracket_adjustment() -> None:
    """Trigger the dynamic left bracket adjustment in profiled_variance_smin.

    Uses beta with one small component so that beta_min_sq < 1, making the
    interval (sqrt(beta_min_sq), 1) non-empty.  For |t| in this range the
    secular equation f_left > 0 at the default epsilon, exercising the while
    loop that tightens the bracket.
    """
    beta = np.array([3.0, 0.1])
    perturbation = GaussianPerturbation(mu=0.0, sigma=1.0)
    c_star = linear_optimal_direction(beta, perturbation, 0.5)
    assert np.isclose(np.linalg.norm(c_star), 1.0)
