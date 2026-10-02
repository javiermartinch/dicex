# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Tests for the GaussianPerturbation distribution."""

import numpy as np
import pytest

from dicex.distributions.base import BasePerturbation
from dicex.distributions.gaussian import GaussianPerturbation


def test_gaussian_params() -> None:
    """Test that params property returns correct values."""
    mu, sigma = 1.0, 2.0
    dist = GaussianPerturbation(mu, sigma)
    assert dist.params == {"mu": mu, "sigma": sigma}


def test_gaussian_params_with_env() -> None:
    """Test that params includes env noise when configured."""
    mu, sigma = 1.0, 2.0
    env_mean = np.array([0.1, 0.2])
    env_cov = np.eye(2) * 0.5
    dist = GaussianPerturbation(mu, sigma, env_mean=env_mean, env_cov=env_cov)
    assert "env_mean" in dist.params
    assert "env_cov" in dist.params


def test_gaussian_invalid_sigma() -> None:
    """Test that negative sigma raises ValueError."""
    with pytest.raises(ValueError, match="sigma cannot be negative"):
        GaussianPerturbation(0.0, -1.0)


def test_gaussian_sample() -> None:
    """Test sampling results (shape and approximate moments)."""
    mu, sigma = 10.0, 1.0
    dist = GaussianPerturbation(mu, sigma, seed=42)
    n = 10000
    samples = dist.sample(n)

    assert samples.shape == (n, 1)
    # Higher tolerance for mean/std because of sampling noise
    assert np.isclose(np.mean(samples), mu, atol=0.1)
    assert np.isclose(np.std(samples), sigma, atol=0.1)


def test_gaussian_sample_env_zero_cov() -> None:
    """Test env noise sampling with zero covariance (constant at mean)."""
    env_mean = np.array([1.0, 2.0, 3.0])
    dist = GaussianPerturbation(0.0, 1.0, env_mean=env_mean)
    samples = dist.sample_env(100, 3)
    assert samples.shape == (100, 3)
    # All samples should be at the mean
    assert np.allclose(samples, env_mean)


def test_gaussian_sample_env_no_config() -> None:
    """Test env noise sampling with no env config (all zeros)."""
    dist = GaussianPerturbation(0.0, 1.0)
    samples = dist.sample_env(100, 3)
    assert samples.shape == (100, 3)
    assert np.allclose(samples, 0.0)


def test_gaussian_sample_env_nonzero_cov() -> None:
    """Test env noise sampling with non-zero covariance."""
    env_mean = np.array([1.0, 2.0])
    env_cov = np.array([[1.0, 0.0], [0.0, 0.5]])
    dist = GaussianPerturbation(0.0, 1.0, env_mean=env_mean, env_cov=env_cov, seed=42)
    n = 50000
    samples = dist.sample_env(n, 2)
    assert samples.shape == (n, 2)
    assert np.isclose(np.mean(samples[:, 0]), 1.0, atol=0.1)
    assert np.isclose(np.mean(samples[:, 1]), 2.0, atol=0.1)
    assert np.isclose(np.var(samples[:, 0]), 1.0, atol=0.1)
    assert np.isclose(np.var(samples[:, 1]), 0.5, atol=0.1)


def test_gaussian_projected_env_mean() -> None:
    """Test m_{β,ε} = β^T m_ε."""
    env_mean = np.array([1.0, 2.0, 3.0])
    dist = GaussianPerturbation(0.0, 1.0, env_mean=env_mean)
    beta = np.array([1.0, 0.5, -1.0])
    # Expected: 1*1 + 0.5*2 + (-1)*3 = 1 + 1 - 3 = -1
    assert np.isclose(dist.projected_env_mean(beta), -1.0)


def test_gaussian_projected_env_var() -> None:
    """Test v_{β,ε} = β^T Σ_ε β."""
    env_cov = np.array([[2.0, 0.0], [0.0, 3.0]])
    dist = GaussianPerturbation(0.0, 1.0, env_cov=env_cov)
    beta = np.array([1.0, 1.0])
    # Expected: [1,1] @ [[2,0],[0,3]] @ [1,1]^T = 2 + 3 = 5
    assert np.isclose(dist.projected_env_var(beta), 5.0)


def test_gaussian_projected_env_no_config() -> None:
    """Test projected values return 0 when env noise not configured."""
    dist = GaussianPerturbation(0.0, 1.0)
    beta = np.array([1.0, 2.0])
    assert dist.projected_env_mean(beta) == 0.0
    assert dist.projected_env_var(beta) == 0.0


def test_gaussian_inheritance() -> None:
    """Test that GaussianPerturbation is a BasePerturbation."""
    dist = GaussianPerturbation(0.0, 1.0)
    assert isinstance(dist, BasePerturbation)


def test_gaussian_full_covariance_sample() -> None:
    """A single class should also support full multivariate covariance."""
    mu = np.array([1.0, -1.0])
    cov = np.array([[2.0, 0.4], [0.4, 1.0]])
    dist = GaussianPerturbation(mu=mu, cov=cov, seed=42)

    samples = dist.sample(20000)

    assert samples.shape == (20000, 2)
    assert np.allclose(np.mean(samples, axis=0), mu, atol=0.1)
    assert np.allclose(np.cov(samples, rowvar=False), cov, atol=0.15)


def test_gaussian_sigma_zero() -> None:
    """Verify behavior when sigma is 0."""
    mu = 5.0
    dist = GaussianPerturbation(mu, 0.0)
    samples = dist.sample(100)
    assert np.allclose(samples, mu)
    # A zero scale yields a point mass at mu, so the projected variance is zero.
    assert dist.projected_env_var(np.array([1.0])) == 0.0


def test_gaussian_near_singular_env_cov() -> None:
    """Verify projected env variance with near singular covariance."""
    env_cov = np.array([[1.0, 1.0], [1.0, 1.0]])  # perfectly correlated
    dist = GaussianPerturbation(0.0, 1.0, env_cov=env_cov)
    beta_null = np.array([1.0, -1.0])
    assert np.isclose(dist.projected_env_var(beta_null), 0.0)
    beta_max = np.array([1.0, 1.0])
    assert np.isclose(dist.projected_env_var(beta_max), 4.0)


def test_gaussian_initialization_errors() -> None:
    """Verify ValueError for invalid init combinations."""
    with pytest.raises(ValueError, match="Exactly one of 'sigma' or 'cov'"):
        GaussianPerturbation(mu=0.0, sigma=1.0, cov=np.eye(2))
    with pytest.raises(ValueError, match="Exactly one of 'sigma' or 'cov'"):
        GaussianPerturbation(mu=0.0, sigma=None, cov=None)
    with pytest.raises(ValueError, match="cov must be a vector or matrix"):
        GaussianPerturbation(mu=0.0, cov=np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="Diagonal covariance entries cannot be negative"):
        GaussianPerturbation(mu=0.0, cov=np.array([-1.0, 1.0]))


def test_gaussian_dim_resolution() -> None:
    """Verify dimensionality inference logic."""
    dist_d = GaussianPerturbation(mu=0.0, sigma=1.0, d=5)
    assert dist_d.sample(1).shape == (1, 5)

    dist_sigma = GaussianPerturbation(mu=0.0, sigma=np.array([1.0, 1.0, 1.0]))
    assert dist_sigma.sample(1).shape == (1, 3)

    dist_env = GaussianPerturbation(mu=0.0, sigma=1.0, env_mean=np.array([0.0, 0.0]))
    assert dist_env.sample(1).shape == (1, 2)


def test_gaussian_mean_sigma_cov_errors() -> None:
    """Verify mismatched dimensions and inaccessible parameters."""
    dist_cov = GaussianPerturbation(mu=np.array([0.0, 0.0]), cov=np.eye(2))
    with pytest.raises(ValueError, match="sigma is not available"):
        dist_cov._sigma_vector(2)  # pyright: ignore[reportPrivateUsage]

    dist_mu_mismatch = GaussianPerturbation(mu=np.array([0.0, 0.0]), sigma=1.0)
    with pytest.raises(ValueError, match="mu has dimension 2 but expected 3"):
        dist_mu_mismatch.sample(1, d=3)

    dist_sigma_mismatch = GaussianPerturbation(mu=0.0, sigma=np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="sigma has dimension 2 but expected 3"):
        dist_sigma_mismatch.sample(1, d=3)

    dist_cov_mismatch = GaussianPerturbation(mu=0.0, cov=np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="cov has dimension 2 but expected 3"):
        dist_cov_mismatch.sample(1, d=3)


def test_gaussian_params_d() -> None:
    """Verify d is present in params if specified."""
    dist = GaussianPerturbation(mu=0.0, sigma=1.0, d=5)
    assert dist.params["d"] == 5


def test_gaussian_params_env() -> None:
    """Verify params includes env_mean and env_cov when set."""
    env_mean = np.array([1.0, 2.0])
    env_cov = np.array([[0.1, 0.0], [0.0, 0.2]])
    dist = GaussianPerturbation(mu=1.0, sigma=0.5, env_mean=env_mean, env_cov=env_cov, d=2)
    params = dist.params
    assert np.array_equal(params["env_mean"], env_mean)
    assert np.array_equal(params["env_cov"], env_cov)


def test_gaussian_params_cov() -> None:
    """Verify params includes cov when the Gaussian is cov-parameterized."""
    cov = np.array([[2.0, 0.5], [0.5, 3.0]])
    dist = GaussianPerturbation(mu=0.0, cov=cov)
    params = dist.params
    assert np.array_equal(params["cov"], cov)
    assert "sigma" not in params


def test_gaussian_cov_inference_and_sampling() -> None:
    """Verify dim inference from cov and 1D cov sampling."""
    # 2D cov dim inference
    dist_cov2d = GaussianPerturbation(mu=0.0, cov=np.eye(4))
    assert dist_cov2d._resolve_dim() == 4  # pyright: ignore[reportPrivateUsage]

    # 1D cov sampling
    dist_cov1d = GaussianPerturbation(mu=0.0, cov=np.array([1.0, 4.0, 9.0]))
    # The dimension 3 is inferred from the covariance.
    samples = dist_cov1d.sample(1)
    assert samples.shape == (1, 3)
