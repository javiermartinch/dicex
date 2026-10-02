# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Tests for the UniformPerturbation distribution."""

import numpy as np
import pytest

from dicex.distributions.base import BasePerturbation
from dicex.distributions.uniform import UniformPerturbation


def test_uniform_params() -> None:
    """Test that params property returns correct values."""
    mu, delta = 1.0, 0.5
    dist = UniformPerturbation(mu, delta)
    assert dist.params == {"mu": mu, "delta": delta}


def test_uniform_invalid_delta() -> None:
    """Test that negative delta raises ValueError."""
    with pytest.raises(ValueError, match="delta cannot be negative"):
        UniformPerturbation(0.0, -1.0)


def test_uniform_sample() -> None:
    """Test sampling results (shape and range)."""
    mu, delta = 5.0, 2.0
    dist = UniformPerturbation(mu, delta, seed=42)
    n = 1000
    samples = dist.sample(n)

    assert samples.shape == (n, 1)
    assert np.all(samples >= mu - delta)
    assert np.all(samples <= mu + delta)
    # Mean should be close to mu
    assert np.isclose(np.mean(samples), mu, atol=0.2)


def test_uniform_inheritance() -> None:
    """Test that UniformPerturbation is a BasePerturbation."""
    dist = UniformPerturbation(0.0, 1.0)
    assert isinstance(dist, BasePerturbation)


def test_uniform_projected_env() -> None:
    """Verify projected env operations for uniform (which has no env_cov)."""
    dist = UniformPerturbation(0.0, 1.0)
    beta = np.array([1.0, 2.0])
    assert dist.projected_env_mean(beta) == 0.0
    assert dist.projected_env_var(beta) == 0.0


def test_uniform_initialization_errors() -> None:
    """Verify ValueError for invalid init combinations."""
    with pytest.raises(ValueError, match="Half-width delta cannot be negative"):
        UniformPerturbation(mu=0.0, delta=np.array([-1.0, 1.0]))


def test_uniform_dim_resolution() -> None:
    """Verify dimensionality inference logic."""
    dist_d = UniformPerturbation(mu=0.0, delta=1.0, d=5)
    assert dist_d.sample(1).shape == (1, 5)

    dist_mu = UniformPerturbation(mu=np.array([0.0, 0.0, 0.0]), delta=1.0)
    assert dist_mu.sample(1).shape == (1, 3)

    dist_delta = UniformPerturbation(mu=0.0, delta=np.array([1.0, 1.0, 1.0]))
    assert dist_delta.sample(1).shape == (1, 3)

    dist_env = UniformPerturbation(mu=0.0, delta=1.0, env_mean=np.array([0.0, 0.0]))
    assert dist_env.sample(1).shape == (1, 2)

    dist_env_cov = UniformPerturbation(mu=0.0, delta=1.0, env_cov=np.eye(4))
    # _resolve_dim based on env_cov
    assert dist_env_cov.sample(1).shape == (1, 4)


def test_uniform_params_d_and_env() -> None:
    """Verify d, env_mean, and env_cov are present in params if specified."""
    dist = UniformPerturbation(mu=0.0, delta=1.0, d=5, env_mean=np.array([0.0]), env_cov=np.array([[1.0]]))
    params = dist.params
    assert params["d"] == 5
    assert "env_mean" in params
    assert "env_cov" in params


def test_uniform_sample_no_rng() -> None:
    """Verify sample and sample_env use default RNG when None is passed, and explicit RNG when passed."""
    dist = UniformPerturbation(mu=0.0, delta=1.0)
    samples = dist.sample(1, rng=None)
    assert samples.shape == (1, 1)

    env_samples = dist.sample_env(1, 1, rng=None)
    assert env_samples.shape == (1, 1)

    rng = np.random.default_rng(42)
    samples_with_rng = dist.sample(1, rng=rng)
    assert samples_with_rng.shape == (1, 1)
    env_samples_with_rng = dist.sample_env(1, 1, rng=rng)
    assert env_samples_with_rng.shape == (1, 1)

    # Test _resolve_dim fallback with explicit d
    d = 7
    assert dist._resolve_dim(d=d) == d  # pyright: ignore[reportPrivateUsage]
