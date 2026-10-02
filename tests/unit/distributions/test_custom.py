# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Tests for the CustomPerturbation distribution."""

import numpy as np
import pytest

from dicex.distributions.base import BasePerturbation
from dicex.distributions.custom import CustomPerturbation
from dicex.exceptions import InvalidParameterError


def test_custom_sample() -> None:
    """Test that custom sample_fn is called correctly."""

    def my_sampler(n: int) -> np.ndarray:
        return np.ones(n) * 42.0

    dist = CustomPerturbation(sample_fn=my_sampler)

    # Without d
    samples = dist.sample(10)
    assert samples.shape == (10,)
    assert np.all(samples == 42.0)

    # With d, falls back to my_sampler(n) due to TypeError
    samples_d = dist.sample(10, d=5)
    assert samples_d.shape == (10,)
    assert np.all(samples_d == 42.0)


def test_custom_sample_with_d() -> None:
    """Test sample_fn that accepts d."""

    def my_sampler_d(n: int, d: int) -> np.ndarray:
        return np.ones((n, d)) * 24.0

    dist = CustomPerturbation(sample_fn=my_sampler_d)
    samples = dist.sample(10, d=3)
    assert samples.shape == (10, 3)
    assert np.all(samples == 24.0)


def test_custom_sample_env() -> None:
    """Test that sample_env uses sample_env_fn or falls back correctly."""

    def env_sampler(n: int, d: int) -> np.ndarray:
        return np.ones((n, d)) * 3.0

    # With sample_env_fn
    dist = CustomPerturbation(sample_fn=np.zeros, sample_env_fn=env_sampler)
    env_samples = dist.sample_env(5, 2)
    assert env_samples.shape == (5, 2)
    assert np.all(env_samples == 3.0)

    # Without sample_env_fn, mean 0, cov 0
    dist2 = CustomPerturbation(sample_fn=np.zeros)
    env_samples2 = dist2.sample_env(5, 2)
    assert env_samples2.shape == (5, 2)
    assert np.all(env_samples2 == 0.0)

    # Without sample_env_fn, custom mean and 1D cov
    dist3 = CustomPerturbation(sample_fn=np.zeros, env_mean=np.array([1.0, 2.0]), env_cov=np.array([0.1, 0.2]))
    env_samples3 = dist3.sample_env(5, 2)
    assert env_samples3.shape == (5, 2)

    # Without sample_env_fn, custom mean and 2D cov
    dist4 = CustomPerturbation(
        sample_fn=np.zeros, env_mean=np.array([1.0, 2.0]), env_cov=np.array([[0.1, 0.0], [0.0, 0.2]])
    )
    env_samples4 = dist4.sample_env(5, 2)
    assert env_samples4.shape == (5, 2)


def test_custom_sample_env_with_rng() -> None:
    """Test sample_env with explicit rng and no sample_env_fn."""
    dist = CustomPerturbation(sample_fn=np.zeros, env_mean=np.array([0.0, 0.0]))
    rng = np.random.default_rng(42)
    samples = dist.sample_env(5, 2, rng=rng)
    assert samples.shape == (5, 2)


def test_custom_sample_env_mean_mismatch() -> None:
    """Test that mismatched env_mean raises InvalidParameterError."""
    dist = CustomPerturbation(sample_fn=np.zeros, env_mean=np.array([1.0, 2.0]))
    with pytest.raises(InvalidParameterError, match="env_mean has length 2, but d=3"):
        dist.sample_env(5, 3)


def test_custom_params() -> None:
    """Test that params property works."""
    dist = CustomPerturbation(
        sample_fn=lambda n: np.random.default_rng(42).normal(size=n),
        params={"name": "dummy"},
    )
    assert dist.params == {"name": "dummy"}


def test_custom_params_with_env() -> None:
    """Test that params includes env_mean and env_cov when set."""
    env_mean = np.array([1.0, 2.0])
    env_cov = np.array([[0.1, 0.0], [0.0, 0.2]])
    dist = CustomPerturbation(sample_fn=np.zeros, env_mean=env_mean, env_cov=env_cov)
    params = dist.params
    assert np.array_equal(params["env_mean"], env_mean)
    assert np.array_equal(params["env_cov"], env_cov)


def test_custom_inheritance() -> None:
    """Test that CustomPerturbation is a BasePerturbation."""
    dist = CustomPerturbation(sample_fn=np.zeros)
    assert isinstance(dist, BasePerturbation)
